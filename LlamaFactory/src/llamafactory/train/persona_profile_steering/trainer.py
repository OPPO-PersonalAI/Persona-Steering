# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expressed or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are modifications by OPPO Inc.
# Licensed under the Apache License, Version 2.0.
"""
Custom Trainer for PersonaSteering Model
Handles user encoder + steering vector + decoder forward pass
Based on trainer_reconstruction.py logic
"""

from typing import TYPE_CHECKING, Any, Optional, Union
import json
import os
import logging
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import PredictionOutput, PREFIX_CHECKPOINT_DIR, TrainOutput
from transformers.trainer import TRAINER_STATE_NAME
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME

from ...extras import logging as llamafactory_logging
from ...extras.constants import IGNORE_INDEX

# Optional SwanLab integration
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    swanlab = None

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments

logger = llamafactory_logging.get_logger(__name__)


class PersonaSteeringTrainer(Trainer):
    """Custom trainer for PersonaSteering model with user encoder and steering vector."""
    
    TWO_STAGE_STATE_NAME = "two_stage_state.json"
    
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        user_tokenizer: Optional["PreTrainedTokenizer"] = None,
        model_args: Optional["ModelArguments"] = None,
        use_two_stage_training: bool = False,
        mlp_only_epochs: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.user_tokenizer = user_tokenizer
        self.model_args = model_args
        
        # Two-stage training configuration
        self.use_two_stage_training = use_two_stage_training
        self.mlp_only_epochs = mlp_only_epochs
        self.training_stage = "mlp_only" if use_two_stage_training else "all"
        self._checkpoint_step_offset = 0
        self._current_two_stage_mlp_steps: Optional[int] = None
        self._setup_file_logging()
        
        # If two-stage training is enabled, freeze user encoder initially
        if self.use_two_stage_training:
            logger.info_rank0("\n" + "=" * 60)
            logger.info_rank0("Two-stage training enabled:")
            logger.info_rank0(f"  Stage 1: Train MLP only (freeze encoder) for {self.mlp_only_epochs} epoch(s)")
            logger.info_rank0(f"  Stage 2: Train all parameters (unfreeze encoder) for remaining epochs")
            logger.info_rank0("=" * 60)
            unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(unwrapped_model, 'freeze_user_encoder'):
                unwrapped_model.freeze_user_encoder()
        # When two-stage is disabled, ensure user_encoder is trainable (explicit unfreeze).
        # Important for Qwen/BERT + LoRA: LoRA params will get requires_grad=True here.
        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        if not self.use_two_stage_training and hasattr(unwrapped_model, 'unfreeze_user_encoder'):
            unwrapped_model.unfreeze_user_encoder()
            lora_hint = " (including LoRA adapters)" if getattr(self.model, 'user_encoder_use_lora', False) else ""
            logger.info_rank0(f"Single-stage training: user_encoder{lora_hint} is set trainable.")

        # ✅ CRITICAL FIX: Ensure gradients can flow through the frozen decoder
        # When the decoder is frozen, PyTorch optimizes by not computing gradients for inputs 
        # unless explicitly told to do so. We need this for the steering vector gradients.
        if hasattr(self.model, "decoder") and self.model.decoder is not None:
            # Make sure we can call the method
            if hasattr(self.model.decoder, "enable_input_require_grads"):
                self.model.decoder.enable_input_require_grads()
                logger.info_rank0("✅ Enabled input_require_grads for frozen decoder to maintain gradient flow to steering vector")
            else:
                logger.warning_rank0("⚠️ Warning: decoder does not have enable_input_require_grads method. Gradients might be broken.")

        # Gradients into user encoder (LoRA / gradient checkpointing)
        if hasattr(self.model, "user_encoder") and self.model.user_encoder is not None:
            ue = self.model.user_encoder
            # HF backbone on UserEncoder (.model)
            hf_encoder = getattr(ue, "model", None)
            
            if hf_encoder and hasattr(hf_encoder, "enable_input_require_grads"):
                hf_encoder.enable_input_require_grads()
                logger.info_rank0("✅ Enabled input_require_grads for User Encoder (ensures gradient flow to LoRA)")
            elif hasattr(ue, "enable_input_require_grads"):
                ue.enable_input_require_grads()

    def _two_stage_state_path(self, target_dir: str) -> str:
        return os.path.join(target_dir, self.TWO_STAGE_STATE_NAME)

    def _write_two_stage_state(
        self,
        target_dir: str,
        *,
        training_stage: str,
        mlp_steps: int,
        stage1_completed: bool,
        checkpoint_global_step: Optional[int] = None,
        checkpoint_step_offset: Optional[int] = None,
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        payload = {
            "training_stage": training_stage,
            "mlp_steps": int(mlp_steps),
            "stage1_completed": bool(stage1_completed),
        }
        if checkpoint_global_step is not None:
            payload["checkpoint_global_step"] = int(checkpoint_global_step)
        if checkpoint_step_offset is not None:
            payload["checkpoint_step_offset"] = int(checkpoint_step_offset)

        with open(self._two_stage_state_path(target_dir), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _read_two_stage_state(self, target_dir: str) -> Optional[dict[str, Any]]:
        state_path = self._two_stage_state_path(target_dir)
        if not os.path.isfile(state_path):
            return None

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"⚠️  Failed to read two-stage state from {state_path}: {e}")

        return None

    def _load_checkpoint_global_step(self, checkpoint_path: str) -> Optional[int]:
        state_path = os.path.join(checkpoint_path, TRAINER_STATE_NAME)
        if not os.path.isfile(state_path):
            return None

        try:
            state = TrainerState.load_from_json(state_path)
            return int(state.global_step)
        except Exception:
            pass

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
            if isinstance(state_dict, dict) and "global_step" in state_dict:
                return int(state_dict["global_step"])
        except Exception as e:
            logger.warning(f"⚠️  Failed to read trainer state from {state_path}: {e}")

        return None

    def _resolve_two_stage_resume(
        self,
        resume_from_checkpoint: Optional[str],
        mlp_steps: int,
    ) -> dict[str, Any]:
        if not resume_from_checkpoint:
            return {"mode": "fresh", "checkpoint": None, "metadata": None}

        metadata = self._read_two_stage_state(resume_from_checkpoint)
        if metadata is not None:
            stage = metadata.get("training_stage")
            stage1_completed = bool(metadata.get("stage1_completed", stage == "all"))
            if stage == "all":
                return {"mode": "stage2_resume", "checkpoint": resume_from_checkpoint, "metadata": metadata}
            if stage == "mlp_only" and stage1_completed:
                return {"mode": "stage2_from_stage1", "checkpoint": resume_from_checkpoint, "metadata": metadata}
            return {"mode": "stage1_resume", "checkpoint": resume_from_checkpoint, "metadata": metadata}

        # Backward compatibility for checkpoints created before two-stage metadata existed.
        global_step = self._load_checkpoint_global_step(resume_from_checkpoint)
        if global_step is None:
            logger.warning(
                f"⚠️  Checkpoint {resume_from_checkpoint} has no {self.TWO_STAGE_STATE_NAME}; "
                "falling back to Stage 1 resume semantics."
            )
            return {"mode": "stage1_resume", "checkpoint": resume_from_checkpoint, "metadata": None}

        if global_step >= mlp_steps:
            logger.warning(
                f"⚠️  Checkpoint {resume_from_checkpoint} has no {self.TWO_STAGE_STATE_NAME}; "
                f"global_step={global_step} >= mlp_steps={mlp_steps}, treating it as Stage 2 initialization."
            )
            return {"mode": "stage2_from_stage1", "checkpoint": resume_from_checkpoint, "metadata": None}

        logger.warning(
            f"⚠️  Checkpoint {resume_from_checkpoint} has no {self.TWO_STAGE_STATE_NAME}; "
            f"global_step={global_step} < mlp_steps={mlp_steps}, treating it as Stage 1 resume."
        )
        return {"mode": "stage1_resume", "checkpoint": resume_from_checkpoint, "metadata": None}

    def _load_steering_weights_only(self, checkpoint_path: str) -> None:
        steering_model_path = os.path.join(checkpoint_path, "steering_model.pt")
        if not os.path.isfile(steering_model_path):
            raise FileNotFoundError(f"Checkpoint does not contain steering weights: {steering_model_path}")

        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = torch.load(
            steering_model_path,
            map_location=self.accelerator.device if hasattr(self, "accelerator") else "cpu"
        )

        required_keys = ["user_encoder", "steering_generator"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        unwrapped_model.user_encoder.load_state_dict(checkpoint["user_encoder"], strict=False)
        unwrapped_model.steering_generator.load_state_dict(checkpoint["steering_generator"], strict=False)
        logger.info_rank0(f"✅ Loaded steering weights from {checkpoint_path} for Stage 2 initialization.")

    def _save_stage_boundary_checkpoint(
        self,
        *,
        trial: Any,
        mlp_steps: int,
    ) -> Optional[str]:
        if not getattr(self.args, "should_save", True):
            return None

        output_dir = self._get_output_dir(trial=trial)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{int(self.state.global_step)}"
        checkpoint_dir = os.path.join(output_dir, checkpoint_folder)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.save_model(checkpoint_dir, _internal_call=True)

        if hasattr(self, "_save_optimizer_and_scheduler"):
            self._save_optimizer_and_scheduler(checkpoint_dir)
        else:
            if getattr(self, "optimizer", None) is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            if getattr(self, "lr_scheduler", None) is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

        if hasattr(self, "_save_rng_state"):
            self._save_rng_state(checkpoint_dir)

        try:
            self.state.save_to_json(os.path.join(checkpoint_dir, TRAINER_STATE_NAME))
        except Exception:
            pass

        self._write_two_stage_state(
            checkpoint_dir,
            training_stage="mlp_only",
            mlp_steps=mlp_steps,
            stage1_completed=True,
            checkpoint_global_step=int(self.state.global_step),
            checkpoint_step_offset=0,
        )
        self._write_two_stage_state(
            self.args.output_dir,
            training_stage="all",
            mlp_steps=mlp_steps,
            stage1_completed=True,
            checkpoint_global_step=int(self.state.global_step),
            checkpoint_step_offset=mlp_steps,
        )

        if hasattr(self, "_rotate_checkpoints"):
            self._rotate_checkpoints(use_mtime=False, output_dir=output_dir)

        logger.info_rank0(f"✅ Saved Stage 1 boundary checkpoint to {checkpoint_dir}")
        return checkpoint_dir
       
    def _setup_file_logging(self):
        """Setup file logging to save training logs."""
        try:
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "training.log")
            

            class FlushFileHandler(logging.FileHandler):
                def emit(self, record):
                    super().emit(record)
                    self.flush()
            
            # File logging (LlamaFactory loggers often do not propagate)

            current_logger = llamafactory_logging.get_logger(__name__)
            has_current_handler = any(
                isinstance(h, logging.FileHandler) and 
                hasattr(h, 'baseFilename') and 
                os.path.abspath(h.baseFilename) == os.path.abspath(log_file)
                for h in current_logger.handlers
            )
            
            if not has_current_handler:
                file_handler = FlushFileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler.setFormatter(formatter)
                current_logger.addHandler(file_handler)
                logger.info_rank0(f"✅ Training logs will be saved to: {log_file}")
            
            # llamafactory root
            try:

                llamafactory_root = logging.getLogger("llamafactory")
                has_root_handler = any(
                    isinstance(h, logging.FileHandler) and 
                    hasattr(h, 'baseFilename') and 
                    os.path.abspath(h.baseFilename) == os.path.abspath(log_file)
                    for h in llamafactory_root.handlers
                )
                
                if not has_root_handler:
                    file_handler2 = FlushFileHandler(log_file, mode='a', encoding='utf-8')
                    file_handler2.setLevel(logging.INFO)
                    formatter2 = logging.Formatter(
                        fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                    file_handler2.setFormatter(formatter2)
                    llamafactory_root.addHandler(file_handler2)
            except Exception:
                pass
                
            # Python root logger
            root_logger = logging.getLogger()
            has_root_file_handler = any(
                isinstance(h, logging.FileHandler) and 
                hasattr(h, 'baseFilename') and 
                os.path.abspath(h.baseFilename) == os.path.abspath(log_file)
                for h in root_logger.handlers
            )
            
            if not has_root_file_handler:
                file_handler3 = FlushFileHandler(log_file, mode='a', encoding='utf-8')
                file_handler3.setLevel(logging.INFO)
                formatter3 = logging.Formatter(
                    fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler3.setFormatter(formatter3)
                root_logger.addHandler(file_handler3)
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to setup file logging: {e}")
            import traceback
            logger.warning(f"   Error details: {traceback.format_exc()}")
        
        # Initialize SwanLab for visualization (if available)
        self.use_swanlab = SWANLAB_AVAILABLE and os.environ.get('USE_SWANLAB', '0') == '1'
        if self.use_swanlab:
            try:
                from datetime import datetime
                should_init_swanlab = False
                try:
                    should_init_swanlab = bool(self.use_swanlab and self.is_world_process_zero())
                except Exception:
                    should_init_swanlab = False
                
                if should_init_swanlab:
                    # Extract experiment name from output directory
                    output_dir = self.args.output_dir
                    # Get the last folder name from the path
                    experiment_name = os.path.basename(os.path.normpath(output_dir))
                    # If empty or just '.', use a default name with timestamp
                    if not experiment_name or experiment_name == '.':
                        experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    swanlab.init(
                        project="PersonaSteer",
                        experiment_name=experiment_name,
                        config={
                            "output_dir": self.args.output_dir,
                            "learning_rate": self.args.learning_rate,
                            "batch_size": self.args.per_device_train_batch_size,
                            "num_epochs": self.args.num_train_epochs,
                            "warmup_steps": self.args.warmup_steps,
                            "logging_steps": self.args.logging_steps,
                        }
                    )
                    logger.info_rank0(f"✅ SwanLab initialized for visualization (experiment: {experiment_name})")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize SwanLab: {e}")
                self.use_swanlab = False

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Compute Language Model Loss with activation steering.
        Based on trainer_reconstruction.py _compute_lm_loss logic.
        """
        # Unwrap DDP/DataParallel
        unwrapped_model = model.module if hasattr(model, "module") else model
        device = next(model.parameters()).device
        
        # Use standard LM loss
        return self._compute_lm_loss(model, inputs, return_outputs, unwrapped_model, device)
    def _compute_lm_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool,
        unwrapped_model: nn.Module,
        device: torch.device,
    ):
        """
        Compute standard Language Model Loss with activation steering.
        """
        # Make sure the decoder is frozen
        if unwrapped_model.decoder is not None:
            unwrapped_model.decoder.eval() 
        
        # Forward: demographic (Stream A) + optional query/profile (Stream B)
        forward_kw = dict(
            user_input_ids=inputs["user_input_ids"].to(device),
            user_attention_mask=inputs.get("user_attention_mask").to(device) if inputs.get("user_attention_mask") is not None else None,
            decoder_input_ids=inputs["input_ids"].to(device),
            decoder_attention_mask=inputs.get("attention_mask").to(device) if inputs.get("attention_mask") is not None else None,
            use_steering=True,
        )
        if "query_input_ids" in inputs and "profile_input_ids" in inputs:
            forward_kw["query_input_ids"] = inputs["query_input_ids"].to(device)
            forward_kw["query_attention_mask"] = inputs.get("query_attention_mask")
            if forward_kw["query_attention_mask"] is not None:
                forward_kw["query_attention_mask"] = forward_kw["query_attention_mask"].to(device)
            forward_kw["profile_input_ids"] = inputs["profile_input_ids"].to(device)
            forward_kw["profile_attention_mask"] = inputs.get("profile_attention_mask")
            if forward_kw["profile_attention_mask"] is not None:
                forward_kw["profile_attention_mask"] = forward_kw["profile_attention_mask"].to(device)
        # IMPORTANT: run forward through the wrapped model so DDP/FSDP hooks stay active.
        model_outputs = model(**forward_kw)
        
        decoder_outputs = model_outputs["decoder_outputs"]
        steering_vector = model_outputs["steering_vector"]

        # Optional decoder prompt dump to file
        try:
            should_log_prompts = (
                hasattr(self, "state")
                and (
                    self.state.global_step < 50 or  # first steps
                    (os.environ.get("LOG_DECODER_PROMPT", "0") == "1" and 
                     self.state.global_step % max(1, self.args.logging_steps) == 0)
                )
                and self.is_world_process_zero()
            )
        except Exception:
            should_log_prompts = False

        if should_log_prompts:
            # Cap examples
            max_examples = 5
            # Prefer model.decoder_tokenizer
            dec_tokenizer = getattr(unwrapped_model, "decoder_tokenizer", None)
            if dec_tokenizer is None:
                # Fallback to trainer tokenizer
                dec_tokenizer = getattr(self, "tokenizer", None)
            usr_tokenizer = getattr(self, "user_tokenizer", None)
            input_ids = inputs["input_ids"].to(device)
            labels = inputs.get("labels", None)
            if labels is not None:
                labels = labels.to(device)
            user_ids = inputs.get("user_input_ids", None)

            # Labels / tokenizer debug
            debug_info = []
            debug_info.append(f"[Debug] labels is None: {labels is None}")
            if labels is not None:
                debug_info.append(f"[Debug] labels shape: {labels.shape}, dtype: {labels.dtype}")
                debug_info.append(f"[Debug] labels device: {labels.device}")
                debug_info.append(f"[Debug] labels has IGNORE_INDEX: {(labels == IGNORE_INDEX).any().item()}")
            debug_info.append(f"[Debug] decoder_tokenizer is None: {dec_tokenizer is None}")
            if dec_tokenizer is not None:
                debug_info.append(f"[Debug] decoder_tokenizer type: {type(dec_tokenizer)}")
            debug_info.append(f"[Debug] unwrapped_model has decoder_tokenizer: {hasattr(unwrapped_model, 'decoder_tokenizer')}")
            if hasattr(unwrapped_model, 'decoder_tokenizer'):
                debug_info.append(f"[Debug] unwrapped_model.decoder_tokenizer: {unwrapped_model.decoder_tokenizer}")
            debug_info.append(f"[Debug] inputs keys: {list(inputs.keys())}")

            bsz = input_ids.shape[0]
            num_show = min(max_examples, bsz)

            # Build log lines
            log_lines = []
            log_lines.append("\n" + "=" * 60)
            log_lines.append(f"[Decoder Prompt Debug] - Step {self.state.global_step}")
            log_lines.append("=" * 60)

            log_lines.append("\n[Debug Info]:")
            log_lines.extend(debug_info)
            log_lines.append("=" * 60)
            
            for i in range(num_show):
                # User text
                if user_ids is not None and usr_tokenizer is not None:
                    user_text = usr_tokenizer.decode(user_ids[i].tolist(), skip_special_tokens=True)
                else:
                    user_text = str(user_ids[i].tolist()) if user_ids is not None else "<no user_input_ids>"
                # Decoder full input
                if dec_tokenizer is not None:
                    full_text = dec_tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=False)
                else:
                    full_text = str(input_ids[i].tolist())
                # Prompt segment from labels mask
                if labels is not None and dec_tokenizer is not None:

                    labels_i = labels[i] if labels.device == device else labels[i].to(device)
                    prompt_mask = (labels_i == IGNORE_INDEX)
                    prompt_ids = input_ids[i][prompt_mask]
                    if prompt_ids.numel() > 0:
                        # skip_special_tokens for readability
                        prompt_text = dec_tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
                    else:
                        prompt_text = "<empty prompt by mask>"
                else:
                    # Missing fields
                    missing = []
                    if labels is None:
                        missing.append("labels")
                    if dec_tokenizer is None:
                        missing.append("decoder_tokenizer")
                    prompt_text = f"<{' and '.join(missing)} unavailable>"
                
                log_lines.append(f"\n[Sample {i+1}/{num_show}]:")
                log_lines.append(f"  User Text (Encoder Input):\n    {user_text}")
                log_lines.append(f"  Decoder Full Input:\n    {full_text}")
                log_lines.append(f"  Decoder Prompt Only (masked by labels):\n    {prompt_text}")            
            log_lines.append("=" * 60 + "\n")
            
            # Append prompt_samples.log
            try:
                log_file = os.path.join(self.args.output_dir, "prompt_samples.log")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(log_lines))
                    f.write("\n")
            except Exception as e:
                logger.warning_rank0(f"⚠️  Failed to write prompt samples to file: {e}")
                

        if not steering_vector.requires_grad:
            steering_vector = steering_vector.requires_grad_(True)

        # Keep steering_vector in graph
        _ = steering_vector.sum() * 0.0

        # Gradient debug (frozen decoder logits may have requires_grad False)

        if hasattr(self, 'state') and (self.state.global_step < 50 or self.state.global_step % self.args.logging_steps == 0):
            accumulation_step = getattr(self, '_accumulation_step', 0) if hasattr(self, '_accumulation_step') else None
            accumulation_info = f" (accumulation step {accumulation_step})" if accumulation_step is not None else ""
            
            logger.info(f"\n[Gradient Debug] Step {self.state.global_step}{accumulation_info}:")
            logger.info(f"  steering_vector.requires_grad: {steering_vector.requires_grad}")
            logger.info(f"  steering_vector.grad_fn: {steering_vector.grad_fn}")
            logger.info(f"  steering_vector.is_leaf: {steering_vector.is_leaf}")
            logger.info(f"  steering_vector.shape: {steering_vector.shape}")
            logger.info(f"  steering_vector.norm: {steering_vector.norm().item():.6f}")
            
            hook_sv = unwrapped_model.steering_hook.steering_vector
            if hook_sv is not None:
                logger.info(f"  hook.steering_vector is same object: {hook_sv is steering_vector}")
                logger.info(f"  hook.steering_vector.requires_grad: {hook_sv.requires_grad}")
                logger.info(f"  hook.steering_vector.grad_fn: {hook_sv.grad_fn}")
            else:
                logger.info(f"  hook.steering_vector is None ⚠️")
            
            hook_stats = unwrapped_model.get_hook_stats()
            logger.info(f"  hook.steering_mode: dynamic")
            if self.state.global_step == 0:
                logger.info_rank0(f"  ✅ Steering mode: dynamic (context-dependent steering)")
            logger.info(f"  hook.call_count: {hook_stats.get('call_count', 0)}")
            logger.info(f"  hook.is_registered: {hook_stats.get('is_registered', False)}")
            
            logger.info(f"  logits.requires_grad: {decoder_outputs.logits.requires_grad}")
            logger.info(f"  logits.grad_fn: {decoder_outputs.logits.grad_fn}")
            logger.info(f"  logits.shape: {decoder_outputs.logits.shape}")
        
        logits = decoder_outputs.logits.float()

        if steering_vector.requires_grad:
            # Dummy op ties steering_vector into autograd without logits diamond (GC-safe).
            _ = steering_vector.sum() * 0.0
            pass
        
        labels = inputs["labels"].to(device)
        
        # LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        
        # Per-sample valid token counts
        batch_size = labels.shape[0]
        valid_tokens_per_sample = []
        loss_per_sample = []
        
        for i in range(batch_size):
            # Valid label tokens
            sample_labels = shift_labels[i]
            valid_mask = (sample_labels != IGNORE_INDEX)
            valid_count = valid_mask.sum().item()
            valid_tokens_per_sample.append(valid_count)
            
            # Per-sample loss
            if valid_count > 0:
                # Same reduction as batch loss
                sample_logits_flat = shift_logits[i][valid_mask].view(-1, shift_logits.size(-1))  # [num_valid, vocab_size]
                sample_labels_flat = sample_labels[valid_mask].view(-1)  # [num_valid]
                
                try:
                    sample_loss = loss_fct(
                        sample_logits_flat,  # [num_valid_tokens, vocab_size]
                        sample_labels_flat  # [num_valid_tokens]
                    ).item()
                    loss_per_sample.append(sample_loss)
                except Exception as e:
                    loss_per_sample.append(float('nan'))
                    if hasattr(self, 'state') and self.state.global_step % self.args.logging_steps == 0:
                        logger.warning(f"  Failed to compute loss for sample {i}: {e}")
                        logger.warning(f"    sample_logits_flat shape: {sample_logits_flat.shape}")
                        logger.warning(f"    sample_labels_flat shape: {sample_labels_flat.shape}")
            else:
                loss_per_sample.append(0.0)  # no valid tokens
        
        # Batch loss
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        # Extra loss term only when GC is off (avoids double-backward with GC).
        if not getattr(self.args, "gradient_checkpointing", False):
            # Tie loss to steering_vector as a backup grad path

            loss = loss + 0.0 * steering_vector.sum()
        if hasattr(self, 'state'):
            should_log_diagnosis = (
                self.state.global_step < 50 or
                self.state.global_step % self.args.logging_steps == 0 or
                loss.item() == 0.0 or
                torch.isnan(loss) or
                torch.isinf(loss) or
                any(v == 0 for v in valid_tokens_per_sample)
            )
            
            if should_log_diagnosis:
                logger.info(f"\n[Loss Diagnosis - Step {self.state.global_step}]:")
                logger.info(f"  Batch loss: {loss.item():.6f}")
                logger.info(f"  Batch size: {batch_size}")
                logger.info(f"  Valid tokens per sample: {valid_tokens_per_sample}")
                logger.info(f"  Loss per sample: {[f'{l:.4f}' for l in loss_per_sample]}")
                logger.info(f"  Min valid tokens: {min(valid_tokens_per_sample) if valid_tokens_per_sample else 0}")
                logger.info(f"  Max valid tokens: {max(valid_tokens_per_sample) if valid_tokens_per_sample else 0}")
                logger.info(f"  Avg valid tokens: {sum(valid_tokens_per_sample)/len(valid_tokens_per_sample) if valid_tokens_per_sample else 0:.1f}")
                logger.info(f"  Samples with 0 valid tokens: {sum(1 for v in valid_tokens_per_sample if v == 0)}")
                
                # Anomalies
                if loss.item() == 0.0:
                    logger.warning(f"  ⚠️  Batch loss is 0! This may indicate all samples have 0 valid tokens.")
                if any(v == 0 for v in valid_tokens_per_sample):
                    zero_indices = [i for i, v in enumerate(valid_tokens_per_sample) if v == 0]
                    logger.warning(f"  ⚠️  Samples {zero_indices} have 0 valid tokens! This will cause loss=0 for those samples.")
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"  ❌ Loss is NaN or Inf!")
                    logger.error(f"  Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                    logger.error(f"  Labels stats: min={labels.min().item()}, max={labels.max().item()}")
                    logger.error(f"  Valid tokens: {sum(valid_tokens_per_sample)} total")
                
                # Logits NaN/Inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    nan_count = torch.isnan(logits).sum().item()
                    inf_count = torch.isinf(logits).sum().item()
                    logger.error(f"  ❌ Found {nan_count} NaN and {inf_count} Inf in logits!")
                
                # Cross-sample loss spread
                if len(loss_per_sample) > 1:
                    # Finite losses only
                    valid_losses = [l for l in loss_per_sample if not (isinstance(l, float) and (l != l or abs(l) == float('inf')))]
                    if len(valid_losses) > 1:
                        loss_std = torch.tensor(valid_losses).std().item()
                        logger.info(f"  Loss std across samples: {loss_std:.4f}")
                        if loss_std > 2.0:
                            logger.warning(f"  ⚠️  High loss variance across samples ({loss_std:.4f}), may indicate data quality issues!")
        
        # Loss / grad debug
        if hasattr(self, 'state') and (self.state.global_step < 50 or self.state.global_step % self.args.logging_steps == 0):
            logger.info(f"  loss.requires_grad: {loss.requires_grad}")
            logger.info(f"  loss.grad_fn: {loss.grad_fn}")
            logger.info(f"  loss.item(): {loss.item():.6f}")
            
            has_grad = any(
                param.grad is not None 
                for name, param in unwrapped_model.named_parameters()
                if ('steering_generator' in name or 'user_encoder' in name) and param.requires_grad
            )
            status_label = "(after backward)" if has_grad else "(before backward)"
            logger.info(f"\n[Parameter Gradient Status {status_label}]:")
            
            # Count trainable params instead of listing all
            trainable_count = 0
            grad_count = 0
            modules_with_grad = set()
            total_grad_norm = 0.0
            
            for name, param in unwrapped_model.named_parameters():
                if ('steering_generator' in name or 'user_encoder' in name) and param.requires_grad:
                    trainable_count += 1
                    if param.grad is not None:
                         grad_count += 1
                         param_norm = param.grad.data.norm(2).item()
                         total_grad_norm += param_norm ** 2
                         # Store module name (e.g., user_encoder.layers.0)
                         module_name = '.'.join(name.split('.')[:3])
                         modules_with_grad.add(module_name)

            total_grad_norm = total_grad_norm ** 0.5

            logger.info(f"  Trainable Params: {trainable_count}")
            logger.info(f"  Params with Grad: {grad_count}")
            logger.info(f"  Total Grad Norm: {total_grad_norm:.6f}")
            logger.info(f"  Active Modules: {list(modules_with_grad)[:5]}..." if len(modules_with_grad) > 5 else f"  Active Modules: {list(modules_with_grad)}")
        
        # Fusion alpha for logging
        alpha_val = None
        if hasattr(unwrapped_model, "steering_generator") and hasattr(unwrapped_model.steering_generator, "last_alpha"):
            alpha_val = unwrapped_model.steering_generator.last_alpha
        elif hasattr(model, "steering_generator") and hasattr(model.steering_generator, "last_alpha"):
            alpha_val = model.steering_generator.last_alpha


        # Trainer metrics
        if hasattr(self, 'state') and (self.state.global_step < 50 or self.state.global_step % self.args.logging_steps == 0):
            current_lr = self.args.learning_rate
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                try:
                    if hasattr(self.lr_scheduler, 'get_last_lr'):
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                    elif hasattr(self.lr_scheduler, 'get_lr'):
                        current_lr = self.lr_scheduler.get_lr()[0]
                except Exception:
                    pass
            
            log_dict = {
                "train/loss": loss.item(),
                "train/learning_rate": current_lr,
            }
            # Optional fusion_alpha
            if alpha_val is not None:
                log_dict["train/fusion_alpha"] = alpha_val

            display_step = getattr(self.state, "global_step", 0) + getattr(self, "_checkpoint_step_offset", 0)
            logger.info(f"[Trainer Log] Step {display_step} (Internal: {self.state.global_step}): {log_dict}")
            self.log(log_dict)
        
        if hasattr(self, 'state') and getattr(self, "use_swanlab", False) and SWANLAB_AVAILABLE:
            try:
                # Rank 0 only
                if self.is_world_process_zero():
                    current_lr = self.args.learning_rate
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                        try:
                            if hasattr(self.lr_scheduler, 'get_last_lr'):
                                current_lr = self.lr_scheduler.get_last_lr()[0]
                            elif hasattr(self.lr_scheduler, 'get_lr'):
                                current_lr = self.lr_scheduler.get_lr()[0]
                        except Exception:
                            pass
                    
                    swanlab_dict = {
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                    }
                    # Optional fusion_alpha
                    if alpha_val is not None:
                        swanlab_dict["train/fusion_alpha"] = alpha_val
                    
                    display_step = getattr(self.state, "global_step", 0) + getattr(self, "_checkpoint_step_offset", 0)
                    swanlab.log(swanlab_dict, step=display_step)
                    
                    if self.state.global_step % self.args.logging_steps == 0:
                        extra_msg = f", alpha={alpha_val:.4f}" if alpha_val is not None else ""
                        logger.info(f"[SwanLab Direct] Logged at step {display_step} (Internal: {self.state.global_step}): loss={loss.item():.6f}, lr={current_lr}{extra_msg}")
            except Exception as e:
                if hasattr(self, 'state') and self.state.global_step % self.args.logging_steps == 0:
                    logger.warning(f"⚠️  SwanLab direct logging error at step {self.state.global_step}: {e}")
        
        return (loss, model_outputs) if return_outputs else loss
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """
        Train the model with two-stage training if enabled.
        Stage 1: Train MLP only (freeze encoder) for first N steps (configurable)
        Stage 2: Train all parameters (unfreeze encoder) for remaining epochs
        """
        if not getattr(self, "use_two_stage_training", False):
            return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)

        total_epochs = self.args.num_train_epochs

        # Stage 1 steps:
        # - If user explicitly sets args.mlp_only_steps, use it.
        # - Otherwise, derive from dataloader length and mlp_only_epochs.
        mlp_steps = None
        if self.model_args is not None:
            mlp_steps = getattr(self.model_args, "mlp_only_steps", None)
        # Fallback mlp_only_steps from training_args
        if mlp_steps is None:
            mlp_steps = getattr(self.args, "mlp_only_steps", None)
        
        if mlp_steps is None:
            try:
                train_dataloader = self.get_train_dataloader()
                steps_per_epoch = max(1, len(train_dataloader) // max(1, self.args.gradient_accumulation_steps))
                mlp_steps = max(1, steps_per_epoch * max(1, int(self.mlp_only_epochs)))
            except Exception as e:
                logger.warning(f"⚠️  Warning: Could not infer stage-1 steps from dataloader ({e}); falling back to 1000")
                mlp_steps = 1000
        else:
            mlp_steps = int(mlp_steps)
            if mlp_steps <= 0:
                raise ValueError("--mlp_only_steps must be a positive integer")
        self._current_two_stage_mlp_steps = mlp_steps
        resume_info = self._resolve_two_stage_resume(resume_from_checkpoint, mlp_steps)
        logger.info_rank0("\n" + "=" * 60)
        logger.info_rank0("Starting Two-Stage Training")
        logger.info_rank0("=" * 60)
        logger.info_rank0(f"Total epochs: {total_epochs}")
        logger.info_rank0(f"Stage 1 (MLP only): {mlp_steps} steps")
        logger.info_rank0(f"Stage 2 (All parameters): {total_epochs} epoch(s)")
        logger.info_rank0(f"Resume mode: {resume_info['mode']}")
        logger.info_rank0("=" * 60)

        # Save args for stage 2
        original_epochs = self.args.num_train_epochs
        original_max_steps = getattr(self.args, "max_steps", -1)
        original_warmup_steps = getattr(self.args, "warmup_steps", None)
        original_warmup_ratio = getattr(self.args, "warmup_ratio", None)
        planned_steps_per_epoch: Optional[int] = None
        try:
            train_dataloader = self.get_train_dataloader()
            planned_steps_per_epoch = max(1, len(train_dataloader) // max(1, self.args.gradient_accumulation_steps))
        except Exception:
            planned_steps_per_epoch = None

        if original_max_steps is not None and original_max_steps > 0:
            planned_total_steps: Optional[int] = int(original_max_steps)
        elif planned_steps_per_epoch is not None:
            planned_total_steps = int(planned_steps_per_epoch * float(original_epochs))
        else:
            planned_total_steps = None

        if planned_total_steps is not None:
            logger.info_rank0(f"Planned total optimizer steps: {planned_total_steps}")

        def configure_stage1() -> None:
            logger.info_rank0("\n" + "=" * 60)
            logger.info_rank0(f"STAGE 1: Training MLP only (freeze encoder) for {mlp_steps} steps")
            logger.info_rank0("=" * 60)

            unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(unwrapped_model, 'freeze_user_encoder'):
                unwrapped_model.freeze_user_encoder()
            self.training_stage = "mlp_only"
            self._checkpoint_step_offset = 0
            self._write_two_stage_state(
                self.args.output_dir,
                training_stage="mlp_only",
                mlp_steps=mlp_steps,
                stage1_completed=False,
                checkpoint_step_offset=0,
            )

            self.args.num_train_epochs = 1000.0
            self.args.max_steps = mlp_steps

            if original_warmup_ratio is not None:
                stage1_warmup_steps = max(1, int(mlp_steps * original_warmup_ratio))
                self.args.warmup_steps = stage1_warmup_steps
                self.args.warmup_ratio = None
                logger.info_rank0(f"  Stage 1 warmup_steps: {stage1_warmup_steps} (based on {original_warmup_ratio} ratio)")
            elif original_warmup_steps is not None:
                try:
                    train_dataloader = self.get_train_dataloader()
                    steps_per_epoch = max(1, len(train_dataloader) // max(1, self.args.gradient_accumulation_steps))
                    original_total_steps = int(steps_per_epoch * original_epochs)

                    if original_total_steps > 0:
                        stage1_warmup_steps = max(1, int(original_warmup_steps * mlp_steps / original_total_steps))
                    else:
                        stage1_warmup_steps = min(original_warmup_steps, max(1, mlp_steps // 10))
                    self.args.warmup_steps = stage1_warmup_steps
                    logger.info_rank0(f"  Stage 1 warmup_steps: {stage1_warmup_steps} (scaled from {original_warmup_steps})")
                except Exception as e:
                    self.args.warmup_steps = min(original_warmup_steps, max(1, mlp_steps // 10))
                    logger.warning(f"  Stage 1 warmup_steps: {self.args.warmup_steps} (fallback due to error: {e})")
            else:
                self.args.warmup_steps = max(1, mlp_steps // 10)
                logger.info_rank0(f"  Stage 1 warmup_steps: {self.args.warmup_steps} (default, 10% of mlp_steps)")

            logger.info_rank0("  Stage 1 configuration:")
            logger.info_rank0(f"    - max_steps: {self.args.max_steps}")
            logger.info_rank0(f"    - num_train_epochs: {self.args.num_train_epochs} (large value, will use max_steps)")
            logger.info_rank0(f"    - warmup_steps: {getattr(self.args, 'warmup_steps', 'None')}")
            logger.info_rank0(f"    - learning_rate: {self.args.learning_rate}")

            self.optimizer = None
            self.lr_scheduler = None
            if hasattr(self, '_lr_scheduler'):
                delattr(self, '_lr_scheduler')

        def configure_stage2(stage2_steps: Optional[int] = None, checkpoint_step_offset: int = mlp_steps) -> None:
            logger.info_rank0("\n" + "=" * 60)
            if stage2_steps is None:
                logger.info_rank0(f"STAGE 2: Training all parameters (unfreeze encoder) for {total_epochs} epoch(s)")
            else:
                logger.info_rank0(
                    f"STAGE 2: Training all parameters (unfreeze encoder) for remaining {stage2_steps} steps"
                )
            logger.info_rank0("=" * 60)

            unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(unwrapped_model, 'unfreeze_user_encoder'):
                unwrapped_model.unfreeze_user_encoder()
            self.training_stage = "all"
            self._checkpoint_step_offset = checkpoint_step_offset
            self._write_two_stage_state(
                self.args.output_dir,
                training_stage="all",
                mlp_steps=mlp_steps,
                stage1_completed=True,
                checkpoint_step_offset=checkpoint_step_offset,
            )

            if stage2_steps is None:
                self.args.num_train_epochs = total_epochs
                self.args.max_steps = -1
            else:
                self.args.num_train_epochs = 1000.0
                self.args.max_steps = int(stage2_steps)

            if original_warmup_ratio is not None:
                if stage2_steps is None:
                    self.args.warmup_ratio = original_warmup_ratio
                    self.args.warmup_steps = 0
                    logger.info_rank0(f"  ✅ Restored warmup_ratio: {original_warmup_ratio}")
                else:
                    stage2_warmup_steps = max(1, int(stage2_steps * original_warmup_ratio))
                    self.args.warmup_steps = stage2_warmup_steps
                    self.args.warmup_ratio = None
                    logger.info_rank0(
                        f"  ✅ Stage 2 warmup_steps: {stage2_warmup_steps} "
                        f"(based on {original_warmup_ratio} ratio over remaining steps)"
                    )
            elif original_warmup_steps is not None and original_warmup_steps > 0:
                if stage2_steps is None or planned_total_steps is None or planned_total_steps <= 0:
                    self.args.warmup_steps = original_warmup_steps
                    self.args.warmup_ratio = None
                    logger.info_rank0(f"  ✅ Restored warmup_steps: {original_warmup_steps}")
                else:
                    stage2_warmup_steps = max(1, int(original_warmup_steps * int(stage2_steps) / planned_total_steps))
                    self.args.warmup_steps = stage2_warmup_steps
                    self.args.warmup_ratio = None
                    logger.info_rank0(
                        f"  ✅ Stage 2 warmup_steps: {stage2_warmup_steps} "
                        f"(scaled from {original_warmup_steps} by remaining/total steps)"
                    )
            else:
                if hasattr(self.args, 'warmup_steps') and self.args.warmup_steps is not None:
                    logger.info_rank0(f"  Stage 2: Clearing Stage 1 warmup_steps ({self.args.warmup_steps}), using default warmup_ratio")
                self.args.warmup_steps = 0
                self.args.warmup_ratio = 0.1
                logger.info_rank0(f"  ✅ Using default warmup_ratio=0.1 (10% of training steps)")

            self.optimizer = None
            self.lr_scheduler = None
            if hasattr(self, '_lr_scheduler'):
                delattr(self, '_lr_scheduler')

            if stage2_steps is not None:
                stage2_max_steps = int(stage2_steps)
            else:
                try:
                    train_dataloader = self.get_train_dataloader()
                    steps_per_epoch = max(1, len(train_dataloader) // max(1, self.args.gradient_accumulation_steps))
                    stage2_max_steps = int(steps_per_epoch * total_epochs)
                    logger.info_rank0(
                        f"  Calculated Stage 2: {steps_per_epoch} steps/epoch × {total_epochs} epochs = {stage2_max_steps} total steps"
                    )
                except Exception as e:
                    logger.warning(f"⚠️  Could not calculate Stage 2 max_steps: {e}")
                    stage2_max_steps = None

            logger.info_rank0("  Stage 2 configuration:")
            logger.info_rank0(f"    - max_steps: {self.args.max_steps} (use num_train_epochs)")
            logger.info_rank0(f"    - num_train_epochs: {self.args.num_train_epochs}")
            logger.info_rank0(f"    - warmup_steps: {getattr(self.args, 'warmup_steps', 'None')}")
            logger.info_rank0(f"    - warmup_ratio: {getattr(self.args, 'warmup_ratio', 'None')}")
            logger.info_rank0(f"    - learning_rate: {self.args.learning_rate}")
            if stage2_max_steps is not None:
                logger.info_rank0(f"    - Expected total steps for Stage 2: {stage2_max_steps}")

        try:
            stage2_output = None

            if resume_info["mode"] in ["fresh", "stage1_resume"]:
                configure_stage1()
                stage1_output = super().train(
                    resume_from_checkpoint=resume_info["checkpoint"] if resume_info["mode"] == "stage1_resume" else None,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    **kwargs,
                )
                self._save_stage_boundary_checkpoint(trial=trial, mlp_steps=mlp_steps)
                stage2_steps = None if planned_total_steps is None else max(0, planned_total_steps - int(mlp_steps))
                if stage2_steps is not None and stage2_steps <= 0:
                    logger.info_rank0("No remaining steps for Stage 2. Skipping Stage 2 training.")
                    stage2_output = stage1_output
                else:
                    configure_stage2(stage2_steps=stage2_steps)
                    stage2_output = super().train(
                        resume_from_checkpoint=None,
                        trial=trial,
                        ignore_keys_for_eval=ignore_keys_for_eval,
                        **kwargs,
                    )
            elif resume_info["mode"] == "stage2_from_stage1":
                self._load_steering_weights_only(resume_info["checkpoint"])
                stage2_steps = None if planned_total_steps is None else max(0, planned_total_steps - int(mlp_steps))
                if stage2_steps is not None and stage2_steps <= 0:
                    logger.info_rank0("No remaining steps for Stage 2. Skipping Stage 2 training.")
                    stage2_output = TrainOutput(
                        global_step=int(getattr(self.state, "global_step", 0)),
                        training_loss=0.0,
                        metrics={},
                    )
                else:
                    configure_stage2(stage2_steps=stage2_steps)
                    stage2_output = super().train(
                        resume_from_checkpoint=None,
                        trial=trial,
                        ignore_keys_for_eval=ignore_keys_for_eval,
                        **kwargs,
                    )
            elif resume_info["mode"] == "stage2_resume":
                metadata = resume_info.get("metadata") or {}
                checkpoint_step_offset = int(metadata.get("checkpoint_step_offset", mlp_steps))
                stage2_steps = None if planned_total_steps is None else max(0, planned_total_steps - int(mlp_steps))
                configure_stage2(stage2_steps=stage2_steps, checkpoint_step_offset=checkpoint_step_offset)
                stage2_output = super().train(
                    resume_from_checkpoint=resume_info["checkpoint"],
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown two-stage resume mode: {resume_info['mode']}")
        
            if getattr(self, "use_swanlab", False) and SWANLAB_AVAILABLE:
                try:
                    swanlab.finish()
                    logger.info_rank0("✅ SwanLab logging completed")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to finish SwanLab: {e}")

            return stage2_output
        finally:
            self.args.num_train_epochs = original_epochs
            self.args.max_steps = original_max_steps
            if original_warmup_steps is not None:
                self.args.warmup_steps = original_warmup_steps
            if original_warmup_ratio is not None:
                self.args.warmup_ratio = original_warmup_ratio
            self._checkpoint_step_offset = 0
    
    def create_optimizer(self):
        """
        Create optimizer with only trainable parameters.
        In two-stage training:
        - Stage 1 (MLP only): Only steering_generator parameters
        - Stage 2 (All): user_encoder + steering_generator parameters
        Ensures decoder parameters are never included in the optimizer.
        """
        if self.optimizer is None:
            unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
            
            # Get trainable parameters based on training stage
            include_user_encoder = not (self.use_two_stage_training and self.training_stage == "mlp_only")
            trainable_params = unwrapped_model.get_trainable_parameters(include_user_encoder=include_user_encoder)
            trainable_param_ids = {id(p) for p in trainable_params}
            
            # Get decay parameter names for trainable parameters only
            all_decay_params = self.get_decay_parameter_names(self.model)
            trainable_param_names = {name for name, param in self.model.named_parameters() 
                                   if id(param) in trainable_param_ids}
            
            # Convert all_decay_params to set for intersection operation
            decay_params = set(all_decay_params) & trainable_param_names
            
            # Group parameters with and without weight decay
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for name, p in self.model.named_parameters() 
                        if name in decay_params and id(p) in trainable_param_ids and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for name, p in self.model.named_parameters() 
                        if name not in decay_params and id(p) in trainable_param_ids and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
            
            # Get optimizer class and kwargs
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            
            # Overwrite `params` if provided by optimizer_kwargs (e.g., for special optimizers)
            if "params" in optimizer_kwargs:
                # For special optimizers, we need to filter their params
                params_from_kwargs = optimizer_kwargs.pop("params")
                if isinstance(params_from_kwargs, list) and len(params_from_kwargs) > 0:
                    # Check if it's a list of dicts (grouped parameters) or list of tensors
                    if isinstance(params_from_kwargs[0], dict):
                        # Filter each group
                        for group in params_from_kwargs:
                            if "params" in group:
                                group["params"] = [p for p in group["params"] if id(p) in trainable_param_ids]
                        optimizer_grouped_parameters = params_from_kwargs
                    else:
                        # List of parameters
                        optimizer_grouped_parameters = [p for p in params_from_kwargs if id(p) in trainable_param_ids]
            
            # Create optimizer
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            # Final verification: ensure no decoder parameters are in optimizer
            if unwrapped_model.decoder is not None:
                decoder_param_ids = {id(p) for p in unwrapped_model.decoder.parameters()}
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if id(p) in decoder_param_ids:
                            raise RuntimeError(
                                "CRITICAL: Decoder parameter found in optimizer! "
                                "This indicates a bug in create_optimizer."
                            )
    
    def _save_checkpoint(self, model, trial, metrics=None, **kwargs):
        """
        Override checkpoint saving to avoid writing full `model.safetensors` (decoder is huge).
        We only persist the trainable parts into `steering_model.pt` plus trainer state
        (optimizer/scheduler/rng/trainer_state) so training can be resumed.
        """

        # Base output directory (handles hyperparameter search runs)
        output_dir = self._get_output_dir(trial=trial)
        checkpoint_step = int(self.state.global_step) + int(getattr(self, "_checkpoint_step_offset", 0))
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{checkpoint_step}"
        checkpoint_dir = os.path.join(output_dir, checkpoint_folder)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 1) Save ONLY the steering weights
        self.save_model(checkpoint_dir, _internal_call=True)

        # 2) Save trainer state (optimizer/scheduler/rng + trainer_state.json)
        if getattr(self.args, "should_save", True):
            # Optimizer & scheduler
            if hasattr(self, "_save_optimizer_and_scheduler"):
                self._save_optimizer_and_scheduler(checkpoint_dir)
            else:
                if getattr(self, "optimizer", None) is not None:
                    torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                if getattr(self, "lr_scheduler", None) is not None:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

            # RNG state
            if hasattr(self, "_save_rng_state"):
                self._save_rng_state(checkpoint_dir)

            # Trainer state JSON
            try:
                self.state.save_to_json(os.path.join(checkpoint_dir, TRAINER_STATE_NAME))
            except Exception:
                # Some older versions might not have save_to_json; ignore.
                pass

            if getattr(self, "use_two_stage_training", False) and self._current_two_stage_mlp_steps is not None:
                self._write_two_stage_state(
                    checkpoint_dir,
                    training_stage=self.training_stage,
                    mlp_steps=self._current_two_stage_mlp_steps,
                    stage1_completed=(self.training_stage == "all"),
                    checkpoint_global_step=int(self.state.global_step),
                    checkpoint_step_offset=int(getattr(self, "_checkpoint_step_offset", 0)),
                )

        # 3) Rotate checkpoints if configured
        if hasattr(self, "_rotate_checkpoints"):
            self._rotate_checkpoints(use_mtime=False, output_dir=output_dir)
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load checkpoint."""
        checkpoint_path = resume_from_checkpoint
        steering_model_path = os.path.join(checkpoint_path, "steering_model.pt")
        
        if os.path.isfile(steering_model_path):
            unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
            checkpoint = torch.load(
                steering_model_path,
                map_location=self.accelerator.device if hasattr(self, 'accelerator') else 'cpu'
            )
            unwrapped_model.user_encoder.load_state_dict(checkpoint['user_encoder'])
            unwrapped_model.steering_generator.load_state_dict(checkpoint['steering_generator'])
        
        # Check if standard transformers checkpoint files exist
        # If they don't exist, skip calling parent method to avoid error
        weights_file = os.path.join(checkpoint_path, WEIGHTS_NAME)
        safe_weights_file = os.path.join(checkpoint_path, SAFE_WEIGHTS_NAME)
        weights_index_file = os.path.join(checkpoint_path, WEIGHTS_INDEX_NAME)
        safe_weights_index_file = os.path.join(checkpoint_path, SAFE_WEIGHTS_INDEX_NAME)
        
        has_standard_checkpoint = any(
            os.path.isfile(f) for f in [
                weights_file,
                safe_weights_file,
                weights_index_file,
                safe_weights_index_file,
            ]
        )
        
        # Only call parent method if standard checkpoint files exist
        if has_standard_checkpoint:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        else:
            # If only custom checkpoint exists, just return without error
            return None
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        """
        Load optimizer and scheduler states from checkpoint.
        Skip loading only when optimizer states are incompatible; otherwise defer to the default Trainer behavior.
        """
        try:
            return super()._load_optimizer_and_scheduler(checkpoint)
        except ValueError as e:
            # If there's a parameter group mismatch error, skip optimizer loading
            if "parameter group" in str(e).lower() and "doesn't match" in str(e).lower():
                logger.info_rank0("\n" + "=" * 60)
                logger.info_rank0(f"⚠️  Warning: Optimizer state mismatch detected: {e}")
                logger.info_rank0("   Skipping optimizer and scheduler state loading.")
                logger.info_rank0("   Starting with fresh optimizer and scheduler state.")
                logger.info_rank0("=" * 60)
                return
            else:
                # Re-raise if it's a different error
                raise
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prediction step for evaluation."""
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        # Keep labels for compute_loss
        labels = inputs.get("labels") if has_labels else None
       
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            decoder_outputs = outputs.get("decoder_outputs") if isinstance(outputs, dict) else None
            if decoder_outputs is not None:
                logits = decoder_outputs.logits
            else:
                logits = None
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model, including user encoder and steering generator.
        Only saves trainable parts (not the frozen decoder) to save space.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Unwrap DDP/DataParallel wrappers (and keep save format stable).
        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # Save user encoder and steering generator
        save_dict = {
            'user_encoder': unwrapped_model.user_encoder.state_dict(),
            'steering_generator': unwrapped_model.steering_generator.state_dict(),
        }
        
        torch.save(save_dict, os.path.join(output_dir, "steering_model.pt"))
        if getattr(self, "use_two_stage_training", False) and self._current_two_stage_mlp_steps is not None:
            self._write_two_stage_state(
                output_dir,
                training_stage=self.training_stage,
                mlp_steps=self._current_two_stage_mlp_steps,
                stage1_completed=(self.training_stage == "all"),
                checkpoint_step_offset=int(getattr(self, "_checkpoint_step_offset", 0)),
            )
        
        # Also save config if needed
        if hasattr(unwrapped_model, 'config') and unwrapped_model.config is not None:
            # Use PreTrainedConfig's to_json_file method for proper serialization
            config_path = os.path.join(output_dir, "config.json")
            if hasattr(unwrapped_model.config, 'to_json_file'):
                unwrapped_model.config.to_json_file(config_path)
            else:
                # Fallback: convert to dict and save
                import json
                config_dict = unwrapped_model.config.to_dict() if hasattr(unwrapped_model.config, 'to_dict') else unwrapped_model.config.__dict__
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)