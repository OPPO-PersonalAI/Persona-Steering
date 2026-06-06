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
Training workflow for PersonaSteering model
"""

from typing import TYPE_CHECKING, Optional, Any, Dict
import os
import re
from datetime import datetime
import torch

from ...extras.logging import get_logger
from ...extras.constants import IGNORE_INDEX
from ...model import load_tokenizer
from ...hparams import DataArguments, FinetuningArguments, ModelArguments
from .trainer import PersonaSteeringTrainer
from .data_collator import PersonaSteeringDataCollator

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

logger = get_logger(__name__)


def format_learning_rate(lr: float) -> str:
    """
    Format learning rate for use in file/directory names.
    
    Args:
        lr: Learning rate (e.g., 2e-5)
        
    Returns:
        Formatted string (e.g., "2e-5" or "0p00002")
    """
    if lr < 1e-3 or lr >= 1e6:
        # Use scientific notation
        lr_str = f"{lr:.2e}"
        # Normalize: 2.00e-05 -> 2e-5, 1.00e+03 -> 1e3
        if "e" in lr_str:
            base, exp = lr_str.split("e")
            base_float = float(base)
            if base_float == int(base_float):
                base = str(int(base_float))
            else:
                # Keep 2 decimal places if not integer, remove trailing zeros
                base = f"{base_float:.2f}".rstrip('0').rstrip('.')
            # Handle exponent: remove leading zeros and + sign
            exp = exp.lstrip("+")
            if exp.startswith("0") and len(exp) > 1:
                exp = exp[1:]
            exp_int = int(exp) if exp else 0
            lr_str = f"{base}e{exp_int}" if exp_int != 0 else base
    else:
        # For regular floats, use 'p' instead of '.' to avoid filesystem issues
        lr_str = str(lr).replace(".", "p")
    return lr_str


def sanitize_path_component(text: str, max_length: int = 50) -> str:
    """
    Sanitize a string for use as a filesystem path component.
    
    Args:
        text: Input string
        max_length: Maximum length of the output string
        
    Returns:
        Sanitized string safe for filesystem use
    """
    # Remove or replace invalid filesystem characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_.')
    return sanitized


def get_decoder_short_name(decoder_name: str) -> str:
    """
    Extract a short name from decoder path.
    
    Args:
        decoder_name: Full decoder path (e.g., "Qwen/Qwen2.5-7B-Instruct")
        
    Returns:
        Short name (e.g., "Qwen2-7B")
    """
    # Extract the last component of the path
    base_name = os.path.basename(decoder_name)
    
    # Remove common suffixes
    base_name = re.sub(r'[-_]Instruct$', '', base_name)
    base_name = re.sub(r'[-_]Chat$', '', base_name)
    
    # Replace multiple underscores/hyphens with single hyphen
    base_name = re.sub(r'[_-]+', '-', base_name)
    
    # Extract model size if present (e.g., "7B", "8B", "13B")
    size_match = re.search(r'(\d+[BM])', base_name)
    if size_match:
        size = size_match.group(1)
        # Extract model name (before size)
        name_match = re.match(r'([A-Za-z0-9]+)', base_name)
        if name_match:
            model_name = name_match.group(1)
            return f"{model_name}-{size}"
    
    # Fallback: use first part of name, limit length
    parts = base_name.split('-')
    if len(parts) > 0:
        return sanitize_path_component(parts[0], max_length=20)
    
    return sanitize_path_component(base_name, max_length=20)


def generate_output_dir(
    base_dir: str = "./output",
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    num_epochs: int = 3,
    steering_layer_idx: Optional[int] = None,
    steering_mlp_architecture: str = "unified",
    use_two_stage_training: bool = False,
    user_encoder_name: Optional[str] = None,
    decoder_name: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generate output directory name based on training parameters.
    
    Args:
        base_dir: Base output directory (e.g., "./output")
        learning_rate: Learning rate
        batch_size: Per device batch size
        num_epochs: Number of training epochs
        steering_layer_idx: Steering layer index (if specified)
        steering_mlp_architecture: (Deprecated/ignored) kept for backward compatibility
        use_two_stage_training: Whether two-stage training is enabled
        user_encoder_name: User encoder name (if specified and not default)
        decoder_name: Decoder name (if specified)
        **kwargs: Additional parameters to include
        
    Returns:
        Full output directory path (e.g., "./output/20240101_120000_enc_BERT_dec_Qwen2-7B_lr_2e-5_bs_8_e_3_layer20_2stage")
    """
    # Start building directory name components
    components = []
    
    # Timestamp at the beginning (format: YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    components.append(timestamp)
    
    # Encoder name (always include if provided)
    if user_encoder_name:
        # If a full path is provided (e.g., "/path/to/local/model"), extract a short name
        looks_like_path = ("/" in user_encoder_name) or ("\\" in user_encoder_name)
        if looks_like_path:
            # Extract the last component of the path
            encoder_base = os.path.basename(user_encoder_name)
            encoder_short = sanitize_path_component(encoder_base, max_length=15)
        else:
            encoder_short = sanitize_path_component(user_encoder_name, max_length=15)
        components.append(f"enc_{encoder_short}")
    
    # Decoder name (always include if provided)
    if decoder_name:
        decoder_short = get_decoder_short_name(decoder_name)
        components.append(f"dec_{decoder_short}")
    
    # Learning rate
    lr_str = format_learning_rate(learning_rate)
    components.append(f"lr_{lr_str}")
    
    # Batch size
    components.append(f"bs_{batch_size}")
    
    # Number of epochs
    components.append(f"e_{num_epochs}")
    
    # Steering layer index (if specified)
    if steering_layer_idx is not None:
        components.append(f"layer{steering_layer_idx}")
    
    # Two-stage training flag
    if use_two_stage_training:
        components.append("2stage")
    
    # Join components
    dir_name = "_".join(components)
    
    # Ensure base_dir exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Full path
    full_path = os.path.join(base_dir, dir_name)
    
    return full_path


def load_persona_steering_model(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
):
    """
    Load PersonaSteering model.
    
    Args:
        model_args: Model arguments
        finetuning_args: Finetuning arguments
        is_trainable: Whether the model is trainable
    """
    # Import from local modules (no longer depends on PersonaSteer)
    from .model import UserEncoderModel, UserEncoderConfig
    
    # Create config
    config = UserEncoderConfig(
        user_encoder_name=getattr(model_args, 'user_encoder_name', 'bert-base-uncased'),
        decoder_name=model_args.model_name_or_path,
        decoder_hidden_size=getattr(model_args, 'decoder_hidden_size', None),
        steering_layer_idx=getattr(model_args, 'steering_layer_idx', -1),
        steering_coeff=getattr(model_args, 'steering_coeff', 1.0),
        user_encoder_use_lora=getattr(model_args, 'user_encoder_use_lora', False),
        user_encoder_lora_r=getattr(model_args, 'user_encoder_lora_r', 8),
        user_encoder_lora_alpha=getattr(model_args, 'user_encoder_lora_alpha', 16),
        user_encoder_lora_dropout=getattr(model_args, 'user_encoder_lora_dropout', 0.05),
        user_encoder_lora_target_modules=getattr(model_args, 'user_encoder_lora_target_modules', None),
        steering_mlp_activation=getattr(model_args, 'steering_mlp_activation', 'gelu'),
        steering_mlp_dropout=getattr(model_args, 'steering_mlp_dropout', 0.1),
        steering_mlp_use_layer_norm=getattr(model_args, 'steering_mlp_use_layer_norm', True),
        fusion_alpha_min=getattr(model_args, 'fusion_alpha_min', 0.1),
        fusion_alpha_max=getattr(model_args, 'fusion_alpha_max', 0.9),
    )
    
    # Load model
    if hasattr(model_args, 'adapter_name_or_path') and model_args.adapter_name_or_path:
        adapter_path = model_args.adapter_name_or_path
        if isinstance(adapter_path, list) and len(adapter_path) > 0:
            adapter_path = adapter_path[0]
        
        # Load or create model
        try:
            model = UserEncoderModel.from_pretrained(adapter_path, config=config)
        except Exception as e:
            allow_fallback = bool(getattr(model_args, "allow_new_model_on_checkpoint_failure", False))
            if allow_fallback:
                logger.warning(
                    f"Failed to load from checkpoint {adapter_path}: {e}. "
                    "allow_new_model_on_checkpoint_failure=True, creating new model."
                )
                model = UserEncoderModel(config=config)
            else:
                raise RuntimeError(
                    f"Failed to load checkpoint from {adapter_path}. "
                    "Set `allow_new_model_on_checkpoint_failure=true` to force fallback to a new model."
                ) from e
    else:
        # New model if no adapter
        model = UserEncoderModel(config=config)
    
    if not is_trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    else:
        model.train()
        if model.decoder is not None:
            for param in model.decoder.parameters():
                param.requires_grad = False
            logger.info_rank0("Decoder is frozen (not trainable).")
    
    return model


def load_persona_steering_dataset(
    data_path: str,
    user_tokenizer,
    decoder_tokenizer,
    max_user_length: int = 512,
    max_length: int = 2048,
    decoder_prompt: Optional[str] = None,
    output_dir: Optional[str] = None,
    task: str = "generation_abstract",
    task_mapping: Optional[Dict[str, str]] = None,
    max_samples_per_user: Optional[int] = None,  # Limit maximum number of samples per user
    max_profile_items: int = 8,  # Dual-stream: max profile items for Stream B
    max_profile_item_len: int = 128,  # Max tokens per profile item
    max_query_length: int = 256,  # Max tokens for query (user_content) in Stream B
    profile_topk_prescreen: bool = False,  # Whether to pre-screen profile items by query relevance
    profile_topk: Optional[int] = None,  # top-k value for pre-screening
    **kwargs
):
    """Load dataset for PersonaSteering training with multi-task support.
    
    Args:
        data_path: One path or comma-separated paths; path substrings may imply task.
        task_mapping: Optional path-prefix -> task map, e.g.
                     {"lamp4": "lamp_4", "lamp5": "lamp_5", "abstract": "generation_abstract"}
    """
    import json
    
    # Parse data_path
    if isinstance(data_path, str):
        # Comma-separated list
        if ',' in data_path:
            data_paths = [p.strip() for p in data_path.split(',')]
        else:
            data_paths = [data_path]
    elif isinstance(data_path, list):
        data_paths = data_path
    else:
        raise ValueError(f"data_path must be str or list, got {type(data_path)}")
    
    # Default path heuristics if no task_mapping
    if task_mapping is None:
        task_mapping = {}
    
    all_data = []
    task_stats = {}
    failed_paths = []

    # Load each split with task tags
    for path in data_paths:
        path = path.strip()
        if not path:
            continue
            
        # Infer task from path
        inferred_task = None
        path_lower = path.lower()
        
        # Path heuristics
        if 'lamp4' in path_lower or 'lamp_4' in path_lower:
            inferred_task = 'lamp_4'
        elif 'lamp5' in path_lower or 'lamp_5' in path_lower:
            inferred_task = 'lamp_5'
        elif 'lamp7' in path_lower or 'lamp_7' in path_lower:
            inferred_task = 'lamp_7'
        elif 'abstract' in path_lower and 'generation' in path_lower:
            inferred_task = 'generation_abstract'
        elif 'topic' in path_lower and 'writing' in path_lower:
            inferred_task = 'topic_writing'
        elif 'product' in path_lower and 'review' in path_lower:
            inferred_task = 'product_review_writing'
        
        # Explicit mapping
        for key, mapped_task in task_mapping.items():
            if key.lower() in path_lower:
                inferred_task = mapped_task
                break
        
        # Default task
        if inferred_task is None:
            inferred_task = task
        
        # Load JSON/JSONL
        logger.info_rank0(f"Loading dataset from: {path} (inferred task: {inferred_task})")
        try:
            if path.endswith('.json') or path.endswith('.jsonl'):
                file_data = []
                with open(path, 'r', encoding='utf-8') as f:
                    if path.endswith('.jsonl'):
                        for line in f:
                            if line.strip():
                                file_data.append(json.loads(line))
                    else:
                        file_data = json.load(f)
                        if isinstance(file_data, dict):
                            # If it's a dict, try to get a list from it
                            file_data = file_data.get('data', file_data.get('train', list(file_data.values())[0] if file_data else []))
            else:
                logger.warning_rank0(f"Unsupported file format: {path}, skipping...")
                continue
            
            # Tag _task if missing
            count = 0
            for item in file_data:
                if not isinstance(item, dict):
                    continue
                # Keep existing task or inferred
                if 'task' not in item:
                    item['task'] = inferred_task
                all_data.append(item)
                count += 1
            
            task_stats[inferred_task] = task_stats.get(inferred_task, 0) + count
            logger.info_rank0(f"  ✓ Loaded {count} samples (task: {inferred_task})")
            
        except FileNotFoundError as e:
            logger.warning_rank0(f"  ⚠️  File not found: {path}, skipping...")
        except Exception as e:
            logger.warning_rank0(f"  ❌ Error loading {path}: {e}, skipping...")
            failed_paths.append((path, str(e)))
    
    if not all_data:
        raise ValueError(f"No data loaded from any of the provided paths: {data_paths}")
    
    from .dataset import SteeringDataset
    logger.info_rank0(f"Using SteeringDataset with multi-task support (default task: {task})")
    if max_samples_per_user is not None:
        logger.info_rank0(f"Limiting to {max_samples_per_user} samples per user")
    dataset = SteeringDataset(
        data=all_data,
        user_tokenizer=user_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        max_user_length=max_user_length,
        max_length=max_length,
        decoder_prompt=decoder_prompt,
        output_dir=output_dir,
        task=task,
        max_samples_per_user=max_samples_per_user,
        max_profile_items=max_profile_items,
        max_profile_item_len=max_profile_item_len,
        max_query_length=max_query_length,
        profile_topk_prescreen=profile_topk_prescreen,
        profile_topk=profile_topk,
    )
    
    # Dataset summary
    logger.info_rank0(f"\n{'='*60}")
    logger.info_rank0(f"Dataset Summary:")
    logger.info_rank0(f"  Total users: {len(all_data)}")
    logger.info_rank0(f"  Total training samples: {len(dataset)}")
    logger.info_rank0(f"  Task distribution (users):")
    for task_name, count in sorted(task_stats.items()):
        logger.info_rank0(f"    - {task_name}: {count} users")
    if failed_paths:
        logger.info_rank0(f"\n  ⚠️  Failed to load {len(failed_paths)} dataset(s):")
        for failed_path, error in failed_paths:
            logger.info_rank0(f"    - {failed_path}: {error}")
    logger.info_rank0(f"{'='*60}\n")
    
    # Write dataset log under output_dir
    if output_dir is not None:
        try:
            # Main process check: use global rank-0 only (avoid multi-node local_rank=0 races)
            is_main_process = False
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    is_main_process = dist.get_rank() == 0
                else:
                    is_main_process = True
            except (ImportError, AttributeError):
                rank = int(os.environ.get("RANK", "0"))
                is_main_process = (rank == 0)
            
            if is_main_process:
                os.makedirs(output_dir, exist_ok=True)
                summary_log = os.path.join(output_dir, "dataset_summary.log")
                with open(summary_log, "w", encoding="utf-8") as f:
                    f.write("Dataset Summary:\n")
                    f.write(f"  Total users: {len(all_data)}\n")
                    f.write(f"  Total training samples: {len(dataset)}\n")
                    f.write("  Task distribution (users):\n")
                    for task_name, count in sorted(task_stats.items()):
                        f.write(f"    - {task_name}: {count} users\n")
                    if failed_paths:
                        f.write(f"\n  ⚠️  Failed to load {len(failed_paths)} dataset(s):\n")
                        for failed_path, error in failed_paths:
                            f.write(f"    - {failed_path}: {error}\n")
                logger.info_rank0(f"✅ Saved dataset summary to {summary_log}")
        except Exception as e:
            logger.warning_rank0(f"Failed to write dataset summary log to {output_dir}: {e}")
            import traceback
            logger.warning_rank0(f"Traceback: {traceback.format_exc()}")
        
    return dataset


def run_persona_steering(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """Main training function for PersonaSteering."""
    # Auto-generate output directory if requested
    auto_output_dir = getattr(training_args, 'auto_output_dir', False)
    if auto_output_dir or (hasattr(training_args, 'output_dir') and not training_args.output_dir):
        base_dir = getattr(training_args, 'output_dir', './output') or './output'
        user_encoder_name = getattr(model_args, 'user_encoder_name', 'bert-base-uncased')
        decoder_name = model_args.model_name_or_path
        steering_layer_idx = getattr(model_args, 'steering_layer_idx', None)
        use_two_stage_training = getattr(model_args, 'use_two_stage_training', False)
        
        rank = int(os.environ.get("RANK", "0"))

        if rank == 0:
            # Rank 0 creates run dir
            generated_output_dir = generate_output_dir(
                base_dir=base_dir,
                learning_rate=training_args.learning_rate,
                batch_size=training_args.per_device_train_batch_size,
                num_epochs=training_args.num_train_epochs,
                steering_layer_idx=steering_layer_idx,
                use_two_stage_training=use_two_stage_training,
                user_encoder_name=user_encoder_name,
                decoder_name=decoder_name,
            )
            # Write run dir for other ranks
            import tempfile
            temp_file = os.path.join(base_dir, ".output_dir.tmp")
            with open(temp_file, "w") as f:
                f.write(generated_output_dir)
        else:
            # Other ranks poll file
            import time
            temp_file = os.path.join(base_dir, ".output_dir.tmp")
            max_wait = 30
            waited = 0
            while not os.path.exists(temp_file) and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1
            if os.path.exists(temp_file):
                with open(temp_file, "r") as f:
                    generated_output_dir = f.read().strip()
            else:
                raise RuntimeError(f"Failed to get output directory from rank 0 after {max_wait} seconds")
        
        training_args.output_dir = generated_output_dir
        
        # Distributed sync
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                # Broadcast output dir
                output_dir_bytes = generated_output_dir.encode('utf-8')
                output_dir_list = [output_dir_bytes]
                dist.broadcast_object_list(output_dir_list, src=0)
                training_args.output_dir = output_dir_list[0].decode('utf-8')
        except Exception:
            # Non-distributed: file sync only
            pass
        
        logger.info_rank0("\n" + "=" * 60)
        logger.info_rank0("Auto output path:")
        logger.info_rank0("=" * 60)
        logger.info_rank0(f"  Base: {base_dir}")
        logger.info_rank0(f"  Output: {generated_output_dir}")
        logger.info_rank0("=" * 60 + "\n")
    
    # Load tokenizers
    decoder_tokenizer_module = load_tokenizer(model_args)
    decoder_tokenizer = decoder_tokenizer_module["tokenizer"]
    
    import logging
    transformers_logger = logging.getLogger("transformers.tokenization_utils_base")
    original_level = transformers_logger.level
    transformers_logger.setLevel(logging.WARNING)
    
    # Load user tokenizer (usually BERT)
    from transformers import AutoTokenizer
    user_encoder_name = getattr(model_args, 'user_encoder_name', 'bert-base-uncased')
    try:
        # Check if user_encoder_name is a local path
        is_local_path = os.path.isabs(user_encoder_name) or os.path.exists(user_encoder_name)
        if is_local_path and os.path.isdir(user_encoder_name):
            # Load from local directory
            user_tokenizer = AutoTokenizer.from_pretrained(user_encoder_name, local_files_only=True)
        else:
            # Load from HuggingFace Hub or relative path
            user_tokenizer = AutoTokenizer.from_pretrained(user_encoder_name)
        if user_tokenizer.pad_token is None:
            user_tokenizer.pad_token = user_tokenizer.unk_token if user_tokenizer.unk_token else user_tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load user tokenizer from {user_encoder_name}: {e}")
        raise
    
    # Load model
    model = load_persona_steering_model(
        model_args, finetuning_args, 
        is_trainable=training_args.do_train,
    )
    
    # Load dataset
    train_dataset = None
    eval_dataset = None
    
    if training_args.do_train:
        dataset_path = getattr(data_args, 'dataset', None)
        if dataset_path is None:
            raise ValueError("dataset must be specified for training")
        
        train_dataset = load_persona_steering_dataset(
            data_path=dataset_path,
            user_tokenizer=user_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            max_user_length=getattr(data_args, 'max_user_length', 512),
            max_length=getattr(data_args, 'cutoff_len', 2048),
            decoder_prompt=getattr(data_args, 'decoder_prompt', None),
            output_dir=training_args.output_dir,
            task=getattr(data_args, 'task', 'generation_abstract'),
            max_samples_per_user=getattr(data_args, 'max_samples_per_user', None),
            max_profile_items=getattr(data_args, 'max_profile_items', 8),
            max_profile_item_len=getattr(data_args, 'max_profile_item_len', 128),
            max_query_length=getattr(data_args, 'max_query_length', 256),
            profile_topk_prescreen=getattr(data_args, 'profile_topk_prescreen', False),
            profile_topk=getattr(data_args, 'profile_topk', None),
        )
        logger.info_rank0(f"Train dataset size: {len(train_dataset)}")
    
    if training_args.do_eval:
        eval_dataset_path = getattr(data_args, 'eval_dataset', None) or getattr(data_args, 'dataset', None)
        if eval_dataset_path:
            eval_dataset = load_persona_steering_dataset(
                data_path=eval_dataset_path,
                user_tokenizer=user_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                max_user_length=getattr(data_args, 'max_user_length', 512),
                max_length=getattr(data_args, 'cutoff_len', 2048),
                decoder_prompt=getattr(data_args, 'decoder_prompt', None),
                output_dir=training_args.output_dir,
                task=getattr(data_args, 'task', 'generation_abstract'),
                max_samples_per_user=getattr(data_args, 'max_samples_per_user', None),
                max_profile_items=getattr(data_args, 'max_profile_items', 8),
                max_profile_item_len=getattr(data_args, 'max_profile_item_len', 128),
                max_query_length=getattr(data_args, 'max_query_length', 256),
                profile_topk_prescreen=getattr(data_args, 'profile_topk_prescreen', False),
                profile_topk=getattr(data_args, 'profile_topk', None),
            )
            logger.info_rank0(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = PersonaSteeringDataCollator(
        user_tokenizer=user_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if getattr(data_args, 'ignore_pad_token_for_loss', True) else decoder_tokenizer.pad_token_id,
    )
    
    # Get two-stage training parameters
    use_two_stage_training = getattr(model_args, 'use_two_stage_training', False)
    mlp_only_epochs = getattr(model_args, 'mlp_only_epochs', 1)
    
    logger.info_rank0("\n" + "=" * 60)
    logger.info_rank0("Using SFT Trainer (Reconstruction task)")
    logger.info_rank0("=" * 60)
    trainer = PersonaSteeringTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        user_tokenizer=user_tokenizer,
        model_args=model_args,
        use_two_stage_training=use_two_stage_training,
        mlp_only_epochs=mlp_only_epochs,
        callbacks=callbacks or [],
    )
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        logger.info_rank0("Training completed!")
    
    # Evaluation
    if training_args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info_rank0("Evaluation completed!")
    
    return trainer
