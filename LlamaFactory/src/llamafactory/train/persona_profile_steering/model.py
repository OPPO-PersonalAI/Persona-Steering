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
Main Model: Integrates User Encoder, Steering Vector Generator, and Frozen LLM Decoder
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from .user_encoder import UserEncoder
from .steering_vector import SteeringVectorGenerator
from .steering_hook import SteeringHook


class UserEncoderConfig(PretrainedConfig):
    """Configuration for User Encoder Model."""
    model_type = "user_encoder"
    
    def __init__(
        self,
        user_encoder_name: str = "bert",
        # User encoder LoRA (optional; recommended for large encoders to avoid AdamW OOM)
        user_encoder_use_lora: bool = False,
        user_encoder_lora_r: int = 8,
        user_encoder_lora_alpha: int = 16,
        user_encoder_lora_dropout: float = 0.05,
        user_encoder_lora_target_modules: Optional[List[str]] = None,
        decoder_name: str = "Qwen/Qwen2.5-7B-Instruct",
        decoder_hidden_size: Optional[int] = None,
        steering_mlp_activation: str = "gelu",
        steering_mlp_dropout: float = 0.1,
        steering_mlp_use_layer_norm: bool = True,
        steering_layer_idx: int = 16,
        steering_coeff: float = 1.0,
        fusion_alpha_min: float = 0.1,
        fusion_alpha_max: float = 0.9,
        **kwargs,
    ):
        # Set name_or_path before super().__init__ so PreTrainedConfig records _name_or_path.
        if 'name_or_path' not in kwargs:
            kwargs['name_or_path'] = decoder_name
        
        super().__init__(**kwargs)
        self.user_encoder_name = user_encoder_name
        self.user_encoder_use_lora = user_encoder_use_lora
        self.user_encoder_lora_r = user_encoder_lora_r
        self.user_encoder_lora_alpha = user_encoder_lora_alpha
        self.user_encoder_lora_dropout = user_encoder_lora_dropout
        self.user_encoder_lora_target_modules = user_encoder_lora_target_modules
        self.decoder_name = decoder_name
        self.decoder_hidden_size = decoder_hidden_size
        self.steering_mlp_activation = steering_mlp_activation
        self.steering_mlp_dropout = steering_mlp_dropout
        self.steering_mlp_use_layer_norm = steering_mlp_use_layer_norm
        self.steering_layer_idx = steering_layer_idx
        self.steering_coeff = steering_coeff
        self.fusion_alpha_min = fusion_alpha_min
        self.fusion_alpha_max = fusion_alpha_max


class UserEncoderModel(nn.Module):
    """
    Main model that integrates:
    1. User Encoder (BERT or Qwen) - trainable
    2. Steering Vector Generator (MLP) - trainable
    3. LLM Decoder (Qwen2.5-7B-Instruct) - frozen
    4. Activation Steering Hook - injects steering vector
    """
    
    def __init__(
        self,
        config: Optional[UserEncoderConfig] = None,
        user_encoder_name: Optional[str] = None,
        decoder_name: Optional[str] = None,
        decoder_model: Optional[PreTrainedModel] = None,
        decoder_hidden_size: Optional[int] = None,
        steering_layer_idx: Optional[int] = None,
    ):
        """
        Initialize the User Encoder Model.
        
        Args:
            config: UserEncoderConfig instance (optional)
            user_encoder_name: Model name for user encoder (BERT or Qwen, if config not provided)
            decoder_name: LLM model name for decoder (if config not provided)
            decoder_model: Pre-loaded decoder model (optional)
            decoder_hidden_size: Hidden size of decoder (auto-detected if None)
            steering_layer_idx: Layer index to inject steering vector (None to use config value, -1 for last layer, 0 for first layer)
        """
        super().__init__()
        
        # Use config if provided, otherwise use individual arguments
        if config is not None:
            user_encoder_name = config.user_encoder_name
            decoder_name = config.decoder_name
            decoder_hidden_size = config.decoder_hidden_size
            if steering_layer_idx is None:
                steering_layer_idx = config.steering_layer_idx
            steering_coeff = getattr(config, 'steering_coeff', 1.0)
            self.config = config
        else:
            # Create default config
            final_steering_layer_idx = steering_layer_idx if steering_layer_idx is not None else -1
            steering_coeff = 1.0
            self.config = UserEncoderConfig(
                user_encoder_name=user_encoder_name or "Qwen/Qwen2-1.5B",
                decoder_name=decoder_name or "Qwen/Qwen2.5-7B-Instruct",
                decoder_hidden_size=decoder_hidden_size,
                steering_layer_idx=final_steering_layer_idx,
                steering_coeff=steering_coeff,
            )
            user_encoder_name = self.config.user_encoder_name
            decoder_name = self.config.decoder_name
            decoder_hidden_size = self.config.decoder_hidden_size
            steering_layer_idx = final_steering_layer_idx
        
        # User Encoder (trainable)
        self.user_encoder = UserEncoder(
            model_name=user_encoder_name,
            use_lora=getattr(self.config, "user_encoder_use_lora", False),
            lora_r=getattr(self.config, "user_encoder_lora_r", 8),
            lora_alpha=getattr(self.config, "user_encoder_lora_alpha", 16),
            lora_dropout=getattr(self.config, "user_encoder_lora_dropout", 0.05),
            lora_target_modules=getattr(self.config, "user_encoder_lora_target_modules", None),
        )
        self.user_encoder_use_lora = getattr(self.config, "user_encoder_use_lora", False)
        
        # Get decoder hidden size
        if decoder_hidden_size is None:
            try:
                decoder_config = AutoConfig.from_pretrained(decoder_name)
                decoder_hidden_size = decoder_config.hidden_size
            except Exception as e:
                decoder_hidden_size = 3584  # Default for Qwen2.5-7B-Instruct
        
        # Steering Vector Generator (trainable)
        steering_generator_kwargs = {
            'input_dim': self.user_encoder.hidden_size,
            'output_dim': decoder_hidden_size,
            'activation': getattr(self.config, 'steering_mlp_activation', 'gelu'),
            'dropout': getattr(self.config, 'steering_mlp_dropout', 0.1),
            'use_layer_norm': getattr(self.config, 'steering_mlp_use_layer_norm', True),
            'fusion_alpha_min': getattr(self.config, 'fusion_alpha_min', 0.1),
            'fusion_alpha_max': getattr(self.config, 'fusion_alpha_max', 0.9),
        }
        self.steering_generator = SteeringVectorGenerator(**steering_generator_kwargs)
        
        # LLM Decoder (frozen)
        if decoder_model is not None:
            self.decoder = decoder_model
        else:
            try:
                attn_implementation = None
                try:
                    import flash_attn
                    attn_implementation = "flash_attention_2"
                except ImportError:
                    pass
                
                decoder_kwargs = {
                    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    "low_cpu_mem_usage": True,
                }
                
                if attn_implementation is not None:
                    decoder_kwargs["attn_implementation"] = attn_implementation
                
                self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name, **decoder_kwargs)
            except Exception as e:
                self.decoder = None
        
        if self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.decoder.eval()
        
        # Steering Hook
        self.steering_hook = SteeringHook(coeff=steering_coeff)
        self.steering_layer_idx = steering_layer_idx
        
        # Tokenizers
        self.user_tokenizer = self.user_encoder.get_tokenizer()
        if self.decoder is not None:
            try:
                self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
            except:
                self.decoder_tokenizer = None
        else:
            self.decoder_tokenizer = None
    
    def register_steering_hook(self):
        """Register the steering hook to the decoder."""
        if self.decoder is not None:
            self.steering_hook.register_hook(
                self.decoder,
                layer_idx=self.steering_layer_idx
            )
    
    def get_hook_stats(self):
        """Get statistics about hook usage."""
        return self.steering_hook.get_hook_stats()
    
    def remove_steering_hook(self):
        """Remove the steering hook."""
        self.steering_hook.remove_hook()
    
    def forward(
        self,
        user_input_ids: torch.Tensor,
        user_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_steering: bool = True,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        profile_input_ids: Optional[torch.Tensor] = None,
        profile_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model (dual-stream: demographic + profile/query).
        
        Args:
            user_input_ids: Demographic token IDs [batch_size, user_seq_len] (Stream A)
            user_attention_mask: Demographic attention mask
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Decoder attention mask
            use_steering: Whether to use steering vector injection
            query_input_ids: Current query token IDs [B, Lq] for Stream B (optional)
            query_attention_mask: Query attention mask (optional)
            profile_input_ids: Profile token IDs [B, N, L] for Stream B (optional)
            profile_attention_mask: Profile attention mask [B, N, L] (optional)
            
        Returns:
            Dictionary containing: user_representation, steering_vector, decoder_outputs
        """
        # 1. Stream A: Demographic as sequence (Query will attend to it)
        demographic_sequence = self.user_encoder(
            input_ids=user_input_ids,
            attention_mask=user_attention_mask,
            return_sequence=True,
        )
        demographic_mask = (user_attention_mask > 0).long() if user_attention_mask is not None else (user_input_ids != 0).long()

        # 2. Query: required for Stream A (Query×Demographic)
        if query_input_ids is not None:
            query_representation = self.user_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )
        else:
            query_representation = self.user_encoder(
                input_ids=user_input_ids,
                attention_mask=user_attention_mask,
            )

        profile_representations = None
        profile_mask = None
        if profile_input_ids is not None:
            B, N, L = profile_input_ids.shape
            profile_flat = profile_input_ids.view(B * N, L)
            profile_flat_mask = profile_attention_mask.view(B * N, L) if profile_attention_mask is not None else None
            profile_rep_flat = self.user_encoder(
                input_ids=profile_flat,
                attention_mask=profile_flat_mask,
            )
            profile_representations = profile_rep_flat.view(B, N, -1)
            profile_mask = (profile_attention_mask.sum(dim=-1) > 0).long() if profile_attention_mask is not None else (profile_input_ids != 0).any(dim=-1).long()
        # Fallback to prevent NaN in Softmax if a user has no profile items
        # If the user has absolutely no profile items, profile_mask might be all zeros.
        # Attention requires at least one valid key/value to attend to per sequence,
        # otherwise Softmax outputs NaN. If a user has no profile, we artificially
        # unmask the first profile item (which contains just pad tokens/zeros), 
        # so that attention has something to attend to, preventing NaNs.
        # Since this path is taken, the attended representation will just be the 
        # projection of the padding token, effectively a constant zero-knowledge vector.
        if profile_mask is not None:
            # Check if any user in the batch has an all-zero profile mask
            all_zero_users = (profile_mask.sum(dim=-1) == 0)
            if all_zero_users.any():
                # For those users, unmask the first profile item
                profile_mask[all_zero_users, 0] = 1

        if demographic_mask is not None:
            all_zero_demo = (demographic_mask.sum(dim=-1) == 0)
            if all_zero_demo.any():
                demographic_mask[all_zero_demo, 0] = 1

        # 3. Generate steering: Stream A = Query×Demographic, Stream B = Query×Profile
        steering_vector = self.steering_generator(
            demographic_sequence=demographic_sequence,
            demographic_mask=demographic_mask,
            query_representation=query_representation,
            context_hidden_states=None,
            profile_representations=profile_representations,
            profile_mask=profile_mask,
        )
        
        # Check for NaN or Inf in steering vector
        # if torch.isnan(steering_vector).any() or torch.isinf(steering_vector).any():
        #     steering_vector = torch.where(
        #         torch.isnan(steering_vector) | torch.isinf(steering_vector),
        #         torch.zeros_like(steering_vector),
        #         steering_vector
        #     )
        
        # # Clip steering vector magnitude
        # max_steering_norm = 10.0
        # steering_norm = torch.norm(steering_vector, dim=-1, keepdim=True)
        # if (steering_norm > max_steering_norm).any():
        #     scale_factor = torch.clamp(max_steering_norm / (steering_norm + 1e-8), max=1.0)
        #     steering_vector = steering_vector * scale_factor
        
        # 3. Ensure steering_vector has the same dtype as decoder
        if self.decoder is not None:
            try:
                decoder_dtype = next(self.decoder.parameters()).dtype
            except StopIteration:
                if hasattr(self.decoder, 'dtype'):
                    decoder_dtype = self.decoder.dtype
                elif hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'embed_tokens'):
                    decoder_dtype = self.decoder.model.embed_tokens.weight.dtype
                else:
                    decoder_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            steering_vector = steering_vector.to(dtype=decoder_dtype)
        

        
        # 4. Set user vector and generator for dynamic steering hook
        if use_steering:
            static_user_vec = steering_vector.squeeze(1)
            if not static_user_vec.requires_grad:
                static_user_vec = static_user_vec.requires_grad_(True)
            self.steering_hook.set_steering(
                user_vector=static_user_vec,
                steering_generator=self.steering_generator,
                context_attention_mask=decoder_attention_mask,
            )
            
            if not self.steering_hook.hook_handle:
                self.register_steering_hook()
        else:
            self.steering_hook.set_steering(
                user_vector=None,
                steering_generator=None,
                context_attention_mask=None,
            )
            if self.steering_hook.hook_handle:
                self.remove_steering_hook()
        
        # 5. Forward through decoder (if provided)
        decoder_outputs = None
        # Keep return field name "steering_vector" for API compatibility.
        steering_vector_ret = static_user_vec.unsqueeze(1) if use_steering else torch.zeros(1, 1, 1).to(query_representation.device)
        if decoder_input_ids is not None and self.decoder is not None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                return_dict=True,
            )
        
        return {
            "user_representation": query_representation,
            "steering_vector": steering_vector,
            "decoder_outputs": decoder_outputs,
        }
    
    def insert_steering_vector_via_special_token(
        self,
        user_input_ids: torch.Tensor,
        user_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: torch.Tensor = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        steering_token_id: Optional[int] = None,
        profile_input_ids: Optional[torch.Tensor] = None,
        profile_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Insert steering vector into decoder input via special token (like CURP).
        This method replaces the hook-based injection with direct embedding insertion.
        
        Args:
            user_input_ids: User input token IDs [batch_size, user_seq_len]
            user_attention_mask: User attention mask [batch_size, user_seq_len]
            decoder_input_ids: Decoder input token IDs [batch_size, decoder_seq_len]
                               Should contain special token (e.g., <USER>) to mark insertion point
            decoder_attention_mask: Decoder attention mask [batch_size, decoder_seq_len]
            steering_token_id: Token ID of special token. If None, will try to auto-detect.
            profile_input_ids: Profile token IDs [B, N, L] for Stream B (optional)
            profile_attention_mask: Profile attention mask [B, N, L] (optional)
            
        Returns:
            Dictionary containing:
                - user_representation: User representation
                - steering_vector: Generated steering vector
                - inputs_embeds: Combined embeddings with steering vector inserted
                - attention_mask: Combined attention mask
        """
        from torch.nn.utils.rnn import pad_sequence
        
        device = next(self.parameters()).device
        batch_size = decoder_input_ids.shape[0]
        
        # 1. Demographic as sequence + query (use pooled demographic as query when no query given)
        demographic_sequence = self.user_encoder(
            input_ids=user_input_ids,
            attention_mask=user_attention_mask,
            return_sequence=True,
        )
        demographic_mask = (user_attention_mask > 0).long() if user_attention_mask is not None else (user_input_ids != 0).long()
        
        if demographic_mask is not None:
            all_zero_demo = (demographic_mask.sum(dim=-1) == 0)
            if all_zero_demo.any():
                demographic_mask[all_zero_demo, 0] = 1
                
        query_representation = self.user_encoder(
            input_ids=user_input_ids,
            attention_mask=user_attention_mask,
        )
        
        profile_representations = None
        profile_mask = None
        if profile_input_ids is not None:
            B, N, L = profile_input_ids.shape
            profile_flat = profile_input_ids.view(B * N, L)
            profile_flat_mask = profile_attention_mask.view(B * N, L) if profile_attention_mask is not None else None
            profile_rep_flat = self.user_encoder(
                input_ids=profile_flat,
                attention_mask=profile_flat_mask,
            )
            profile_representations = profile_rep_flat.view(B, N, -1)
            profile_mask = (profile_attention_mask.sum(dim=-1) > 0).long() if profile_attention_mask is not None else (profile_input_ids != 0).any(dim=-1).long()
            
        if profile_mask is not None:
            all_zero_users = (profile_mask.sum(dim=-1) == 0)
            if all_zero_users.any():
                profile_mask[all_zero_users, 0] = 1

        steering_vector = self.steering_generator(
            demographic_sequence=demographic_sequence,
            demographic_mask=demographic_mask,
            query_representation=query_representation,
            context_hidden_states=None,
            profile_representations=profile_representations,
            profile_mask=profile_mask,
        )
        steering_vector = steering_vector.squeeze(1)
        
        # Check for NaN or Inf
        if torch.isnan(steering_vector).any() or torch.isinf(steering_vector).any():
            steering_vector = torch.where(
                torch.isnan(steering_vector) | torch.isinf(steering_vector),
                torch.zeros_like(steering_vector),
                steering_vector
            )
        
        # Clip steering vector magnitude
        max_steering_norm = 10.0
        steering_norm = torch.norm(steering_vector, dim=-1, keepdim=True)
        if (steering_norm > max_steering_norm).any():
            scale_factor = torch.clamp(max_steering_norm / (steering_norm + 1e-8), max=1.0)
            steering_vector = steering_vector * scale_factor
        
        # 3. Ensure steering_vector has the same dtype as decoder
        if self.decoder is not None:
            try:
                decoder_dtype = next(self.decoder.parameters()).dtype
            except StopIteration:
                if hasattr(self.decoder, 'dtype'):
                    decoder_dtype = self.decoder.dtype
                elif hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'embed_tokens'):
                    decoder_dtype = self.decoder.model.embed_tokens.weight.dtype
                else:
                    decoder_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            steering_vector = steering_vector.to(dtype=decoder_dtype)
        
        # 4. Get decoder embeddings
        decoder_embeddings = self.decoder.get_input_embeddings()(decoder_input_ids)  # [batch_size, seq_len, hidden_size]
        
        # 5. Find special token position
        if steering_token_id is None:
            # Try to auto-detect common special token IDs
            if self.decoder_tokenizer is not None:
                try:
                    steering_token_id = self.decoder_tokenizer.convert_tokens_to_ids("<USER>")
                    if steering_token_id == self.decoder_tokenizer.unk_token_id:
                        steering_token_id = self.decoder_tokenizer.convert_tokens_to_ids("[USER]")
                except:
                    pass
        
        if steering_token_id is None:
            raise ValueError("steering_token_id must be provided or auto-detected. "
                           "Please ensure <USER> or [USER] token exists in tokenizer.")
        
        # Find insertion position
        match_mask = (decoder_input_ids == steering_token_id)
        if not match_mask.any(dim=1).all():
            raise ValueError(f"Special token {steering_token_id} not found in all samples. "
                            f"Please add the special token to decoder_input_ids.")
        
        insert_pos = match_mask.float().argmax(dim=1)  # [batch_size]
        
        # 6. Split decoder embeddings: before, at special token, after
        before_emb = []
        after_emb = []
        before_mask = []
        after_mask = []
        
        for b in range(batch_size):
            idx = insert_pos[b].item()
            before_emb.append(decoder_embeddings[b, :idx])  # [L_before, hidden_size]
            after_emb.append(decoder_embeddings[b, idx + 1:])  # [L_after, hidden_size]
            if decoder_attention_mask is not None:
                before_mask.append(decoder_attention_mask[b, :idx])
                after_mask.append(decoder_attention_mask[b, idx + 1:])
        
        # 7. Insert steering vector (expand to [batch_size, 1, hidden_size] for concatenation)
        steering_emb = steering_vector.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 8. Concatenate: [before, steering_vector, after]
        combined_embeddings_list = []
        combined_masks_list = []
        
        for b in range(batch_size):
            combined_emb = torch.cat([before_emb[b], steering_emb[b], after_emb[b]], dim=0)
            combined_embeddings_list.append(combined_emb)
            
            if decoder_attention_mask is not None:
                steering_mask = torch.ones(1, dtype=decoder_attention_mask.dtype, device=device)
                combined_mask = torch.cat([before_mask[b], steering_mask, after_mask[b]], dim=0)
                combined_masks_list.append(combined_mask)
        
        # 9. Pad to same length (left padding for decoder)
        inputs_embeds = pad_sequence(
            combined_embeddings_list, 
            batch_first=True, 
            padding_value=0.0
        )
        
        if decoder_attention_mask is not None:
            final_attention_mask = pad_sequence(
                combined_masks_list,
                batch_first=True,
                padding_value=0
            )
        else:
            final_attention_mask = None
        
        return {
            "user_representation": query_representation,
            "steering_vector": steering_vector,
            "inputs_embeds": inputs_embeds,
            "attention_mask": final_attention_mask,
        }
    
    def freeze_user_encoder(self):
        """Freeze user encoder parameters."""
        for param in self.user_encoder.parameters():
            param.requires_grad = False
        self.user_encoder.eval()
    
    def unfreeze_user_encoder(self):
        """Unfreeze user encoder parameters."""
        if getattr(self, "user_encoder_use_lora", False):
            for name, param in self.user_encoder.named_parameters():
                param.requires_grad = ("lora_" in name)
        else:
            for param in self.user_encoder.parameters():
                param.requires_grad = True
        self.user_encoder.train()
    
    def get_trainable_parameters(self, include_user_encoder: bool = True):
        """
        Get trainable parameters.
        
        Args:
            include_user_encoder: If True, include user encoder parameters. 
                                 If False, only return steering generator parameters.
        
        Returns:
            List of trainable parameters
        """
        trainable_params = []
        if include_user_encoder:
            trainable_params.extend(list(self.user_encoder.parameters()))
        trainable_params.extend(list(self.steering_generator.parameters()))
        return trainable_params
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for the model.
        
        This method is called by Transformers Trainer when gradient_checkpointing is enabled.
        It enables gradient checkpointing on the decoder (if supported) to save memory.
        
        Args:
            gradient_checkpointing_kwargs: Optional kwargs for gradient checkpointing
        """
        # Enable gradient checkpointing on decoder if it supports it
        if self.decoder is not None:
            if hasattr(self.decoder, 'gradient_checkpointing_enable'):
                self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            elif hasattr(self.decoder, 'enable_gradient_checkpointing'):
                # Some models use enable_gradient_checkpointing instead
                self.decoder.enable_gradient_checkpointing()
        
        # User encoder typically doesn't need gradient checkpointing (it's smaller)
        # But if it's a large model and supports it, we can enable it
        if hasattr(self.user_encoder, 'model') and hasattr(self.user_encoder.model, 'gradient_checkpointing_enable'):
            self.user_encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the model.
        """
        if self.decoder is not None:
            if hasattr(self.decoder, 'gradient_checkpointing_disable'):
                self.decoder.gradient_checkpointing_disable()
            elif hasattr(self.decoder, 'disable_gradient_checkpointing'):
                self.decoder.disable_gradient_checkpointing()
        
        if hasattr(self.user_encoder, 'model') and hasattr(self.user_encoder.model, 'gradient_checkpointing_disable'):
            self.user_encoder.model.gradient_checkpointing_disable()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[UserEncoderConfig] = None,
        **kwargs
    ):
        """
        Load model from checkpoint.
        
        Args:
            pretrained_model_name_or_path: Path to checkpoint directory
            config: UserEncoderConfig instance (optional, will load from checkpoint if available)
            **kwargs: Additional arguments passed to __init__
        
        Returns:
            UserEncoderModel instance
        """
        import os
        
        # Load config if available
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path) and config is None:
            try:
                config = UserEncoderConfig.from_json_file(config_path)
            except Exception as e:
                pass
        
        # Create model instance
        if config is not None:
            model = cls(config=config, **kwargs)
        else:
            model = cls(**kwargs)
        
        # Load weights
        steering_model_path = os.path.join(pretrained_model_name_or_path, "steering_model.pt")
        if os.path.exists(steering_model_path):
            checkpoint = torch.load(steering_model_path, map_location='cpu')
            if 'user_encoder' in checkpoint:
                model.user_encoder.load_state_dict(checkpoint['user_encoder'])
            if 'steering_generator' in checkpoint:
                model.steering_generator.load_state_dict(checkpoint['steering_generator'])
        
        return model