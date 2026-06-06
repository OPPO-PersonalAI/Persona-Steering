#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 OPPO Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Activation Steering Hook: Inject steering vector into decoder hidden states (dynamic mode only).
"""

import torch
from typing import Optional, Callable


class SteeringHook:
    """
    Hook to inject steering vector into decoder's hidden states during forward pass.
    Uses dynamic steering: user vector (Q) attends to context hidden states (K/V) to produce
    a context-dependent delta added at the hooked layer.
    """

    def __init__(self, steering_vector: Optional[torch.Tensor] = None, enable_debug: bool = False, coeff: float = 1.0):
        """
        Initialize steering hook.

        Args:
            steering_vector: User vector [batch_size, hidden_dim] (set via set_steering before use)
            enable_debug: Whether to enable debug logging
            coeff: Coefficient for scaling steering vector (default: 1.0)
        """
        self.steering_vector = steering_vector
        self.steering_generator = None
        self.context_attention_mask = None
        self.hook_handle = None
        self.enable_debug = enable_debug
        self.coeff = coeff
        self.call_count = 0
        self.last_hidden_shape = None
        self.last_steering_shape = None

    def set_steering(
        self,
        user_vector: torch.Tensor,
        steering_generator,
        context_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Set user vector and generator for dynamic steering (required before forward).

        Args:
            user_vector: User representation [batch_size, hidden_dim]
            steering_generator: Module with compute_dynamic_vector(context_hidden_states, user_projected_vector)
            context_attention_mask: Optional decoder padding mask [batch_size, seq_len]
        """
        self.steering_vector = user_vector
        self.steering_generator = steering_generator
        self.context_attention_mask = context_attention_mask
    
    def create_hook_fn(self, layer_idx: int = -1) -> Callable:
        """
        Create a hook function that injects steering vector into specified layer.
        
        Args:
            layer_idx: Layer index to inject steering vector (-1 for last layer)
            
        Returns:
            Hook function
        """

        def hook_fn(module, input, output):
            """
            Hook function to modify hidden states.
            
            Args:
                module: The module being hooked
                input: Input to the module
                output: Output from the module
            """
            # Increment call count for verification
            self.call_count += 1
            
            if self.steering_vector is None:
                if self.enable_debug:
                    print(f"[Hook Debug] Hook called (count: {self.call_count}) but steering_vector is None")
                return output
            
            base_steering_vector = self.steering_vector
            if base_steering_vector.requires_grad is False:
                # Detached steering_vector; warn only
                if self.enable_debug:
                    print(f"[Hook Debug] Warning: steering_vector.requires_grad is False")
            
            # Handle different output formats
            if isinstance(output, tuple):
                # For transformer layers, output is usually (hidden_states, ...)
                hidden_states = output[0]
            elif isinstance(output, torch.Tensor):
                hidden_states = output
            else:
                return output

            self.last_hidden_shape = hidden_states.shape
            self.last_steering_shape = base_steering_vector.shape

            if self.steering_generator is None:
                if self.enable_debug:
                    print(f"[Hook Debug] steering_generator is None, skipping steering")
                return output

            # Dynamic steering: user (Q) attends to context (K/V) -> delta [B, seq_len, H]
            if not hidden_states.requires_grad and torch.is_grad_enabled():
                hidden_states.requires_grad_(True)

            try:
                final_steering_vector = self.steering_generator.compute_dynamic_vector(
                    context_hidden_states=hidden_states,
                    user_projected_vector=base_steering_vector,
                    context_attention_mask=self.context_attention_mask,
                )
            except AttributeError:
                if self.enable_debug:
                    print(f"[Hook Debug] Steering generator missing compute_dynamic_vector method")
                raise AttributeError("Steering generator missing compute_dynamic_vector")
            except Exception as e:
                if self.enable_debug:
                    print(f"[Hook Error] compute_dynamic_vector failed: {e}")
                raise e

            steering_to_add = (self.coeff * final_steering_vector).to(dtype=hidden_states.dtype, device=hidden_states.device)

            if hidden_states.shape[-1] != steering_to_add.shape[-1]:
                if self.enable_debug:
                    print(f"[Hook Debug] Dimension mismatch: hidden={hidden_states.shape[-1]}, steering={steering_to_add.shape[-1]}")
                return output

            modified_hidden = hidden_states + steering_to_add
            if self.enable_debug and self.call_count <= 3:
                print(f"[Hook Debug] Hook called (count: {self.call_count})")
                print(f"  Hidden shape: {hidden_states.shape}, Steering shape: {steering_to_add.shape}")

            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            return modified_hidden
        
        return hook_fn
        
    def register_hook(
        self,
        model: torch.nn.Module,
        layer_name: str = None,
        layer_idx: int = -1
    ):
        """
        Register hook to the specified layer of the model.
        
        Args:
            model: The decoder model
            layer_name: Name of the layer to hook (if None, hooks the last transformer layer)
            layer_idx: Index of the layer to hook (-1 for last layer)
        """
        # Remove existing hook if any
        self.remove_hook()
        
        # Reset call count
        self.call_count = 0
        
        # Find the target layer
        if layer_name:
            target_layer = dict(model.named_modules())[layer_name]
            if self.enable_debug:
                print(f"[Hook Debug] Using specified layer: {layer_name}")
        else:
            # Find transformer layers (for LLaMA, typically 'model.layers')
            # Try common patterns: model.layers, transformer.layers, decoder.layers, etc.
            transformer_layers = None
            layer_key = None
            
            # Common patterns for transformer layers
            possible_keys = ['model.layers', 'transformer.layers', 'decoder.layers', 'layers']
            
            for key in possible_keys:
                if key in dict(model.named_modules()):
                    transformer_layers = dict(model.named_modules())[key]
                    layer_key = key
                    break
            
            # If not found, search for ModuleList containing layers
            if transformer_layers is None:
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                        # Check if it looks like transformer layers
                        first_layer = module[0]
                        if hasattr(first_layer, 'self_attn') or hasattr(first_layer, 'attention'):
                            transformer_layers = module
                            layer_key = name
                            break
            
            if transformer_layers is None:
                raise ValueError(
                    "Could not find transformer layers in the model. "
                    "Please specify layer_name explicitly."
                )
            
            if self.enable_debug:
                print(f"[Hook Debug] Found transformer layers: {layer_key}, total layers: {len(transformer_layers)}")
            
            # Validate and normalize layer index
            num_layers = len(transformer_layers)
            if layer_idx < 0:
                # Negative index: convert to positive (e.g., -1 -> last layer)
                actual_idx = num_layers + layer_idx
            else:
                actual_idx = layer_idx
            
            # Bounds checking
            if actual_idx < 0 or actual_idx >= num_layers:
                raise IndexError(
                    f"Layer index {layer_idx} (actual: {actual_idx}) is out of bounds. "
                    f"Model has {num_layers} layers (valid indices: 0 to {num_layers - 1}, or -1 to -{num_layers})"
                )
            
            # Get the target layer (last layer by default)
            target_layer = transformer_layers[actual_idx]
            
            if self.enable_debug:
                print(f"[Hook Debug] Targeting layer index: {layer_idx} (0-indexed: {actual_idx})")
                print(f"[Hook Debug] Hooking layer directly to modify the final residual stream: {type(target_layer).__name__}")
            # Hooking the layer itself will modify its final output (the residual stream)
        
        # Register forward hook
        hook_fn = self.create_hook_fn(layer_idx)
        self.hook_handle = target_layer.register_forward_hook(hook_fn)
        
        if self.enable_debug:
            print(f"[Hook Debug] Hook registered successfully")
        
        return self.hook_handle
    
    def get_hook_stats(self):
        """Get statistics about hook usage."""
        return {
            'call_count': self.call_count,
            'last_hidden_shape': self.last_hidden_shape,
            'last_steering_shape': self.last_steering_shape,
            'is_registered': self.hook_handle is not None,
        }
    
    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
