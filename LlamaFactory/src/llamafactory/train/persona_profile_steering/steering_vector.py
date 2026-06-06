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
Steering Vector Generator: MLP to map user representation to decoder hidden dimension.

Unified architecture:
    - Stream A (Explicit): Demographic -> long-term preference vector.
    - Stream B (Implicit): Query attend to Profile -> task-relevant vector.
    - Fusion: vec_A + vec_B (or weighted).
"""

import torch
import torch.nn as nn
from typing import Optional


class ProfileQueryAttention(nn.Module):
    """
    Stream B: Current input (Query) attends to Profile (K/V) -> task-relevant vector.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.o_proj = nn.Linear(output_dim, output_dim)
        self.scale = output_dim ** -0.5
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        nn.init.zeros_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        query_rep: torch.Tensor,      # [B, input_dim]
        profile_rep: torch.Tensor,   # [B, N, input_dim]
        profile_mask: torch.Tensor,   # [B, N], 1=valid, 0=pad
    ) -> torch.Tensor:
        B, N, _ = profile_rep.shape
        q = self.query_proj(query_rep).unsqueeze(1)   # [B, 1, output_dim]
        k = self.k_proj(profile_rep)                   # [B, N, output_dim]
        v = self.v_proj(profile_rep)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(profile_mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        out = (attn @ v).squeeze(1)
        return self.o_proj(out)   # [B, output_dim]


class QueryDemographicAttention(nn.Module):
    """
    Stream A (replacement): Query attend to Demographic sequence -> only query-relevant demographic.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.o_proj = nn.Linear(output_dim, output_dim)
        self.scale = output_dim ** -0.5
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        nn.init.zeros_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        query_rep: torch.Tensor,           # [B, input_dim]
        demographic_sequence: torch.Tensor, # [B, L, input_dim]
        demographic_mask: torch.Tensor,    # [B, L], 1=valid 0=pad
    ) -> torch.Tensor:
        q = self.query_proj(query_rep).unsqueeze(1)
        k = self.k_proj(demographic_sequence)
        v = self.v_proj(demographic_sequence)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(demographic_mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        out = (attn @ v).squeeze(1)
        return self.o_proj(out)   # [B, output_dim]


class SteeringVectorGenerator(nn.Module):
    """
    Steering Vector Generator: Maps user representation to decoder hidden dimension.
    
    Supports both static (user_only) and dynamic (user + context) generation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        fusion_alpha_min: float = 0.1,
        fusion_alpha_max: float = 0.9,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fusion_alpha_min = fusion_alpha_min
        self.fusion_alpha_max = fusion_alpha_max

        # 1. Input Normalization (Critical for Stability)
        # Prevents high-norm vectors from user encoder or decoder from saturating/disturbing the MLP
        self.user_norm = nn.LayerNorm(input_dim)
        self.context_norm = nn.LayerNorm(output_dim)

        # 2. Dynamic Steering: Cross-Attention (User as Q, Context as K/V) — replaces Concat+MLP
        # User representation attends to decoder context to produce one steering vector per sample.
        self.cross_attn_q_proj = nn.Linear(output_dim, output_dim)
        self.cross_attn_k_proj = nn.Linear(output_dim, output_dim)
        self.cross_attn_v_proj = nn.Linear(output_dim, output_dim)
        self.cross_attn_o_proj = nn.Linear(output_dim, output_dim)
        self._scale = output_dim ** -0.5
        # Optional: bound steering magnitude (set to None to disable)
        # self.dynamic_tanh = nn.Tanh()

        # Small init so initial steering is small
        for proj in (self.cross_attn_q_proj, self.cross_attn_k_proj, self.cross_attn_v_proj):
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        nn.init.zeros_(self.cross_attn_o_proj.weight)
        if self.cross_attn_o_proj.bias is not None:
            nn.init.zeros_(self.cross_attn_o_proj.bias)

        # Stream B: Query attend to Profile -> task-relevant vector
        self.profile_query_attn = ProfileQueryAttention(input_dim, output_dim)
        # Stream A (replacement): Query attend to Demographic -> only query-relevant demographic
        self.query_demographic_attn = QueryDemographicAttention(input_dim, output_dim)

        # Adaptive Fusion Gate: decides fusion weight between demographic (A) and history (B)
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(output_dim)
        
        # For logging/debugging
        self.last_alpha = None

    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "swish": nn.SiLU(),
        }
        return activations.get(activation.lower(), nn.GELU())
    
    def forward(self,
                demographic_sequence: torch.Tensor,
                demographic_mask: torch.Tensor,
                query_representation: torch.Tensor,
                context_hidden_states: Optional[torch.Tensor] = None,
                profile_representations: Optional[torch.Tensor] = None,
                profile_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate steering vector (Stream A = Query×Demographic, Stream B = Query×Profile, fused).
        
        Args:
            demographic_sequence: [Batch, L_demo, User_Dim] from encoder(return_sequence=True)
            demographic_mask: [Batch, L_demo] 1=valid 0=pad
            query_representation: Current query [Batch, User_Dim]
            context_hidden_states: Decoder hidden states (optional, for dynamic)
            profile_representations: Profile items [Batch, N, User_Dim] for Stream B (optional)
            profile_mask: [Batch, N] 1=valid 0=pad (optional)
            
        Returns:
            Steering vector [Batch, 1, Decoder_Dim] (or [Batch, Seq_Len, Decoder_Dim] if context provided)
        """
        query_norm = self.user_norm(query_representation)
        
        # Reset last_alpha for current batch
        self.last_alpha = None
        
        # 1. Compute Stream A (Explicit Demographic)
        vec_A = self.query_demographic_attn(query_norm, demographic_sequence, demographic_mask)

        # 2. Compute Stream B (Implicit History) and perform Adaptive Fusion
        if profile_representations is not None and profile_mask is not None:
            vec_B = self.profile_query_attn(query_norm, profile_representations, profile_mask)
            
            # Adaptive Fusion Gate
            gate_input = torch.cat([vec_A, vec_B], dim=-1) # [B, 2*Dim]
            alpha = self.fusion_gate(gate_input) # [B, 1]
            alpha = torch.clamp(alpha, min=self.fusion_alpha_min, max=self.fusion_alpha_max)
            try:
                self.last_alpha = alpha.mean().detach().cpu().item()
            except Exception:
                self.last_alpha = None
            
            # Weighted sum for stability
            raw_steering = alpha * vec_A + (1 - alpha) * vec_B
            steering = self.fusion_norm(raw_steering)
        else:
            # Fallback
            steering = self.fusion_norm(vec_A)

        if context_hidden_states is None:
            return steering.unsqueeze(1)
        return self.compute_dynamic_vector(context_hidden_states, steering)

    def compute_dynamic_vector(
        self,
        context_hidden_states: torch.Tensor,
        user_projected_vector: torch.Tensor,
        context_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute dynamic steering vector via Cross-Attention: User (Q) attends to Context (K/V).
        Returns one vector per sample, broadcast to [B, Seq_Len, Dim] for adding to hidden states.

        Args:
            context_hidden_states: [Batch, Seq_Len, Dim]
            user_projected_vector: [Batch, Dim]
            context_attention_mask: Optional decoder padding mask [Batch, Seq_Len]

        Returns:
            Dynamic vector: [Batch, Seq_Len, Dim]
        """
        out_dtype = context_hidden_states.dtype
        compute_dtype = self.cross_attn_q_proj.weight.dtype
        _, seq_len, _ = context_hidden_states.shape

        # Decoder hidden states are often bfloat16 while steering MLP weights stay float32;
        # run dynamic attention in compute_dtype, then cast back for residual add.
        # also ensure they are on the same device as the steering generator.
        compute_device = self.cross_attn_q_proj.weight.device
        context = context_hidden_states.to(device=compute_device, dtype=compute_dtype)
        user_pv = user_projected_vector.to(device=compute_device, dtype=compute_dtype)

        # Normalize context for stable attention
        context = self.context_norm(context)  # [B, L, H]

        # Q from user: [B, H] -> [B, 1, H]
        if user_pv.dim() == 2:
            user_q = user_pv.unsqueeze(1)
        else:
            user_q = user_pv
        q = self.cross_attn_q_proj(user_q)  # [B, 1, H]
        k = self.cross_attn_k_proj(context)  # [B, L, H]
        v = self.cross_attn_v_proj(context)  # [B, L, H]

        # Attention: scores [B, 1, L], output [B, 1, H]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        if context_attention_mask is not None:
            mask = context_attention_mask.to(device=scores.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            elif mask.dim() != 3:
                raise ValueError(
                    f"context_attention_mask must have shape [B, L] or [B, 1, L], got {tuple(mask.shape)}"
                )
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.matmul(attn_weights, v)  # [B, 1, H]
        out = self.cross_attn_o_proj(out)
        # out = self.dynamic_tanh(out)

        # Broadcast to [B, seq_len, H] so we add the same steering vector at every position
        delta = out.expand(-1, seq_len, -1)
        return delta.to(out_dtype)
