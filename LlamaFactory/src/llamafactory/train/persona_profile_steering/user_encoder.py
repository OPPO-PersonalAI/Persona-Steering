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
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""""User encoder: Qwen via AutoModel; BERT/RoBERTa/etc. via AutoModel."""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def _apply_lora_to_encoder(
    base: nn.Module,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[List[str]],
    default_targets: List[str],
) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise ImportError(
            "LoRA requested for user encoder, but `peft` is not available. Install with: pip install peft"
        ) from e

    for p in base.parameters():
        p.requires_grad = False

    targets = lora_target_modules if lora_target_modules else default_targets
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(targets),
        bias="none",
    )
    return get_peft_model(base, lora_cfg)


class UserEncoder(nn.Module):
    """
    User encoder: Qwen-family (mean pool) or encoder models (BERT, RoBERTa, etc.) via AutoModel.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = None,
        use_qwen: bool = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        self.tokenizer = None
        self.model_name_or_path = model_name
        self.cache_dir = None

        if use_qwen is None:
            use_qwen = "qwen" in model_name.lower() or "qwen2" in model_name.lower()

        self.use_qwen = use_qwen
        self.model_name = model_name
        self.use_lora = bool(use_lora)

        base = AutoModel.from_pretrained(model_name)

        if self.use_qwen:
            if self.use_lora:
                default_targets = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
                self.model = _apply_lora_to_encoder(
                    base,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_target_modules=lora_target_modules,
                    default_targets=default_targets,
                )
            else:
                self.model = base
        else:
            if self.use_lora:
                default_targets = ["query", "key", "value"]
                self.model = _apply_lora_to_encoder(
                    base,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_target_modules=lora_target_modules,
                    default_targets=default_targets,
                )
            else:
                self.model = base

        self.config = self.model.config
        self.hidden_size = hidden_size or getattr(self.config, "hidden_size", None) or getattr(
            self.config, "d_model", None
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        return_sequence: bool = False,
    ):
        if self.use_qwen:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            if return_sequence:
                return outputs.last_hidden_state
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (outputs.last_hidden_state * attention_mask_expanded).sum(dim=1)
                sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
                representation = sum_embeddings / sum_mask
            else:
                representation = outputs.last_hidden_state.mean(dim=1)
        else:
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids
            outputs = self.model(**model_inputs)
            if return_sequence:
                return outputs.last_hidden_state
            representation = outputs.last_hidden_state[:, 0, :]
        return representation

    def get_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer

        return AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            do_lower_case=True if "uncased" in self.model_name_or_path.lower() else False,
            cache_dir=self.cache_dir,
            use_fast=True,
            trust_remote_code=True,
        )
