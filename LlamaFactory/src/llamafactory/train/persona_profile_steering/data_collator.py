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

"""
Data collator for PersonaSteering training
Handles batching of user inputs and decoder inputs
Based on SteeringDataCollator from trainer_reconstruction.py
"""

from typing import Any, Dict, List, Optional
import torch

from ...extras.constants import IGNORE_INDEX


class PersonaSteeringDataCollator:
    """Data collator for PersonaSteering model with dynamic padding."""
    
    def __init__(
        self,
        user_tokenizer,
        decoder_tokenizer,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = IGNORE_INDEX,
    ):
        self.user_tokenizer = user_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        
        # Get pad token IDs
        self.user_pad_token_id = getattr(user_tokenizer, "pad_token_id", None) or 0
        self.decoder_pad_token_id = getattr(decoder_tokenizer, "pad_token_id", None) or 0
        
        # Get padding side
        self.user_padding_side = getattr(user_tokenizer, "padding_side", "right")
        self.decoder_padding_side = getattr(decoder_tokenizer, "padding_side", "right")
    
    @staticmethod
    def _pad_1d(t: torch.Tensor, target_len: int, pad_value: int, side: str) -> torch.Tensor:
        """Pad 1D tensor to target length."""
        pad_len = int(target_len) - int(t.size(0))
        if pad_len <= 0:
            return t
        pad = t.new_full((pad_len,), pad_value)
        if side == "left":
            return torch.cat([pad, t], dim=0)
        return torch.cat([t, pad], dim=0)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features with dynamic padding.
        
        Expected features format:
        {
            'user_input_ids': torch.Tensor (1D),  # Full user input (demographic + history)
            'user_attention_mask': torch.Tensor (1D),
            'demographic_input_ids': torch.Tensor (1D),  # Only demographic information (for MSE trainer)
            'demographic_attention_mask': torch.Tensor (1D),
            'input_ids': torch.Tensor (1D),  # decoder input
            'attention_mask': torch.Tensor (1D),  # decoder attention
            'labels': torch.Tensor (1D),
        }
        """
        batch = {}

        # Dynamic padding: pad to the max length within the batch.
        if "user_input_ids" in features[0]:
            user_max_len = max(int(f["user_input_ids"].size(0)) for f in features)
            batch["user_input_ids"] = torch.stack(
                [
                    self._pad_1d(f["user_input_ids"], user_max_len, self.user_pad_token_id, self.user_padding_side)
                    for f in features
                ]
            )
            batch["user_attention_mask"] = torch.stack(
                [
                    self._pad_1d(f["user_attention_mask"], user_max_len, 0, self.user_padding_side)
                    for f in features
                ]
            )
        
        # Handle demographic_input_ids (only demographic information, for MSE trainer)
        if "demographic_input_ids" in features[0]:
            demographic_max_len = max(int(f["demographic_input_ids"].size(0)) for f in features)
            batch["demographic_input_ids"] = torch.stack(
                [
                    self._pad_1d(f["demographic_input_ids"], demographic_max_len, self.user_pad_token_id, self.user_padding_side)
                    for f in features
                ]
            )
            batch["demographic_attention_mask"] = torch.stack(
                [
                    self._pad_1d(f["demographic_attention_mask"], demographic_max_len, 0, self.user_padding_side)
                    for f in features
                ]
            )

        # Dual-stream: profile [max_profile_items, max_profile_item_len] is already fixed per sample
        if "profile_input_ids" in features[0]:
            batch["profile_input_ids"] = torch.stack([f["profile_input_ids"] for f in features])
            batch["profile_attention_mask"] = torch.stack([f["profile_attention_mask"] for f in features])

        # Dual-stream: query (variable length, pad to batch max)
        if "query_input_ids" in features[0]:
            query_max_len = max(int(f["query_input_ids"].size(0)) for f in features)
            batch["query_input_ids"] = torch.stack([
                self._pad_1d(f["query_input_ids"], query_max_len, self.user_pad_token_id, self.user_padding_side)
                for f in features
            ])
            batch["query_attention_mask"] = torch.stack([
                self._pad_1d(f["query_attention_mask"], query_max_len, 0, self.user_padding_side)
                for f in features
            ])

        if "input_ids" in features[0]:
            dec_max_len = max(int(f["input_ids"].size(0)) for f in features)
            
            # Apply pad_to_multiple_of if specified
            if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
                dec_max_len = ((dec_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            
            batch["input_ids"] = torch.stack(
                [
                    self._pad_1d(f["input_ids"], dec_max_len, self.decoder_pad_token_id, self.decoder_padding_side)
                    for f in features
                ]
            )
            batch["attention_mask"] = torch.stack(
                [
                    self._pad_1d(f["attention_mask"], dec_max_len, 0, self.decoder_padding_side)
                    for f in features
                ]
            )
            # Labels: pad with -100 so padded positions are ignored by CrossEntropyLoss
            batch["labels"] = torch.stack(
                [
                    self._pad_1d(f["labels"], dec_max_len, self.label_pad_token_id, self.decoder_padding_side)
                    for f in features
                ]
            )

        # Process rejected data for DPO training (if available)
        if "rejected_input_ids" in features[0] and features[0].get("has_rejected", torch.tensor(False)).item():
            # Check if any sample has rejected data
            has_rejected_samples = [f.get("has_rejected", torch.tensor(False)).item() for f in features]
            
            if any(has_rejected_samples):
                # Find max length for rejected sequences
                rejected_features = [f for f in features if f.get("has_rejected", torch.tensor(False)).item()]
                rejected_max_len = max(int(f["rejected_input_ids"].size(0)) for f in rejected_features)
                
                # Apply pad_to_multiple_of if specified
                if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
                    rejected_max_len = ((rejected_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
                
                # Create rejected tensors for all samples
                rejected_input_ids_list = []
                rejected_attention_mask_list = []
                rejected_labels_list = []
                has_rejected_list = []
                
                for f in features:
                    if f.get("has_rejected", torch.tensor(False)).item():
                        rejected_input_ids_list.append(
                            self._pad_1d(f["rejected_input_ids"], rejected_max_len, self.decoder_pad_token_id, self.decoder_padding_side)
                        )
                        rejected_attention_mask_list.append(
                            self._pad_1d(f["rejected_attention_mask"], rejected_max_len, 0, self.decoder_padding_side)
                        )
                        rejected_labels_list.append(
                            self._pad_1d(f["rejected_labels"], rejected_max_len, self.label_pad_token_id, self.decoder_padding_side)
                        )
                        has_rejected_list.append(True)
                    else:
                        # Create dummy tensors for samples without rejected data
                        dummy_rejected = torch.full((rejected_max_len,), self.decoder_pad_token_id, dtype=torch.long)
                        rejected_input_ids_list.append(dummy_rejected)
                        rejected_attention_mask_list.append(torch.zeros_like(dummy_rejected))
                        rejected_labels_list.append(torch.full_like(dummy_rejected, self.label_pad_token_id))
                        has_rejected_list.append(False)
                
                batch["rejected_input_ids"] = torch.stack(rejected_input_ids_list)
                batch["rejected_attention_mask"] = torch.stack(rejected_attention_mask_list)
                batch["rejected_labels"] = torch.stack(rejected_labels_list)
                batch["has_rejected"] = torch.tensor(has_rejected_list, dtype=torch.bool)

        return batch
