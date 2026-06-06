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

# Portions of this file are modifications by OPPO Inc.
# Licensed under the Apache License, Version 2.0.

"""
Dataset for PersonaSteering training (Multi-Task Support)
- Supports multiple LongLaMP tasks: generation_abstract, topic_writing, product_review_writing
- Supports LaMP generation tasks: lamp_4 (News title), lamp_5 (Paper title), lamp_7 (Tweet paraphrase)
- Each user has multiple items from 'profile' that are used to create training samples
- All samples from same user share identical steering vector (Demographic Information)
- Task-specific processing logic for different data formats
"""

import os
import re
import logging
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
from collections import Counter

from ...extras.constants import IGNORE_INDEX


class SteeringDataset(Dataset):
    """Dataset for training User Encoder with multiple task support."""
    
    _assistant_mask_warning_shown = False
    
    # Task configurations: demographic prefix, output field name, and default system prompt
    TASK_CONFIGS = {
        "generation_abstract": {
            "demographic_prefix": "Researcher ID:",
            "output_field": "abstract",
            "default_system_prompt": "You are an expert academic researcher. Your task is to generate high-quality research paper abstracts based on given titles and research contexts. Write clear, concise, and informative abstracts that accurately summarize the research content."
        },
        "topic_writing": {
            "demographic_prefix": "Writer ID:",
            "output_field": "output",
            "default_system_prompt": "You are a skilled writer. Your task is to write engaging and well-structured content on various topics. Write in a clear, coherent style that matches the user's writing preferences and historical writing patterns."
        },
        "product_review_writing": {
            "demographic_prefix": "Reviewer ID:",
            "output_field": "reviewText",
            "default_system_prompt": "You are a helpful product reviewer. Your task is to write honest, detailed, and helpful product reviews. Write reviews that reflect the user's review style and preferences based on their historical reviews."
        },
        "lamp_4": {
            "demographic_prefix": "User ID:",
            "output_field": "output",
            "default_system_prompt": "You are a news editor. Your task is to generate concise and engaging news titles based on news article content. Create titles that are informative, attention-grabbing, and match the user's reading preferences."
        },
        "lamp_5": {
            "demographic_prefix": "Researcher ID:",
            "output_field": "output",
            "default_system_prompt": "You are an academic researcher. Your task is to generate appropriate research paper titles based on abstract content. Create titles that are clear, descriptive, and match the researcher's academic writing style."
        },
        "lamp_7": {
            "demographic_prefix": "User ID:",
            "output_field": "output",
            "default_system_prompt": "You are a social media assistant. Your task is to paraphrase tweets while maintaining the original meaning and style. Create paraphrased versions that match the user's communication style and preferences."
        }
    }
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        user_tokenizer=None,
        decoder_tokenizer=None,
        max_user_length: int = 512,
        max_length: int = 2048,
        decoder_prompt: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_filename: Optional[str] = None,
        task: str = "generation_abstract",  # Task type: generation_abstract, topic_writing, product_review_writing, lamp_4, lamp_5, lamp_7
        max_samples_per_user: Optional[int] = None,  # Limit maximum number of samples per user
        max_profile_items: int = 8,  # Max number of profile items for Stream B (Query attend to Profile)
        max_profile_item_len: int = 128,  # Max tokens per profile item
        max_query_length: int = 256,  # Max tokens for current query (user_content) in Stream B
        profile_topk_prescreen: bool = False,  # If True, pre-select top-k profile items by query relevance
        profile_topk: Optional[int] = None,  # top-k value for pre-screening; defaults to max_profile_items
    ):
        if data is None or len(data) == 0:
            raise ValueError("'data' must be provided and non-empty")
        
        # Validate task
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task}. Supported tasks: {list(self.TASK_CONFIGS.keys())}")
        
        self.task = task  # default task
        self.task_config = self.TASK_CONFIGS[task]
        expanded_data = []
        
        # Per-task counts
        task_stats = {}
        
        # Per-sample task override supported
        for user_data in data:
            # task field or default
            sample_task = user_data.get('task', task)
            
            # Validate task
            if sample_task not in self.TASK_CONFIGS:
                raise ValueError(f"Unknown task in data: {sample_task}. Supported tasks: {list(self.TASK_CONFIGS.keys())}")
            
            # Task config
            sample_task_config = self.TASK_CONFIGS[sample_task]
            
            # Count by task
            task_stats[sample_task] = task_stats.get(sample_task, 0) + 1
            
            # Demographics / steering prefix uses task
            demographic_info = self._get_demographic_info(user_data, sample_task, sample_task_config)
            
            # ✅ Limit profile size per user if max_samples_per_user is set
            if max_samples_per_user is not None and max_samples_per_user > 0:
                original_profile = user_data.get('profile', [])
                if isinstance(original_profile, list) and len(original_profile) > max_samples_per_user:
                    # Create a copy to avoid modifying original data
                    user_data = user_data.copy()
                    user_data['profile'] = original_profile[:max_samples_per_user]
            
            # Process based on task type
            if sample_task == "generation_abstract":
                samples = self._process_abstract_generation(user_data, demographic_info)
            elif sample_task == "topic_writing":
                samples = self._process_topic_writing(user_data, demographic_info)
            elif sample_task == "product_review_writing":
                samples = self._process_product_review(user_data, demographic_info)
            elif sample_task == "lamp_4":
                samples = self._process_lamp_4(user_data, demographic_info)
            elif sample_task == "lamp_5":
                samples = self._process_lamp_5(user_data, demographic_info)
            elif sample_task == "lamp_7":
                samples = self._process_lamp_7(user_data, demographic_info)
            else:
                raise ValueError(f"Task {sample_task} processing not implemented")
            
            # Tag sample with _task
            for sample in samples:
                sample['_task'] = sample_task
            
            expanded_data.extend(samples)
        

        if len(expanded_data) == 0:
            raise ValueError(f"No valid samples generated for task '{task}'. Check data format and profile structure.")
        
        # Log demographic info source statistics
        demo_source_stats = {}
        for sample in expanded_data[:100]:  # Check first 100 samples
            demo_info = sample.get('Demographic Information', '')
            if demo_info.startswith(self.task_config['demographic_prefix']):
                demo_source_stats['user_id'] = demo_source_stats.get('user_id', 0) + 1
            elif 'research' in demo_info.lower() or 'academic' in demo_info.lower():
                demo_source_stats['profile_extracted'] = demo_source_stats.get('profile_extracted', 0) + 1
            else:
                demo_source_stats['field'] = demo_source_stats.get('field', 0) + 1
        
        if demo_source_stats:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Demographic Information sources (first 100 samples): {demo_source_stats}")
            
        self.data = expanded_data
        self.user_tokenizer = user_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_user_length = max_user_length
        self.max_length = max_length
        self.decoder_prompt = decoder_prompt
        self.output_dir = output_dir
        self.max_samples_per_user = max_samples_per_user
        self.max_profile_items = max_profile_items
        self.max_profile_item_len = max_profile_item_len
        self.max_query_length = max_query_length
        self.profile_topk_prescreen = bool(profile_topk_prescreen)
        self.profile_topk = max_profile_items if profile_topk is None else int(profile_topk)
        if self.profile_topk <= 0:
            self.profile_topk = max_profile_items
        self._token_pattern = re.compile(r"\w+")
        self.dataset_logger = self._setup_dataset_logger(output_dir, log_filename) if output_dir else None
        
        # Logging config
        try:
            self.dataset_log_first_n = int(os.environ.get("DATASET_LOG_FIRST_N", "3"))
        except Exception:
            self.dataset_log_first_n = 3
        try:
            self.dataset_log_every_n = int(os.environ.get("DATASET_LOG_EVERY_N", "0"))
        except Exception:
            self.dataset_log_every_n = 0
        
        # Validation
        if self.decoder_tokenizer is not None:
            # Ensure left truncation for generation tasks
            try:
                self.decoder_tokenizer.truncation_side = "left"
            except Exception:
                pass
        
        self.column_names = [
            'user_input_ids',
            'user_attention_mask',
            'demographic_input_ids',
            'demographic_attention_mask',
            'profile_input_ids',
            'profile_attention_mask',
            'query_input_ids',
            'query_attention_mask',
            'input_ids',
            'attention_mask',
            'labels'
        ]
        
        # Log sample prompts per task
        if self.decoder_tokenizer is not None:
            self._log_task_samples()
            self._log_mask_statistics_summary()

    def _simple_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [t.lower() for t in self._token_pattern.findall(text)]

    def _query_profile_overlap_score(self, query_text: str, profile_text: str) -> float:
        """
        Lightweight lexical relevance score for pre-screening profile items.
        Uses weighted token overlap (query-side weighting).
        """
        q_tokens = self._simple_tokenize(query_text)
        p_tokens = self._simple_tokenize(profile_text)
        if not q_tokens or not p_tokens:
            return 0.0

        q_counter = Counter(q_tokens)
        p_counter = Counter(p_tokens)
        overlap = 0.0
        for token, q_count in q_counter.items():
            overlap += min(q_count, p_counter.get(token, 0))
        return overlap / max(1.0, float(len(q_tokens)))

    def _select_profile_items_for_query(self, profile_items: List[Dict[str, Any]], task: str, query_text: str) -> List[Dict[str, Any]]:
        """
        Select profile items for Stream B.
        - Default: keep original order and truncate to max_profile_items.
        - Optional pre-screen: score items by query relevance, keep top-k, then preserve original order.
        """
        if not isinstance(profile_items, list) or len(profile_items) == 0:
            return []

        if not self.profile_topk_prescreen:
            return profile_items[: self.max_profile_items]

        k = min(len(profile_items), self.profile_topk, self.max_profile_items)
        if k <= 0:
            return []

        scored = []
        for idx, item in enumerate(profile_items):
            text = self._profile_item_to_text(item, task)
            score = self._query_profile_overlap_score(query_text, text)
            scored.append((idx, score))

        top_indices = set(idx for idx, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k])
        selected = [item for idx, item in enumerate(profile_items) if idx in top_indices]
        return selected[: self.max_profile_items]
    
    def _log_task_samples(self):
        """Log up to 3 samples per task with full prompts (including assistant)."""
        if self.dataset_logger is None:
            return
        
        # Group by task
        task_samples = {}
        for idx, sample in enumerate(self.data):
            task = sample.get('_task', self.task)
            if task not in task_samples:
                task_samples[task] = []
            task_samples[task].append((idx, sample))
        
        # Up to 3 samples per task
        self.dataset_logger.info("\n" + "="*80)
        self.dataset_logger.info("Dataset Sample Prompts by Task")
        self.dataset_logger.info("="*80)
        
        for task_name in sorted(task_samples.keys()):
            samples = task_samples[task_name]
            sample_count = min(3, len(samples))
            
            self.dataset_logger.info(f"\n{'='*80}")
            self.dataset_logger.info(f"Task: {task_name} (showing {sample_count} of {len(samples)} samples)")
            self.dataset_logger.info(f"{'='*80}")
            
            # First 3 samples
            for i, (idx, sample) in enumerate(samples[:sample_count]):
                try:
                    # Task system content
                    task = sample.get('_task', task_name)
                    system_content = self._get_system_content(task=task, sample=sample)
                    
                    user_content = sample.get('user_content', '')
                    assistant_content = sample.get('abstract', '')
                    
                    # Log segments
                    self.dataset_logger.info(f"\n--- Sample {i+1} (Dataset Index: {idx}) ---")
                    self.dataset_logger.info(f"\n[Demographic Information (for User Encoder)]:")
                    demo_info = sample.get('Demographic Information', 'N/A')
                    self.dataset_logger.info(f"{demo_info}")
                    
                    self.dataset_logger.info(f"\n[Prompt Components (before formatting)]:")
                    self.dataset_logger.info(f"  System Content:\n    {system_content}")
                    self.dataset_logger.info(f"  User Content:\n    {user_content}")
                    self.dataset_logger.info(f"  Assistant Content:\n    {assistant_content}")
                    
                    # Build chat structure
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                    
                    self.dataset_logger.info(f"\n[Messages Structure (dict format)]:")
                    for msg in messages:
                        self.dataset_logger.info(f"  {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                    
                    # chat_template without tokenizing
                    if hasattr(self.decoder_tokenizer, 'apply_chat_template') and self.decoder_tokenizer.chat_template is not None:
                        formatted_prompt = self.decoder_tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        
                        # Full formatted prompt
                        self.dataset_logger.info(f"\n[Formatted Prompt (after apply_chat_template)]:")
                        self.dataset_logger.info(f"{formatted_prompt}")
                        
                        # Tokenizer / template info
                        tokenizer_name = getattr(self.decoder_tokenizer, 'name_or_path', 'unknown')
                        self.dataset_logger.info(f"\n[Tokenizer Info]:")
                        self.dataset_logger.info(f"  Tokenizer: {tokenizer_name}")
                        self.dataset_logger.info(f"  Has chat_template: {self.decoder_tokenizer.chat_template is not None}")
                        
                        # Token length
                        try:
                            tokenized = self.decoder_tokenizer(
                                formatted_prompt,
                                return_tensors='pt',
                                add_special_tokens=False
                            )
                            token_count = tokenized['input_ids'].shape[1]
                            self.dataset_logger.info(f"  Tokenized length: {token_count} tokens")
                        except Exception as e:
                            self.dataset_logger.info(f"  Tokenized length: (failed to tokenize: {e})")
                    else:
                        # Fallback manual format
                        formatted_prompt = f"System: {system_content}\n\nUser: {user_content}\n\nAssistant: {assistant_content}"
                        self.dataset_logger.info(f"\n[Formatted Prompt (fallback format)]:")
                        self.dataset_logger.info(f"{formatted_prompt}")
                        self.dataset_logger.warning(f"  ⚠️  Tokenizer does not support chat_template, using fallback format")
                    
                    self.dataset_logger.info(f"\n{'─'*80}")
                    
                except Exception as e:
                    self.dataset_logger.warning(f"Failed to log sample {idx} for task {task_name}: {e}")
                    import traceback
                    self.dataset_logger.warning(f"Traceback: {traceback.format_exc()}")
        
        self.dataset_logger.info(f"\n{'='*80}\n")
    
    def _log_mask_statistics_summary(self):
        """Log dataset mask statistics summary."""
        if self.dataset_logger is None:
            return
        
        # Sample first 100 for mask stats
        sample_size = min(100, len(self.data))
        mask_stats = {
            'total_tokens': [],
            'masked_tokens': [],
            'loss_tokens': [],
            'loss_token_ratios': [],
        }
        
        self.dataset_logger.info("\n" + "="*80)
        self.dataset_logger.info("Computing Mask Statistics Summary (sampling first 100 samples)...")
        self.dataset_logger.info("="*80)
        
        for idx in range(sample_size):
            try:
                # Raw fields only
                sample = self.data[idx]
                task = sample.get('_task', self.task)
                system_content = self._get_system_content(task=task, sample=sample)
                user_content = sample['user_content']
                assistant_content = sample['abstract']
                
                # Mask stats
                prompt_messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
                prompt_text = self.decoder_tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_encoded = self.decoder_tokenizer(
                    prompt_text,
                    padding=False,
                    truncation=False,
                    return_tensors='pt',
                )
                prompt_length = prompt_encoded['input_ids'].shape[1]
                
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
                full_text = self.decoder_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                full_encoded = self.decoder_tokenizer(
                    full_text,
                    padding=False,
                    truncation=False,
                    return_tensors='pt',
                )
                full_length = full_encoded['input_ids'].shape[1]
                
                # Length after truncation
                if full_length > self.max_length:
                    truncated_length = self.max_length
                    dropped = full_length - self.max_length
                    if prompt_length >= dropped:
                        assistant_start = prompt_length - dropped
                    else:
                        assistant_start = 0
                else:
                    truncated_length = full_length
                    assistant_start = prompt_length
                
                # Mask stats
                total_tokens = truncated_length
                masked_tokens = assistant_start
                loss_tokens = total_tokens - masked_tokens
                loss_ratio = loss_tokens / total_tokens if total_tokens > 0 else 0.0
                
                mask_stats['total_tokens'].append(total_tokens)
                mask_stats['masked_tokens'].append(masked_tokens)
                mask_stats['loss_tokens'].append(loss_tokens)
                mask_stats['loss_token_ratios'].append(loss_ratio)
                
            except Exception as e:
                if idx < 5:
                    self.dataset_logger.warning(f"Failed to compute mask stats for sample {idx}: {e}")
                continue
        
        # Aggregate
        if mask_stats['loss_token_ratios']:
            import numpy as np
            avg_loss_ratio = np.mean(mask_stats['loss_token_ratios'])
            median_loss_ratio = np.median(mask_stats['loss_token_ratios'])
            min_loss_ratio = np.min(mask_stats['loss_token_ratios'])
            max_loss_ratio = np.max(mask_stats['loss_token_ratios'])
            std_loss_ratio = np.std(mask_stats['loss_token_ratios'])
            
            avg_total = np.mean(mask_stats['total_tokens'])
            avg_masked = np.mean(mask_stats['masked_tokens'])
            avg_loss = np.mean(mask_stats['loss_tokens'])
            
            self.dataset_logger.info(f"\n[Mask Statistics Summary (from {len(mask_stats['loss_token_ratios'])} samples)]:")
            self.dataset_logger.info(f"  Average total tokens: {avg_total:.1f}")
            self.dataset_logger.info(f"  Average masked tokens: {avg_masked:.1f}")
            self.dataset_logger.info(f"  Average loss tokens: {avg_loss:.1f}")
            self.dataset_logger.info(f"  Loss token ratio:")
            self.dataset_logger.info(f"    Mean: {avg_loss_ratio*100:.2f}%")
            self.dataset_logger.info(f"    Median: {median_loss_ratio*100:.2f}%")
            self.dataset_logger.info(f"    Min: {min_loss_ratio*100:.2f}%")
            self.dataset_logger.info(f"    Max: {max_loss_ratio*100:.2f}%")
            self.dataset_logger.info(f"    Std: {std_loss_ratio*100:.2f}%")
            
            # Warn outliers
            if avg_loss_ratio < 0.1:
                self.dataset_logger.warning(f"  ⚠️  CRITICAL: Average loss token ratio is very low ({avg_loss_ratio*100:.2f}%)!")
                self.dataset_logger.warning(f"     This means most tokens are masked, loss may be unreliable!")
            elif avg_loss_ratio > 0.9:
                self.dataset_logger.warning(f"  ⚠️  WARNING: Average loss token ratio is very high ({avg_loss_ratio*100:.2f}%)!")
                self.dataset_logger.warning(f"     This may indicate prompt masking is not working correctly!")
            
            # Count bad samples
            low_ratio_count = sum(1 for r in mask_stats['loss_token_ratios'] if r < 0.1)
            high_ratio_count = sum(1 for r in mask_stats['loss_token_ratios'] if r > 0.9)
            if low_ratio_count > 0:
                self.dataset_logger.warning(f"  ⚠️  {low_ratio_count} samples have loss ratio < 10%")
            if high_ratio_count > 0:
                self.dataset_logger.warning(f"  ⚠️  {high_ratio_count} samples have loss ratio > 90%")
        
        self.dataset_logger.info("="*80 + "\n")
    
    def _get_system_content(self, task: str = None, sample: Dict[str, Any] = None) -> str:
        """
        Build task-specific system content.

        Precedence:
        1. decoder_prompt if set
        2. Task default prompt (includes task description)
        3. Generic default prompt

        Args:
            task: Task id (defaults to self.task)
            sample: Optional sample dict for future use

        Returns:
            System content string
        """
        if task is None:
            task = self.task
        
        # 1. decoder_prompt
        if self.decoder_prompt and self.decoder_prompt.strip():
            return self.decoder_prompt.strip()
        
        # 2. Task default
        task_config = self.TASK_CONFIGS.get(task, self.TASK_CONFIGS.get(self.task, {}))
        default_prompt = task_config.get(
            "default_system_prompt", 
            "You are a helpful assistant."
        )
        
        return default_prompt
    
    def _get_demographic_info(self, user_data: Dict[str, Any], task: str = None, task_config: Dict[str, str] = None) -> str:
        """
        Get Demographic Information for steering vector.
        Priority: 
        1. "Demographic Information" field in data (if exists and valid)
        2. Extract from profile (simplified version)
        3. Fallback to user ID
        
        Args:
            user_data: User data dictionary
            task: Task type for this sample (if None, use self.task)
            task_config: Task configuration dict (if None, use self.task_config)
            
        Returns:
            Demographic information string
        """
        # task_config or default
        if task_config is None:
            task_config = self.task_config
        if task is None:
            task = self.task
        
        # 1. Priority: Use "Demographic Information" field if exists
        if "Demographic Information" in user_data:
            demo_info = user_data["Demographic Information"]
            if demo_info and isinstance(demo_info, str) and len(demo_info.strip()) >= 5:
                return demo_info.strip()
            elif demo_info and not isinstance(demo_info, str):
                # Handle non-string types (e.g., dict, list)
                demo_str = str(demo_info).strip()
                if len(demo_str) >= 5:
                    return demo_str
        
        # 2. Extract from profile (simplified version)
        profile = user_data.get('profile', [])
        if profile and isinstance(profile, list) and len(profile) > 0:
            # Generate simplified demographic info based on profile
            if task in ["lamp_4", "lamp_5", "generation_abstract"]:
                # Academic tasks: extract from titles/abstracts
                titles = []
                for item in profile[:5]:  # Only use first 5 items
                    if isinstance(item, dict):
                        title = item.get('title', '').strip()
                        if title:
                            titles.append(title)
                
                if titles:
                    # Simplified version: indicate researcher with publications
                    return f"This person likes research in the fields covered by their publications.\nThis person focuses on academic work.\nA researcher who has published works in relevant academic domains."
            else:
                # Other tasks: extract from text content
                texts = []
                for item in profile[:5]:  # Only use first 5 items
                    if isinstance(item, dict):
                        text = item.get('text', '').strip() or item.get('content', '').strip()
                        if text:
                            texts.append(text[:100])  # Truncate to 100 chars
                
                if texts:
                    # Simplified version: indicate user with historical data
                    return f"This person has interests related to their historical data.\nA user with relevant historical information."
        
        # 3. Fallback: Use user ID (original behavior)
        user_name = user_data.get('id', 'Unknown')
        return f"{task_config['demographic_prefix']} {user_name}"
    
    def _process_abstract_generation(self, user_data: Dict[str, Any], demographic_info: str) -> List[Dict[str, Any]]:
        """Process generation_abstract task."""
        base_input = user_data.get('input', '')
        profiles = user_data.get('profile', [])
        target_output = user_data.get('output', '').strip()
        
        if not isinstance(profiles, list):
            return []
        
        samples = []
        
        # Type 1: historical papers as paired examples
        for i, paper in enumerate(profiles):
            if not isinstance(paper, dict):
                continue
            
            title = paper.get('title', '').strip()
            abstract = paper.get('abstract', paper.get('text', '')).strip()
            
            if not title or not abstract:
                continue
            
            user_content_for_this_paper = (
                f'Generate an abstract for the title "{title}" using the following items: (from the paper).'
            )
            
            # Exclude current item from profile to prevent data leakage
            context_profiles = profiles[:i] + profiles[i+1:]

            samples.append({
                'Demographic Information': demographic_info,
                'user_content': user_content_for_this_paper,
                'abstract': abstract,
                'profile': context_profiles,
            })
        
        # Type 2: current row as supervised pair
        if target_output:
            samples.append({
                'Demographic Information': demographic_info,
                'user_content': base_input,
                'abstract': target_output,
                'profile': profiles,
            })
        
        return samples

    def _process_topic_writing(self, user_data: Dict[str, Any], demographic_info: str) -> List[Dict[str, Any]]:
        """Process topic_writing task."""
        base_input = user_data.get('input', '')
        profiles = user_data.get('profile', [])
        target_output = user_data.get('output', '').strip()
        
        if not isinstance(profiles, list):
            return []
        
        samples = []
        
        # Type 1: historical rows (summary -> request, content -> target)
        for i, item in enumerate(profiles):
            if not isinstance(item, dict):
                continue
            
            if "input" in item and "output" in item:
                content = item["output"].strip()
                user_content_for_this = item.get("input", "").strip() or base_input
            elif "summary" in item and "content" in item:
                content = item["content"].strip()
                summary = item.get("summary", "").strip()
                user_content_for_this = f"Generate the content for a reddit post: {summary}" if summary else base_input
            else:
                continue
            
            if not content:
                continue
            
            # Exclude current item from profile to prevent data leakage
            context_profiles = profiles[:i] + profiles[i+1:]

            samples.append({
                'Demographic Information': demographic_info,
                'user_content': user_content_for_this,
                'abstract': content,
                'profile': context_profiles,
            })
        
        # Type 2: current row as supervised pair
        if target_output:
            samples.append({
                'Demographic Information': demographic_info,
                'user_content': base_input,
                'abstract': target_output,
                'profile': profiles,
            })
        
        return samples

    def _process_product_review(self, user_data: Dict[str, Any], demographic_info: str) -> List[Dict[str, Any]]:
        """Process product_review_writing task."""
        base_input = user_data.get('input', '')
        profiles = user_data.get('profile', [])
        target_output = user_data.get('output', '').strip()
        
        if not isinstance(profiles, list):
            return []
        
        samples = []
        
        # Type 1: historical reviews (product context -> reviewText)
        for i, review in enumerate(profiles):
            if not isinstance(review, dict):
                continue
            
            review_text = review.get('reviewText', '').strip()
            if not review_text:
                continue
            
            desc = review.get('description', '')[:400].strip()
            overall = review.get('overall', '')
            summary = review.get('summary', '').strip()
            user_content_for_this = (
                f'Generate the review text written by a reviewer who has given an overall rating of "{overall}" '
                f'for a product with description "{desc}". The summary of the review text is "{summary}".'
            )
            
            # Exclude current item from profile to prevent data leakage
            context_profiles = profiles[:i] + profiles[i+1:]

            samples.append({
                'Demographic Information': demographic_info,
                'user_content': user_content_for_this,
                'abstract': review_text,
                'profile': context_profiles,
            })
        
        # Type 2: current row as supervised pair
        if target_output:
            samples.append({
                'Demographic Information': demographic_info,
                'user_content': base_input,
                'abstract': target_output,
                'profile': profiles,
            })
        
        return samples

    def _process_lamp_4(self, user_data: Dict[str, Any], demographic_info: str) -> List[Dict[str, Any]]:
        """Process LaMP-4: News title generation."""
        base_input = user_data.get('input', '')
        profiles = user_data.get('profile', [])
        target_output = user_data.get('output', '').strip()
        
        if not isinstance(profiles, list):
            return []
        
        samples = []
        
        # Type 1: historical news (text -> title)
        for i, item in enumerate(profiles):
            if not isinstance(item, dict):
                continue
            
            text = item.get('text', '').strip()
            title = item.get('title', '').strip()
            if not text or not title:
                continue
            
            user_content_for_this = f"Generate a headline for the following article: {text}"
            
            # Exclude current item from profile to prevent data leakage
            context_profiles = profiles[:i] + profiles[i+1:]

            samples.append({
                'Demographic Information': demographic_info,
                'user_content': user_content_for_this,
                'abstract': title,
                'profile': context_profiles,
            })
        
        # Type 2: current row as supervised pair
        if target_output:
            samples.append({
                'Demographic Information': demographic_info,
                'user_content': base_input,
                'abstract': target_output,
                'profile': profiles,
            })
        
        return samples

    def _process_lamp_5(self, user_data: Dict[str, Any], demographic_info: str) -> List[Dict[str, Any]]:
        """Process LaMP-5: Paper title generation."""
        base_input = user_data.get('input', '')
        profiles = user_data.get('profile', [])
        target_output = user_data.get('output', '').strip()
        
        if not isinstance(profiles, list):
            return []
        
        samples = []
        
        # Type 1: historical papers (abstract -> title)
        for i, item in enumerate(profiles):
            if not isinstance(item, dict):
                continue
            
            abstract_text = item.get('abstract', '').strip()
            title = item.get('title', '').strip()
            if not abstract_text or not title:
                continue
            
            user_content_for_this = f"Generate a title for the following abstract of a paper: {abstract_text}"
            
            # Exclude current item from profile to prevent data leakage
            context_profiles = profiles[:i] + profiles[i+1:]

            samples.append({
                'Demographic Information': demographic_info,
                'user_content': user_content_for_this,
                'abstract': title,
                'profile': context_profiles,
            })
        
        # Type 2: current row as supervised pair
        if target_output:
            samples.append({
                'Demographic Information': demographic_info,
                'user_content': base_input,
                'abstract': target_output,
                'profile': profiles,
            })
        
        return samples

    def _process_lamp_7(self, user_data: Dict[str, Any], demographic_info: str) -> List[Dict[str, Any]]:
        """Process LaMP-7: Tweet paraphrase."""
        base_input = user_data.get('input', '')
        profiles = user_data.get('profile', [])
        target_output = user_data.get('output', '').strip()
        
        if not isinstance(profiles, list):
            return []
        
        samples = []
        
        
        # Type 2: current row as supervised pair
        profiles = user_data.get('profile', [])
        if target_output:
            samples.append({
                'Demographic Information': demographic_info,
                'user_content': base_input,
                'abstract': target_output,
                'profile': profiles,
            })
        
        return samples

    def _profile_item_to_text(self, item: Dict[str, Any], task: str) -> str:
        """Convert one profile item to a string for encoding (Stream B)."""
        if not isinstance(item, dict):
            return ""
        if task == "generation_abstract":
            title = item.get("title", "").strip()
            abstract = item.get("abstract", item.get("text", "")).strip()
            return (f"{title} {abstract}")[:800]
        if task == "topic_writing":
            if "content" in item:
                return (item.get("summary", "") + " " + item.get("content", ""))[:800]
            return (item.get("input", "") + " " + item.get("output", ""))[:800]
        if task == "product_review_writing":
            return (item.get("reviewText", item.get("text", "")) or "")[:800]
        if task == "lamp_4":
            return (item.get("text", "") + " " + item.get("title", ""))[:800]
        if task == "lamp_5":
            return (item.get("abstract", "") + " " + item.get("title", ""))[:800]
        if task == "lamp_7":
            return str(item.get("input", item.get("output", "")))[:800]
        return str(item.get("input", item.get("output", "")))[:800]

    def _setup_dataset_logger(self, output_dir: Optional[str], log_filename: Optional[str] = None) -> Optional[logging.Logger]:
        if output_dir is None:
            return None
        
        logger = logging.getLogger('dataset_debug')
        logger.setLevel(logging.INFO)
        if logger.handlers:
            return logger
        
        os.makedirs(output_dir, exist_ok=True)
        if log_filename is None:
            log_filename = 'token_ids_debug.log'
        log_file = os.path.join(output_dir, log_filename)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
        
        logger.propagate = False
        return logger
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. Prepare User Encoder Input (Steering Vector Source)
            # Using "Demographic Information" string constructed in __init__
        user_input_text = sample['Demographic Information']
        
        # Strip duplicate eos special strings
        if isinstance(user_input_text, str):
            # Remove <|endoftext|>
            import re
            # Variants
            user_input_text = re.sub(r'<\|endoftext\|>+', '', user_input_text)
            # Other eos strings
            user_input_text = re.sub(r'<\|eot_id\|>+', '', user_input_text)
            # Trim whitespace
            user_input_text = ' '.join(user_input_text.split())
            user_input_text = user_input_text.strip()
        
        user_encoded = self.user_tokenizer(
            user_input_text,
            max_length=self.max_user_length,
            padding=False,
            truncation=True,
            return_tensors='pt',
        )
        
        # For this logic, demographic == user_input
        demographic_encoded = user_encoded

        # ---------- Dual-stream: Profile (Stream B K/V) and Query (Stream B Q) ----------
        task = sample.get('_task', self.task)
        query_text = sample.get("user_content", "")
        profile_items = self._select_profile_items_for_query(
            sample.get('profile', []),
            task=task,
            query_text=query_text,
        )
        profile_texts = [
            self._profile_item_to_text(it, task) for it in profile_items
            if self._profile_item_to_text(it, task)
        ]
        user_pad_id = getattr(self.user_tokenizer, "pad_token_id", None) or 0

        profile_encoded_list = []
        profile_mask_list = []
        for i in range(self.max_profile_items):
            if i < len(profile_texts) and profile_texts[i].strip():
                enc = self.user_tokenizer(
                    profile_texts[i],
                    max_length=self.max_profile_item_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                profile_encoded_list.append(enc["input_ids"].squeeze(0))
                profile_mask_list.append(enc["attention_mask"].squeeze(0))
            else:
                # Instead of tokenizing an empty string, create zero tensors directly
                # to guarantee they contribute nothing and have an all-zero attention mask
                zeros_ids = torch.zeros(self.max_profile_item_len, dtype=torch.long)
                zeros_mask = torch.zeros(self.max_profile_item_len, dtype=torch.long)
                if user_pad_id is not None:
                    zeros_ids.fill_(user_pad_id)
                profile_encoded_list.append(zeros_ids)
                profile_mask_list.append(zeros_mask)
        profile_input_ids = torch.stack(profile_encoded_list)
        profile_attention_mask = torch.stack(profile_mask_list)

        query_encoded = self.user_tokenizer(
            query_text,
            max_length=self.max_query_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        query_input_ids = query_encoded["input_ids"].squeeze(0)
        query_attention_mask = query_encoded["attention_mask"].squeeze(0)

        # 2. Prepare Decoder Input (LLM SFT)
        # System content
        system_content = self._get_system_content(task=task, sample=sample)
        
        user_content = sample['user_content']
        assistant_content = sample['abstract']
        
        # Build chat structure
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        if not hasattr(self.decoder_tokenizer, 'apply_chat_template') or self.decoder_tokenizer.chat_template is None:
            raise ValueError(
                "Decoder tokenizer must support chat_template (e.g., Qwen2, Llama3). "
            )

        # Assistant start = len(tokenize(prompt_messages))
        # Tokenize prompt only (system + user, without assistant)
        prompt_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        prompt_text = self.decoder_tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_encoded = self.decoder_tokenizer(
            prompt_text,
            padding=False,
            truncation=False,
            return_tensors='pt',
        )
        prompt_length = prompt_encoded['input_ids'].shape[1]

        # Tokenize full conversation (system + user + assistant)
        full_text = self.decoder_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_encoded = self.decoder_tokenizer(
            full_text,
            padding=False,
            truncation=False,
            return_tensors='pt',
        )
        full_input_ids = full_encoded['input_ids'].squeeze(0)

        # Assistant from prompt_length
        assistant_start_full = prompt_length

        # Truncate (Left Truncation preferred for generation ending)
        if full_input_ids.numel() > self.max_length:
            dropped = full_input_ids.numel() - self.max_length
            input_ids = full_input_ids[-self.max_length:]
            
            # Truncation vs assistant_start
            if assistant_start_full >= dropped:
                assistant_start_idx = assistant_start_full - dropped
            else:
                # Prompt fully truncated
                assistant_start_idx = 0
        else:
            dropped = 0
            input_ids = full_input_ids
            assistant_start_idx = assistant_start_full

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Use assistant_start_idx
        if assistant_start_idx is not None and assistant_start_idx > 0:
            labels[:int(assistant_start_idx)] = IGNORE_INDEX
        else:
            # Edge: whole sequence is assistant
            if assistant_start_idx == 0:
                # All assistant: no mask change
                pass
            else:
                # Unexpected: warn
                if not SteeringDataset._assistant_mask_warning_shown:
                    if self.dataset_logger:
                        self.dataset_logger.warning(
                            f"Could not determine assistant start position in sample {idx}. "
                            f"Prompt length: {prompt_length}, Full length: {full_input_ids.numel()}"
                        )
                    SteeringDataset._assistant_mask_warning_shown = True
        
        # Mask stats
        total_tokens = labels.numel()
        masked_tokens = (labels == IGNORE_INDEX).sum().item()
        loss_tokens = total_tokens - masked_tokens
        loss_token_ratio = loss_tokens / total_tokens if total_tokens > 0 else 0.0
        masked_token_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Optional verbose mask logging
        log_mask_stats = os.environ.get('LOG_MASK_STATS', '1') == '1'
        mask_stats_first_n = int(os.environ.get('MASK_STATS_FIRST_N', '10'))
        
        if log_mask_stats and (idx < mask_stats_first_n):
            if self.dataset_logger:
                self.dataset_logger.info(f"\n[Sample {idx} - Mask Statistics]")
                self.dataset_logger.info(f"  Total tokens: {total_tokens}")
                self.dataset_logger.info(f"  Masked tokens (IGNORE_INDEX): {masked_tokens} ({masked_token_ratio*100:.2f}%)")
                self.dataset_logger.info(f"  Loss tokens: {loss_tokens} ({loss_token_ratio*100:.2f}%)")
                self.dataset_logger.info(f"  Assistant start idx: {assistant_start_idx}")
                self.dataset_logger.info(f"  Prompt length (before truncation): {prompt_length}")
                self.dataset_logger.info(f"  Full length (before truncation): {full_input_ids.numel()}")
                self.dataset_logger.info(f"  Truncated length: {input_ids.numel()}")
                if dropped > 0:
                    self.dataset_logger.info(f"  Dropped tokens: {dropped}")
                
                # Warn if loss-token ratio odd
                if loss_token_ratio < 0.1:
                    self.dataset_logger.warning(f"  ⚠️  Warning: Loss token ratio is very low ({loss_token_ratio*100:.2f}%), loss may be unreliable!")
                elif loss_token_ratio > 0.9:
                    self.dataset_logger.warning(f"  ⚠️  Warning: Loss token ratio is very high ({loss_token_ratio*100:.2f}%), prompt may not be masked correctly!")
        
        # Debug logging (optional, simplified)
        print_token_ids = os.environ.get('PRINT_TOKEN_IDS', '0') == '1'
        should_log = False
        if print_token_ids and self.dataset_logger:
             if idx < self.dataset_log_first_n:
                should_log = True
        
        if should_log:
            self.dataset_logger.info(f"\n[Sample {idx}]")
            self.dataset_logger.info(f"  User Enc Input: {user_input_text}")
            self.dataset_logger.info(f"  Messages User Content: {user_content[:100]}...")
            self.dataset_logger.info(f"  Messages Assistant: {assistant_content[:100]}...")
            self.dataset_logger.info(f"  Input IDs Shape: {input_ids.shape}")
            self.dataset_logger.info(f"  Labels Ignored (-100): {masked_tokens}/{total_tokens}")
            self.dataset_logger.info(f"  Loss tokens: {loss_tokens}/{total_tokens} ({loss_token_ratio*100:.2f}%)")

        # Require some loss tokens
        if loss_tokens == 0:
            if self.dataset_logger:
                self.dataset_logger.error(f"\n❌ ERROR: Sample {idx} has 0 loss tokens!")
                self.dataset_logger.error(f"  Assistant content: {assistant_content[:200]}...")
                self.dataset_logger.error(f"  Assistant content length: {len(assistant_content)} chars")
                self.dataset_logger.error(f"  Assistant start idx: {assistant_start_idx}")
                self.dataset_logger.error(f"  Total tokens: {total_tokens}")
                self.dataset_logger.error(f"  Masked tokens: {masked_tokens}")
                self.dataset_logger.error(f"  Prompt length: {prompt_length}")
                self.dataset_logger.error(f"  Full length: {full_input_ids.numel()}")
                self.dataset_logger.error(f"  Truncated length: {input_ids.numel()}")
                self.dataset_logger.error(f"  Dropped tokens: {dropped}")
                self.dataset_logger.error(f"  This sample will cause loss=0 or NaN!")
            # Log only; do not raise

        return {
            'user_input_ids': user_encoded['input_ids'].squeeze(0),
            'user_attention_mask': user_encoded['attention_mask'].squeeze(0),
            'demographic_input_ids': demographic_encoded['input_ids'].squeeze(0),
            'demographic_attention_mask': demographic_encoded['attention_mask'].squeeze(0),
            'profile_input_ids': profile_input_ids,
            'profile_attention_mask': profile_attention_mask,
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }