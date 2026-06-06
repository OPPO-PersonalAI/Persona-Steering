#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 OPPO. All rights reserved.
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
import os
import sys

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)
from eval_text_metrics import evaluate_data  # noqa: E402

from transformers import AutoTokenizer, AutoModelForCausalLM
from data.datasets_llama import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset,convert_to_llama_dataset,create_preprocessor_chatgpt
from prompts.prompts_llama import create_prompt_generator
#  from test_llama import perform, run, run_vllm
import argparse
import json
import logging
import time
import re
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--name", required = True)
parser.add_argument("--validation_data", required = True)
parser.add_argument("--model_path", required = True, help="Local model path")
parser.add_argument("--tokenizer", required = False)
parser.add_argument("--task", required = True)
parser.add_argument("--output_dir", required = True)
parser.add_argument("--use_profile", action="store_true", default=False,
                   help="Use profile retrieval (default: False)")
parser.add_argument("--use_demographic", action="store_true", default=False,
                   help="Use Demographic Information as baseline (default: False)")
parser.add_argument("--demographic_in_context", action="store_true", default=False,
                   help="When profile and demographic are both on, inject demographic into RAG context")
parser.add_argument("--demographic_position", type=str, default="before_profile",
                   choices=["before_profile", "after_profile"],
                   help="Where to place demographic in the fused context")
parser.add_argument("--retriever", default = "bm25")
parser.add_argument("--num_support_profile", type = int, default = 1,
                   help="Number of profiles to retrieve (-1 = all, default: 1)")
parser.add_argument("--is_ranked", action = "store_true")
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument("--max_length", type = int, default = 128)
parser.add_argument("--max_new_tokens", type = int, default = 128, help="Max new tokens to generate")
parser.add_argument("--max_input_length", type = int, default = 2048, help="Max input length")
parser.add_argument("--do_sample", action = "store_true", help="Use sampling during generation")
parser.add_argument("--temperature", type = float, default = 0.7, help="Sampling temperature")
parser.add_argument("--top_p", type = float, default = 0.9, help="Top-p sampling parameter")


def remove_repetitions(text, min_repeat_length=20):
    """
    Detect and remove repeated text segments (especially trailing repeats).

    Args:
        text: Input text.
        min_repeat_length: Minimum repeat length in characters.

    Returns:
        Cleaned text.
    """
    if not text or len(text) < 2 * min_repeat_length:
        return text
    
    text_lower = text.lower()
    text_length = len(text_lower)
    
    # Find trailing repeated segment
    max_repetition_length = None
    for repetition_length in range(min_repeat_length, int(text_length / 2)):
        # Check for suffix repetition
        same = True
        for i in range(repetition_length):
            if text_lower[text_length - repetition_length - i - 1] != text_lower[text_length - i - 1]:
                same = False
                break
        
        if same:
            max_repetition_length = repetition_length
    
    if max_repetition_length is None:
        return text
    
    # Drop repeats, keep the final segment
    cleaned_text = text
    cleaned_text_lower = text_lower
    while cleaned_text_lower.endswith(text_lower[-max_repetition_length:]):
        if len(cleaned_text) <= max_repetition_length:
            break
        cleaned_text = cleaned_text[:-max_repetition_length]
        cleaned_text_lower = cleaned_text_lower[:-max_repetition_length]
    
    return cleaned_text.strip()


def postprocess_generated_text(generated_text, task):
    """
    Post-process generated text: strip prefixes, placeholders, keep answer only.

    Args:
        generated_text: Raw model output.
        task: Task name (generation_abstract, topic_writing, product_review_writing, email_generation).

    Returns:
        Cleaned text.
    """
    if not generated_text:
        return generated_text
    
    # Remove repeated segments first
    generated_text = remove_repetitions(generated_text)

    # Task-specific prefixes to strip
    task_prefixes = {
        "generation_abstract": [
            "abstract:",
            "abstract",
            "summary:",
            "summary",
            "the abstract:",
            "the abstract is:",
            "here is the abstract:",
            "generated abstract:",
        ],
        "topic_writing": [
            "content:",
            "content",
            "text:",
            "text",
            "writing:",
            "writing",
            "the content:",
            "here is the content:",
            "generated content:",
        ],
        "product_review_writing": [
            "review:",
            "review",
            "review text:",
            "review text",
            "the review:",
            "here is the review:",
            "generated review:",
            "product review:",
        ],
        "email_generation": [
            "email:",
            "email",
            "email text:",
            "email text",
            "email body:",
            "email body",
            "the email:",
            "here is the email:",
            "generated email:",
        ],
    }
    
    prefixes = task_prefixes.get(task, [])
    generated_lower = generated_text.lower()

    # Strip task-specific prefixes
    for prefix in prefixes:
        if generated_lower.startswith(prefix):
            # After colon if present
            if ':' in generated_text:
                parts = generated_text.split(':', 1)
                if len(parts) > 1:
                    generated_text = parts[1].strip()
                    break
            # Otherwise take text after first sentence break
            elif '. ' in generated_text:
                parts = generated_text.split('. ', 1)
                if len(parts) > 1:
                    generated_text = parts[1].strip()
                    break
    
    # Placeholder lines (underscores, dashes, etc.)
    # Entire string placeholder-only?
    placeholder_only_chars = generated_text.replace('_', '').replace('-', '').replace(' ', '').replace('.', '').replace('=', '')
    if not placeholder_only_chars or len(placeholder_only_chars) < 3:
        generated_text = ""
    else:
        # Line-level placeholder filter
        lines = generated_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Skip placeholder-only lines
            placeholder_chars = line_stripped.replace('_', '').replace('-', '').replace(' ', '').replace('.', '').replace('=', '')
            if not placeholder_chars or len(placeholder_chars) < 3:
                continue
            cleaned_lines.append(line_stripped)

        if cleaned_lines:
            generated_text = '\n'.join(cleaned_lines)
        else:
            generated_text = ""

    # Generic prefixes (all tasks)
    generic_prefixes = [
        "based on",
        "according to",
        "considering",
        "following",
        "here is",
        "here's",
        "the answer is",
        "the result is",
        "answer:",
        "result:",
        "output:",
    ]
    
    generated_lower = generated_text.lower()
    for prefix in generic_prefixes:
        if generated_lower.startswith(prefix):
            if ':' in generated_text:
                parts = generated_text.split(':', 1)
                if len(parts) > 1:
                    generated_text = parts[1].strip()
                    break
            elif '. ' in generated_text:
                parts = generated_text.split('. ', 1)
                if len(parts) > 1:
                    generated_text = parts[1].strip()
                    break
    
    # First non-empty paragraph when multiple blocks exist
    if '\n\n' in generated_text:
        paragraphs = [p.strip() for p in generated_text.split('\n\n') if p.strip()]
        if paragraphs:
            generated_text = paragraphs[0]
    
    # Strip common closing boilerplate
    ending_phrases = [
        "i hope this helps",
        "let me know if",
        "please note that",
        "note:",
        "note that",
    ]
    
    for ending in ending_phrases:
        if ending in generated_lower:
            idx = generated_lower.find(ending)
            generated_text = generated_text[:idx].strip()
            break
    
    # Unwrap surrounding quotes
    if generated_text.startswith('"') and generated_text.endswith('"'):
        generated_text = generated_text[1:-1].strip()
    elif generated_text.startswith("'") and generated_text.endswith("'"):
        generated_text = generated_text[1:-1].strip()
    
    # Final trim
    generated_text = generated_text.strip()
    
    return generated_text


def inject_demographic_into_context(source_text, demographic_info, position="before_profile"):
    """Inject demographic block into an existing retrieval/context prompt."""
    demo_clean = " ".join(str(demographic_info).split()) if demographic_info is not None else ""
    if not demo_clean:
        return source_text

    demo_block = f"[Demographic Information]\n{demo_clean}\n\n"
    src = source_text or ""

    if position == "after_profile":
        return f"{src.rstrip()}\n\n{demo_block}".rstrip()

    marker_candidates = [
        "***User Profile***",
        "Following are given profiles",
        "Following are profile",
    ]
    insert_at = -1
    for marker in marker_candidates:
        idx = src.find(marker)
        if idx != -1:
            insert_at = idx
            break

    if insert_at != -1:
        return f"{src[:insert_at]}{demo_block}{src[insert_at:]}".rstrip()
    return f"{demo_block}{src}".rstrip()


def run_local_model(prompted_dataset, model, tokenizer, output_dir, args):
    """
    Local model inference (replaces run_vllm).
    Returns: [{"generated_text": "...", "output": "..."}, ...]
    """
    results = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Save retrieved prompts when using profile
    retrieved_data = [] if args.use_profile else None
    demographic_data = [] if args.use_demographic else None

    print("=" * 80)
    print("Starting local model generation...")
    print(f"Dataset size: {len(prompted_dataset)}")
    print(
        f"Generation: max_new_tokens={args.max_new_tokens}, do_sample=True, "
        f"temperature=1.0, top_p=0.9"
    )
    print(f"Max input length: {args.max_input_length}")
    print("=" * 80)

    if len(prompted_dataset) > 0:
        first_item = prompted_dataset[0]
        first_input = first_item['source']
        first_input_tokens = len(tokenizer.encode(first_input))
        print("\n[debug] First sample:")
        print(f"  Input length: {len(first_input)} chars, {first_input_tokens} tokens")
        print(f"  Input preview (first 200 chars): {first_input[:200]}...")
        if first_input_tokens > args.max_input_length:
            print(
                f"  Warning: input tokens ({first_input_tokens}) exceed "
                f"max_input_length ({args.max_input_length}); will truncate"
            )
        print()

    for idx, item in enumerate(tqdm(prompted_dataset, desc="Generating")):
        input_text = item['source']
        target_text = item.get('target', '')
        
        # System prompt + chat template
        system_content = "You are a helpful assistant."
        if args.task == "generation_abstract":
            system_content = "You are an expert academic researcher. Your task is to generate high-quality research paper abstracts."
        elif args.task == "product_review_writing":
            system_content = "You are a helpful product reviewer. Your task is to write honest, detailed, and helpful product reviews."
        elif args.task in ["topic_writing", "email_generation"]:
            system_content = "You are a skilled writer. Your task is to write engaging and well-structured content."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": input_text}
        ]
        
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"System: {system_content}\n\nUser: {input_text}\n\nAssistant: "
            
        # Tokenize input
        input_ids = tokenizer.encode(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=args.max_input_length
        ).to(model.device)
        
        input_token_count = input_ids.shape[1]
        
        if idx < 3:
            print(f"\n[debug] Sample {idx + 1}:")
            print(f"  Input tokens: {input_token_count}")
            print(f"  Target length: {len(target_text)} chars")

        max_retries = 2  # initial attempt + one retry
        generated_text = ""
        generated_token_count = 0
        

        for attempt in range(max_retries):
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

                generated_text = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
                generated_token_count = output_ids.shape[1] - input_token_count
                generated_text = postprocess_generated_text(generated_text, args.task)

                if generated_text and len(generated_text.strip()) > 0:
                    break

                if attempt < max_retries - 1:
                    logging.warning(
                        f"Sample {idx + 1} attempt {attempt + 1} empty; retrying..."
                    )

        if not generated_text or len(generated_text.strip()) == 0:
            logging.warning(
                f"Sample {idx + 1} empty after {max_retries - 1} retries"
            )

        if idx < 3:
            print(f"  Generated tokens: {generated_token_count}")
            print(f"  Generated length: {len(generated_text)} chars")
            print(f"  Generated preview: {generated_text[:150]}...")
            print(f"  Target preview: {target_text[:150]}...")

        # evaluate_data format
        result_item = {
            "generated_text": generated_text,
            "output": target_text
        }
        results.append(result_item)
        
        if args.use_profile:
            retrieved_data.append({
                "source": input_text,
                "target": target_text,
                "generated_text": generated_text
            })
        
        if args.use_demographic:
            demographic_data.append({
                "source": input_text,
                "target": target_text,
                "generated_text": generated_text
            })
    
    print("\n" + "=" * 80)
    print("Generation complete.")
    print(f"Total samples: {len(results)}")
    print("=" * 80)

    # Output dir: {output_dir}/{name}/
    output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.use_profile and retrieved_data:
        retrieved_file = os.path.join(output_dir, f"{args.task}_retrieved_prompts.json")
        with open(retrieved_file, 'w', encoding='utf-8') as f:
            json.dump(retrieved_data, f, indent=2, ensure_ascii=False)
        print(f"Saved retrieved prompts: {retrieved_file}")

    if args.use_demographic and demographic_data:
        demographic_file = os.path.join(output_dir, f"{args.task}_demographic_prompts.json")
        with open(demographic_file, 'w', encoding='utf-8') as f:
            json.dump(demographic_data, f, indent=2, ensure_ascii=False)
        print(f"Saved demographic prompts: {demographic_file}")

    results_file = os.path.join(output_dir, f"{args.task}_generated.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results: {results_file}")
    
    return results


if __name__ == "__main__":

    logging.basicConfig(filename='zero_shot_llama.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    opts = parser.parse_args()
    retriever,task = opts.retriever,opts.task
    
    print("=" * 80)
    print("Run arguments:")
    print("=" * 80)
    print(f"  name: {opts.name}")
    print(f"  validation_data: {opts.validation_data}")
    print(f"  model_path: {opts.model_path}")
    print(f"  task: {opts.task}")
    print(f"  output_dir: {opts.output_dir}")
    print(f"  use_profile: {opts.use_profile}")
    print(f"  use_demographic: {opts.use_demographic}")
    print(f"  demographic_in_context: {opts.demographic_in_context}")
    if opts.demographic_in_context:
        print(f"  demographic_position: {opts.demographic_position}")
    if opts.use_profile:
        print(f"  retriever: {opts.retriever}")
        print(f"  num_support_profile: {opts.num_support_profile}")
        print(f"  is_ranked: {opts.is_ranked}")
    elif opts.use_demographic:
        print("  Using Demographic Information baseline")
    print(f"  max_new_tokens: {opts.max_new_tokens}")
    print(f"  max_input_length: {opts.max_input_length}")
    print(f"  do_sample: {opts.do_sample}")
    if opts.do_sample:
        print(f"  temperature: {opts.temperature}")
        print(f"  top_p: {opts.top_p}")
    print("="*80)
    
    print(f"\nLoading model from {opts.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(opts.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token ({tokenizer.eos_token})")
    
    print(f"  Tokenizer vocab_size: {len(tokenizer)}")
    
    model = AutoModelForCausalLM.from_pretrained(
        opts.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    print("Model loaded.\n")
    
    logging.info("Creating prompt")
    start = time.process_time()
    if opts.demographic_in_context and not (opts.use_profile and opts.use_demographic):
        raise ValueError("demographic_in_context requires both use_profile and use_demographic")

    if opts.use_profile:
        print("Creating prompt generator (with profile)...")
        prompt_generator, contriver = create_prompt_generator(
            opts.num_support_profile,
            opts.retriever,
            opts.is_ranked,
            opts.max_length,
            tokenizer=tokenizer,
            cache_dir=opts.cache_dir,
        )
        logging.info("Got prompt")
        print(f"  Profile retriever: {opts.retriever}")
        nprof = opts.num_support_profile if opts.num_support_profile != -1 else "all"
        print(f"  Profiles retrieved: {nprof}")
        if opts.cache_dir and opts.retriever == "contriver":
            print(f"  Local Contriever cache: {opts.cache_dir}")
    elif opts.use_demographic:
        prompt_generator, contriver = None, None
        print("Demographic Information mode (no profile retrieval)\n")
    else:
        prompt_generator, contriver = None, None
        print("No profile (baseline mode)\n")
    
    logging.info("use_profile: "+str(opts.use_profile))
    logging.info("use_demographic: "+str(opts.use_demographic))
    logging.info(f"Task value: '{task}', type: {type(task)}")
    
    print(f"Loading validation data: {opts.validation_data}")
    if opts.use_demographic and not opts.use_profile:
        print("Demographic Information baseline...")
        if task == "generation_abstract":
            eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, False, task, None, use_demographic=True)
        elif task == "topic_writing":
            eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, False, task, None, use_demographic=True)
        elif task == "product_review_writing":
            eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, False, task, None, use_demographic=True)
        elif task == "email_generation":
            eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, False, task, None, use_demographic=True)
        else:
            raise ValueError(f"Unknown task: '{task}'. Supported tasks: generation_abstract, topic_writing, product_review_writing, email_generation")
    elif task == "generation_abstract":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
    elif task == "topic_writing":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
    elif task == "product_review_writing":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
    elif task == "email_generation":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
    else:
        raise ValueError(f"Unknown task: '{task}'. Supported tasks: generation_abstract, topic_writing, product_review_writing, email_generation")
    
    print(f"Dataset size: {len(eval_dataset)} samples")
    logging.info("Created eval dataset")

    logging.info("Getting into convert")
    print("\nConverting dataset format...")
    prompted_dataset = convert_to_llama_dataset(eval_dataset)
    logging.info("Converted to llama dataset")
    logging.info("Added to retrieved json file")

    if opts.use_profile and opts.use_demographic and opts.demographic_in_context:
        print("Fusion mode: injecting Demographic Information into RAG context...")
        with open(opts.validation_data, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        demo_map = {}
        for i, item in enumerate(raw_data):
            sample_id = item.get('id', f"sample_{i}")
            demo_map[str(sample_id)] = item.get('Demographic Information', '')

        for i, item in enumerate(prompted_dataset):
            sample_id = str(item.get('id', f"sample_{i}"))
            demo_info = demo_map.get(sample_id, '')
            if demo_info and str(demo_info).strip():
                item['source'] = inject_demographic_into_context(
                    item.get('source', ''),
                    demo_info,
                    position=opts.demographic_position,
                )
    
    results = run_local_model(prompted_dataset, model, tokenizer, opts.output_dir, opts)
    logging.info("Got results")
    
    print("\nEvaluating...")
    metrics = evaluate_data(results,opts.name)
    logging.info("Got metrics")
    
    print("\n" + "="*80)
    print("Metrics:")
    print("="*80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*80)
    
    elapsed_time = round((time.process_time() - start), 2)
    logging.info(f"{elapsed_time}s ")
    print(f"\nElapsed: {elapsed_time}s")