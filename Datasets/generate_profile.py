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

import argparse
import os
import json
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI
import shutil

def print_progress(current, total, prefix="Progress", bar_length=50):
    if total == 0:
        return
    percent = float(current) / total
    filled_length = int(bar_length * percent)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    percent_str = f"{percent * 100:.1f}%"
    sys.stdout.write(f'\r{prefix}: [{bar}] {current}/{total} ({percent_str})')
    sys.stdout.flush()
    if current == total:
        print()

PROMPT_TEMPLATE = """Analyze the following complete history of a person's work/content to create a concise and accurate persona profile called "Demographic Information".

Based on their complete history, infer their characteristics, preferences, and style. Focus ONLY on aspects that can be credibly determined from the provided content:

**Key Aspects to Analyze:**
1. **Domain & Topics**: What are their primary areas of interest? Identify specific domains, themes, and recurring topics.
2. **Style & Approach**: 
   - Writing style (formal vs casual, technical vs accessible)
   - Content focus (theoretical vs practical, detailed vs concise)
   - Tone and perspective (critical vs supportive, analytical vs descriptive)
3. **Depth & Sophistication**: Level of expertise, complexity, and sophistication in their work
4. **Continuity & Evolution**: Consistency in interests, evolution over time, temporal patterns
5. **Preferred Keywords**: 3-5 specific terms, topics, or themes they consistently focus on
6. **Avoided Topics**: 3-5 areas they clearly don't engage with (only if evident from absence in their history)

**Output Requirements:**
- Start with "This person likes [keyword1], [keyword2], [keyword3]..." - use specific terms from their actual work/content
- Then "This person dislikes [keyword1], [keyword2]..." - only include if clearly evident from what they avoid
- Follow with a 2-3 sentence summary describing:
  * Their primary focus and domain expertise
  * Their style and approach
  * Key characteristics that distinguish their work

**Important Guidelines:**
- Be specific and evidence-based - only include information directly inferable from the history
- Use terminology that reflects their actual areas of interest
- Focus on work-related characteristics, not personal attributes that cannot be determined
- If the history is very diverse, note the breadth; if focused, note the specialization
- Consider the temporal span and evolution of their interests
- Adapt the analysis to the type of content (research papers, reviews, emails, tweets, etc.)

Complete History:
{profile}

Example Output Format:
{{
    "Demographic Information": "This person likes wireless sensor networks, energy-efficient protocols, distributed systems, network optimization, and signal processing. This person dislikes pure theoretical mathematics, social sciences, and non-technical humanities. A researcher specializing in wireless sensor networks with a strong focus on energy-efficient protocol design and network optimization. Their work spans from low-level protocol architectures to high-level system management, demonstrating both theoretical rigor and practical application. They consistently explore application-specific solutions that balance performance, energy consumption, and system reliability."
}}

IMPORTANT: Output ONLY the raw JSON string with key "Demographic Information". Do not include any preamble or explanation.
"""

def format_publications(profile, max_papers: int = 50):
    """Format profile history for the prompt (caps list length). Supports LaMP/LongLaMP item shapes."""
    
    formatted_list = []
    
    if not isinstance(profile, list) or len(profile) == 0:
        return "No publication history available."
    
    total_papers = len(profile)
    papers = profile[:max_papers] if total_papers > max_papers else profile
    
    if total_papers > max_papers:
        formatted_list.append(f"Total publications: {total_papers} (showing first {max_papers})\n")
    else:
        formatted_list.append(f"Total publications: {total_papers}\n")
    
    if len(papers) > 0 and not isinstance(papers[0], dict):
        return f"No valid publication history available. Profile items are not dictionaries. First item type: {type(papers[0])}"
    
    first_pub = papers[0] if len(papers) > 0 else {}
    available_keys = set(first_pub.keys()) if isinstance(first_pub, dict) else set()

    is_product_review = 'overall' in available_keys and 'reviewText' in available_keys
    is_topic_writing = ('input' in available_keys and 'output' in available_keys) or ('summary' in available_keys and 'content' in available_keys)
    is_email = 'title' in available_keys and 'text' in available_keys
    is_lamp7 = 'text' in available_keys and 'date' in available_keys and 'id' in available_keys
    is_abstract = 'title' in available_keys and 'abstract' in available_keys
    
    valid_papers = 0
    for i, pub in enumerate(papers, 1):
        if not isinstance(pub, dict):
            continue
        
        if is_product_review:
            title = pub.get('description', pub.get('Description', 'N/A'))
            abstract = pub.get('reviewText', pub.get('reviewText', pub.get('summary', pub.get('Summary', 'N/A'))))
            rating = pub.get('overall', pub.get('Overall', 'N/A'))
            date = 'N/A'
            if title != 'N/A' or abstract != 'N/A':
                formatted_list.append(f"Review {i} (Rating: {rating}):\nProduct Description: {title}\nReview Text: {abstract}")
                valid_papers += 1
        
        elif is_topic_writing:
            title = pub.get('summary', pub.get('Summary', pub.get('input', pub.get('Input', 'N/A'))))
            abstract = pub.get('content', pub.get('Content', pub.get('output', pub.get('Output', 'N/A'))))
            date = pub.get('date', pub.get('Date', pub.get('year', pub.get('Year', 'N/A'))))
            if title != 'N/A' or abstract != 'N/A':
                formatted_list.append(f"Writing Sample {i} ({date}):\nSummary/Input: {title}\nContent/Output: {abstract}")
                valid_papers += 1
        
        elif is_email:
            title = pub.get('title', pub.get('Title', 'N/A'))
            abstract = pub.get('text', pub.get('Text', 'N/A'))
            date = pub.get('date', pub.get('Date', pub.get('year', pub.get('Year', 'N/A'))))
            if title != 'N/A' or abstract != 'N/A':
                formatted_list.append(f"Email {i} ({date}):\nSubject: {title}\nBody: {abstract}")
                valid_papers += 1
        
        elif is_lamp7:
            title = 'Tweet/Post'
            abstract = pub.get('text', pub.get('Text', 'N/A'))
            date = pub.get('date', pub.get('Date', 'N/A'))
            if abstract != 'N/A':
                formatted_list.append(f"Post {i} ({date}):\nContent: {abstract}")
                valid_papers += 1
        
        elif is_abstract:
            title = pub.get('title', pub.get('Title', 'N/A'))
            abstract = pub.get('abstract', pub.get('Abstract', 'N/A'))
            date = pub.get('date', pub.get('Date', pub.get('year', pub.get('Year', 'N/A'))))
            if title != 'N/A' or abstract != 'N/A':
                formatted_list.append(f"Publication {i} ({date}):\nTitle: {title}\nAbstract: {abstract}")
                valid_papers += 1
        
        else:
            title = (pub.get('title') or pub.get('Title') or 
                    pub.get('name') or pub.get('Name') or 'N/A')
            abstract = (pub.get('abstract') or pub.get('Abstract') or 
                       pub.get('description') or pub.get('Description') or 
                       pub.get('summary') or pub.get('Summary') or 
                       pub.get('text') or pub.get('Text') or 
                       pub.get('content') or pub.get('Content') or 'N/A')
            date = (pub.get('date') or pub.get('Date') or 
                   pub.get('year') or pub.get('Year') or 
                   pub.get('published_date') or pub.get('publishedDate') or 'N/A')
            
            if title != 'N/A' or abstract != 'N/A':
                formatted_list.append(f"Item {i} ({date}):\nTitle: {title}\nContent: {abstract}")
                valid_papers += 1
    
    if valid_papers == 0:
        if len(papers) > 0 and isinstance(papers[0], dict):
            sample_keys = list(papers[0].keys())
            return f"No valid publication history available. Publications exist but lack recognizable content fields. Available keys in first publication: {sample_keys}"
        else:
            return f"No valid publication history available. Profile structure is invalid."
    
    return "\n\n".join(formatted_list)

def normalize_summary(s):
    text = s.get("content") if isinstance(s, dict) else str(s)
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "Demographic Information" in parsed:
            return parsed["Demographic Information"]
        elif isinstance(parsed, str):
            return parsed
    except json.JSONDecodeError:
        pass
    
    return text

async def summarize_profiles_inplace(
    input_file: str,
    api_key: str,
    base_url: str | None,
    lm_model: str,
    batch_size: int,
    max_papers: int,
    dry_run: bool = False,
):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading: {input_file}")
    print("Loading large JSON file, this may take a while...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from file.")

    def is_invalid_demographic_info(demo_info):
        if not demo_info:
            return True
        if isinstance(demo_info, str):
            demo_str = demo_info.strip()
            if not demo_str:
                return True
            invalid_patterns = [
                "likes N/A",
                "dislikes N/A", 
                "undefined focus",
                "lack of available publication",
                "cannot be determined",
                "inability to characterize"
            ]
            invalid_count = sum(1 for pattern in invalid_patterns if pattern.lower() in demo_str.lower())
            if invalid_count >= 2:
                return True
        return False

    pending_data = [
        item for item in data 
        if is_invalid_demographic_info(item.get("Demographic Information"))
    ]
    print(f"Total: {len(data)}, Pending (no/invalid Demographic Information): {len(pending_data)}")
    
    if len(pending_data) > 0:
        missing_count = sum(1 for item in pending_data if not item.get("Demographic Information"))
        invalid_count = sum(1 for item in pending_data if item.get("Demographic Information") and is_invalid_demographic_info(item.get("Demographic Information")))
        print(f"  - missing: {missing_count}, invalid (e.g. N/A): {invalid_count}")
    
    if len(pending_data) == 0:
        print("No pending data to process. Exiting.")
        return

    if dry_run:
        print(f"Dry run: would process {len(pending_data)} samples in "
              f"{(len(pending_data) + batch_size - 1) // batch_size} batches.")
        return

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (use --input and env, or pass --api-key).")

    backup_path = input_path.with_suffix(input_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)
        print(f"Backup created: {backup_path}")

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)

    async def request_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=lm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = resp.choices[0].message.content if resp and resp.choices else ""
                return content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"OpenAI request failed after {max_retries} attempts: {e}")
                    return ""
                await asyncio.sleep(base_delay * (2 ** attempt))

    print("\n" + "="*60)
    print("First 5 pending samples:")
    print("="*60)
    for idx, item in enumerate(pending_data[:5], 1):
        profile = item.get('profile', [])
        profile_count = len(profile) if isinstance(profile, list) else 0
        input_text = item.get('input', '')[:100] if item.get('input') else 'N/A'
        demo_info = item.get('Demographic Information', '')
        print(f"\nSample {idx}:")
        print(f"  ID: {item.get('id', 'N/A')}")
        print(f"  Profile items: {profile_count}")
        if profile_count > 0:
            first_paper = profile[0] if isinstance(profile, list) else {}
            if isinstance(first_paper, dict):
                print(f"  First item keys: {list(first_paper.keys())}")
                first_title = (first_paper.get('title') or first_paper.get('Title') or 'N/A')
                if len(str(first_title)) > 80:
                    first_title = str(first_title)[:80] + "..."
                print(f"  First title: {first_title}")
            else:
                print(f"  First item type: {type(first_paper)}, value: {first_paper}")
        if demo_info:
            demo_preview = str(demo_info)[:100] + "..." if len(str(demo_info)) > 100 else str(demo_info)
            print(f"  Current Demographic Information: {demo_preview}")
        if input_text != 'N/A' and len(input_text) > 0:
            print(f"  Input preview: {input_text}...")
    print("="*60 + "\n")

    total_batches = (len(pending_data) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batches of up to {batch_size} samples")
    print(f"Up to {max_papers} papers per sample\n")

    for i in range(0, len(pending_data), batch_size):
        batch = pending_data[i : i + batch_size]
        batch_num = i // batch_size + 1
        
        print_progress(i, len(pending_data), f"batch {batch_num}/{total_batches}")

        print(f"\nFormatting prompts for batch {batch_num}...")
        prompts = []
        for item_idx, item in enumerate(batch):
            profile = item.get('profile', [])
            profile_count = len(profile) if isinstance(profile, list) else 0
            formatted = format_publications(profile, max_papers=max_papers)
            
            if item_idx == 0:
                print(f"  First sample: {profile_count} papers")
                print(f"  Formatted (first 500 chars): {formatted[:500]}")
                if "No valid" in formatted or formatted.startswith("No publication history"):
                    print("  Warning: formatting returned an error message")
                    print(f"  Raw profile type: {type(profile)}")
                    if isinstance(profile, list) and len(profile) > 0:
                        print(f"  First profile entry: {profile[0]}")
                elif profile_count > 0 and isinstance(profile, list) and len(profile) > 0:
                    first_item = profile[0]
                    if isinstance(first_item, dict):
                        has_content = False
                        for key in ['title', 'abstract', 'content', 'text', 'reviewText', 'description', 'summary', 'input', 'output']:
                            if key in first_item and first_item[key] and str(first_item[key]).strip() and str(first_item[key]).strip() != 'N/A':
                                has_content = True
                                break
                        if not has_content:
                            print("  Warning: first profile item has no usable text fields")
                            print(f"  Keys present: {list(first_item.keys())}")

            prompt = PROMPT_TEMPLATE.format(profile=formatted)
            prompts.append(prompt)
            if item_idx == 0:
                print(f"  Prompt length: {len(prompt)} chars")

        print(f"Sending {len(batch)} API requests...")
        tasks = [asyncio.create_task(request_with_retry(p)) for p in prompts]
        summaries = await asyncio.gather(*tasks)

        failed_count = 0
        for item, summary in zip(batch, summaries):
            normalized = normalize_summary(summary)
            if not normalized or not normalized.strip():
                failed_count += 1
            item["Demographic Information"] = normalized

        if failed_count > 0:
            print(f"\nError: batch {batch_num} had {failed_count}/{len(batch)} failed requests")
            print("Saving progress and exiting...")
            tmp_path = input_path.with_suffix(input_path.suffix + ".tmp")
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, input_path)
            print(f"Saved progress to: {input_file}")
            print("Check API key, quota, and base URL, then rerun.")
            sys.exit(1)

        print(f"Batch {batch_num} done: {i + len(batch)} / {len(pending_data)}")
        print()

        tmp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, input_path)

    print("\nFinal save...")
    tmp_path = input_path.with_suffix(input_path.suffix + ".tmp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, input_path)
    print(f"Success. In-place updated: {input_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill Demographic Information in LaMP / LongLaMP JSON via an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--input", "-i",
        default=os.environ.get("LAMP_INPUT_FILE"),
        help="JSON file to update in place (or set LAMP_INPUT_FILE)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (default: OPENAI_API_KEY env)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL (default: OPENAI_BASE_URL env; omit for provider default)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LM_MODEL", "gpt-4o-mini"),
        help="Chat model name (default: gpt-4o-mini or LM_MODEL env)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "20")),
        help="Samples per API batch (default: 20)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=int(os.environ.get("MAX_PAPERS", "50")),
        help="Max profile items per sample in the prompt (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count pending samples only; no API calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input:
        print("ERROR: pass --input or set LAMP_INPUT_FILE", file=sys.stderr)
        sys.exit(2)
    asyncio.run(
        summarize_profiles_inplace(
            input_file=args.input,
            api_key=args.api_key or "",
            base_url=args.base_url or None,
            lm_model=args.model,
            batch_size=args.batch_size,
            max_papers=args.max_papers,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()