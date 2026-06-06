#!/usr/bin/env python3
"""
LLM-as-Judge evaluation for personalized text generation (profile steering).

Method: G-Eval style multi-dimensional rubric scoring (1-5 Likert) with
chain-of-thought reasoning, adapted for persona/style alignment tasks.

Supports OpenAI-compatible APIs (OpenAI, DeepSeek, Azure, local vLLM, etc.).

Usage:
  # Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) or pass --api_key / --base_url.

  # Evaluate one file (dry-run first 3 samples)
  python llm_judge_eval.py --path lamp_eval_profile_steering_results/qwen/a+b/coeff_1/LaMP_4_steer_layer19_coeff1.0_predictions.json --limit 3

  # Full run with concurrency
  python llm_judge_eval.py --path lamp_eval_profile_steering_results/qwen/a+b/coeff_1 --workers 8

  # Batch: coeff 0.2–3 under a+b, summary to downloaded_results/
  python llm_judge_eval.py --path lamp_eval_profile_steering_results/qwen/a+b \
    --coeff_min 0.2 --coeff_max 3 \
    --summary_out qwen_a+b_llm_judge_summary.json --workers 8

  # Compare two steering configs (pairwise win-rate)
  python llm_judge_eval.py --compare path/to/baseline.json path/to/steered.json --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# API config: env vars and CLI only (no hardcoded secrets)
# ---------------------------------------------------------------------------
DEFAULT_API_KEY: str | None = None
DEFAULT_BASE_URL: str | None = None

# ---------------------------------------------------------------------------
# Task descriptions & rubrics (G-Eval inspired, personalization-focused)
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS: dict[str, str] = {
    "LaMP_4": "Generate a news article headline that matches the user's historical headline style.",
    "LaMP_5": "Generate a scholarly paper title that matches the user's historical title style.",
    "LaMP_7": "Generate a social-media post (tweet) that matches the user's historical posting style.",
    "topic_writing": "Continue or write a Reddit-style post in the user's personal writing voice.",
    "product_review_writing": "Write a product review in the user's distinctive review style.",
    "generation_abstract": "Write an academic paper abstract in the user's scholarly writing style.",
}

DIMENSIONS = [
    "style_alignment",   # surface writing style match to reference
    "task_fulfillment",  # completes the assigned generation task
    "coherence",         # fluent, logically structured, not garbled
    "content_quality",   # substantive, on-topic, minimal meta-chatter
]

DIMENSION_LABELS: dict[str, str] = {
    "style_alignment": "Style & Persona Alignment",
    "task_fulfillment": "Task Fulfillment",
    "coherence": "Coherence & Fluency",
    "content_quality": "Content Quality",
}

TRUNCATION_POLICY = """
## Important: ignore truncation
The GENERATED output may be cut off mid-sentence due to max-length limits — this is NOT
the model's fault. When scoring:
- Evaluate only the portion that is present; treat an abrupt ending as neutral.
- Do NOT lower style_alignment, task_fulfillment, coherence, content_quality, or overall because the text ends incomplete, mid-word, or mid-sentence.
- Do NOT mention truncation as a weakness in your reasoning unless it causes internal
  garbling (e.g., repetition loops) within the visible text.
""".strip()

RUBRIC_TEXT = """
Score each dimension from 1 to 5 (integers only):

**style_alignment** — How well does the GENERATED text match the user's personal
writing style in the REFERENCE? Consider tone, vocabulary, sentence structure,
formatting habits, humor/sarcasm level, and distinctive phrases. Do NOT penalize
for differing factual content or for the output being cut off at the end.
  1 = completely different voice; 2 = weak resemblance; 3 = partial match;
  4 = strong match; 5 = near-indistinguishable style

**task_fulfillment** — Does the generation accomplish the task described below?
Penalize excessive meta-commentary ("Here are a few options", "Title:", instructions
to the user, or answering the wrong task). Do NOT penalize an incomplete ending caused
by length truncation — judge whether the visible content moves toward the task.
  1 = fails task entirely; 3 = partially fulfills; 5 = fully and cleanly fulfills

**coherence** — Is the visible portion readable and logically ordered? Penalize
repetition loops or incoherent fragments within the text, but NOT an abrupt stop at the end.
  1 = unreadable; 3 = understandable with issues; 5 = polished and fluent

**content_quality** — Is the visible output substantive, relevant, and appropriately
detailed for the task (not empty, off-topic, or mostly template filler)?
Do not penalize for missing content that would appear after a truncation point.
  1 = no useful content; 3 = adequate; 5 = high-quality, informative output
""".strip()

SYSTEM_PROMPT = """You are an expert evaluator for personalized text generation systems.
You judge whether a model successfully mimics a user's writing style while completing
a specific generation task. Be objective, consistent, and critical of common failure
modes (meta-instructions, multiple draft options, style drift) — but NEVER penalize
outputs that were cut off by a length limit (truncation).
Always respond with valid JSON only — no markdown fences."""

SCORING_USER_TEMPLATE = """## Task
{task_description}

## Reference (user's historical writing style / ground truth)
{reference}

## Generated output (model prediction to evaluate)
{prediction}

## Rubric
{rubric}

{truncation_policy}

Evaluate the GENERATED output. Provide brief reasoning then scores.

Respond with this exact JSON schema:
{{
  "reasoning": "<2-4 sentences summarizing key strengths and weaknesses>",
  "style_alignment": <int 1-5>,
  "task_fulfillment": <int 1-5>,
  "coherence": <int 1-5>,
  "content_quality": <int 1-5>,
  "overall": <int 1-5>
}}

The "overall" score is your holistic judgment (1-5), not necessarily the average."""

PAIRWISE_SYSTEM_PROMPT = """You are an expert judge comparing two personalized text
generations for the same user. Decide which better matches the user's style while
fulfilling the task. Ignore truncation / abrupt endings at the end of either response.
Respond with valid JSON only."""

PAIRWISE_USER_TEMPLATE = """## Task
{task_description}

## Reference (user's style / ground truth)
{reference}

## Response A
{response_a}

## Response B
{response_b}

{truncation_policy}

Which response is better for personalized generation?
- "A" if A is clearly better
- "B" if B is clearly better
- "tie" if roughly equal

JSON schema:
{{
  "reasoning": "<brief comparison>",
  "winner": "A" | "B" | "tie"
}}"""


# ---------------------------------------------------------------------------
# Data loading (compatible with evaluate_jsonl.py)
# ---------------------------------------------------------------------------

def infer_task_from_path(path: Path) -> str:
    """Infer task name from file path (e.g. results/LaMP_4/.../LaMP-4_generated.json)."""
    blob = "/".join(path.parts).lower().replace("-", "_")
    patterns = [
        ("generation_abstract", "generation_abstract"),
        ("product_review_writing", "product_review_writing"),
        ("topic_writing", "topic_writing"),
        ("lamp_7", "LaMP_7"),
        ("lamp_5", "LaMP_5"),
        ("lamp_4", "LaMP_4"),
    ]
    for needle, task in patterns:
        if needle in blob:
            return task
    return "unknown"


def load_predictions(path: str | Path) -> tuple[str, list[dict]]:
    path = Path(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
            task = rows[0].get("task") if rows else None
            return task or infer_task_from_path(path), rows
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        task = data.get("task") or infer_task_from_path(path)
        return task, data["predictions"]
    if isinstance(data, list):
        return infer_task_from_path(path), data
    raise ValueError(f"Unrecognized format: {path}")


def extract_fields(item: dict, index: int | None = None) -> tuple[str, str, str]:
    sample_id = str(item.get("id", item.get("sample_id", "")))
    if not sample_id and index is not None:
        sample_id = f"__idx_{index}"
    reference = (
        item.get("reference")
        or item.get("output")
        or item.get("gold")
        or ""
    )
    prediction = (
        item.get("prediction")
        or item.get("generated_text")
        or item.get("pred")
        or ""
    )
    return sample_id, reference.strip(), prediction.strip()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible)
# ---------------------------------------------------------------------------

@dataclass
class JudgeConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = DEFAULT_API_KEY
    base_url: str | None = DEFAULT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 5
    retry_delay: float = 2.0
    max_ref_chars: int = 2000
    max_pred_chars: int = 1500


def get_client(cfg: JudgeConfig):
    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install openai", file=sys.stderr)
        raise
    kwargs: dict[str, Any] = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs)


def call_llm(
    client,
    cfg: JudgeConfig,
    system: str,
    user: str,
) -> str:
    last_err: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.retry_delay * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {cfg.max_retries} retries: {last_err}")


def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    # strip markdown fences if model ignores instruction
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def validate_scores(parsed: dict) -> dict:
    result: dict[str, Any] = {
        "reasoning": str(parsed.get("reasoning", "")),
    }
    for dim in DIMENSIONS:
        val = parsed.get(dim)
        if val is None:
            raise ValueError(f"Missing dimension: {dim}")
        score = int(round(float(val)))
        result[dim] = max(1, min(5, score))
    overall = parsed.get("overall")
    if overall is not None:
        result["overall"] = max(1, min(5, int(round(float(overall)))))
    else:
        result["overall"] = round(
            sum(result[d] for d in DIMENSIONS) / len(DIMENSIONS)
        )
    return result


# ---------------------------------------------------------------------------
# Single-sample & pairwise judging
# ---------------------------------------------------------------------------

def judge_sample(
    client,
    cfg: JudgeConfig,
    task: str,
    sample_id: str,
    reference: str,
    prediction: str,
) -> dict:
    if not reference or not prediction:
        return {
            "id": sample_id,
            "skipped": True,
            "reason": "empty reference or prediction",
        }
    task_desc = TASK_DESCRIPTIONS.get(task, f"Personalized text generation task: {task}")
    user_msg = SCORING_USER_TEMPLATE.format(
        task_description=task_desc,
        reference=truncate(reference, cfg.max_ref_chars),
        prediction=truncate(prediction, cfg.max_pred_chars),
        rubric=RUBRIC_TEXT,
        truncation_policy=TRUNCATION_POLICY,
    )
    raw = call_llm(client, cfg, SYSTEM_PROMPT, user_msg)
    parsed = parse_json_response(raw)
    scores = validate_scores(parsed)
    scores["id"] = sample_id
    return scores


def judge_pairwise(
    client,
    cfg: JudgeConfig,
    task: str,
    sample_id: str,
    reference: str,
    response_a: str,
    response_b: str,
) -> dict:
    task_desc = TASK_DESCRIPTIONS.get(task, f"Task: {task}")
    user_msg = PAIRWISE_USER_TEMPLATE.format(
        task_description=task_desc,
        reference=truncate(reference, cfg.max_ref_chars),
        response_a=truncate(response_a, cfg.max_pred_chars),
        response_b=truncate(response_b, cfg.max_pred_chars),
        truncation_policy=TRUNCATION_POLICY,
    )
    raw = call_llm(client, cfg, PAIRWISE_SYSTEM_PROMPT, user_msg)
    parsed = parse_json_response(raw)
    winner = str(parsed.get("winner", "tie")).upper()
    if winner not in ("A", "B", "TIE"):
        winner = "TIE"
    return {
        "id": sample_id,
        "reasoning": parsed.get("reasoning", ""),
        "winner": winner if winner != "TIE" else "tie",
    }


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def cache_path_for(pred_path: Path) -> Path:
    return pred_path.with_name(pred_path.stem + "_llm_judge.jsonl")


def load_cache(cache_path: Path, num_items: int | None = None) -> dict[str, dict]:
    """Load deduplicated cache rows keyed by sample id (last write wins)."""
    del num_items  # kept for call-site compatibility; full-file dedupe is authoritative
    done: dict[str, dict] = {}
    if not cache_path.exists():
        return done
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("id", ""))
            if not sid or row.get("error"):
                continue
            done[sid] = row
    return done


def load_scored_rows(cache_path: Path, num_items: int | None = None) -> list[dict]:
    """Load deduplicated scored rows from cache (last write wins per id)."""
    return list(load_cache(cache_path, num_items=num_items).values())


def repair_cache_ids(pred_path: Path) -> bool:
    """Rewrite cache rows with empty ids to stable __idx_{i} keys."""
    cache_path = cache_path_for(pred_path)
    if not cache_path.exists():
        return False
    _, items = load_predictions(pred_path)
    rows: list[dict] = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return False
    if len(rows) > len(items):
        rows = rows[-len(items):]
    changed = False
    repaired: list[dict] = []
    for idx, row in enumerate(rows):
        sid = str(row.get("id", ""))
        if not sid:
            row = dict(row)
            row["id"] = f"__idx_{idx}"
            changed = True
        repaired.append(row)
    if changed:
        cache_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in repaired) + "\n",
            encoding="utf-8",
        )
    return changed


def append_cache(cache_path: Path, row: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("skipped") and not r.get("error")]
    if not valid:
        return {"samples": 0}
    agg: dict[str, Any] = {"samples": len(valid)}
    for dim in DIMENSIONS + ["overall"]:
        vals = [r[dim] for r in valid if dim in r]
        if vals:
            agg[dim] = round(sum(vals) / len(vals), 4)
            agg[f"{dim}_std"] = round(
                (sum((v - agg[dim]) ** 2 for v in vals) / len(vals)) ** 0.5, 4
            )
    # histogram for overall
    dist = {i: 0 for i in range(1, 6)}
    for r in valid:
        if "overall" in r:
            dist[r["overall"]] = dist.get(r["overall"], 0) + 1
    agg["overall_distribution"] = dist
    return agg


def summary_path_for(pred_path: Path) -> Path:
    return pred_path.with_name(pred_path.stem + "_llm_judge_summary.json")


# ---------------------------------------------------------------------------
# Evaluate one predictions file
# ---------------------------------------------------------------------------

def evaluate_file(
    pred_path: Path,
    cfg: JudgeConfig,
    limit: int | None = None,
    offset: int = 0,
    workers: int = 4,
    force: bool = False,
    dry_run: bool = False,
) -> dict | None:
    task, items = load_predictions(pred_path)
    cache_path = cache_path_for(pred_path)
    done = {} if force else load_cache(cache_path, num_items=len(items))

    pending: list[tuple[int, dict]] = []
    for i, item in enumerate(items):
        if i < offset:
            continue
        sid, ref, pred = extract_fields(item, index=i)
        if sid in done:
            continue
        pending.append((i, item))
        if limit is not None and len(pending) >= limit:
            break

    print(f"\n[{pred_path.name}] task={task}, total={len(items)}, "
          f"cached={len(done)}, pending={len(pending)}")

    if dry_run:
        for _, item in pending[:3]:
            sid, ref, pred = extract_fields(item)
            print(f"  sample {sid}: ref={len(ref)} chars, pred={len(pred)} chars")
        return None

    if not pending:
        all_rows = load_scored_rows(cache_path)
        summary = aggregate_scores(all_rows)
        summary["task"] = task
        summary["source"] = str(pred_path)
        sp = summary_path_for(pred_path)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  All cached. Summary -> {sp}")
        return summary

    client = get_client(cfg)

    def _judge_one(idx_item: tuple[int, dict]) -> dict:
        idx, item = idx_item
        sid, ref, pred = extract_fields(item, index=idx)
        try:
            return judge_sample(client, cfg, task, sid, ref, pred)
        except Exception as e:
            return {"id": sid, "error": str(e)}

    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_judge_one, p): p for p in pending}
        try:
            from tqdm import tqdm
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Judging")
        except ImportError:
            iterator = as_completed(futures)
        for fut in iterator:
            row = fut.result()
            append_cache(cache_path, row)
            new_rows.append(row)
            if row.get("error"):
                print(f"  ERROR id={row.get('id')}: {row['error']}")

    all_rows = load_scored_rows(cache_path)
    summary = aggregate_scores(all_rows)
    summary["task"] = task
    summary["source"] = str(pred_path)
    summary["model"] = cfg.model
    sp = summary_path_for(pred_path)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Done. Summary -> {sp}")
    print(f"  overall={summary.get('overall', 'N/A')}, "
          f"style_alignment={summary.get('style_alignment', 'N/A')}")
    return summary


# ---------------------------------------------------------------------------
# Pairwise comparison of two runs
# ---------------------------------------------------------------------------

def compare_files(
    path_a: Path,
    path_b: Path,
    cfg: JudgeConfig,
    limit: int | None = None,
    workers: int = 4,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    task_a, items_a = load_predictions(path_a)
    task_b, items_b = load_predictions(path_b)
    task = task_a if task_a != "unknown" else task_b

    map_a = {extract_fields(it)[0]: it for it in items_a}
    map_b = {extract_fields(it)[0]: it for it in items_b}
    common_ids = sorted(set(map_a) & set(map_b))
    if limit:
        common_ids = common_ids[:limit]

    out_path = path_a.parent / f"pairwise_{path_a.stem}_vs_{path_b.stem}_judge.jsonl"
    client = get_client(cfg)
    wins = {label_a: 0, label_b: 0, "tie": 0}

    def _one(sid: str) -> dict:
        _, ref, _ = extract_fields(map_a[sid])
        _, _, pred_a = extract_fields(map_a[sid])
        _, _, pred_b = extract_fields(map_b[sid])
        result = judge_pairwise(client, cfg, task, sid, ref, pred_a, pred_b)
        w = result.get("winner", "tie")
        if w == "A":
            wins[label_a] += 1
        elif w == "B":
            wins[label_b] += 1
        else:
            wins["tie"] += 1
        result["label_a"] = label_a
        result["label_b"] = label_b
        return result

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, sid): sid for sid in common_ids}
        try:
            from tqdm import tqdm
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Pairwise")
        except ImportError:
            iterator = as_completed(futures)
        for fut in iterator:
            row = fut.result()
            rows.append(row)
            append_cache(out_path, row)

    n = len(rows) or 1
    summary = {
        "task": task,
        "samples": len(rows),
        "path_a": str(path_a),
        "path_b": str(path_b),
        "label_a": label_a,
        "label_b": label_b,
        "win_rate_a": round(wins[label_a] / n, 4),
        "win_rate_b": round(wins[label_b] / n, 4),
        "tie_rate": round(wins["tie"] / n, 4),
        "wins": wins,
    }
    summary_file = out_path.with_suffix(".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nPairwise: {label_a} wins {wins[label_a]}, {label_b} wins {wins[label_b]}, "
          f"ties {wins['tie']} -> {summary_file}")
    return summary


# ---------------------------------------------------------------------------
# Directory walk
# ---------------------------------------------------------------------------

def is_predictions_file(name: str) -> bool:
    lower = name.lower()
    if "_llm_judge" in lower or "_metrics" in lower:
        return False
    if any(x in lower for x in (
        "_prompts", "_timing", "_stats", "_retrieved", "run.log",
    )):
        return False
    if lower.endswith("_generated.json") or lower.endswith("_generated.jsonl"):
        return True
    return (
        (name.endswith(".json") or name.endswith(".jsonl"))
        and "predictions" in lower
    )


def parse_coeff_dir(dirname: str) -> float | None:
    """coeff_0p2 -> 0.2, coeff_1p5 -> 1.5, coeff_3 -> 3.0"""
    if not dirname.startswith("coeff_"):
        return None
    token = dirname[len("coeff_"):].replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def coeff_in_range(
    dirname: str,
    coeff_min: float | None,
    coeff_max: float | None,
) -> bool:
    val = parse_coeff_dir(dirname)
    if val is None:
        return coeff_min is None and coeff_max is None
    if coeff_min is not None and val < coeff_min - 1e-9:
        return False
    if coeff_max is not None and val > coeff_max + 1e-9:
        return False
    return True


def collect_prediction_files(
    root: Path,
    coeff_min: float | None = None,
    coeff_max: float | None = None,
) -> list[Path]:
    files: list[Path] = []
    if not root.is_dir():
        return files

    # Steering sweep layout: qwen/a+b/coeff_0p2/*.json (flat per coeff dir)
    has_coeff_layout = any(
        p.is_dir() and parse_coeff_dir(p.name) is not None
        for p in root.iterdir()
    )
    if has_coeff_layout or parse_coeff_dir(root.name) is not None:
        if parse_coeff_dir(root.name) is not None:
            coeff_dirs = [root]
        else:
            coeff_dirs = sorted(
                p for p in root.iterdir()
                if p.is_dir() and coeff_in_range(p.name, coeff_min, coeff_max)
            )
        for coeff_dir in coeff_dirs:
            for fpath in sorted(coeff_dir.iterdir()):
                if fpath.is_file() and is_predictions_file(fpath.name):
                    files.append(fpath)
        return files

    # Nested layout: <study_root>/<task>/<method>/*_generated.json
    for fpath in sorted(root.rglob("*")):
        if fpath.is_file() and is_predictions_file(fpath.name):
            files.append(fpath)
    return files


def build_master_summary(
    all_summaries: dict[str, dict],
    study_root: Path,
) -> dict:
    """Pivot summaries into coeff × task tables for easy comparison."""
    by_coeff_task: dict[str, dict[str, dict]] = {}
    rows: list[dict] = []

    for rel_path, summary in all_summaries.items():
        parts = Path(rel_path).parts
        coeff_name = parts[0] if parts else "unknown"
        coeff_val = parse_coeff_dir(coeff_name)
        task = summary.get("task", "unknown")
        method = parts[1] if len(parts) > 2 and not parts[0].startswith("coeff_") else ""
        entry = {
            "coeff": coeff_name,
            "coeff_value": coeff_val,
            "method": method,
            "task": task,
            "file": rel_path,
            "samples": summary.get("samples", 0),
            "style_alignment": summary.get("style_alignment"),
            "task_fulfillment": summary.get("task_fulfillment"),
            "coherence": summary.get("coherence"),
            "content_quality": summary.get("content_quality"),
            "overall": summary.get("overall"),
        }
        rows.append(entry)
        by_coeff_task.setdefault(coeff_name, {})[task] = entry

    # Mean overall per coeff (across tasks)
    coeff_overall: dict[str, float] = {}
    for coeff_name, tasks in by_coeff_task.items():
        scores = [t["overall"] for t in tasks.values() if t.get("overall") is not None]
        if scores:
            coeff_overall[coeff_name] = round(sum(scores) / len(scores), 4)

    return {
        "study_root": str(study_root),
        "files_evaluated": len(all_summaries),
        "coeff_overall_mean": coeff_overall,
        "by_file": all_summaries,
        "by_coeff_task": by_coeff_task,
        "flat_rows": sorted(
            rows,
            key=lambda r: (r.get("coeff_value") or 0, r.get("task", "")),
        ),
    }


def write_summary_csv(rows: list[dict], csv_path: Path) -> None:
    fields = [
        "coeff", "coeff_value", "method", "task", "samples",
        "style_alignment", "task_fulfillment", "coherence",
        "content_quality", "overall", "file",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_directory(
    root: Path,
    cfg: JudgeConfig,
    limit: int | None,
    offset: int,
    workers: int,
    force: bool,
    dry_run: bool,
    coeff_min: float | None = None,
    coeff_max: float | None = None,
    summary_out: Path | None = None,
) -> dict:
    pred_files = collect_prediction_files(root, coeff_min, coeff_max)
    if not pred_files:
        print(f"No prediction files found under {root}", file=sys.stderr)
        return {}

    coeff_filter = ""
    if coeff_min is not None or coeff_max is not None:
        coeff_filter = f" (coeff {coeff_min}–{coeff_max})"
    print(f"Found {len(pred_files)} prediction files{coeff_filter}")

    all_summaries: dict[str, dict] = {}
    for fpath in pred_files:
        rel = str(fpath.relative_to(root))
        summary = evaluate_file(
            fpath, cfg, limit=limit, offset=offset,
            workers=workers, force=force, dry_run=dry_run,
        )
        if summary:
            all_summaries[rel] = summary

    if all_summaries and not dry_run:
        master = build_master_summary(all_summaries, root)
        agg_path = summary_out or (root / "aggregated_llm_judge_summary.json")
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)
        csv_path = agg_path.with_suffix(".csv")
        write_summary_csv(master["flat_rows"], csv_path)
        print(f"\nAggregated summary -> {agg_path}")
        print(f"CSV summary         -> {csv_path}")
    return all_summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> JudgeConfig:
    return JudgeConfig(
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY,
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        max_ref_chars=args.max_ref_chars,
        max_pred_chars=args.max_pred_chars,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation for personalized generation (G-Eval style)",
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Predictions .json/.jsonl file or directory to evaluate",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("PATH_A", "PATH_B"),
        help="Pairwise comparison mode: two prediction files",
    )
    parser.add_argument("--label_a", type=str, default="A")
    parser.add_argument("--label_b", type=str, default="B")
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Judge model (default: gpt-4o-mini)",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="OpenAI-compatible base URL (e.g. https://api.deepseek.com)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument(
        "--max_ref_chars", type=int, default=2000,
        help="Truncate reference text to this length",
    )
    parser.add_argument(
        "--max_pred_chars", type=int, default=1500,
        help="Truncate prediction text to this length",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per file (for testing / cost control)",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-judge even if cached")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only print sample stats, no API calls",
    )
    parser.add_argument(
        "--coeff_min", type=float, default=None,
        help="Only evaluate coeff_* dirs >= this value (e.g. 0.2)",
    )
    parser.add_argument(
        "--coeff_max", type=float, default=None,
        help="Only evaluate coeff_* dirs <= this value (e.g. 3.0)",
    )
    parser.add_argument(
        "--summary_out", type=str, default=None,
        help="Path for aggregated JSON summary (default: <path>/aggregated_...)",
    )
    args = parser.parse_args()
    cfg = build_config(args)

    if not cfg.api_key and not args.dry_run:
        print(
            "ERROR: Set OPENAI_API_KEY or pass --api_key\n"
            "Example:\n"
            "  set OPENAI_API_KEY=sk-...\n"
            "  set OPENAI_BASE_URL=https://api.deepseek.com\n"
            "  python llm_judge_eval.py --path ... --limit 5",
            file=sys.stderr,
        )
        sys.exit(1)

    target = Path(args.path)
    if args.compare:
        compare_files(
            Path(args.compare[0]), Path(args.compare[1]),
            cfg, limit=args.limit, workers=args.workers,
            label_a=args.label_a, label_b=args.label_b,
        )
        return

    if target.is_file():
        evaluate_file(
            target, cfg, limit=args.limit, offset=args.offset,
            workers=args.workers, force=args.force, dry_run=args.dry_run,
        )
    elif target.is_dir():
        summary_out = Path(args.summary_out) if args.summary_out else None
        if summary_out and not summary_out.is_absolute():
            summary_out = Path.cwd() / summary_out
        evaluate_directory(
            target, cfg, limit=args.limit, offset=args.offset,
            workers=args.workers, force=args.force, dry_run=args.dry_run,
            coeff_min=args.coeff_min, coeff_max=args.coeff_max,
            summary_out=summary_out,
        )
    else:
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
