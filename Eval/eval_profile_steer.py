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
"""
LaMP / LongLaMP evaluation for LlamaFactory persona_profile_steering checkpoints.

Metrics match eval_text_metrics (per-sample rouge-score, NLTK METEOR, sacrebleu
sentence BLEU / 100, arithmetic mean). Generation uses dual-stream steering
(aligned with SteeringDataset).

Optional: rank profile items by BM25 over the current input (--profile_bm25,
--profile_bm25_k K).

Dependencies:
  pip install -r requirements.txt && pip install -r requirements-eval.txt
  Automatic metrics: per-row output in questions_file (eval_text_metrics).
  LongLaMP-Benchmark/eval/evaluation.py: only when using --golds_file without
  per-row output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if "DISABLE_VERSION_CHECK" not in os.environ:
    os.environ["DISABLE_VERSION_CHECK"] = "1"


def find_and_add_project_root(start_dir: str) -> Optional[str]:
    d = start_dir
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "PersonaSteer")) or os.path.isdir(os.path.join(d, "LlamaFactory")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
        d = os.path.dirname(d)
    return None


project_root = find_and_add_project_root(current_dir)


def find_llamafactory_src() -> Optional[str]:
    roots = [project_root] if project_root else []
    roots += [
        os.path.abspath(os.path.join(current_dir, "..", "..", "..")),
        os.path.abspath(os.path.join(current_dir, "..", "..")),
    ]
    for r in roots:
        if not r:
            continue
        cand = os.path.join(r, "LlamaFactory", "src")
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "llamafactory")):
            return cand
    return None


UserEncoderConfig: Any = None
UserEncoderModel: Any = None
BM25Okapi: Any = None
compute_per_sample_metrics: Any = None
evaluate_data: Any = None
_DEPS_READY = False
_LAMP_EVAL_CLS: Any = None


def find_longlamp_evaluation_py() -> Optional[str]:
    """Locate evaluation.py from a cloned LongLaMP-Benchmark (or LaMP) repo."""
    candidates: list[str] = []
    env_root = os.environ.get("LONGLAMP_BENCHMARK_ROOT") or os.environ.get("LONGLAMP_ROOT")
    if env_root:
        candidates.append(os.path.join(env_root, "eval", "evaluation.py"))
    roots = []
    if project_root:
        roots.append(project_root)
    roots.append(os.path.abspath(os.path.join(current_dir, "..")))
    roots.append(os.path.abspath(os.path.join(current_dir, "../..")))
    for root in roots:
        candidates.extend([
            os.path.join(root, "LongLaMP-Benchmark", "eval", "evaluation.py"),
            os.path.join(root, "LongLaMP", "eval", "evaluation.py"),
            os.path.join(root, "LaMP", "eval", "evaluation.py"),
        ])
    return next((p for p in candidates if os.path.isfile(p)), None)


def get_lamp_evaluation_class() -> Any:
    """
    LaMPEvaluation lives in LongLaMP-Benchmark (not in PersonaSteer).
    Only needed when --golds_file is used without per-row output in questions.
    """
    global _LAMP_EVAL_CLS
    if _LAMP_EVAL_CLS is not None:
        return _LAMP_EVAL_CLS

    evaluation_path = find_longlamp_evaluation_py()
    if evaluation_path is None:
        raise FileNotFoundError(
            "Could not find LongLaMP-Benchmark eval/evaluation.py. "
            "Clone https://github.com/LaMP-Benchmark/LongLaMP next to PersonaSteer, "
            "or set LONGLAMP_BENCHMARK_ROOT to the repo root. "
            "If questions_file already has per-row output, metrics use eval_text_metrics "
            "and this file is not required."
        )
    spec = importlib.util.spec_from_file_location("longlamp_evaluation_module", evaluation_path)
    evaluation_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(evaluation_module)
    _LAMP_EVAL_CLS = evaluation_module.LaMPEvaluation
    return _LAMP_EVAL_CLS


def bootstrap_eval_deps(verbose: bool = True) -> None:
    """Load LlamaFactory + eval_text_metrics (pip install -r requirements*.txt)."""
    global UserEncoderConfig, UserEncoderModel, BM25Okapi
    global compute_per_sample_metrics, evaluate_data, _DEPS_READY
    if _DEPS_READY:
        return

    lf_src = find_llamafactory_src()
    if lf_src and lf_src not in sys.path:
        sys.path.insert(0, lf_src)
        if verbose:
            print(f"Added LlamaFactory src: {lf_src}")

    from llamafactory.train.persona_profile_steering.model import (  # noqa: WPS433
        UserEncoderConfig as _UserEncoderConfig,
        UserEncoderModel as _UserEncoderModel,
    )

    UserEncoderConfig = _UserEncoderConfig
    UserEncoderModel = _UserEncoderModel

    try:
        from rank_bm25 import BM25Okapi as _BM25Okapi  # noqa: WPS433
    except ImportError:
        _BM25Okapi = None
    BM25Okapi = _BM25Okapi

    from eval_text_metrics import (  # noqa: WPS433
        compute_per_sample_metrics as _compute_per_sample_metrics,
        evaluate_data as _evaluate_data,
    )

    compute_per_sample_metrics = _compute_per_sample_metrics
    evaluate_data = _evaluate_data
    _DEPS_READY = True


def get_demographic_information(sample: Dict[str, Any], profile: List[Dict[str, Any]], task_name: str) -> str:
    if "Demographic Information" in sample:
        demo_info = sample["Demographic Information"]
        if demo_info and len(str(demo_info).strip()) >= 5:
            return str(demo_info).strip()
    longlamp_configs = {
        "generation_abstract": "Researcher ID:",
        "topic_writing": "Writer ID:",
        "product_review_writing": "Reviewer ID:",
        "email_generation": "User ID:",
    }
    if task_name in longlamp_configs:
        return f"{longlamp_configs[task_name]} {sample.get('id', 'unknown')}"
    if not profile:
        return "A general user."
    if task_name in ["LaMP_1", "LaMP_2", "LaMP_3"]:
        titles = [item.get("title", "") for item in profile[:5] if isinstance(item, dict) and item.get("title")]
        if titles:
            return (
                "This person likes research in the fields covered by their publications.\n"
                "This person focuses on academic work.\n"
                "A researcher who has published works in relevant academic domains."
            )
        return "A general user."
    texts = []
    for item in profile[:5]:
        if isinstance(item, dict):
            t = item.get("text", "") or item.get("content", "") or str(item)
            if t:
                texts.append(t[:100])
    if texts:
        return (
            "This person has interests related to their historical data.\n"
            "A user with relevant historical information."
        )
    return "A general user."


def create_demographic_prompt(inp: str, demographic_info: str, task: str) -> str:
    task_prefixes = {
        "LaMP-1": "Researcher Profile", "LaMP-2": "User Profile", "LaMP-3": "Reviewer Profile",
        "LaMP-4": "Writer Profile", "LaMP-5": "Researcher Profile", "LaMP-6": "User Profile", "LaMP-7": "User Profile",
        "generation_abstract": "Researcher Profile", "topic_writing": "Writer Profile",
        "product_review_writing": "Reviewer Profile", "email_generation": "User Profile",
    }
    normalized = task.replace("_", "-") if task.startswith("LaMP") else task
    prefix = task_prefixes.get(normalized, "User Profile")
    demo_clean = " ".join(str(demographic_info).split()) if isinstance(demographic_info, str) else str(demographic_info).strip()
    return f"{prefix}: {demo_clean}\n\nTask: {inp}"


def remove_repetitions(text: str, min_repeat_length: int = 20) -> str:
    if not text or len(text) < 2 * min_repeat_length:
        return text
    tl, n = text.lower(), len(text)
    mlen = None
    for rl in range(min_repeat_length, n // 2):
        if all(tl[n - rl - i - 1] == tl[n - i - 1] for i in range(rl)):
            mlen = rl
    if mlen:
        ct, ctl = text, tl
        while ctl.endswith(tl[-mlen:]) and len(ct) > mlen:
            ct, ctl = ct[:-mlen], ctl[:-mlen]
        return ct.strip()
    return text.strip()


def postprocess_generated_text(generated_text: str, task: str) -> str:
    if not generated_text:
        return generated_text
    generated_text = remove_repetitions(generated_text)
    task_prefixes = {
        "generation_abstract": ["abstract:", "abstract", "summary:", "summary", "the abstract:", "the abstract is:", "here is the abstract:", "generated abstract:"],
        "topic_writing": ["content:", "content", "text:", "text", "writing:", "writing", "the content:", "here is the content:", "generated content:"],
        "product_review_writing": ["review:", "review", "review text:", "review text", "the review:", "here is the review:", "generated review:", "product review:"],
        "email_generation": ["email:", "email", "email text:", "email text", "email body:", "email body", "the email:", "here is the email:", "generated email:"],
    }
    gl = generated_text.lower()
    for prefix in task_prefixes.get(task, []):
        if gl.startswith(prefix):
            if ":" in generated_text:
                parts = generated_text.split(":", 1)
                if len(parts) > 1:
                    generated_text, gl = parts[1].strip(), parts[1].strip().lower()
                    break
            elif ". " in generated_text:
                parts = generated_text.split(". ", 1)
                generated_text, gl = parts[1].strip(), parts[1].strip().lower()
                break
            elif len(generated_text) > len(prefix):
                generated_text = generated_text[len(prefix) :].strip()
                gl = generated_text.lower()
                break
    ph = generated_text.replace("_", "").replace("-", "").replace(" ", "").replace(".", "").replace("=", "")
    if not ph or len(ph) < 3:
        generated_text = ""
    else:
        lines = [ln.strip() for ln in generated_text.split("\n") if ln.strip()]
        lines = [ln for ln in lines if len(ln.replace("_", "").replace("-", "").replace(" ", "").replace(".", "").replace("=", "")) >= 3]
        generated_text = "\n".join(lines) if lines else ""
    gl = generated_text.lower()
    for prefix in ["based on", "according to", "considering", "following", "here is", "here's", "the answer is", "the result is", "answer:", "result:", "output:", "sure, here is"]:
        if gl.startswith(prefix):
            if ":" in generated_text:
                parts = generated_text.split(":", 1)
                if len(parts) > 1:
                    generated_text, gl = parts[1].strip(), parts[1].strip().lower()
                    break
            elif ". " in generated_text:
                parts = generated_text.split(". ", 1)
                generated_text, gl = parts[1].strip(), parts[1].strip().lower()
                break
            elif "," in generated_text:
                parts = generated_text.split(",", 1)
                generated_text, gl = parts[1].strip(), parts[1].strip().lower()
                break
    if "\n\n" in generated_text:
        paras = [p.strip() for p in generated_text.split("\n\n") if p.strip()]
        if paras:
            skip_phrases = ("the length of", "should be between", "meets the given requirements", "here is a", "here is the", "following is")
            chosen = None
            for para in paras:
                pl = para.lower()
                if any(s in pl for s in skip_phrases):
                    continue
                if len(para) > 20:
                    chosen = para
                    break
            generated_text = chosen or paras[0]
    gl = generated_text.lower()
    for ending in ["i hope this helps", "let me know if", "please note that", "note:", "note that"]:
        if ending in gl:
            generated_text = generated_text[: gl.find(ending)].strip()
            break
    if generated_text.startswith('"') and generated_text.endswith('"'):
        generated_text = generated_text[1:-1].strip()
    elif generated_text.startswith("'") and generated_text.endswith("'"):
        generated_text = generated_text[1:-1].strip()
    return generated_text.strip()


def build_per_user_metrics(
    preds: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in preds:
        sample_id = str(p.get("id", ""))
        pred_text = p.get("generated_text", "") or ""
        gold_text = p.get("output", "") or ""
        metrics = compute_per_sample_metrics(
            pred_text,
            gold_text,
            compute_bleu=True,
            compute_meteor=True,
            prefer_fast_rouge=True,
        )
        rows.append(
            {
                "id": sample_id,
                "generated_text": pred_text,
                "output": gold_text,
                "metric_type": "multi_metrics",
                **metrics,
                "has_gold": bool(str(gold_text).strip()),
            }
        )
    return rows


def map_eval_task_to_profile_key(task_name: str) -> str:
    t = task_name.replace("-", "_")
    if t in ("LaMP_1", "LaMP_2", "LaMP_3"):
        return "generation_abstract"
    if t == "LaMP_4":
        return "lamp_4"
    if t == "LaMP_5":
        return "lamp_5"
    if t == "LaMP_6":
        return "topic_writing"
    if t == "LaMP_7":
        return "lamp_7"
    if task_name in ("generation_abstract", "topic_writing", "product_review_writing"):
        return task_name
    if task_name == "email_generation":
        return "topic_writing"
    return "topic_writing"


def profile_item_to_text(item: Dict[str, Any], task: str) -> str:
    if not isinstance(item, dict):
        return ""
    if task == "generation_abstract":
        title, ab = item.get("title", "").strip(), item.get("abstract", item.get("text", "")).strip()
        return (f"{title} {ab}")[:800]
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


def rank_profile_by_bm25(
    query_text: str,
    profile: List[Dict[str, Any]],
    task_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Rank profile items by BM25; return top_k (skip items with empty profile_item_to_text).
    Matches LaMP prompts: BM25Okapi + query.split() / corpus tokenization.
    """
    if BM25Okapi is None:
        raise ImportError("Profile BM25 requires package: pip install rank-bm25")
    if not isinstance(profile, list) or not profile or top_k <= 0:
        return []
    profs_nonempty: List[Dict[str, Any]] = []
    tokenized_corpus: List[List[str]] = []
    for it in profile:
        t = profile_item_to_text(it, task_key)
        if not (t or "").strip():
            continue
        profs_nonempty.append(it)
        tokenized_corpus.append(t.lower().split())
    if not tokenized_corpus:
        return []
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = (query_text or "").lower().split()
    n = min(top_k, len(profs_nonempty))
    return bm25.get_top_n(tokenized_query, profs_nonempty, n=n)


def select_profile_items_for_steering(
    query_text: str,
    profile: List[Dict[str, Any]],
    task_key: str,
    max_profile_slots: int,
    use_bm25: bool,
    bm25_top_k: int,
) -> List[Dict[str, Any]]:
    """Profile subset for steering, capped at max_profile_slots."""
    if max_profile_slots <= 0:
        return []
    if not isinstance(profile, list):
        return []
    if use_bm25:
        k = min(bm25_top_k, max_profile_slots)
        return rank_profile_by_bm25(query_text, profile, task_key, k)
    return profile[:max_profile_slots]


def clean_demographic_string(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"<\|endoftext\|>+", "", s)
    s = re.sub(r"<\|eot_id\|>+", "", s)
    return " ".join(s.split()).strip()


def infer_decoder_hidden_size(decoder_name: str, checkpoint: Optional[Dict[str, Any]], default_value: Optional[int]) -> int:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(decoder_name, local_files_only=True)
        h = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if h is not None:
            print(f"decoder_hidden_size from config: {h}")
            return int(h)
    except Exception as e:
        print(f"decoder config: {e}")
    if checkpoint and "steering_generator" in checkpoint:
        st = checkpoint["steering_generator"]
        for key in ("fusion_norm.weight", "user_projector.0.weight", "layer_norm.weight"):
            if key in st:
                d = int(st[key].shape[0])
                print(f"decoder_hidden_size from {key}: {d}")
                return d
    if default_value is not None:
        return int(default_value)
    raise ValueError("Cannot infer decoder_hidden_size; pass --decoder_hidden_size")


def compute_steering_vector(
    model: UserEncoderModel,
    demographic_text: str,
    query_text: str,
    profile: List[Dict[str, Any]],
    task_key: str,
    device: torch.device,
    max_profile_items: int,
    max_profile_item_len: int,
    max_query_length: int,
    max_user_length: int,
    use_profile_bm25: bool = False,
    profile_bm25_top_k: int = 8,
) -> torch.Tensor:
    ut = model.user_tokenizer
    demo = ut(clean_demographic_string(demographic_text), max_length=max_user_length, padding=False, truncation=True, return_tensors="pt")
    demo = {k: v.to(device) for k, v in demo.items()}
    dem_seq = model.user_encoder(
        input_ids=demo["input_ids"],
        attention_mask=demo.get("attention_mask"),
        token_type_ids=demo.get("token_type_ids"),
        return_sequence=True,
    )
    dem_mask = (demo["attention_mask"] > 0).long()
    q = ut(query_text or "", max_length=max_query_length, padding=False, truncation=True, return_tensors="pt")
    q = {k: v.to(device) for k, v in q.items()}
    q_rep = model.user_encoder(
        input_ids=q["input_ids"],
        attention_mask=q.get("attention_mask"),
        token_type_ids=q.get("token_type_ids"),
        return_sequence=False,
    )
    prof_items = select_profile_items_for_steering(
        query_text, profile, task_key, max_profile_items, use_profile_bm25, profile_bm25_top_k
    )
    texts = [t for t in (profile_item_to_text(it, task_key) for it in prof_items) if t]
    enc_ids, enc_m = [], []
    
    # Fix: fill with zero tensors instead of encoding empty strings to avoid [CLS][SEP] noise
    user_pad_id = getattr(ut, "pad_token_id", None) or 0
    for i in range(max_profile_items):
        seg = texts[i] if i < len(texts) else ""
        if seg.strip():
            e = ut(seg, max_length=max_profile_item_len, padding="max_length", truncation=True, return_tensors="pt")
            enc_ids.append(e["input_ids"].squeeze(0))
            enc_m.append(e["attention_mask"].squeeze(0))
        else:
            zeros_ids = torch.zeros(max_profile_item_len, dtype=torch.long)
            zeros_mask = torch.zeros(max_profile_item_len, dtype=torch.long)
            if user_pad_id is not None:
                zeros_ids.fill_(user_pad_id)
            enc_ids.append(zeros_ids)
            enc_m.append(zeros_mask)
            
    pid = torch.stack(enc_ids).unsqueeze(0).to(device)
    pm = torch.stack(enc_m).unsqueeze(0).to(device)
    
    if pm.sum() == 0:
        pm[:, 0, 0] = 1
    if dem_mask.sum() == 0:
        dem_mask[:, 0] = 1
        
    B, N, L = pid.shape
    flat = model.user_encoder(input_ids=pid.view(B * N, L), attention_mask=pm.view(B * N, L), return_sequence=False)
    prof_rep = flat.view(B, N, -1)
    prof_mask_1d = (pm.sum(dim=-1) > 0).long()
    return model.steering_generator(
        demographic_sequence=dem_seq,
        demographic_mask=dem_mask,
        query_representation=q_rep,
        context_hidden_states=None,
        profile_representations=prof_rep,
        profile_mask=prof_mask_1d,
    )


def generate_with_profile_steering_model(
    model: UserEncoderModel,
    input_text: str,
    profile: List[Dict[str, Any]],
    task_name: str,
    device: torch.device,
    sample: Optional[Dict[str, Any]] = None,
    mode: str = "steer",
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_profile_items: int = 8,
    max_profile_item_len: int = 128,
    max_query_length: int = 256,
    max_user_length: int = 512,
    use_profile_bm25: bool = False,
    profile_bm25_top_k: int = 8,
    decoder_prompt: Optional[str] = None,
    steering_query_text: Optional[str] = None,
) -> str:
    if mode not in ("steer", "no_steer"):
        raise ValueError(f"Unsupported mode: {mode}. Use 'steer' or 'no_steer'.")
    tok = model.decoder_tokenizer
    if tok is None:
        raise RuntimeError("decoder_tokenizer is None")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    task_key = map_eval_task_to_profile_key(task_name)
    sample = sample or {}
    query_for_steering = steering_query_text if steering_query_text is not None else input_text
    
    # Use task-specific system prompt and chat template if supported
    try:
        from llamafactory.train.persona_profile_steering.dataset import SteeringDataset
        task_config = SteeringDataset.TASK_CONFIGS.get(task_key, {})
        system_content = task_config.get("default_system_prompt", "You are a helpful assistant.")
    except Exception:
        system_content = "You are a helpful assistant."

    if decoder_prompt is not None:
        system_content = decoder_prompt

    prompt_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": input_text}
    ]
    
    if hasattr(tok, 'apply_chat_template') and tok.chat_template is not None:
        prompt = tok.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"System: {system_content}\n\nUser: {input_text}\n\nAssistant: "

    inputs = tok(prompt, return_tensors="pt", padding=False, truncation=True, max_length=4096).to(device)
    steering_on = False
    # Avoid reading stale alpha from previous sample when steering is disabled.
    if hasattr(model, "steering_generator"):
        model.steering_generator.last_alpha = None
    try:
        with torch.no_grad():
            if mode == "steer":
                demo = get_demographic_information(sample, profile, task_name)
                if demo and len(demo.strip()) >= 5:
                    sv = compute_steering_vector(
                        model,
                        demo,
                        query_for_steering,
                        profile,
                        task_key,
                        device,
                        max_profile_items,
                        max_profile_item_len,
                        max_query_length,
                        max_user_length,
                        use_profile_bm25=use_profile_bm25,
                        profile_bm25_top_k=profile_bm25_top_k,
                    )
                    vec = sv.squeeze(1).to(next(model.decoder.parameters()).dtype)
                    model.remove_steering_hook()
                    model.steering_hook.set_steering(user_vector=vec, steering_generator=model.steering_generator)
                    model.register_steering_hook()
                    steering_on = True
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            out = model.decoder.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
                eos_token_id=tok.eos_token_id,
            )
    finally:
        if steering_on:
            model.remove_steering_hook()
    il = inputs["input_ids"].shape[1]
    text = tok.decode(out[0][il:], skip_special_tokens=True).strip()
    return postprocess_generated_text(text, task_name)


def load_lamp_data(questions_file: str, demographic_file: Optional[str] = None) -> List[Dict[str, Any]]:
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if demographic_file and os.path.exists(demographic_file):
        dm: Dict[Any, str] = {}
        try:
            with open(demographic_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    iid = item.get("id")
                    if iid is not None:
                        dm[iid] = item.get("Demographic Information", "")
            for it in data:
                iid = it.get("id")
                if iid in dm:
                    it["Demographic Information"] = dm[iid]
            print(f"Merged demographic: {len(dm)}")
        except Exception as e:
            print(f"demographic file: {e}")
    return data


def process_samples_worker(
    gpu_id: int,
    samples: List[Dict[str, Any]],
    task_name: str,
    mode: str,
    checkpoint_path: str,
    decoder_name: str,
    user_encoder_name: str,
    user_encoder_use_lora: bool,
    user_encoder_lora_r: int,
    user_encoder_lora_alpha: int,
    user_encoder_lora_dropout: float,
    user_encoder_lora_target_modules: List[str],
    steering_layer_idx: int,
    steering_coeff: float,
    decoder_hidden_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    demographic_map: Dict[Any, str],
    fusion_alpha_min: float,
    fusion_alpha_max: float,
    ablation_mode: str,
    max_profile_items: int,
    max_profile_item_len: int,
    max_query_length: int,
    max_user_length: int,
    use_profile_bm25: bool,
    profile_bm25_top_k: int,
    log_fusion_alpha: bool,
    fusion_alpha_log_every: int,
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{gpu_id}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = UserEncoderConfig(
        user_encoder_name=user_encoder_name,
        user_encoder_use_lora=user_encoder_use_lora,
        user_encoder_lora_r=user_encoder_lora_r,
        user_encoder_lora_alpha=user_encoder_lora_alpha,
        user_encoder_lora_dropout=user_encoder_lora_dropout,
        user_encoder_lora_target_modules=user_encoder_lora_target_modules,
        decoder_name=decoder_name,
        decoder_hidden_size=decoder_hidden_size,
        steering_layer_idx=steering_layer_idx,
        steering_coeff=steering_coeff,
        fusion_alpha_min=fusion_alpha_min,
        fusion_alpha_max=fusion_alpha_max,
        ablation_mode=ablation_mode,
    )
    model = UserEncoderModel(config=cfg).to(device)
    if "steering_generator" in ckpt:
        model.steering_generator.load_state_dict(ckpt["steering_generator"], strict=False)
    if ckpt.get("user_encoder"):
        model.user_encoder.load_state_dict(ckpt["user_encoder"], strict=False)
    model.eval()
    if model.decoder_tokenizer and model.decoder_tokenizer.pad_token is None:
        model.decoder_tokenizer.pad_token = model.decoder_tokenizer.eos_token
        model.decoder_tokenizer.pad_token_id = model.decoder_tokenizer.eos_token_id
    preds: List[Dict[str, Any]] = []
    alpha_values: List[float] = []
    logged = 0
    for s in samples:
        inp, prof = s.get("input", ""), s.get("profile", [])
        sid, gold = s.get("id", ""), s.get("output", "")
        if sid in demographic_map:
            s["Demographic Information"] = demographic_map[sid]
        if not inp:
            preds.append({"id": sid, "generated_text": "", "output": gold, "alpha": None})
            continue
        try:
            out = generate_with_profile_steering_model(
                model,
                inp,
                prof,
                task_name,
                device,
                s,
                mode,
                max_new_tokens,
                temperature,
                top_p,
                max_profile_items,
                max_profile_item_len,
                max_query_length,
                max_user_length,
                use_profile_bm25=use_profile_bm25,
                profile_bm25_top_k=profile_bm25_top_k,
            )
            alpha = getattr(model.steering_generator, "last_alpha", None)
            alpha_value = None
            if alpha is not None:
                try:
                    alpha_value = float(alpha)
                    alpha_values.append(alpha_value)
                    logged += 1
                    if log_fusion_alpha and fusion_alpha_log_every > 0 and (logged % fusion_alpha_log_every == 0):
                        print(
                            f"[GPU{gpu_id}] fusion alpha progress: n={logged}, "
                            f"last={alpha_values[-1]:.4f}, mean={sum(alpha_values) / len(alpha_values):.4f}"
                        )
                except Exception:
                    pass
            preds.append({"id": sid, "generated_text": out, "output": gold, "alpha": alpha_value})
        except Exception as e:
            print(f"[GPU{gpu_id}] {sid}: {e}")
            preds.append({"id": sid, "generated_text": "", "output": gold, "alpha": None})
    return {"preds": preds, "alpha_values": alpha_values}


def evaluate_lamp_task(
    model: Optional[UserEncoderModel],
    questions_file: str,
    golds_file: Optional[str],
    task_name: str,
    device: torch.device,
    output_dir: str,
    modes: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    demographic_file: Optional[str],
    num_gpus: int,
    gpu_ids: Optional[List[int]],
    checkpoint_path: Optional[str],
    decoder_name: Optional[str],
    user_encoder_name: Optional[str],
    user_encoder_use_lora: bool,
    user_encoder_lora_r: int,
    user_encoder_lora_alpha: int,
    user_encoder_lora_dropout: float,
    user_encoder_lora_target_modules: Optional[List[str]],
    steering_layer_idx: int,
    steering_coeff: float,
    decoder_hidden_size: int,
    fusion_alpha_min: float,
    fusion_alpha_max: float,
    ablation_mode: str,
    max_profile_items: int,
    max_profile_item_len: int,
    max_query_length: int,
    max_user_length: int,
    use_profile_bm25: bool = False,
    profile_bm25_top_k: int = 8,
    log_fusion_alpha: bool = False,
    fusion_alpha_log_every: int = 50,
    save_per_user_metrics: bool = True,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}\nEvaluating {task_name}\n{'=' * 60}")
    if use_profile_bm25:
        print(f"Profile selection: BM25 top-{profile_bm25_top_k} (max slots {max_profile_items})")
    questions = load_lamp_data(questions_file, demographic_file)
    seen = set()
    for idx, q in enumerate(questions):
        oid = q.get("id", "")
        if not oid or (isinstance(oid, str) and not str(oid).strip()):
            q["id"] = f"sample_{idx}"
        elif str(oid) in seen:
            q["id"] = f"sample_{idx}_orig_{oid}"
        else:
            q["id"] = str(oid)
        seen.add(q["id"])
    if golds_file and os.path.exists(golds_file):
        with open(golds_file, "r", encoding="utf-8") as f:
            gd = json.load(f)
        items = gd.get("golds", gd if isinstance(gd, list) else [])
        gm = {str(x.get("id")): x.get("output", "") for x in items if x.get("id") is not None}
        for q in questions:
            qid = str(q.get("id", ""))
            if qid in gm:
                q["output"] = gm[qid]
    dmap: Dict[Any, str] = {}
    if demographic_file and os.path.exists(demographic_file):
        try:
            with open(demographic_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    iid = item.get("id")
                    if iid is not None:
                        dmap[iid] = item.get("Demographic Information", "")
        except Exception as e:
            print(f"dmap: {e}")
    avail = torch.cuda.device_count()
    use_multi = num_gpus > 1 and avail > 1
    if use_multi:
        gids = list(range(min(num_gpus, avail))) if gpu_ids is None else [g for g in gpu_ids if g < avail]
        if not gids:
            gids, use_multi = [0], False
        print(f"Multi-GPU: {gids}")
    else:
        gids = [0]
        if num_gpus > 1:
            print(f"Single GPU (requested {num_gpus}, have {avail})")
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    tmods = user_encoder_lora_target_modules or []
    all_results: Dict[str, Any] = {}
    for mode in modes:
        print(f"\n--- {mode} ---")
        preds: List[Dict[str, Any]] = []
        alpha_values: List[float] = []
        alpha_logged = 0
        if use_multi and checkpoint_path and decoder_name:
            cs = (len(questions) + len(gids) - 1) // len(gids)
            chunks = [questions[i : i + cs] for i in range(0, len(questions), cs)][: len(gids)]
            with ProcessPoolExecutor(max_workers=len(gids)) as ex:
                futs = [
                    ex.submit(
                        process_samples_worker,
                        gid,
                        ch,
                        task_name,
                        mode,
                        checkpoint_path,
                        decoder_name,
                        user_encoder_name or "bert-base-uncased",
                        user_encoder_use_lora,
                        user_encoder_lora_r,
                        user_encoder_lora_alpha,
                        user_encoder_lora_dropout,
                        tmods,
                        steering_layer_idx,
                        steering_coeff,
                        decoder_hidden_size,
                        max_new_tokens,
                        temperature,
                        top_p,
                        dmap,
                        fusion_alpha_min,
                        fusion_alpha_max,
                        ablation_mode,
                        max_profile_items,
                        max_profile_item_len,
                        max_query_length,
                        max_user_length,
                        use_profile_bm25,
                        profile_bm25_top_k,
                        log_fusion_alpha,
                        fusion_alpha_log_every,
                    )
                    for gid, ch in zip(gids, chunks)
                ]
                parts = [f.result() for f in tqdm(futs, desc="workers")]
            by_id: Dict[str, Dict[str, Any]] = {}
            for part in parts:
                alpha_values.extend(part.get("alpha_values", []))
                for p in part.get("preds", []):
                    by_id[str(p["id"])] = p
            for q in questions:
                qid = str(q["id"])
                preds.append(
                    by_id.get(
                        qid,
                        {"id": qid, "generated_text": "", "output": q.get("output", ""), "alpha": None},
                    )
                )
        else:
            if model is None:
                raise ValueError("model required for single-GPU")
            for s in tqdm(questions, desc=mode):
                inp, prof = s.get("input", ""), s.get("profile", [])
                sid, gold = s.get("id", ""), s.get("output", "")
                if sid in dmap:
                    s["Demographic Information"] = dmap[sid]
                if not inp:
                    preds.append({"id": sid, "generated_text": "", "output": gold, "alpha": None})
                    continue
                try:
                    out = generate_with_profile_steering_model(
                        model,
                        inp,
                        prof,
                        task_name,
                        device,
                        s,
                        mode,
                        max_new_tokens,
                        temperature,
                        top_p,
                        max_profile_items,
                        max_profile_item_len,
                        max_query_length,
                        max_user_length,
                        use_profile_bm25=use_profile_bm25,
                        profile_bm25_top_k=profile_bm25_top_k,
                    )
                    alpha = getattr(model.steering_generator, "last_alpha", None)
                    alpha_value = None
                    if alpha is not None:
                        try:
                            alpha_value = float(alpha)
                            alpha_values.append(alpha_value)
                            alpha_logged += 1
                            if log_fusion_alpha and fusion_alpha_log_every > 0 and (alpha_logged % fusion_alpha_log_every == 0):
                                print(
                                    f"[fusion alpha] mode={mode} n={alpha_logged}, "
                                    f"last={alpha_values[-1]:.4f}, mean={sum(alpha_values) / len(alpha_values):.4f}"
                                )
                        except Exception:
                            pass
                    preds.append({"id": sid, "generated_text": out, "output": gold, "alpha": alpha_value})
                except Exception as e:
                    print(f"{sid}: {e}")
                    preds.append({"id": sid, "generated_text": "", "output": gold, "alpha": None})
        cf = f"{steering_coeff:.1f}" if abs(steering_coeff % 1) < 1e-10 else str(steering_coeff)
        pfn = os.path.join(output_dir, f"{task_name}_{mode}_layer{steering_layer_idx}_coeff{cf}_predictions.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(pfn, "w", encoding="utf-8") as f:
            json.dump({"task": task_name, "predictions": preds, "steering_layer": steering_layer_idx, "steering_coeff": steering_coeff}, f, indent=2, ensure_ascii=False)
        print(f"Saved {pfn}")

        if save_per_user_metrics:
            per_user_rows = build_per_user_metrics(preds)
            upfn = os.path.join(
                output_dir,
                f"{task_name}_{mode}_layer{steering_layer_idx}_coeff{cf}_per_user_metrics.json",
            )
            with open(upfn, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": task_name,
                        "mode": mode,
                        "metric_type": "multi_metrics",
                        "rows": per_user_rows,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"Saved per-user metrics: {upfn}")

        ed = [{"generated_text": p.get("generated_text", ""), "output": p["output"]} for p in preds if p.get("output")]
        if ed:
            all_results[mode] = evaluate_data(ed, f"{task_name}_{mode}")
            print(all_results[mode])
        elif golds_file and os.path.exists(golds_file):
            tmp = pfn.replace(".json", "_for_lamp.json")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"task": task_name, "golds": [{"id": p["id"], "output": p.get("generated_text", "")} for p in preds]}, f, indent=2, ensure_ascii=False)
            LampEval = get_lamp_evaluation_class()
            all_results[mode] = LampEval(single_gold_json_file_addr=golds_file).evaluate_task(tmp, task_name)
            if os.path.exists(tmp):
                os.remove(tmp)
            print(all_results[mode])
        else:
            all_results[mode] = {"skipped": True}
        if alpha_values:
            alpha_stats = {
                "count": len(alpha_values),
                "mean": sum(alpha_values) / len(alpha_values),
                "min": min(alpha_values),
                "max": max(alpha_values),
            }
            print(
                f"[fusion alpha] mode={mode} count={alpha_stats['count']} "
                f"mean={alpha_stats['mean']:.4f} min={alpha_stats['min']:.4f} max={alpha_stats['max']:.4f}"
            )
            if isinstance(all_results.get(mode), dict):
                all_results[mode]["fusion_alpha_stats"] = alpha_stats
            else:
                all_results[mode] = {
                    "result": all_results.get(mode),
                    "fusion_alpha_stats": alpha_stats,
                }
    return all_results


def main() -> None:
    p = argparse.ArgumentParser(description="Eval persona_profile_steering on LaMP / LongLaMP")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--decoder_name", required=True)
    p.add_argument("--user_encoder_name", default="bert-base-uncased")
    p.add_argument("--user_encoder_use_lora", action="store_true")
    p.add_argument("--user_encoder_lora_r", type=int, default=8)
    p.add_argument("--user_encoder_lora_alpha", type=int, default=16)
    p.add_argument("--user_encoder_lora_dropout", type=float, default=0.05)
    p.add_argument("--user_encoder_lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--steering_layer_idx", type=int, default=-1)
    p.add_argument("--steering_coeff", type=float, default=1.0)
    p.add_argument("--decoder_hidden_size", type=int, default=None)
    p.add_argument("--fusion_alpha_min", type=float, default=0.0)
    p.add_argument("--fusion_alpha_max", type=float, default=1.0)
    p.add_argument(
        "--ablation_mode",
        type=str,
        default="ab",
        choices=["ab", "a_only", "b_only"],
        help="Explicit steering ablation mode.",
    )
    p.add_argument("--log_fusion_alpha", action="store_true", help="Print fusion alpha progress and summary statistics.")
    p.add_argument("--fusion_alpha_log_every", type=int, default=50, help="Print fusion alpha every N valid steering samples.")
    p.add_argument("--max_profile_items", type=int, default=8)
    p.add_argument("--max_profile_item_len", type=int, default=128)
    p.add_argument("--max_query_length", type=int, default=256)
    p.add_argument("--max_user_length", type=int, default=512)
    p.add_argument(
        "--profile_bm25",
        action="store_true",
        help="Use BM25 over current input to pick the top-k most relevant profile items for steering.",
    )
    p.add_argument(
        "--profile_bm25_k",
        type=int,
        default=8,
        help="Keep top-k profile items after BM25 ranking (capped by --max_profile_items). Ignored without --profile_bm25.",
    )
    p.add_argument("--task_name", required=True, choices=[
        "LaMP_1", "LaMP_2", "LaMP_3", "LaMP_4", "LaMP_5", "LaMP_6", "LaMP_7",
        "generation_abstract", "topic_writing", "product_review_writing", "email_generation",
    ])
    p.add_argument("--questions_file", required=True)
    p.add_argument("--golds_file", default=None)
    p.add_argument("--demographic_file", default=None)
    p.add_argument("--output_dir", default="./lamp_eval_profile_steering_results")
    p.add_argument("--modes", nargs="+", default=["steer"], choices=["steer", "no_steer"])
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--gpu_ids", type=int, nargs="+", default=None)
    p.add_argument(
        "--disable_per_user_metrics",
        action="store_true",
        help="Disable saving per-user metrics file.",
    )
    args = p.parse_args()
    bootstrap_eval_deps()
    if args.profile_bm25 and BM25Okapi is None:
        p.error("BM25 profile selection requires: pip install rank-bm25")
    if args.profile_bm25_k < 1:
        p.error("--profile_bm25_k must be >= 1")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ue = ckpt.get("user_encoder", {})
    if any("lora_" in k.lower() for k in ue.keys()) and not args.user_encoder_use_lora:
        args.user_encoder_use_lora = True
        print("Auto-enabled user_encoder_use_lora")
    dec_h = infer_decoder_hidden_size(args.decoder_name, ckpt, args.decoder_hidden_size)
    lmods = [x.strip() for x in args.user_encoder_lora_target_modules.split(",") if x.strip()]
    cfg = UserEncoderConfig(
        user_encoder_name=args.user_encoder_name,
        user_encoder_use_lora=args.user_encoder_use_lora,
        user_encoder_lora_r=args.user_encoder_lora_r,
        user_encoder_lora_alpha=args.user_encoder_lora_alpha,
        user_encoder_lora_dropout=args.user_encoder_lora_dropout,
        user_encoder_lora_target_modules=lmods,
        decoder_name=args.decoder_name,
        decoder_hidden_size=dec_h,
        steering_layer_idx=args.steering_layer_idx,
        steering_coeff=args.steering_coeff,
        fusion_alpha_min=args.fusion_alpha_min,
        fusion_alpha_max=args.fusion_alpha_max,
        ablation_mode=args.ablation_mode,
    )
    model = UserEncoderModel(config=cfg).to(device)
    if "steering_generator" in ckpt:
        model.steering_generator.load_state_dict(ckpt["steering_generator"], strict=False)
    if ue:
        model.user_encoder.load_state_dict(ue, strict=False)
    model.eval()
    kw = dict(
        questions_file=args.questions_file,
        golds_file=args.golds_file,
        task_name=args.task_name,
        device=device,
        output_dir=args.output_dir,
        modes=args.modes,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        demographic_file=args.demographic_file,
        user_encoder_name=args.user_encoder_name,
        user_encoder_use_lora=args.user_encoder_use_lora,
        user_encoder_lora_r=args.user_encoder_lora_r,
        user_encoder_lora_alpha=args.user_encoder_lora_alpha,
        user_encoder_lora_dropout=args.user_encoder_lora_dropout,
        user_encoder_lora_target_modules=lmods,
        steering_layer_idx=args.steering_layer_idx,
        steering_coeff=args.steering_coeff,
        decoder_hidden_size=dec_h,
        fusion_alpha_min=args.fusion_alpha_min,
        fusion_alpha_max=args.fusion_alpha_max,
        ablation_mode=args.ablation_mode,
        max_profile_items=args.max_profile_items,
        max_profile_item_len=args.max_profile_item_len,
        max_query_length=args.max_query_length,
        max_user_length=args.max_user_length,
        use_profile_bm25=args.profile_bm25,
        profile_bm25_top_k=args.profile_bm25_k,
        log_fusion_alpha=args.log_fusion_alpha,
        fusion_alpha_log_every=args.fusion_alpha_log_every,
        save_per_user_metrics=not args.disable_per_user_metrics,
    )
    if args.num_gpus > 1 and torch.cuda.device_count() > 1:
        model.cpu()
        torch.cuda.empty_cache()
        res = evaluate_lamp_task(None, num_gpus=args.num_gpus, gpu_ids=args.gpu_ids, checkpoint_path=args.checkpoint, decoder_name=args.decoder_name, **kw)
    else:
        res = evaluate_lamp_task(model, num_gpus=1, gpu_ids=None, checkpoint_path=args.checkpoint, decoder_name=args.decoder_name, **kw)
    cf = f"{args.steering_coeff:.1f}" if abs(args.steering_coeff % 1) < 1e-10 else str(args.steering_coeff)
    sf = os.path.join(args.output_dir, f"{args.task_name}_layer{args.steering_layer_idx}_coeff{cf}_summary.json")
    with open(sf, "w", encoding="utf-8") as f:
        json.dump({"task": args.task_name, "modes": args.modes, "results": res}, f, indent=2, ensure_ascii=False)
    print(f"Summary -> {sf}")


if __name__ == "__main__":
    main()