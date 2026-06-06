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
Text metrics shared by aggregate_best_per_user_from_sweep.py and eval scripts.

Scoring matches aggregate_best_per_user_from_sweep: rouge-score by default,
NLTK METEOR when available, sacrebleu sentence BLEU / 100 when compute_bleu=True.

Corpus-level summaries are the arithmetic mean of per-sample scores (not
sacrebleu corpus_bleu), so numbers match per-user JSON / sweep aggregation.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from rouge_score import rouge_scorer  # pyright: ignore[reportMissingImports]
except ImportError:
    rouge_scorer = None  # type: ignore[assignment]


_ROUGE_SCORER_INSTANCE = None
_SLOW_ROUGE_WARNED = False

AGG_METRIC_KEYS = ("bleu", "rouge-1", "rouge-2", "rouge-L", "rouge-LSum", "meteor")


def _compute_meteor_one(pred: str, gold: str) -> float:
    """Compute METEOR with NLTK; return 0.0 if NLTK/resources are unavailable."""
    if not str(gold).strip():
        return 0.0
    try:
        from nltk import word_tokenize  # pyright: ignore[reportMissingImports]
        from nltk.translate.meteor_score import meteor_score  # pyright: ignore[reportMissingImports]

        ref = word_tokenize(str(gold).lower())
        hyp = word_tokenize(str(pred).lower())
        if not ref or not hyp:
            return 0.0
        return float(meteor_score([ref], hyp))
    except Exception:
        return 0.0


def _get_rouge_scorer():
    global _ROUGE_SCORER_INSTANCE
    if rouge_scorer is None:
        return None
    if _ROUGE_SCORER_INSTANCE is None:
        _ROUGE_SCORER_INSTANCE = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )
    return _ROUGE_SCORER_INSTANCE


def compute_per_sample_metrics(
    pred: str,
    gold: str,
    *,
    compute_bleu: bool = False,
    compute_meteor: bool = True,
    prefer_fast_rouge: bool = True,
) -> Dict[str, float]:
    """
    Per-sample metrics (same logic as aggregate_best_per_user_from_sweep._compute_per_sample_metrics).
    """
    global _SLOW_ROUGE_WARNED
    has_gold = bool(str(gold).strip())
    out: Dict[str, float] = {
        "bleu": 0.0,
        "rouge-1": 0.0,
        "rouge-2": 0.0,
        "rouge-L": 0.0,
        "rouge-LSum": 0.0,
    }
    if not has_gold:
        out["meteor"] = 0.0
        return out

    scorer = _get_rouge_scorer()
    if prefer_fast_rouge and scorer is not None:
        s = scorer.score(gold, pred)
        out["rouge-1"] = float(s["rouge1"].fmeasure)
        out["rouge-2"] = float(s["rouge2"].fmeasure)
        out["rouge-L"] = float(s["rougeL"].fmeasure)
        out["rouge-LSum"] = out["rouge-L"]
    else:
        if not _SLOW_ROUGE_WARNED:
            print(
                "Warning: rouge-score not available, ROUGE metrics stay 0.0. "
                "Install rouge-score: pip install rouge-score",
                flush=True,
            )
            _SLOW_ROUGE_WARNED = True

    if compute_bleu:
        bleu_ok = False
        try:
            from sacrebleu.metrics import BLEU  # pyright: ignore[reportMissingImports]

            b = BLEU(effective_order=True)
            out["bleu"] = float(b.sentence_score(pred, [gold]).score) / 100.0
            bleu_ok = True
        except Exception:
            pass
        if not bleu_ok:
            try:
                import sacrebleu  # pyright: ignore[reportMissingImports]

                out["bleu"] = float(sacrebleu.sentence_bleu(pred, [gold]).score) / 100.0
                bleu_ok = True
            except Exception:
                pass
        if not bleu_ok:
            out["bleu"] = 0.0

    if compute_meteor:
        out["meteor"] = _compute_meteor_one(pred, gold) if has_gold else 0.0
    else:
        out["meteor"] = 0.0

    return out


def mean_metrics_over_eval_data(
    data: List[Dict[str, Any]],
    *,
    compute_bleu: bool = True,
    compute_meteor: bool = True,
    prefer_fast_rouge: bool = True,
) -> Dict[str, float]:
    """
    Mean of per-sample metrics over items with non-empty generated_text and output
    (same filter as legacy generation_metrics.evaluate_data).
    """
    keys = AGG_METRIC_KEYS
    sums = {k: 0.0 for k in keys}
    n = 0
    for d in data:
        pred = d.get("generated_text", "") or ""
        gold = d.get("output", "") or ""
        if not pred or not gold:
            continue
        m = compute_per_sample_metrics(
            pred,
            gold,
            compute_bleu=compute_bleu,
            compute_meteor=compute_meteor,
            prefer_fast_rouge=prefer_fast_rouge,
        )
        for k in keys:
            sums[k] += float(m.get(k, 0.0))
        n += 1
    if n == 0:
        return {k: 0.0 for k in keys}
    return {k: sums[k] / n for k in keys}


def evaluate_data(
    data: List[Dict[str, Any]],
    name: str,
    *,
    append_metrics_txt: bool = True,
    compute_bleu: bool = True,
    compute_meteor: bool = True,
    prefer_fast_rouge: bool = True,
) -> Dict[str, float]:
    """
    Drop-in replacement for metrics.generation_metrics.evaluate_data: mean per-sample
    metrics plus optional append to metrics.txt (same filename as legacy).
    """
    result = mean_metrics_over_eval_data(
        data,
        compute_bleu=compute_bleu,
        compute_meteor=compute_meteor,
        prefer_fast_rouge=prefer_fast_rouge,
    )
    if append_metrics_txt and name:
        try:
            with open("metrics.txt", "a", encoding="utf-8") as f:
                f.write(name + ":" + str(result) + "\n")
        except Exception:
            pass
    return result
