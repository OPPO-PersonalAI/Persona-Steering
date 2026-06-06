# PersonaSteering

PersonaSteering is a personalization model for transformer-based text generation that conditions generation on a user's profile and task instruction to produce outputs aligned with individual users.

## Method Overview

<p align="center">
  <img src="assets/main_00.png" alt="PersonaSteer method diagram" width="90%">
</p>

*Method diagram: dual-stream user encoding (demographic information + profile history), adaptive fusion, and dynamic steering injected at a target transformer layer.

## Key Features & Architecture

- **Dual-Stream Steering Vector Generation**: 
  - **Stream A (Trait/Demographic)**: The current query attends to the user's demographic information sequence to extract long-term preferences.
  - **Stream B (Resdual/History)**: The current query attends to the user's historical profile records (using Cross-Attention) to extract task-relevant, short-term preferences.
  - **Adaptive Fusion Gate**: Dynamically weights and combines Stream A and Stream B to form the final user representation.
- **Dynamic Steering via Activation Hooks**: Instead of simple static addition, the fused steering vector acts as a Query that dynamically attends to the frozen Decoder's hidden states (Key/Value) at a specified transformer layer. This produces a context-dependent delta that is injected directly into the residual stream.
- **Two-Stage Training**:
  - **Stage 1**: Freezes the User Encoder and trains only the Steering Generator (MLP/Attention components) to stabilize initial learning.
  - **Stage 2**: Unfreezes the User Encoder (or its LoRA adapters) for full end-to-end joint optimization.
- **Multi-Task Support**: Natively supports multiple LongLaMP and LaMP tasks (e.g., `generation_abstract`, `topic_writing`, `product_review_writing`, `lamp_4`, `lamp_5`, `lamp_7`) with task-specific profile extraction, pre-screening, and left-truncation prompt masking.

The training pipeline is compatible with the [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) style configuration (`stage: persona_profile_steering`).

This repository ships a **bundled LlamaFactory** tree with PersonaSteer training stages already integrated under `LlamaFactory/src/llamafactory/train/persona_*`. You do **not** need to merge a separate `Train/` folder.

---

## Quick start 

```bash
git clone <your-repo-url> PersonaSteer
cd PersonaSteer

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-eval.txt

cp .env.example .env   # fill OPENAI_API_KEY if using Datasets/ or LLM Judge

python scripts/smoke_check.py   # no GPU; verifies layout and CLI --help
```

**Typical pipeline**

1. **Smoke test (no GPU):** `python scripts/smoke_check.py`
2. **Data:** download [LongLaMP-Benchmark](https://github.com/LaMP-Benchmark/LongLaMP) / [LaMP](https://github.com/LaMP-Benchmark/LaMP) and point YAML `dataset:` to your `*_questions.json` files.
3. Optional demographic fill: `python Datasets/generate_profile.py -i /path/to/train.json --dry-run`
4. **Train (GPU):** edit model and dataset paths in `custom_persona_steering.yaml`, then:
   ```bash
   cd LlamaFactory
   llamafactory-cli train examples/train_profile_persona_steering/custom_persona_steering.yaml
   ```
5. **Evaluate:** `python Eval/eval_profile_steer.py --help` (input JSON schema in **Data formats** below).
6. **LLM Judge:** `python Eval/llm_judge_eval.py --path predictions.json --limit 5`

---

## LlamaFactory integration

PersonaSteer code lives inside the bundled `LlamaFactory/` directory:

| Path | Role |
|------|------|
| `LlamaFactory/src/llamafactory/train/persona_profile_steering/` | Dual-stream training |
| `LlamaFactory/examples/train_profile_persona_steering/` | Example YAML + `run_dual_stream_train.sh` |

Install and train from the bundled tree:

```bash
cd LlamaFactory
pip install -e .
llamafactory-cli train examples/train_profile_persona_steering/custom_persona_steering.yaml
```

Optional extras (DeepSpeed, SwanLab, etc.) are listed in LlamaFactory's `requirements/` and documentation.

---

## Repository layout

| Path | Description |
|------|-------------|
| `LlamaFactory/` | Bundled LLaMA-Factory + PersonaSteer `persona_*` training stages |
| `LlamaFactory/examples/train_profile_persona_steering/` | Dual-stream YAML and launch script |
| `Datasets/` | Data prep: `generate_profile.py`, `generate_profile.sh` |
| `Eval/` | Inference + metrics + LLM Judge (see Evaluation below) |
| `scripts/` | Pipeline wrappers + `smoke_check.py` (no-GPU sanity test) |
| `Datasets/download.py` | Download published eval artifacts from Hugging Face |
| `.env.example` | Environment variable template (copy to `.env`, not committed) |

---

## Supported tasks

Matches current `dataset.py` implementation. LaMP1–3 are not supported (extension required).

LongLaMP:

| `task` | Meaning | Typical `profile` fields |
|--------|---------|-------------------------|
| `generation_abstract` | Generate an abstract from title/keywords | `title`, `abstract` |
| `topic_writing` | Generate body from summary/topic | `input`/`output` or `summary`/`content` |
| `product_review_writing` | Product reviews | `overall`, `summary`, `description`, `reviewText` |

LaMP:

| `task` | Meaning | Typical `profile` fields |
|-------|---------|-------------------------|
| `lamp_4` | News headline generation | `title`, `text` |
| `lamp_5` | Generate a paper title from an abstract | `title`, `abstract` |
| `lamp_7` | Tweet rewriting | `text`, `date`, `id` |

Demographic prefixes (e.g., `Researcher ID:`, `Writer ID:`) are added automatically when `Demographic Information` is missing. If you provide a valid `Demographic Information` string, it is used as Stream A input directly.

---

## Data formats

PersonaSteer uses the **same per-row JSON schema** for training and inference. Files can be:

- **`.json`**: a JSON **array** of objects (typical for LongLaMP / LaMP `*_questions.json`)
- **`.jsonl`**: one JSON object per line (supported for training data loading)

Implementation: `LlamaFactory/src/llamafactory/train/persona_profile_steering/dataset.py` (training) and `Eval/eval_profile_steer.py` (inference).

### Common fields (every row)

| Field | Type | Training | Inference (`eval_profile_steer.py`) | Notes |
|-------|------|----------|--------------------------------------|-------|
| `id` | string / scalar | required | required | User identifier; converted with `str()` |
| `input` | string | required | required | Task prompt / instruction for the **current** sample |
| `profile` | list of objects | required | required | User history; field names depend on `task` (see below) |
| `output` | string | **required** | **required for metrics** | Supervised target (training label; gold reference at eval) |
| `Demographic Information` | string | optional | optional | Long-term persona text for Stream A; auto-filled or synthesized if missing |
| `task` | string | optional | inferred from `--task_name` | Per-row override: `generation_abstract`, `topic_writing`, `product_review_writing`, `lamp_4`, `lamp_5`, `lamp_7` |

`profile` must be a **list of dicts**. Empty lists are allowed but give weak personalization.

### `profile` fields by task

| `task` | Each `profile[i]` should contain |
|--------|-----------------------------------|
| `generation_abstract` | `title`, `abstract` (or `text` as abstract fallback) |
| `topic_writing` | `input` + `output`, **or** `summary` + `content` |
| `product_review_writing` | `reviewText` (required); often `overall`, `summary`, `description` |
| `lamp_4` | `text`, `title` |
| `lamp_5` | `abstract`, `title` |
| `lamp_7` | `text`; often `date`, `id` |

---

### Training data format

**File layout:** point YAML `dataset:` to one or more comma-separated `.json` / `.jsonl` paths. Task can be set globally in YAML (`task: ...`) and/or per row (`"task": "lamp_4"`). If a path contains `LaMP4`, `longlamp`, etc., the loader may **infer** `task` from the path when the row omits it.

**What one row means:** one user’s **current** example plus their **history** in `profile`.

**How rows become training samples** (`SteeringDataset` expands each row):

1. **History pairs:** for each item in `profile`, build a supervised example from that item (query derived from the item; label = item’s target field). The current item is **removed** from `profile` in that example to avoid leakage.
2. **Current row:** if `output` is non-empty, add one more example using row-level `input` → `output` with the **full** `profile` as context.

Supervised label field internally follows `TASK_CONFIGS[...]["output_field"]` (`abstract`, `output`, or `reviewText`).

**Example — `generation_abstract` (one row in your JSON file):**

```json
{
  "id": "researcher_42",
  "task": "generation_abstract",
  "input": "Generate an abstract for the title \"Neural Topic Models\" using the following items: ...",
  "profile": [
    {"title": "Paper A", "abstract": "We study ..."},
    {"title": "Paper B", "abstract": "This work ..."}
  ],
  "output": "We propose a method for ...",
  "Demographic Information": "This person likes neural networks, topic models, ..."
}
```

From this single row the trainer may emit up to **3** training examples: two from `profile` history + one from the current `input`/`output` pair.

**Example — `lamp_4`:**

```json
{
  "id": "user_7",
  "input": "Generate a headline for the following article: ...",
  "profile": [
    {"title": "Markets rally", "text": "Stocks rose today ..."},
    {"title": "Policy shift", "text": "The central bank ..."}
  ],
  "output": "Stocks climb on policy news"
}
```

**Training-only notes:**

- `max_samples_per_user` in YAML truncates `profile` length per user before expansion.
- `profile_topk_prescreen` / `profile_topk` optionally keep the most query-relevant profile items (Stream B).
- Without valid `Demographic Information`, the loader falls back to a short text derived from `profile`, or `"{TaskPrefix} {id}"`.

---

### Inference data format (input)

`Eval/eval_profile_steer.py` reads **`--questions_file`**: a JSON **array** with the **same row schema** as training.

| CLI | Format |
|-----|--------|
| `--questions_file` | JSON array; each element needs `id`, `input`, `profile`; `output` = gold for metric computation |
| `--demographic_file` | Optional JSONL; each line `{"id": "...", "Demographic Information": "..."}` merged into questions by `id` |
| `--golds_file` | Optional external gold file (LaMP-style); used only when per-row `output` is missing |

**Inference row (minimal):**

```json
{
  "id": "user_7",
  "input": "Generate a headline for the following article: ...",
  "profile": [
    {"title": "Past headline", "text": "Article body ..."}
  ],
  "output": "Gold headline text"
}
```

At generation time the script uses row-level `input`, `profile`, and `Demographic Information` (from the row or `--demographic_file`). It does **not** re-run training-time profile expansion; one JSON row → one prediction.

---

### Inference data format 

`eval_profile_steer.py` writes under `--output_dir`:

**Predictions file** — `{task}_{mode}_layer{L}_coeff{C}_predictions.json`:

```json
{
  "task": "LaMP_4",
  "steering_layer": 19,
  "steering_coeff": 1.0,
  "predictions": [
    {
      "id": "user_7",
      "generated_text": "Model headline ...",
      "output": "Gold headline text",
      "alpha": 0.62
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `generated_text` | Model prediction |
| `output` | Gold reference copied from the input row (for metrics) |
| `alpha` | Fusion gate value when steering is enabled (may be `null`) |

**Per-user metrics file** (unless `--disable_per_user_metrics`) — `*_per_user_metrics.json` with ROUGE / METEOR / BLEU per `id`.

**Summary file** — `{task}_layer{L}_coeff{C}_summary.json` with aggregated results across modes (`steer`, `no_steer`).

### Downstream metrics and LLM Judge

`Eval/eval_text_metrics.py` and `Eval/llm_judge_eval.py` accept:

- A **wrapper JSON** with a `predictions` list (as above), or
- A flat **JSON / JSONL** list of rows.

They resolve text via these aliases:

| Role | Accepted keys |
|------|----------------|
| Prediction | `generated_text`, `prediction`, `pred` |
| Reference | `output`, `reference`, `gold` |

For automatic metrics, **reference must come from the same row** as the prediction (do not align by file index across separate files).

`llm_judge_eval.py` example:

```bash
python Eval/llm_judge_eval.py --path path/to/LaMP_4_steer_layer19_coeff1.0_predictions.json --limit 10
```

---

> **Legacy format:** Older Reddit-style JSONL (`prompt` / `chosen` / `User-Generated Content`) is **not** supported by the current `dataset.py`. Convert to `input` + `profile` + `output` before training or evaluation.

---

## Data acquisition and construction
1. Overview

   This section explains how to obtain or build the JSON/JSONL training data used by PersonaSteer and how to link it into a LlamaFactory training YAML.

2. Download official data

   - Obtain raw JSON from LongLaMP-Benchmark / LaMP (observe licenses). After download, the dataset directory should contain the `*_questions.json` files that match your training config.

   - Example local paths:

     ```text
     LongLaMP: abstract_generation_temporal/train_questions.json
     LaMP:    LaMP4/ or LaMP5/ or LaMP7/ (each contains train_questions.json, dev_questions.json, ...)
     ```

   - In your LlamaFactory YAML, set the `dataset:` field to the absolute or relative path to the JSON / JSONL file.

3. Required fields

   See **[Data formats](#data-formats)** for the full training / inference schema, profile field tables, and output JSON layout. Minimum per row: `id`, `input`, `profile`, `output` (gold required at eval for automatic metrics).

4. (Optional) Auto-fill `Demographic Information` with an LLM

   The repo provides `Datasets/generate_profile.py`, a helper script that can generate a `Demographic Information` field from a sample's `profile` using a Chat API and write results back to the JSON file.

   - Script: `Datasets/generate_profile.py`
   - Dependency: `pip install openai` (install into the LlamaFactory environment)
   - Environment variables used by the script:
     - `OPENAI_API_KEY` (required)
     - `OPENAI_BASE_URL` (optional; set to your gateway if needed)
     - `LM_MODEL` (default: `gpt-4o-mini`)
     - `LAMP_INPUT_FILE` (path to the JSON to process)
     - `BATCH_SIZE` (default: 20), `MAX_PAPERS` (default: 50)

   Behavior:

   - On first run the script creates a `.bak` backup file (if absent).
   - It processes only samples where `Demographic Information` is missing or considered invalid.
   - Progress is saved after each batch; the script exits on failure so you can resume after fixing network/quota issues.

   Example (Bash):

   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_BASE_URL="YOUR-BASE_URL"
   export LM_MODEL="gpt-5o-mini"
   export LAMP_INPUT_FILE="/path/to/train_questions.json"
   export BATCH_SIZE=20
   export MAX_PAPERS=50

   python Datasets/generate_profile.py -i /path/to/train_questions.json --dry-run
   python Datasets/generate_profile.py -i /path/to/train_questions.json
   ```

   PowerShell (Windows) equivalent:

   ```powershell
   $env:OPENAI_API_KEY = 'your-key'
   python Datasets/generate_profile.py --input C:\path\to\train_questions.json --dry-run
   ```

5. Batch scan helper (`generate_profile.sh`)

   `Datasets/generate_profile.sh` is a Bash helper that scans multiple roots for LongLaMP/LaMP question files and runs `generate_profile.py` on each discovered file.

   - Default search roots: `/path/to/LaMP` and `/path/to/longlamp` (override via `--dir` on the CLI).
   - Requires `OPENAI_API_KEY` (and other environment variables) to be set.
   - Logs are written to `Datasets/logs/` by default.

   Example usage:

   ```bash
   cd Datasets
   chmod +x generate_profile.sh
   export OPENAI_API_KEY="sk-..."
   ./generate_profile.sh --all --dir /path/to/your/dataset/root
   # or single file
   ./generate_profile.sh -f /path/to/dev_questions.json
   ```

6. Linking to training

   After constructing your data, set the YAML `dataset:` field to the JSON or JSONL file path. If `Demographic Information` exists, the trainer will include it as part of the User Encoder input (see `LlamaFactory/src/llamafactory/train/persona_profile_steering/dataset.py`); otherwise the code composes user context automatically from task prefixes + `id` + `profile`.

## Training

1. Edit `LlamaFactory/examples/train_profile_persona_steering/custom_persona_steering.yaml`.
2. Set `task`, `dataset`, `model_name_or_path`, `user_encoder_name`, `output_dir`, `steering_layer_idx`, etc.
3. From the LlamaFactory root directory run:

```bash
cd LlamaFactory
pip install -e .
llamafactory-cli train examples/train_profile_persona_steering/custom_persona_steering.yaml
```

Or from the repo root: `bash scripts/02_train.sh`.

---

## Evaluation

| Script | Purpose |
|--------|---------|
| `Eval/eval_profile_steer.py` | **Main:** load PersonaSteer checkpoint, generate, ROUGE/METEOR/BLEU (metrics via `eval_text_metrics`; optional `LongLaMP-Benchmark/eval/evaluation.py` only for `--golds_file` fallback) |
| `Eval/eval_text_metrics.py` | Metric helpers (used by other eval scripts) |
| `Eval/llm_judge_eval.py` | LLM-as-Judge (G-Eval style rubric); requires `OPENAI_API_KEY` |
| `Eval/eval_longlamp.py` | LongLaMP evaluation (**optional**; needs `data/` + `prompts/` from LongLaMP-Benchmark) |

Prediction files should include model output and a **same-row** reference (`output` or `gold`) for automatic metrics.

```bash
export PYTHONPATH="$(pwd)/LlamaFactory/src:$PYTHONPATH"
python Eval/eval_profile_steer.py --help

export OPENAI_API_KEY="your-key"
python Eval/llm_judge_eval.py --path /path/to/predictions.json --limit 5
```

---

## Reference dataset paths (optional)

If you have cloned benchmark repos or copied data locally, for example:

- LongLaMP abstract: `LongLaMP-Benchmark/longLaMP/abstract_generation_temporal/train_questions.json`
- LaMP4/5/7: `LaMP/data/LaMP/LaMP{4,5,7}/train_questions.json`

Paths above are relative to the PersonaSteer root; use absolute paths if preferred.

---

## Recommended layout for benchmarks

Place LongLaMP-Benchmark and LaMP next to this repo or under `Datasets/data/`:

```
PersonaSteer/
  Datasets/
  Eval/
  LlamaFactory/
  LongLaMP-Benchmark/    # optional, for eval_longlamp.py
  LaMP/                  # optional
```

Notes:
- LongLaMP generation metrics often live under `longLaMP/metrics/` (case-sensitive `longLaMP`).
- LaMP’s evaluation entry is typically `LaMP/eval/evaluation.py` but forks may vary; adapt if needed.

The `Eval` scripts prefer to find `LongLaMP-Benchmark` and `LaMP` at the same directory level as `Eval/`, and will fallback to older monorepo-style locations.

### Training data paths

In LlamaFactory YAML set `dataset:` / `eval_dataset:` to your local JSON path. Data can live anywhere as long as the JSON shape matches `persona_profile_steering/dataset.py`.

### Evaluation dependencies

**Python packages** (required):

```bash
pip install -r requirements.txt && pip install -r requirements-eval.txt
```

**LongLaMP `evaluation.py`** (optional): lives in the cloned [LongLaMP-Benchmark](https://github.com/LaMP-Benchmark/LongLaMP) repo at `eval/evaluation.py`. PersonaSteer does **not** vendor this file. If `questions_file` rows include `output`, automatic metrics use bundled `eval_text_metrics.py` and you do not need LongLaMP's evaluator. It is only loaded when you use `--golds_file` without per-row gold in the questions file.

```bash
git clone https://github.com/LaMP-Benchmark/LongLaMP.git LongLaMP-Benchmark
# optional: export LONGLAMP_BENCHMARK_ROOT=/path/to/LongLaMP-Benchmark
```

**LlamaFactory imports:** `Eval/eval_profile_steer.py` imports `llamafactory.train.persona_profile_steering.*`. Use the bundled tree:

```bash
export PYTHONPATH="$(pwd)/LlamaFactory/src:$PYTHONPATH"
export LLAMAFACTORY_SRC="$(pwd)/LlamaFactory/src"
python Eval/eval_profile_steer.py --help
```

Or run `bash scripts/03_eval_profile_steer.sh --help`.

### Quick sanity checks

```bash
test -d LlamaFactory/src/llamafactory/train/persona_profile_steering && echo OK persona training
python Datasets/generate_profile.py --help 2>/dev/null || python Datasets/generate_profile.py -h
python Eval/eval_profile_steer.py --help
```

---

## License

The training framework follows the license headers from LlamaFactory files. When using this repository, please also comply with the licenses of models and datasets you use.
