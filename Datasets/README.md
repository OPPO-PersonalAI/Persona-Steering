# Datasets

Optional data-prep utilities for PersonaSteer training.

## `generate_profile.py`

Fills the `Demographic Information` field in LaMP / LongLaMP JSON files using an OpenAI-compatible Chat API.

```bash
pip install openai

export OPENAI_API_KEY="your-key"
# optional: export OPENAI_BASE_URL="https://api.openai.com/v1"

python generate_profile.py --input /path/to/train_questions.json --dry-run
python generate_profile.py --input /path/to/train_questions.json
```

- Creates `*.bak` on first run.
- Saves after each batch; safe to resume after API errors.
- CLI flags override environment variables (`LAMP_INPUT_FILE`, `LM_MODEL`, etc.).

## `generate_profile.sh`

Batch driver for many `*_questions.json` files under a data root:

```bash
chmod +x generate_profile.sh
export OPENAI_API_KEY="your-key"
./generate_profile.sh --all --dir /path/to/data
./generate_profile.sh -f /path/to/dev_questions.json
```

Place cloned LaMP / LongLaMP trees under `Datasets/data/` or pass `--dir` explicitly.

Logs: `Datasets/logs/`.
