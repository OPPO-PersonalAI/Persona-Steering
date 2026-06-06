#!/usr/bin/env bash
# LLM-as-Judge on prediction JSON — pass through all args to llm_judge_eval.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Set OPENAI_API_KEY first (see .env.example)" >&2
  exit 2
fi
python "${ROOT}/Eval/llm_judge_eval.py" "$@"
