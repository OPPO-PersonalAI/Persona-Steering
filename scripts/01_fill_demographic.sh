#!/usr/bin/env bash
# Fill Demographic Information for one JSON file.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-}"
if [ -z "$INPUT" ]; then
  echo "Usage: $0 /path/to/train_questions.json" >&2
  exit 2
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Set OPENAI_API_KEY first (see .env.example)" >&2
  exit 2
fi
python "${ROOT}/Datasets/generate_profile.py" --input "$INPUT" "${@:2}"
