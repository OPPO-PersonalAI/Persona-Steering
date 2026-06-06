#!/usr/bin/env bash
# PersonaSteer checkpoint evaluation — pass through all args to eval_profile_steer.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/LlamaFactory/src:${PYTHONPATH:-}"
export LLAMAFACTORY_SRC="${ROOT}/LlamaFactory/src"
python "${ROOT}/Eval/eval_profile_steer.py" "$@"
