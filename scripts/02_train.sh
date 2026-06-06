#!/usr/bin/env bash
# Dual-stream PersonaSteer training (edit YAML paths first).
set -euo pipefail
LF="$(cd "$(dirname "${BASH_SOURCE[0]}")/../LlamaFactory" && pwd)"
CONFIG="${LF}/examples/train_profile_persona_steering/custom_persona_steering.yaml"
export PYTHONPATH="${LF}/src:${PYTHONPATH:-}"
cd "$LF"
llamafactory-cli train "$CONFIG"
