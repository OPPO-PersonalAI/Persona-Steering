#!/bin/bash
# ============================================
# PersonaSteering 8-GPU Multi-Task Training
# ============================================

export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_SWANLAB=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${LLAMAFACTORY_ROOT}/src:${PYTHONPATH:-}"

CONFIG_FILE="${SCRIPT_DIR}/custom_persona_steering.yaml"

echo "============================================"
echo "PersonaSteering 8-GPU Multi-Task Training"
echo "============================================"
echo "Config file: ${CONFIG_FILE}"
echo "LlamaFactory root: ${LLAMAFACTORY_ROOT}"
echo "Number of GPUs: 8"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "SwanLab: ${USE_SWANLAB}"
echo "============================================"

llamafactory-cli train "${CONFIG_FILE}"
