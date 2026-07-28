#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Convert a NeMo-RL HF-shard DCP checkpoint (shard-NNNNN-...safetensors plus
# .hf_metadata/) into a standard HuggingFace layout that vLLM can serve.
#
# These checkpoints are NOT standard torch DCP (no .metadata blob), so
# torch.distributed.checkpoint.format_utils.dcp_to_torch_save can't read
# them. Conversion is a key-rename + repack — see convert_hfdcp_to_hf.py.
#
# Usage:
#   bash convert_dcp_to_hf.sh <ckpt_name>
#
#   <ckpt_name> is a directory under DCP_WEIGHTS_ROOT (default:
#   training_evaluation/weights/), e.g. dapo-7b-tight or sft-32b-cot-100pct.
#
# Output: <SCRIPT_DIR>/weights/<ckpt_name>/  (HF format, vLLM-loadable).
#
# Optional env vars:
#   DCP_WEIGHTS_ROOT  Where source DCP ckpt dirs live.
#   HF_OUT_ROOT       Where to write converted HF dirs (default: <script_dir>/weights).
#   BASE_MODEL_DIR    Override base-model dir for tokenizer/processor copy.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ckpt_name>"
    echo "Example: $0 dapo-7b-tight"
    exit 1
fi

CKPT_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CONVERTER="${SCRIPT_DIR}/convert_hfdcp_to_hf.py"

DCP_WEIGHTS_ROOT="${DCP_WEIGHTS_ROOT:-training_evaluation/weights}"
HF_OUT_ROOT="${HF_OUT_ROOT:-${SCRIPT_DIR}/weights}"

DCP_CKPT_PATH="${DCP_WEIGHTS_ROOT}/${CKPT_NAME}/model"
HF_OUT_PATH="${HF_OUT_ROOT}/${CKPT_NAME}"

if [[ ! -d "$DCP_CKPT_PATH" ]]; then
    echo "ERROR: source DCP path not found: $DCP_CKPT_PATH"
    exit 2
fi
if [[ ! -f "$PYTHON_CONVERTER" ]]; then
    echo "ERROR: python converter not found: $PYTHON_CONVERTER"
    exit 3
fi

# Derive base model dir from ckpt name (-7b- / -32b-). Tokenizer + processor
# files are copied from this local HF base model dir.
case "$CKPT_NAME" in
    *-7b-*|*7B*)
        BASE_DEFAULT="${DCP_WEIGHTS_ROOT}/Qwen2.5-VL-7B-Instruct" ;;
    *-32b-*|*32B*)
        BASE_DEFAULT="${DCP_WEIGHTS_ROOT}/Qwen2.5-VL-32B-Instruct" ;;
    *)
        echo "ERROR: cannot infer base model from ckpt name '$CKPT_NAME'."
        echo "       Expected '-7b-' or '-32b-' substring."
        exit 4
        ;;
esac
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$BASE_DEFAULT}"
if [[ ! -f "${BASE_MODEL_DIR}/tokenizer.json" ]]; then
    echo "ERROR: base model dir missing tokenizer.json: $BASE_MODEL_DIR"
    echo "       Set BASE_MODEL_DIR to a valid Qwen2.5-VL HF dir."
    exit 5
fi

mkdir -p "$HF_OUT_ROOT"

echo "Converting:"
echo "  ckpt_name      : $CKPT_NAME"
echo "  base_model_dir : $BASE_MODEL_DIR"
echo "  src            : $DCP_CKPT_PATH"
echo "  dst            : $HF_OUT_PATH"
echo ""

# Use the local vllm-env's python (already has safetensors + torch).
VENV_PY="python"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PY}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
echo "  python         : $PYTHON_BIN"
echo ""

"$PYTHON_BIN" "$PYTHON_CONVERTER" \
    --src "$DCP_CKPT_PATH" \
    --dst "$HF_OUT_PATH" \
    --base-model-dir "$BASE_MODEL_DIR"

echo ""
echo "Done. HF checkpoint at: $HF_OUT_PATH"
