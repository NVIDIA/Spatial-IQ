#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Convert every DCP checkpoint under DCP_WEIGHTS_ROOT to HF format.
# Skips checkpoints that are already converted (config.json present).
#
# Usage:
#   NEMO_RL_DIR=/path/to/NeMo-RL bash convert_all.sh
#
# Env vars (forwarded to convert_dcp_to_hf.sh):
#   NEMO_RL_DIR        Required. Path to NeMo-RL clone.
#   DCP_WEIGHTS_ROOT   Where the source DCP ckpt dirs live.
#   HF_OUT_ROOT        Where to write converted HF dirs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${SCRIPT_DIR}/convert_dcp_to_hf.sh"
DCP_WEIGHTS_ROOT="${DCP_WEIGHTS_ROOT:-training_evaluation/weights}"
HF_OUT_ROOT="${HF_OUT_ROOT:-${SCRIPT_DIR}/weights}"

if [[ ! -d "$DCP_WEIGHTS_ROOT" ]]; then
    echo "ERROR: DCP_WEIGHTS_ROOT not found: $DCP_WEIGHTS_ROOT"
    exit 1
fi

# Ckpt names = subdirs matching the (sft|dapo)-(7b|32b) pattern (so we skip
# Qwen2.5-VL-*-Instruct bases, which are already in HF format).
# Use a shell glob — `find -type d -printf` has been observed to return empty
# on this Lustre mount with cross-user-owned dirs.
CKPTS=()
shopt -s nullglob
for d in "$DCP_WEIGHTS_ROOT"/*/; do
    name="$(basename "$d")"
    case "$name" in
        sft-7b-*|sft-32b-*|dapo-7b-*|dapo-32b-*) CKPTS+=("$name") ;;
    esac
done
shopt -u nullglob
IFS=$'\n' CKPTS=($(printf '%s\n' "${CKPTS[@]}" | sort))
unset IFS

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "No DCP checkpoints found under $DCP_WEIGHTS_ROOT"
    exit 0
fi

echo "Found ${#CKPTS[@]} DCP checkpoint(s):"
for c in "${CKPTS[@]}"; do echo "  - $c"; done
echo ""

for c in "${CKPTS[@]}"; do
    OUT="${HF_OUT_ROOT}/${c}"
    if [[ -f "${OUT}/config.json" ]]; then
        echo "[$c] already converted at $OUT — skipping"
        continue
    fi
    echo "[$c] converting..."
    DCP_WEIGHTS_ROOT="$DCP_WEIGHTS_ROOT" HF_OUT_ROOT="$HF_OUT_ROOT" \
        bash "$CONVERTER" "$c"
    echo "[$c] done"
    echo ""
done

echo "All conversions complete. HF checkpoints under: $HF_OUT_ROOT"
