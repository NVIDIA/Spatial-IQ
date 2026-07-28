#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# One-time setup: clone VLMEvalKit and create a virtualenv for evaluation.
#
# Usage:
#   bash setup.sh            # uses default venv location
#   VENV_DIR=/custom/path bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVALKIT_DIR="${SCRIPT_DIR}/VLMEvalKit"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/venv}"

echo "=== Setting up VLMEvalKit evaluation environment ==="

# 0. Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# 1. Clone VLMEvalKit if not present
if [ ! -d "$VLMEVALKIT_DIR" ]; then
    echo "Cloning VLMEvalKit..."
    git clone https://github.com/open-compass/VLMEvalKit.git "$VLMEVALKIT_DIR"
else
    echo "VLMEvalKit already cloned at $VLMEVALKIT_DIR"
fi

# 2. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    uv venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 3. Install VLMEvalKit (editable, no-deps) + only the deps we need.
#    The full requirements.txt includes API-only packages (google-genai, etc.)
#    that aren't needed for local Qwen2.5-VL evaluation and may fail to install.
echo "Installing VLMEvalKit (editable, --no-deps)..."
uv pip install --no-deps -e "$VLMEVALKIT_DIR"

echo "Installing required dependencies..."
uv pip install -r "${SCRIPT_DIR}/requirements.txt"

# 4. Download pretrained Qwen2.5-VL-7B-Instruct baseline checkpoint
BASELINE_DIR="${SCRIPT_DIR}/weights/Qwen2.5-VL-7B-Instruct"
if [ ! -d "$BASELINE_DIR" ]; then
    echo "Downloading Qwen2.5-VL-7B-Instruct baseline weights..."
    mkdir -p "${SCRIPT_DIR}/weights"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir='${BASELINE_DIR}',
    resume_download=True,
)
print('Download complete.')
"
else
    echo "Baseline weights already present at $BASELINE_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:  source $VENV_DIR/bin/activate"
echo "Then run evaluations with:      cd VLMEvalKit && python run.py --config ../eval_config.json"
