#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Install OCR tooling for the main-task image evaluator.
#
# Creates a conda env at evaluation/main_ocr/.venv-ocr/ containing:
#   - tesseract binary (conda-forge)
#   - python 3.11
#   - pytesseract, paddleocr+paddlepaddle (CPU), openai client, pillow, tqdm, numpy
#
# Tesseract trained-data is bundled with the conda package. PaddleOCR will
# auto-download its detection/recognition models on first use into ~/.paddleocr
# (override with PADDLE_OCR_BASE_DIR).
#
# Usage:
#   bash evaluation/main_ocr/setup_ocr.sh
#
# Then either activate:
#   conda activate evaluation/main_ocr/.venv-ocr
# or call python directly:
#   evaluation/main_ocr/.venv-ocr/bin/python evaluate_main_ocr.py ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/.venv-ocr"

if [ -d "${ENV_DIR}" ]; then
    echo "Conda env already present at ${ENV_DIR} (delete to recreate)."
else
    echo "Creating conda env at ${ENV_DIR}..."
    conda create -y -p "${ENV_DIR}" -c conda-forge \
        python=3.11 \
        tesseract=5.5
fi

# Use the env's pip directly — avoids 'conda activate' shenanigans inside scripts.
PIP="${ENV_DIR}/bin/pip"
PY="${ENV_DIR}/bin/python"

echo "Upgrading pip..."
"${PIP}" install --upgrade pip

echo "Installing python OCR deps..."
"${PIP}" install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "Verifying installation..."
PATH="${ENV_DIR}/bin:${PATH}" "${PY}" - <<'PY'
import shutil, sys

print(f"  python: {sys.executable}")
tesseract_bin = shutil.which("tesseract")
print(f"  tesseract: {tesseract_bin}")

import pytesseract  # noqa: F401
print(f"  pytesseract: {pytesseract.__version__}")

# PaddleOCR is heavy and prints to stderr on import — keep this terse.
from paddleocr import PaddleOCR  # noqa: F401
import paddleocr
print(f"  paddleocr: {paddleocr.__version__}")

import openai
print(f"  openai: {openai.__version__}")
PY

echo ""
echo "Setup complete."
echo "Next: run evaluate_main_ocr.py --model gemini   (or hunyuan, qwen-image-edit)"
