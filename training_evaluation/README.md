<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# Training Evaluation

Evaluate Qwen2.5-VL fine-tuned checkpoints on standard VLM benchmarks using
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Checkpoints

`eval_config.json` points at checkpoints under `inference/qwen_finetuned/weights/`.
The baselines (`Qwen2.5-VL-7B-Instruct` / `-32B-Instruct`) are HuggingFace
snapshots; the fine-tuned checkpoints are produced by converting NeMo-RL DCP
outputs to an HF layout with `inference/qwen_finetuned/convert_dcp_to_hf.sh`.

| Name | Description |
|------|-------------|
| `Qwen2.5-VL-{7B,32B}-Instruct` | Baseline (HuggingFace) |
| `sft-{7b,32b}-plain-100pct` | SFT, plain answers, 100% data |
| `sft-{7b,32b}-cot-100pct` | SFT, chain-of-thought, 100% data |
| `dapo-{7b,32b}-tight` | DAPO (RL) |

## Benchmarks

The config evaluates on 10 benchmarks covering general VLM capabilities:

| Benchmark | Type | What it tests |
|-----------|------|---------------|
| MMBench_DEV_EN_V11 | MCQ | General multimodal understanding |
| MME | Y/N + MCQ | Perception & cognition |
| SEEDBench_IMG | MCQ | Comprehensive image understanding |
| MMMU_DEV_VAL | MCQ | Multi-discipline reasoning |
| MathVista_MINI | Mixed | Mathematical reasoning |
| HallusionBench | Y/N | Hallucination resistance |
| AI2D_TEST | MCQ | Diagram understanding |
| OCRBench | Open | OCR accuracy |
| RealWorldQA | MCQ | Real-world spatial/visual QA |
| MMStar | MCQ | Multi-modal evaluation (no shortcuts) |

Edit `eval_config.json` to add/remove benchmarks or models.

## Quick Start

### 1. One-time setup

```bash
cd training_evaluation/
bash setup.sh
```

This installs [uv](https://github.com/astral-sh/uv), clones VLMEvalKit, creates a
virtualenv at `training_evaluation/venv/`, and downloads the baseline checkpoint.

### 2. Run evaluations

Activate the environment and run VLMEvalKit with `eval_config.json`:

```bash
source venv/bin/activate
cd VLMEvalKit
python run.py --config ../eval_config.json
```

`eval_config.json` lists which models × benchmarks to run; edit it to add or
remove either. Larger (32B) checkpoints typically need multiple GPUs.

### 3. Check results

Results are saved to `training_evaluation/results/` with the structure:

```
results/
├── Qwen2.5-VL-7B-Instruct/
│   ├── Qwen2.5-VL-7B-Instruct_MMBench_DEV_EN_V11.csv
│   ├── Qwen2.5-VL-7B-Instruct_MME.csv
│   └── ...
├── sft-7b-plain-100pct/
│   └── ...
├── sft-7b-cot-100pct/
│   └── ...
└── dapo-7b-tight/
    └── ...
```

`.csv` files contain the final metrics; `.xlsx` files contain per-sample
predictions.

## Customization

### Adding a new checkpoint

Add an entry to the `"model"` section of `eval_config.json`:

```json
"my-new-checkpoint": {
    "class": "Qwen2VLChat",
    "model_path": "/path/to/checkpoint",
    "min_pixels": 1003520,
    "max_pixels": 12845056,
    "use_custom_prompt": false
}
```

### Adding a new benchmark

Add an entry to the `"data"` section. Use `{}` for default settings or specify
overrides:

```json
"ScienceQA_TEST": {
    "class": "ImageMCQDataset",
    "dataset": "ScienceQA_TEST"
}
```

### LLM-based answer extraction

By default VLMEvalKit uses exact matching for MCQ benchmarks. To use GPT-based
answer extraction (more accurate), set `OPENAI_API_KEY` in your environment or
create `VLMEvalKit/.env`.
