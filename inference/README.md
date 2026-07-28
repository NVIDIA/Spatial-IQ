<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# Inference

Per-model clients that run the SpatialIQ benchmark and write prediction files
(`taskN_pred_*`) next to each view, which `evaluation/` then scores.

Two backends:

- **API models** (`claude/`, `gemini/`, `gpt/`, `kimi/`, `nemotron/`) call an
  OpenAI-compatible, Bearer-token chat-completions endpoint.
- **Local models** (`qwen/`, `qwen3b/`, `glm/`, `hunyuan/`, `vla0/`,
  `qwen_finetuned/`) call a local
  [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible server.

## Prompts

The exact task prompts are in `prompt_*_Type1.json` (text / MCQ 3-, 4-, 5-option /
image). Every client loads these — they are the benchmark's task definitions.

## API models

Set the endpoint and key via environment, then run a client directly:

```bash
export INFERENCE_API_BASE="https://<your-openai-compatible-endpoint>/v1/chat/completions"

# one API key per line; the 1-based index N selects which line to use
printf '%s\n' "$YOUR_API_KEY" > inference/api_keys.txt
# ...or export NVIDIA_API_KEY / INFERENCE_API_KEY instead

python inference/gemini/inference_text_gemini.py <folder> [N]
```

- `<folder>`: a single view/sample folder, or a parent folder (samples auto-discovered).
- `N` *(optional)*: 1-based line index into `api_keys.txt` (default `1`), or a literal key string.

The model id sent to the endpoint is the `MODEL` constant at the top of each
client (the version used in the paper, e.g. `gemini-3-pro-image-preview`); edit
it to target a different model.

| Folder | Clients |
|--------|---------|
| `claude/` | text, mcq, image |
| `gemini/` | text, mcq, image |
| `gpt/` | text, mcq, image |
| `kimi/` | text, mcq |
| `nemotron/` | text |

## Local models (vLLM)

Start a vLLM OpenAI-compatible server for the model, then point the client at it:

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
python inference/qwen/inference_text_qwen.py <folder> --model <served-model-name>
```

| Folder | Notes |
|--------|-------|
| `qwen/`, `qwen3b/` | Qwen2.5-VL variants |
| `glm/` | GLM |
| `hunyuan/` | HunyuanImage (image-output); weights/venv located via `SPATIALIQ_INFERENCE_DIR` |
| `vla0/` | VLA-0; checkpoint located via `SPATIALIQ_INFERENCE_DIR` |
| `qwen_finetuned/` | fine-tuned Qwen2.5-VL; `convert_dcp_to_hf.sh` converts NeMo-RL DCP checkpoints to an HF layout |

## Script types

| Script | Input → Output | Writes |
|--------|---------------|--------|
| `inference_text_*.py`  | image → integer/text answer     | `taskN_pred_text_<slug>.txt`  |
| `inference_mcq_*.py`   | image(s) + choice images → A–E  | `taskN_pred_mcq_<slug>.txt`   |
| `inference_image_*.py` | image → annotated image         | `taskN_pred_image_<slug>.png` |

`inference_mcq_NCQ.py` runs the N-option MCQ variants (3/4/5 choices) across models.

## Common options

| Flag | Description |
|------|-------------|
| `--tasks LIST` | Comma-separated tasks to run, e.g. `1,3,main` (default: all) |
| `--override` | Overwrite existing prediction files |
| `--api-key KEY_OR_N` | API key or 1-based index into `api_keys.txt` (API models) |
| `--base-url URL` / `--model NAME` | vLLM endpoint / served model name (local models) |

Prediction files land next to each view; score them with
`evaluation/run_evaluation.py`.
