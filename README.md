<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# Spatial-IQ

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under the [BSD-3-Clause](LICENSE) License.

A benchmark for evaluating multimodal LLMs on **3D spatial reasoning**: a
procedurally generated dataset of stacked-block scenes, per-model inference
clients (API and local), evaluation across three answer modalities
(free-response text, image-option MCQ, image editing) plus a separate
OCR-based scoring path for image-output models, and a paper-aligned analysis
pipeline that reproduces the manuscript tables and figures.

Companion paper: **Spatial-IQ: Deconstructing Spatial Intelligence via
Hierarchical Capability Tests**.

📄 **[Paper (arXiv) →](https://arxiv.org/abs/2607.22864)** &nbsp;|&nbsp; 🌐 **[Project Page →](https://nvidia.github.io/Spatial-IQ/)** &nbsp;|&nbsp; 📊 **[Dataset (HuggingFace) →](https://huggingface.co/datasets/patrickqrim/spatial-iq)**

---

## What's in this repo

| Path | What it does |
|------|--------------|
| [`data_generation/`](data_generation/) | Isaac Sim 4.5 renderer (`blocks_render_image.py`) that produces the dataset: block scenes, multi-view images, GT masks per task, MCQ choice images, and `gt.json` / `scene.json` metadata. Optional S3 upload driven by env vars. |
| [`inference/`](inference/) | Per-model inference clients under `claude/`, `gemini/`, `gpt/`, `kimi/`, `nemotron/`, `qwen/`, `qwen3b/`, `qwen_finetuned/`, `glm/`, `hunyuan/`, `vla0/`. Each folder contains `inference_{text,mcq,image}_<model>.py`. Shared prompt specifications live in `prompt_*_Type1.json`. |
| [`evaluation/`](evaluation/) | Per-modality scoring: `evaluation_text.py`, `evaluation_mcq.py`, `evaluation_image.py`. Entry point: `run_evaluation.py`. Also `evaluation_human_*.py` for human-baseline results and `trained_ckpts/evaluate_text_ckpts.py` for fine-tuned checkpoints. |
| [`evaluation/main_ocr/`](evaluation/main_ocr/) | Separate OCR-based pipeline that re-extracts the main object count from image-output models' annotated outputs using Tesseract + PaddleOCR + Qwen2.5-VL consensus. |
| [`training/`](training/) | SpatialIQ SFT + RLVR (GRPO / DAPO-tight) training recipe: NeMo-RL YAML configs, the CoT system prompt, Pydantic-validated dataset manifest, data-preparation and checkpoint-evaluation utilities. The training runner itself is [NeMo-RL](https://github.com/NVIDIA-NeMo/RL); see [`training/README.md`](training/README.md). |
| [`training_evaluation/`](training_evaluation/) | [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)-based evaluation of fine-tuned Qwen2.5-VL checkpoints against standard benchmarks (MMBench, MMMU, OCRBench, etc.). |
| [`analyses/`](analyses/) | Paper-aligned analysis pipeline (`spatial-IQ_analyses.py`) that turns scored `eval_raw_*.csv` files into the manuscript/appendix tables, statistics, and figures. Ships with the paper's raw model outputs under `analyses/inference_results/` and a machine-readable manifest (`analyses/analyses_results/paper_output_manifest.csv`) mapping every paper figure/table to the code that produces it. Appendix F statistics come from `analyses/hierarchy_dependency.py`, which the main pipeline calls. |
| [`docs/`](docs/) | Sphinx source for the [project page](https://nvidia.github.io/Spatial-IQ/) — `index.md`, `conf.py`, `_static/` (CSS and pre-rendered figures). Auto-deployed to GitHub Pages on push to `main`. |
| [`inference_results/`](inference_results/) | Subset manifest (`subset_eval.txt`) listing evaluation-set view paths. |

---

## Dataset structure

The Spatial-IQ dataset is a tree of per-view directories under
`<dataset_root>/eval_set/`:

```
data_<timestamp>/level_<L>/sample_<NN>/view_<VV>/offset<O>_fov<F>_dist<D>/
    view.png                    # input image
    gt.json                     # ground-truth metadata (counts, MCQ correct letters, ...)
    scene.json                  # scene-level metadata
    task<N>_gt_mask.npy         # per-task pixel masks
    task<N>_gt_image_*.png      # MCQ choice images
```

Each leaf directory (one `offset*_fov*_dist*` folder) is one **view** — the
unit of inference and evaluation. A subset file such as
[`inference_results/subset_eval.txt`](inference_results/subset_eval.txt) is a
list of these leaf paths, relative to the dataset root.

The dataset itself is distributed via
[Hugging Face](https://huggingface.co/datasets/patrickqrim/spatial-iq); to
generate a fresh dataset from scratch instead, use
[`data_generation/blocks_render_image.py`](data_generation/blocks_render_image.py).

---

## Benchmark tasks

Each view has **11 spatial sub-tasks (`task1`–`task11`) plus a composite
target task (`task_main`)**:

- **`task_main`** — How many blocks are in the scene, including occluded
  blocks that are logically required to support visible ones (gravity
  assumption)?
- **`task1` … `task11`** — Atomic spatial-reasoning sub-tasks: column counts,
  layer counts, visible/hidden object counts, top-layer identification,
  direct-support inference, support-column inference, and MCQ variants for
  each. See [Table 5 of the paper](analyses/) for the exact question text per
  task.

A model can answer each task in one of three answer modalities:

| Modality | Prediction file written next to view | Evaluator | Score |
|---|---|---|---|
| **text** | `task<N>_pred_text_<slug>.txt` | `evaluation/evaluation_text.py` | exact-match / Pearson correlation |
| **MCQ** | `task<N>_pred_mcq_<slug>.txt` | `evaluation/evaluation_mcq.py` | letter match (A–E), 3-/4-/5-way |
| **image** | `task<N>_pred_image_<slug>.png` | `evaluation/evaluation_image.py` | silhouette score against GT masks |
| **main (OCR)** | `task_main_pred_image_<slug>.png` | `evaluation/main_ocr/evaluate_main_ocr.py` | Tesseract + PaddleOCR + Qwen-VL consensus vs `total_blocks` |

`<slug>` is the inference-script directory name (e.g. `gemini`, `claude`,
`vla0`). Slugs flow through every step: inference writes them, evaluation
reads them, output CSVs are named after them.

---

## Setup

You only need to install the components you actually plan to run.

### Requirements files

Every install step points at a scoped `requirements.txt` — there is no
top-level or `inference/`-level `requirements.txt`. Install only the
components you plan to run:

- `analyses/requirements.txt` — paper-figure analysis pipeline
- `evaluation/requirements.txt` — text / MCQ / image scoring
- `evaluation/main_ocr/requirements.txt` — OCR consensus pipeline (installed by `bash evaluation/main_ocr/setup_ocr.sh`)
- `inference/gemini/requirements.txt` — API client HTTP dependency (`requests`)
- `inference/qwen/requirements_image_edit.txt` — Diffusers-based Qwen-Image-Edit
- `training_evaluation/requirements.txt` — VLMEvalKit dependency set (installed by `bash training_evaluation/setup.sh`)

### API models (`claude/`, `gemini/`, `gpt/`, `kimi/`, `nemotron/`)

The API clients only need `requests`:

```bash
pip install -r inference/gemini/requirements.txt
```

Drop one or more API keys into `inference/api_keys.txt` (one key per line —
this file is gitignored via [.gitignore](.gitignore) and is never bundled
with the repository). Each client accepts an optional 1-based line index
argument to select which key to use. Alternatively export
`INFERENCE_API_KEY` in the environment.

Set the endpoint via `INFERENCE_API_BASE` (an OpenAI-compatible
`chat/completions` URL). The model id sent to the endpoint is the `MODEL`
constant at the top of each client — edit it to target a different model.

### Local models (`qwen/`, `qwen3b/`, `glm/`, `hunyuan/`, `vla0/`, `qwen_finetuned/`)

Each local-model client expects a running
[vLLM](https://github.com/vllm-project/vllm)-compatible OpenAI server that
serves the model weights. Typical setup for one model:

1. Create a virtualenv and `pip install vllm transformers` (plus any
   model-specific extras — e.g. `inference/qwen/requirements_image_edit.txt`
   for the Qwen-Image-Edit pipeline).
2. Download the model weights (see [Downloading checkpoints](#downloading-checkpoints)).
3. Launch a vLLM server, e.g. `python -m vllm.entrypoints.openai.api_server
   --model <path-to-weights> --port 8000`.
4. Point the client at it via `--base-url http://127.0.0.1:8000/v1`.

### Scoring (`evaluation/`)

```bash
pip install -r evaluation/requirements.txt
```

### OCR scoring (`evaluation/main_ocr/`)

The OCR path uses Tesseract + PaddleOCR + a local Qwen2.5-VL vLLM endpoint
side-by-side. Create a dedicated environment:

```bash
bash evaluation/main_ocr/setup_ocr.sh
```

This creates a conda env at `evaluation/main_ocr/.venv-ocr/` with Tesseract
5.5 (conda-forge) plus the Python OCR stack (`pytesseract`, `paddleocr`,
`paddlepaddle`, `openai`, `pillow`, `tqdm`, `numpy`).

### VLMEvalKit (`training_evaluation/`)

```bash
cd training_evaluation
bash setup.sh    # clones VLMEvalKit and creates a local virtualenv
```

`eval_config.json` controls which models × which benchmarks run. MathVista's
GPT-graded scoring requires `OPENAI_API_KEY` in the environment.

---

## Downloading checkpoints

All model weights are gitignored. Each local-model directory expects its
weights at a specific path; the table below maps that path to the upstream
source. Skip this section entirely if you only run API models.

| Target directory | Source HF repo |
|---|---|
| `inference/glm/weights_glm-4.6v-flash/` | [`zai-org/GLM-4.6V-Flash`](https://huggingface.co/zai-org/GLM-4.6V-Flash) |
| `inference/qwen/weights_qwen3.5-27b/` | [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) |
| `inference/qwen3b/weights_qwen2.5-3b/` | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| `inference/qwen/weights_image_edit/` | [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| `inference/hunyuan/weights_hunyuan-instruct-distil/` | [`tencent/HunyuanImage-3.0-Instruct-Distil`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) (~160 GB) |
| `inference/qwen_finetuned/weights/Qwen2.5-VL-7B-Instruct/` | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (base for 7B fine-tunes) |
| `inference/qwen_finetuned/weights/Qwen2.5-VL-32B-Instruct/` | [`Qwen/Qwen2.5-VL-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) (base for 32B fine-tunes) |

Install the HuggingFace CLI once and, for gated repos, log in:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # only needed for gated repos
```

Point the HF cache at a filesystem with enough space (home quotas are often
too small):

```bash
export HF_HOME=/path/with/enough/space/.cache/huggingface
```

Then download each model, e.g.:

```bash
# Qwen2.5-VL-3B (~7 GB, fits on one GPU)
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir inference/qwen3b/weights_qwen2.5-3b

# GLM-4.6V-Flash
huggingface-cli download zai-org/GLM-4.6V-Flash \
    --local-dir inference/glm/weights_glm-4.6v-flash
```

### Fine-tuned checkpoint conversion

Fine-tuned Qwen2.5-VL checkpoints (`sft-{7b,32b}-{plain,cot}-100pct`,
`dapo-{7b,32b}-tight`) come out of NeMo-RL as DCP-format shards. The
converter under `inference/qwen_finetuned/` reshapes them into a
vLLM-loadable HF layout — point `DCP_WEIGHTS_ROOT` at the directory
containing the raw DCP run outputs and run the converter for the checkpoint
you want. See [`inference/qwen_finetuned/`](inference/qwen_finetuned/) for
details.

---

## Workflows

### 1. Run inference + evaluation on a subset

Direct single-process invocation:

```bash
# Inference — writes taskN_pred_<mode>_<slug>.* next to each view
python inference/gemini/inference_mcq_gemini.py <path/to/view/or/subset>

# Evaluation — scores prediction files and writes JSON + CSV
python evaluation/run_evaluation.py inference_results/subset_eval.txt \
    --pred-slug gemini --mode mcq
```

Outputs:

- `evaluation_results/subset_eval/eval_<mode>_<slug>.json` — per-task aggregate
- `evaluation_results/subset_eval/eval_raw_<slug>_<mode>.csv` — per-view rows
  (this CSV is the input to `analyses/`)

### 2. OCR-score the main count for image-output models

```bash
# After bash evaluation/main_ocr/setup_ocr.sh
evaluation/main_ocr/.venv-ocr/bin/python evaluation/main_ocr/evaluate_main_ocr.py \
    --model <slug> --subset inference_results/subset_eval.txt
```

Produces `evaluation_results/main_ocr_eval/eval_main_ocr_<slug>_<subset>.{csv,json}`
plus a `_review.csv` with rows where the three OCR systems disagreed.

### 3. Evaluate fine-tuned checkpoints against VLMEvalKit benchmarks

```bash
cd training_evaluation
source venv/bin/activate                              # created by setup.sh
cd VLMEvalKit && python run.py --config ../eval_config.json
```

Results land in `training_evaluation/results/<model>/<benchmark>.{xlsx,csv,pkl}`.

### 4. Generate the paper tables and figures

The [`analyses/`](analyses/) subproject turns scored `eval_raw_*.csv` files
into the paper-aligned analysis tables, statistics, and figures:

```bash
cd analyses
pip install -r requirements.txt
python spatial-IQ_analyses.py --stage all
```

`--stage` can be `all`, `analysis`, `figures`, or `manifest`; `--groups`
selects which modalities are processed (`text`, `mcq5`, `mcq345`, `image`,
`trained_text`). To analyze a new model, drop its scored `eval_raw_*.csv`
into `analyses/inference_results/` and rerun — the script discovers models
by filename pattern (`eval_raw_<slug>_text.csv`, `eval_raw_<slug>_mcq.csv`,
`eval_raw_image_<slug>.csv`, plus `eval_raw_human_frq.csv` /
`eval_raw_human_mcq.csv` for the human baseline). Outputs land under
`analyses/analyses_results/tables/` and `analyses/analyses_results/figures/`
with paper-aligned filenames (e.g. `figure_04_text_task_accuracy.png`);
`analyses/analyses_results/paper_output_manifest.csv` is the machine-readable
map from every paper figure/table to its generating CSV.

Appendix F hierarchy-dependency statistics (per-edge and joint within-scene lifts, joint relation integrity, bundle-to-Object-Counting couplings, the responder-controlled robustness checks, and their significance) are written by the same `--stage all` run as `tables/tableF_*.csv`, and are listed in the manifest under `Appendix F`. The module can also be run on its own — it writes the identical CSVs and adds a stdout summary of the headline numbers:

```bash
python analyses/hierarchy_dependency.py --validate
```

### 5. Generate a new dataset

```bash
cd data_generation
# Inside Isaac Sim 4.5's Python environment:
./python.sh blocks_render_image.py
```

Configure levels, samples-per-level, view counts, environment USDs, and
difficulty knobs at the top of `blocks_render_image.py`. See
[`data_generation/README_blocks_render_image.md`](data_generation/README_blocks_render_image.md)
for the full configuration reference (env vars, S3 upload options,
texture handling).

---

## Key environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `INFERENCE_API_KEY` | API inference clients | Fallback API key if `inference/api_keys.txt` is empty. |
| `INFERENCE_API_BASE` | API inference clients | OpenAI-compatible `chat/completions` endpoint URL. |
| `HF_HOME` | HuggingFace CLI / downloads | Redirect the HuggingFace cache off home-directory storage. |
| `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_REGION` | `data_generation/blocks_render_image.py` | Optional S3 upload destination when `UPLOAD_TO_S3=1`. Nothing is hardcoded. |
| `ASSET_USD_BASE_DIR`, `BLOCK_TEXTURES_PATH`, `FACTORY_BOX_TEXTURES_DIR` | `data_generation/blocks_render_image.py` | Override the paths to USD assets and block textures used during rendering. |
| `OPENAI_API_KEY` | `training_evaluation/` VLMEvalKit | Required for MathVista's GPT-graded answer extraction. |

---

## Project Page

The Spatial-IQ project page is live at **[nvidia.github.io/Spatial-IQ](https://nvidia.github.io/Spatial-IQ/)**.

It is built with [Sphinx](https://www.sphinx-doc.org/) and MyST-Markdown from
the [`docs/`](docs/) folder, and deployed automatically to GitHub Pages via
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) on every push to
`main`.

To build the page locally:

```bash
uv sync
uv run sphinx-build -b html docs docs/_build/html
uv run python -m http.server -d docs/_build/html 8765
# then visit http://127.0.0.1:8765/
```

---

## License

This project is released under the **BSD-3-Clause** license. See
[LICENSE](LICENSE) for the full text.

Each source file in this repository carries an SPDX header
(`SPDX-License-Identifier: BSD-3-Clause`) identifying it as NVIDIA-authored
code licensed under BSD-3-Clause.

## Third-Party Notices

Runtime dependencies are installed by the end user (via `pip`, `conda`, or
the NVIDIA Isaac Sim installer) and are not redistributed by this
repository. The corresponding upstream license notices are reproduced in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for attribution.

## Contributing

This repository is a reproducibility artifact for the paper and does **not**
accept external contributions. Please fork the repository and cite the paper
for any derivative work.

## Citation

If you use this code or the Spatial-IQ benchmark in your research, please
cite the paper:

```bibtex
@misc{rim2026spatialiq,
  title         = {Spatial-IQ: Deconstructing Spatial Intelligence via Hierarchical Capability Tests},
  author        = {Rim, Patrick and Long, Tom and Prashnani, Ekta and Rosenholtz, Ruth and
                   Boudaoud, Ben and Xenopoulos, Peter and Wong, Alex and Kim, Joohwan and
                   Jung, Jae-Hyun},
  year          = {2026},
  eprint        = {2607.22864},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2607.22864},
  url           = {https://arxiv.org/abs/2607.22864},
}
```
