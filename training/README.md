<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# SpatialIQ Training Recipe

This folder documents the SFT and RLVR (GRPO / DAPO-tight) recipe used to
produce the fine-tuned checkpoints reported in the paper, and provides the
data-preparation and evaluation utilities that surround the training runner.

The training runner itself lives in
[NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL) — see the *How to train
with this recipe* section below.

## Contents

```
training/
├── configs/                       # NeMo-RL YAML configs used in the paper
│   ├── sft_spatialiq.yaml
│   ├── sft_spatialiq_main.yaml
│   ├── grpo_spatialiq.yaml
│   └── grpo_spatialiq_main.yaml
├── prompts/
│   └── spatialiq_cot.txt          # System prompt for SFT-CoT and evaluation
├── schema.py                      # Pydantic manifest schema (SpatialIQManifest)
├── generate_manifest.py           # Isaac Sim dataset -> manifest.json
├── prepare_data.py                # manifest.json -> train.jsonl / val.jsonl
├── eval_accuracy.py               # Score a HuggingFace checkpoint on the val set
├── verify_cpu.py                  # Environment smoke test (CPU only)
└── README.md                      # (this file)
```

## Recipe overview

- **SFT-CoT.** Supervised fine-tuning on task1–task9 subtasks with a
  chain-of-thought target listing sub-task answers in the order required for
  additive composition, followed by the integer total. Trains the model to
  expose the intermediate reasoning steps before emitting the final count.
- **DAPO-tight.** A brief SFT-CoT warmup on 10% of the training data followed
  by RL with verifiable rewards under the [DAPO
  recipe](https://arxiv.org/abs/2503.14476) on the remaining 90%. The reward
  gates on structured-answer tag presence + integer match — no per-sub-task
  rewards.

The paper's headline recipe is applied identically at both 7B and 32B; only
memory-driven infrastructure knobs differ. See the paper for the full
ablation.

## How to train with this recipe

The recipe is expressed by the configs in `configs/` and the prompt in
`prompts/`, not by any code in this folder. To reproduce the training runs:

1. **Clone NeMo-RL** and follow its own setup docs to build the environment:

    ```bash
    git clone https://github.com/NVIDIA-NeMo/RL.git
    ```

2. **Prepare data** — turn an Isaac Sim dataset (see `../data_generation/`)
   into the JSONL files consumed by NeMo-RL:

    ```bash
    python training/generate_manifest.py <dataset_root>
    python training/prepare_data.py <dataset_root> \
        --output-dir <dataset_root>/data_sft \
        --split 0.9 --seed 42
    ```

3. **Run SFT** with NeMo-RL's SFT entry point and this repo's config:

    ```bash
    # Inside the NeMo-RL environment
    uv run python examples/run_sft.py \
        --config /path/to/spatial-iq/training/configs/sft_spatialiq.yaml \
        checkpointing.checkpoint_dir=./results/sft_spatialiq \
        data.train_data_path=<dataset_root>/data_sft/train.jsonl \
        data.val_data_path=<dataset_root>/data_sft/val.jsonl
    ```

4. **Run GRPO / DAPO-tight** initialised from the SFT-CoT checkpoint:

    ```bash
    export NEMO_RL_INIT_WEIGHTS_PATH=./results/sft_spatialiq/step_XXX/policy/weights
    uv run python examples/run_grpo.py \
        --config /path/to/spatial-iq/training/configs/grpo_spatialiq.yaml \
        checkpointing.checkpoint_dir=./results/grpo_spatialiq
    ```

The config files bake in the recipe details that mattered — hard format gate
on the reward, KL coefficient, learning rate, prompt / generation counts,
dynamic sampling — so pointing the NeMo-RL runners at them should reproduce
the paper's training runs modulo compute-scale knobs.

## Evaluation

`eval_accuracy.py` loads a HuggingFace-format checkpoint (base or fine-tuned)
and scores it against a val JSONL:

```bash
python training/eval_accuracy.py \
    --checkpoint <path/to/hf/checkpoint/or/hub-id> \
    --val-jsonl <dataset_root>/data_sft/val.jsonl
```

`verify_cpu.py` is a CPU-only smoke test to confirm the Python environment
(datasets + Pydantic + transformers) loads a manifest correctly.

## Fine-tuned checkpoints referenced by the paper

We do **not** redistribute fine-tuned checkpoints as part of this repository.
The paper's reported numbers were produced with the recipe above; independent
uploads of matching checkpoints exist on the Hugging Face Hub for user
convenience, distributed under the uploader's own terms.
