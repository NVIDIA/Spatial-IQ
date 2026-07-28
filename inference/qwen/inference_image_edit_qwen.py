#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
SpatialIQ image tasks using Qwen/Qwen-Image-Edit (Diffusers), producing edited PNGs.
Same prompts as inference_image_gemini.py; outputs taskN_pred_image_<slug>.png per view.
Task 10: the official pipeline accepts one input image; we edit the second view only and
keep the two-view wording in the prompt (reference view is not passed as pixels).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, List, Optional, Set, Tuple

from PIL import Image
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from view_utils import find_and_group_views as _find_and_group_views, resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

MODEL_SLUG = "qwen_image_edit"
_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_image_edit")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFERENCE_RESULTS_DIR = os.path.join(REPO_ROOT, "inference_results")

# ---------------------------------------------------------------------------
# Prompts & constants (same as inference_image_qwen.py / inference_image_gemini.py)
# ---------------------------------------------------------------------------

DEFINITIONS_BLOCK = """
DEFINITIONS:
1. A column is a vertical stack of one or more blocks, where each block is directly supported by the block immediately beneath it. A single block also counts as a column.
2. A layer is a horizontal group of one or more blocks at the same vertical height. A single block also counts as a layer.
3. A visible block is a block with at least one face fully or partially visible.
4. A cluster is a group of one or more blocks that are connected through face adjacency, possibly across multiple layers. Two blocks are considered connected if they share a face (i.e., are laterally adjacent in the 3D scene); diagonal contact does not count. A cluster includes all blocks that are connected directly or indirectly through such face contacts.
5. A supporting block is any block that is beneath a given block in the same column. A directly supporting block is the supporting block in immediate contact beneath a given block. If a block is directly supported by the ground, then it has no supporting blocks.
6. A hidden block is a block with no visible faces in the image. Any hidden block must be a supporting block of at least one visible block. Otherwise, such a block does not exist in a valid structure.
"""

CRITICAL_INSTRUCTION = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. Answer the following questions using the absolute minimum number of words possible. Output only the exact value, count, or object name. Do not include full sentences, explanations, pleasantries, or context. Do not describe your reasoning, steps, or analysis.
"""

TASK_PROMPTS = {
    "task1": CRITICAL_INSTRUCTION + """YOUR TASK: Color the block structure(s) in this image with #0000FF.
As a hint, for this task, ALL blocks should be colored with some color (i.e., not left uncolored).
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.
""",
    "task2": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: Color the TOP-most block(s) in the whole structure (i.e., the blocks that are in the top-most layer in the image) in this image with #0000FF.
As a hint, for this task, at LEAST ONE block should be colored with some color (i.e., not left uncolored).
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.
""",
    "task3": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: Color all blocks within each column with a new unique color.
As a hint, for this task, ALL blocks should be colored with some color (i.e., not left uncolored).
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
    "task4": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: Color all blocks within each layer (i.e., all blocks at the same height) with a new unique color.
As a hint, for this task, ALL blocks should be colored with some color (i.e., not left uncolored).
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
    "task5": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: Color each visible block with a new unique color.
As a hint, for this task, ALL blocks should be colored with some color (i.e., not left uncolored).
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
    "task6": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: Color the block(s) that are directly supporting (i.e., directly BELOW) the top-most block(s).
As a hint, for this task, it could be true that NO blocks are colored (i.e., all blocks are left uncolored). This happens when the maximum height of all columns is 1.
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
    "task7": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: Color the entire supporting column(s) that are directly supporting the top-most block(s). Exclude the top-most block(s) themselves.
As a hint, for this task, it could be true that NO blocks are colored (i.e., all blocks are left uncolored). This happens when the maximum height of all columns is 1.
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
    "task8": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: Color all blocks that are being supported by (i.e., directly above) a non-visible block.
As a hint, for this task, it could be true that NO blocks are colored (i.e., all blocks are left uncolored). This happens when there are no non-visible blocks in the scene.
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
    "task9": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: Identify separate clusters of blocks in the 3D scene and assign a single, unique color to each cluster. Clusters are defined as directly adjacent blocks that share a common flat face (i.e., horizontally or vertically, but not diagonally).
As a hint, for this task, ALL blocks should be colored with some color (i.e., not left uncolored).
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
    "task10": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: You are given two images that are two views of the same 3D scene. Color all blocks that are visible in the second view but are NOT visible in the first view. So you should give me the second image (WITHOUT modifying the entire image other than the color of the blocks) where you color the blocks that are visible in it but NOT in the first image.
As a hint, for this task, it could be true that NO blocks are colored (i.e., all blocks are left uncolored). This happens when all blocks visible in the second view are also visible in the first view.
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
    "task11": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: You are given two images that are two views of the same 3D scene. Color all blocks that are visible in the second view but are NOT visible in the first view. So you should give me the second image (WITHOUT modifying the entire image other than the color of the blocks) where you color the blocks that are visible in it but NOT in the first image.
As a hint, for this task, it could be true that NO blocks are colored (i.e., all blocks are left uncolored). This happens when all blocks visible in the second view are also visible in the first view.
REMINDER: ONLY color the blocks to indicate your answer. Do NOT alter or move the block structures, and do NOT remove any part of the background (non-block areas) of the original image.""",
}

NUM_VIEWS = 4
SPATIALIQ_TASK_NAMES = [f"task{i}" for i in range(1, 12)]
SPATIALIQ_ALL_TASKS = set(SPATIALIQ_TASK_NAMES) | {"main"}
CROSS_VIEW_TASKS = {"task10", "task11"}


def _get_output_dir(data_dir: str) -> str:
    """Map a dataset directory to the corresponding output directory under inference_results/."""
    data_prefix = os.environ.get("SPATIALIQ_DATA_PREFIX", "").rstrip("/")
    if data_prefix and data_dir.startswith(data_prefix + "/"):
        rel = data_dir[len(data_prefix) + 1:]
        out = os.path.join(INFERENCE_RESULTS_DIR, rel)
        os.makedirs(out, exist_ok=True)
        return out
    m = re.search(r'.*/(?=dataset/)', data_dir)
    if m:
        rel = data_dir[m.end():]
        out = os.path.join(INFERENCE_RESULTS_DIR, rel)
        os.makedirs(out, exist_ok=True)
        return out
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def parse_tasks_arg(tasks_str: str) -> set:
    if not (tasks_str or "").strip():
        return SPATIALIQ_ALL_TASKS
    out: set = set()
    for part in tasks_str.replace(" ", "").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "main":
            out.add("main")
        elif part.isdigit():
            n = int(part)
            if 1 <= n <= 11:
                out.add(f"task{n}")
        elif part.startswith("task") and part[4:].isdigit():
            n = int(part[4:])
            if 1 <= n <= 11:
                out.add(f"task{n}")
        else:
            raise ValueError(f"Invalid task: {part!r}. Use 1-11, task1-task11, or main.")
    return out if out else SPATIALIQ_ALL_TASKS


def _get_paired_view_path(img_path: str, step: int) -> Optional[str]:
    """Given a view.png path, compute the path to the view offset by `step` positions."""
    offset_dir = os.path.dirname(img_path)
    view_dir = os.path.dirname(offset_dir)
    sample_dir = os.path.dirname(view_dir)
    view_dirname = os.path.basename(view_dir)
    offset_basename = os.path.basename(offset_dir)
    m = re.search(r"(\d+)$", view_dirname)
    if not m:
        return None
    view_idx = int(m.group(1))
    paired_idx = ((view_idx - 1 + step) % NUM_VIEWS) + 1
    paired_view_dirname = re.sub(r"\d+$", f"{paired_idx:0{len(m.group(1))}d}", view_dirname)
    return os.path.join(sample_dir, paired_view_dirname, offset_basename, "view.png")


def load_prompts(prompts_file: Optional[str]) -> Tuple[dict, str]:
    """Load task prompts from a JSON file.
    The JSON should have a "task_prompts" key containing task1..task10 and main.
    Other top-level keys (e.g. CRITICAL_INSTRUCTION, DEFINITIONS) are variables
    that can be referenced inside task prompt strings as {VARIABLE_NAME}.
    Returns (task_prompts_dict, main_task_prompt).
    """
    if not prompts_file:
        raise SystemExit("--prompts is required: hardcoded prompts have been removed. Pass a JSON prompts file.")
    with open(prompts_file, "r") as f:
        data = json.load(f)

    variables = {k: v for k, v in data.items() if k != "task_prompts"}
    raw_prompts = data.get("task_prompts", {})
    resolved = {k: v.format_map(variables) for k, v in raw_prompts.items()}

    main_prompt = resolved.get("main", None)
    if main_prompt is None:
        raise SystemExit("--prompts JSON is missing 'main' key under 'task_prompts'.")
    return resolved, main_prompt


def _load_rgb(path: str) -> Image.Image:
    if "_gt" in path:
        raise ValueError(f"Refusing to use ground-truth image as model input: {path}")
    img = Image.open(path).convert("RGB")
    return img


def _run_edit(
    pipe: Any,
    pil_image: Image.Image,
    prompt: str,
    *,
    num_inference_steps: int,
    true_cfg_scale: float,
    negative_prompt: str,
    seed: int,
) -> Image.Image:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(
        image=pil_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        generator=gen,
    )
    return out.images[0]


def _collect_work_items(
    samples: List[Tuple[str, List[Tuple[int, str]]]],
    tasks_set: Set[str],
    override: bool,
    pred_stem_suffix: str,
    seed: int,
    task_prompts: dict,
) -> List[dict]:
    """Build a flat list of independent work items that can be farmed to any GPU."""
    items: List[dict] = []
    task_names_single = [t for t in SPATIALIQ_TASK_NAMES if t not in CROSS_VIEW_TASKS and t in tasks_set]
    run_main = "main" in tasks_set

    for label, views in samples:
        image_paths = [v[1] for v in views]
        output_dirs = [_get_output_dir(os.path.dirname(p)) for p in image_paths]

        for v_idx_pos, (v_idx, img_path) in enumerate(views):
            out_dir = output_dirs[v_idx_pos]
            for task_name in task_names_single:
                out_path = os.path.join(out_dir, f"{task_name}_pred{pred_stem_suffix}.png")
                if not override and os.path.isfile(out_path):
                    continue
                items.append({
                    "img_path": img_path,
                    "prompt": task_prompts[task_name],
                    "out_path": out_path,
                    "label": f"{label} view_{v_idx} {task_name}",
                    "seed": seed,
                })

        for cv_task, step in [("task10", 1), ("task11", 2)]:
            if cv_task not in tasks_set:
                continue
            for v_idx_pos, (v_idx, img_path) in enumerate(views):
                paired_path = _get_paired_view_path(img_path, step)
                if paired_path is None or not os.path.isfile(paired_path):
                    continue
                out_dir = output_dirs[v_idx_pos]
                out_path = os.path.join(out_dir, f"{cv_task}_pred{pred_stem_suffix}.png")
                if not override and os.path.isfile(out_path):
                    continue
                paired_idx = ((v_idx - 1 + step) % NUM_VIEWS) + 1
                items.append({
                    "img_path": paired_path,
                    "prompt": task_prompts.get(cv_task, TASK_PROMPTS.get(cv_task, "")),
                    "out_path": out_path,
                    "label": f"{label} {cv_task} view_{v_idx}+view_{paired_idx}",
                    "seed": seed + v_idx_pos,
                })

        if run_main:
            main_prompt = task_prompts.get("main", "")
            if not main_prompt:
                continue
            for v_idx_pos, (v_idx, img_path) in enumerate(views):
                out_dir = output_dirs[v_idx_pos]
                out_path = os.path.join(out_dir, f"task_main_pred{pred_stem_suffix}.png")
                if not override and os.path.isfile(out_path):
                    continue
                items.append({
                    "img_path": img_path,
                    "prompt": main_prompt,
                    "out_path": out_path,
                    "label": f"{label} view_{v_idx} main",
                    "seed": seed,
                })
    return items


def _gpu_worker(
    gpu_id: int,
    work_items: List[dict],
    ckpt: str,
    torch_dtype: Any,
    num_inference_steps: int,
    true_cfg_scale: float,
    negative_prompt: str,
    cpu_offload: bool,
) -> None:
    """Run in a subprocess: load the pipeline on one GPU and process assigned items."""
    import torch
    from diffusers import QwenImageEditPipeline

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading pipeline ({len(work_items)} items) -> {device}")
    t0 = time.monotonic()
    pipe = QwenImageEditPipeline.from_pretrained(ckpt, torch_dtype=torch_dtype)
    if cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
    else:
        pipe.to(device)
    print(f"[GPU {gpu_id}] Pipeline ready ({time.monotonic() - t0:.1f}s)")

    for i, item in enumerate(work_items, 1):
        t1 = time.monotonic()
        try:
            pil = _load_rgb(item["img_path"])
            gen = torch.Generator(device=device).manual_seed(item["seed"])
            out = pipe(
                image=pil,
                prompt=item["prompt"],
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=gen,
            )
            out.images[0].save(item["out_path"])
            elapsed = time.monotonic() - t1
            print(f"[GPU {gpu_id}] ({i}/{len(work_items)}) {item['label']} -> {os.path.basename(item['out_path'])} ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.monotonic() - t1
            print(f"[GPU {gpu_id}] ({i}/{len(work_items)}) {item['label']} ERROR: {e} ({elapsed:.1f}s)")

    print(f"[GPU {gpu_id}] Done.")


def run_spatialiq_image_edit(
    samples: List[Tuple[str, List[Tuple[int, str]]]],
    ckpt: str,
    torch_dtype: Any,
    tasks_set: Set[str],
    override: bool,
    *,
    num_inference_steps: int,
    true_cfg_scale: float,
    negative_prompt: str,
    seed: int,
    pred_stem_suffix: str,
    num_gpus: int,
    cpu_offload: bool,
    task_prompts: dict,
) -> None:
    import torch
    import torch.multiprocessing as mp

    items = _collect_work_items(samples, tasks_set, override, pred_stem_suffix, seed, task_prompts)
    if not items:
        print("All outputs already exist. Nothing to do.")
        return

    num_gpus = min(num_gpus, len(items))
    print(f"Work items: {len(items)}, GPUs: {num_gpus}")
    print(f"  Steps: {num_inference_steps}, CFG: {true_cfg_scale}, suffix: {pred_stem_suffix}\n")

    if num_gpus <= 1:
        _gpu_worker(0, items, ckpt, torch_dtype, num_inference_steps,
                    true_cfg_scale, negative_prompt, cpu_offload)
        return

    # Split items across GPUs round-robin
    shards: List[List[dict]] = [[] for _ in range(num_gpus)]
    for i, item in enumerate(items):
        shards[i % num_gpus].append(item)

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(
            target=_gpu_worker,
            args=(gpu_id, shards[gpu_id], ckpt, torch_dtype,
                  num_inference_steps, true_cfg_scale, negative_prompt, cpu_offload),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed = [p for p in procs if p.exitcode != 0]
    if failed:
        print(f"\nWARNING: {len(failed)} GPU worker(s) exited with errors.")
    else:
        print("\nAll GPU workers finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen-Image-Edit (Diffusers) on SpatialIQ view.png tasks; save edited PNGs."
    )
    parser.add_argument(
        "folder",
        help="Sample folder, tree root, or a .txt file with one folder path per line.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("QWEN_IMAGE_EDIT_CHECKPOINT", _DEFAULT_CKPT),
        metavar="DIR",
        help=f"Local HF folder (default: {_DEFAULT_CKPT} or QWEN_IMAGE_EDIT_CHECKPOINT).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated tasks (default: all except main). Use 1-10, taskN, main.",
    )
    parser.add_argument("--override", action="store_true", help="Overwrite existing PNGs.")
    parser.add_argument(
        "--prompts",
        type=str,
        default=os.path.join(REPO_ROOT, "inference", "prompt_image_Type1.json"),
        metavar="FILE",
        help="Path to a JSON file with task prompts (default: prompt_image_Type1.json).",
    )
    parser.add_argument(
        "--pred-stem-suffix",
        type=str,
        default=f"_image_{MODEL_SLUG}",
        metavar="SUFFIX",
        help=f'Stem suffix after taskN_pred (default: _image_{MODEL_SLUG} → task1_pred_image_{MODEL_SLUG}.png). '
        'Use "" for evaluation_image.py default names (task1_pred.png).',
    )
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps (default 50).")
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="true_cfg_scale (default 4.0, per Qwen-Image-Edit README).",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=" ",
        help='Negative prompt (default single space, per README).',
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (task10 uses seed+view_offset).")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="enable_sequential_cpu_offload() for lower VRAM (slower).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Pipeline torch dtype (default bfloat16).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        dest="num_gpus",
        metavar="N",
        help="Number of GPUs for data-parallel inference (default: all visible GPUs). "
        "Each GPU loads its own pipeline copy and processes a shard of work items.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "Missing dependencies for image edit. Diffusers needs Python >= 3.10.\n"
            "  cd inference/qwen\n"
            "  uv venv --python 3.12 .venv-image-edit\n"
            "  uv pip install --python .venv-image-edit/bin/python -r requirements_image_edit.txt\n"
            f"Original error: {e}"
        ) from e

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isdir(ckpt):
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() or 1
    print(f"Checkpoint: {ckpt}")
    print(f"GPUs available: {torch.cuda.device_count()}, using: {num_gpus}")

    folders = _resolve_folders(args.folder)

    if args.tasks is None:
        tasks_set = SPATIALIQ_ALL_TASKS.copy()
    else:
        try:
            tasks_set = parse_tasks_arg(args.tasks)
        except ValueError as e:
            raise SystemExit(f"--tasks: {e}")

    try:
        task_prompts, _main_task_prompt = load_prompts(args.prompts)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"--prompts: {e}")
    print(f"Loaded prompts from: {args.prompts}\n")

    samples = _scan_folders_cached(args.folder, folders)
    if not samples:
        raise SystemExit("No view.png files found in any of the specified folder(s).")

    run_spatialiq_image_edit(
        samples,
        ckpt,
        torch_dtype,
        tasks_set,
        override=args.override,
        num_inference_steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        pred_stem_suffix=args.pred_stem_suffix,
        num_gpus=num_gpus,
        cpu_offload=args.cpu_offload,
        task_prompts=task_prompts,
    )


if __name__ == "__main__":
    main()
