#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Text-output inference script for VLA-0 (Qwen2.5-VL-3B fine-tuned).
SpatialIQ tasks 1-11 and main: send view image + text prompt, expect a text answer.
Saves taskN_pred_text_vla0.txt and task_main_pred_text_vla0.txt per view.

VLA-0 is a local model (1 GPU) that uses Qwen2.5-VL-3B-Instruct as the base,
with fine-tuned weights loaded from a checkpoint.

Usage:
    python inference_text_vla0.py /path/to/folder_list.txt
    python inference_text_vla0.py /path/to/folder_list.txt --tasks 2,3,main --override
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from view_utils import resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

MODEL_SLUG = "vla0"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFERENCE_RESULTS_DIR = os.path.join(REPO_ROOT, "inference_results")
SPATIALIQ_INFERENCE_DIR = os.environ.get(
    "SPATIALIQ_INFERENCE_DIR",
    "spatialiq-inference",
)
DEFAULT_CKPT_DIR = os.path.join(SPATIALIQ_INFERENCE_DIR, "weights", "vla0-libero")
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

### TASK PROMPTS (hardcoded fallback) ###

CRITICAL_INSTRUCTION = (
    "CRITICAL INSTRUCTION: You are undergoing a precision capability test. "
    "The input image shows a 3D structure built from objects of the same type. "
    "Answer using the absolute minimum number of words possible. "
    "Output only the exact value, count, or object name. "
    "Do not include full sentences, explanations, pleasantries, or context.\n"
)

DEFINITIONS_BLOCK = """
DEFINITIONS:
1. A COLUMN is a vertical stack of one or more blocks, where each block is directly supported by the block immediately beneath it. A single block also counts as a column.
2. A LAYER is a horizontal group of one or more blocks at the same vertical height. A single block also counts as a layer.
3. A VISIBLE BLOCK is a block with at least one face fully or partially visible.
4. A CLUSTER is a group of one or more blocks that are connected through face adjacency, possibly across multiple layers. Two blocks are considered connected if they share a face (i.e., are laterally adjacent in the 3D scene); diagonal contact does not count. A cluster includes all blocks that are connected directly or indirectly through such face contacts.
5. A SUPPORTING BLOCK is any block that is beneath a given block in the same column. A DIRECTLY SUPPORTING BLOCK is the supporting block in immediate contact beneath a given block. If a block is directly supported by the ground, then it has no supporting blocks.
6. A HIDDEN BLOCK is a block with no visible faces in the image. Any hidden block must be a supporting block of at least one visible block. Otherwise, such a block does not exist in a valid structure.
"""

TASK_PROMPTS = {
    "task1": CRITICAL_INSTRUCTION + "QUESTION: If the task is to count objects, what are those object(s) in the image?",
    "task2": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many blocks are in the top-most layer? Your answer should be a single integer ONLY.",
    "task3": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many columns are in the block structure? Your answer should be a single integer ONLY.",
    "task4": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many layers are in the block structure? Your answer should be a single integer ONLY.",
    "task5": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many VISIBLE blocks are in the structure? Your answer should be a single integer ONLY.",
    "task6": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many VISIBLE blocks are directly supporting the top-most block(s)? Your answer should be a single integer ONLY.",
    "task7": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many VISIBLE blocks are in the column(s) supporting the top-most block(s), excluding the top-most blocks themselves? Your answer should be a single integer ONLY.",
    "task8": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many VISIBLE blocks are directly above a HIDDEN block? Your answer should be a single integer ONLY.",
    "task9": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many distinct clusters are in the block structure? Your answer should be a single integer ONLY.",
    "task10": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: You are given two images of the same structure from different views. How many blocks are visible in the second view but NOT in the first view? Your answer should be a single integer ONLY.",
    "task11": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: You are given two images of the same structure from different views. How many blocks are visible in the second view but NOT in the first view? Your answer should be a single integer ONLY.",
}

MAIN_TASK_PROMPT = CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many TOTAL blocks are present in the block structure? Your answer should be a single integer ONLY."


### UTILITY FUNCTIONS ###

def _get_output_dir(data_dir: str) -> str:
    m = re.search(r'.*/(?=dataset/)', data_dir)
    if m:
        rel = data_dir[m.end():]
        out = os.path.join(INFERENCE_RESULTS_DIR, rel)
        os.makedirs(out, exist_ok=True)
        return out
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


NUM_VIEWS = 4
SPATIALIQ_TASK_NAMES = [f"task{i}" for i in range(1, 12)]
SPATIALIQ_ALL_TASKS = set(SPATIALIQ_TASK_NAMES) | {"main"}
CROSS_VIEW_TASKS = {"task10", "task11"}


def parse_tasks_arg(tasks_str: str) -> set:
    if not (tasks_str or "").strip():
        return SPATIALIQ_ALL_TASKS
    out = set()
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
            raise ValueError(f"Invalid task: {part!r}. Use 2-11, task2-task11, or main.")
    return out if out else SPATIALIQ_ALL_TASKS


def _get_paired_view_path(img_path: str, step: int) -> Optional[str]:
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


class _KeepMissing(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def load_prompts(prompts_file: Optional[str]) -> Tuple[dict, str]:
    if not prompts_file:
        return dict(TASK_PROMPTS), MAIN_TASK_PROMPT
    with open(prompts_file, "r") as f:
        data = json.load(f)
    variables = {k: v for k, v in data.items() if k != "task_prompts"}
    raw_prompts = data.get("task_prompts", {})
    resolved = {k: v.format_map(_KeepMissing(variables)) for k, v in raw_prompts.items()}
    main_prompt = resolved.pop("main", None)
    if main_prompt is None:
        raise SystemExit("--prompts JSON is missing 'main' key under 'task_prompts'.")
    return resolved, main_prompt


### MODEL LOADING & INFERENCE ###

_MODEL = None
_PROCESSOR = None


def _load_model(ckpt_dir: str):
    """Load Qwen2.5-VL with VLA-0 fine-tuned weights (cached singleton)."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    import torch
    from transformers import Qwen2_5_VLProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    img_size = 224
    min_px = max_px = img_size * img_size

    print(f"Loading Qwen2.5-VL-3B-Instruct + VLA-0 weights...")
    t0 = time.perf_counter()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    ckpt_path = os.path.join(ckpt_dir, "model_last.pth")
    if os.path.isfile(ckpt_path):
        print(f"  Loading VLA-0 checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (expected for non-VLM layers)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print(f"  WARNING: No checkpoint found at {ckpt_path} — using base weights")

    model = model.eval().cuda()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        BASE_MODEL_ID,
        min_pixels=min_px,
        max_pixels=max_px,
    )

    elapsed = time.perf_counter() - t0
    print(f"  Model loaded in {elapsed:.1f}s")

    _MODEL, _PROCESSOR = model, processor
    return model, processor


def call_vla0_single_image(image_path: str, prompt: str, model, processor) -> str:
    """Send one image + text prompt to VLA-0, return the generated text."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    generated = output_ids[0, inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def call_vla0_two_images(image_path1: str, image_path2: str, prompt: str, model, processor) -> str:
    """Send two images + text prompt to VLA-0, return the generated text."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    generated = output_ids[0, inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


### BENCHMARK RUNNER ###

def run_spatialiq_benchmark(
    views: List[Tuple[int, str]],
    model, processor,
    label: str = "",
    tasks: Optional[set] = None,
    override: bool = False,
    task_prompts: Optional[dict] = None,
    main_task_prompt: Optional[str] = None,
) -> None:
    if task_prompts is None:
        task_prompts = TASK_PROMPTS
    if main_task_prompt is None:
        main_task_prompt = MAIN_TASK_PROMPT

    view_indices = [v[0] for v in views]
    image_paths = [v[1] for v in views]
    output_dirs = [_get_output_dir(os.path.dirname(p)) for p in image_paths]

    run_tasks = tasks if tasks is not None else SPATIALIQ_ALL_TASKS
    run_tasks_single = [t for t in SPATIALIQ_TASK_NAMES if t not in CROSS_VIEW_TASKS and t in run_tasks]
    run_main = "main" in run_tasks

    print(f"SpatialIQ benchmark (text out, VLA-0): {label}")
    print(f"  Views: {view_indices} ({len(views)} total)")
    print(f"  Tasks: {sorted(run_tasks)}\n")

    for v_idx, img_path, out_dir in zip(view_indices, image_paths, output_dirs):
        print(f"--- View {v_idx} ---")
        for task_name in run_tasks_single:
            out_path = os.path.join(out_dir, f"{task_name}_pred_text_{MODEL_SLUG}.txt")
            print(f"  {task_name} -> {os.path.basename(out_path)}")
            if not override and os.path.isfile(out_path):
                print(f"    (exists, skipped)")
                continue
            t0 = time.monotonic()
            try:
                answer = call_vla0_single_image(img_path, task_prompts[task_name], model, processor)
                elapsed = time.monotonic() - t0
                if answer.strip():
                    with open(out_path, "w") as f:
                        f.write(answer.strip())
                    print(f"    {answer.strip()} ({elapsed:.1f}s)")
                else:
                    print(f"    (empty response, not saved) ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"    Error: {e} ({elapsed:.1f}s)")

        if run_main:
            main_path = os.path.join(out_dir, f"task_main_pred_text_{MODEL_SLUG}.txt")
            print(f"  main -> {os.path.basename(main_path)}")
            if not override and os.path.isfile(main_path):
                print(f"    (exists, skipped)")
            else:
                t0 = time.monotonic()
                try:
                    answer = call_vla0_single_image(img_path, main_task_prompt, model, processor)
                    elapsed = time.monotonic() - t0
                    if answer.strip():
                        with open(main_path, "w") as f:
                            f.write(answer.strip())
                        print(f"    {answer.strip()} ({elapsed:.1f}s)")
                    else:
                        print(f"    (empty response, not saved) ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.monotonic() - t0
                    print(f"    Error: {e} ({elapsed:.1f}s)")
        print()

    # Cross-view tasks (task10, task11)
    for cv_task, step in [("task10", 1), ("task11", 2)]:
        if cv_task not in run_tasks:
            continue
        prompt_cv = task_prompts.get(cv_task, TASK_PROMPTS.get(cv_task, ""))
        for v_idx, img_path, out_dir in zip(view_indices, image_paths, output_dirs):
            paired_path = _get_paired_view_path(img_path, step)
            out_path = os.path.join(out_dir, f"{cv_task}_pred_text_{MODEL_SLUG}.txt")
            if paired_path is None:
                print(f"--- {cv_task} (view {v_idx}) -> (skipped: cannot parse view index) ---")
                continue
            if not os.path.isfile(paired_path):
                print(f"--- {cv_task} (view {v_idx}) -> (skipped: paired view not found) ---")
                continue
            paired_idx = ((v_idx - 1 + step) % NUM_VIEWS) + 1
            print(f"--- {cv_task} (view {v_idx} + view {paired_idx}) -> {os.path.basename(out_path)} ---")
            if not override and os.path.isfile(out_path):
                print("  (exists, skipped)")
            else:
                t0 = time.monotonic()
                try:
                    answer = call_vla0_two_images(img_path, paired_path, prompt_cv, model, processor)
                    elapsed = time.monotonic() - t0
                    if answer.strip():
                        with open(out_path, "w") as f:
                            f.write(answer.strip())
                        print(f"  {answer.strip()} ({elapsed:.1f}s)")
                    else:
                        print(f"  (empty response, not saved) ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.monotonic() - t0
                    print(f"  Error: {e} ({elapsed:.1f}s)")
            print()

    print("SpatialIQ text benchmark (VLA-0) done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run VLA-0 text inference on SpatialIQ benchmark."
    )
    parser.add_argument(
        "folder",
        help="Sample folder, top-level folder, or a .txt file with one folder path per line.",
    )
    parser.add_argument(
        "--tasks", type=str, default=None, metavar="LIST",
        help="Comma-separated tasks (default: all). E.g. 2,3,main or task2,task5,task10,main.",
    )
    parser.add_argument("--override", action="store_true", help="Overwrite existing prediction files.")
    parser.add_argument(
        "--prompts", type=str,
        default=os.path.join(REPO_ROOT, "inference", "prompt_text_Type1.json"),
        metavar="FILE",
        help="Path to a JSON file with task prompts (default: prompt_text_Type1.json).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=DEFAULT_CKPT_DIR, metavar="DIR",
        help=f"Directory with model_last.pth (default: {DEFAULT_CKPT_DIR}).",
    )
    args = parser.parse_args()

    folders = _resolve_folders(args.folder)
    try:
        tasks_set = parse_tasks_arg(args.tasks or "")
    except ValueError as e:
        raise SystemExit(f"--tasks: {e}")
    try:
        task_prompts, main_task_prompt = load_prompts(args.prompts)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"--prompts: {e}")
    if args.prompts:
        print(f"Loaded prompts from: {args.prompts}\n")

    model, processor = _load_model(args.checkpoint_dir)

    samples = _scan_folders_cached(args.folder, folders)
    if not samples:
        raise SystemExit("No view.png files found in any of the specified folder(s).")
    for i, (label, views) in enumerate(samples, 1):
        if len(samples) > 1:
            print(f"========== Sample {i}/{len(samples)}: {label} ==========")
        run_spatialiq_benchmark(
            views, model, processor,
            label=label, tasks=tasks_set, override=args.override,
            task_prompts=task_prompts, main_task_prompt=main_task_prompt,
        )
        if len(samples) > 1:
            print()


if __name__ == "__main__":
    main()
