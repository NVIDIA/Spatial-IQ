#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Image-output inference for HunyuanImage 3.0 Instruct-Distil (local model).
Loads the model via HF AutoModelForCausalLM with device_map='auto' (multi-GPU sharding).
For each SpatialIQ task, sends view.png + task prompt -> generates edited/annotated image.
Saves taskN_pred_image_hunyuan.png per view.
Main task: generates image with integer count; also saves CoT-extracted text prediction.

Usage:
    python inference_image_hunyuan.py /path/to/folder_list.txt \
        --model-path ./weights/HunyuanImage-3-Instruct-Distil \
        --diff-steps 8

    python inference_image_hunyuan.py /path/to/sample_dir \
        --model-path ./weights/HunyuanImage-3-Instruct-Distil \
        --tasks 1,2,main --override
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from typing import Any, List, Optional, Tuple

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from view_utils import resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference_image_hunyuan")

MODEL_SLUG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFERENCE_RESULTS_DIR = os.path.join(REPO_ROOT, "inference_results")

# ---------------------------------------------------------------------------
# Task prompts (same as inference_image_gpt.py / inference_image_edit_qwen.py)
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

CRITICAL_INSTRUCTION = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. Answer the following questions using the absolute minimum number of words possible. Output only the exact value, count, or object name. Do not include full sentences, explanations, pleasantries, or context.
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

MAIN_TASK_PROMPT = CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """YOUR TASK: How many TOTAL blocks are present in the block structure? Generate an image with your answer (ONLY a single integer) written in black against a white background."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_VIEWS = 4
SPATIALIQ_TASK_NAMES = [f"task{i}" for i in range(1, 12)]
SPATIALIQ_ALL_TASKS = set(SPATIALIQ_TASK_NAMES) | {"main"}
CROSS_VIEW_TASKS = {"task10", "task11"}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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
    """Parse --tasks string (e.g. '1,2,10,main') into set of task names."""
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
    """Load task prompts from a JSON file with {VARIABLE} substitution.
    Falls back to hardcoded prompts when prompts_file is None.
    """
    if not prompts_file:
        return dict(TASK_PROMPTS), MAIN_TASK_PROMPT
    with open(prompts_file, "r") as f:
        data = json.load(f)
    variables = {k: v for k, v in data.items() if k != "task_prompts"}
    raw_prompts = data.get("task_prompts", {})
    resolved = {k: v.format_map(variables) for k, v in raw_prompts.items()}
    main_prompt = resolved.pop("main", None)
    if main_prompt is None:
        raise SystemExit("--prompts JSON is missing 'main' key under 'task_prompts'.")
    critical = variables.get("CRITICAL_INSTRUCTION", "")
    if critical and main_prompt.startswith(critical):
        main_prompt = main_prompt[len(critical):]
    return resolved, main_prompt


def _extract_last_integer(text: str) -> Optional[str]:
    """Extract the last standalone integer from CoT reasoning text."""
    matches = re.findall(r"\b(\d+)\b", text)
    return matches[-1] if matches else None


def _validate_image_path(path: str) -> None:
    """Refuse to load ground-truth images."""
    if "_gt" in os.path.basename(path):
        raise ValueError(f"Refusing to use ground-truth image as model input: {path}")


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------


def load_hunyuan_model(
    model_path: str,
    *,
    attn_impl: str = "sdpa",
    moe_impl: str = "eager",
) -> Any:
    """Load HunyuanImage 3.0 Instruct-Distil via HF AutoModelForCausalLM.
    Uses device_map='auto' to shard across all visible GPUs.
    """
    from transformers import AutoModelForCausalLM

    logger.info("Loading HunyuanImage from: %s", model_path)
    t0 = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
        moe_impl=moe_impl,
        moe_drop_tokens=True,
    )
    model.load_tokenizer(model_path)

    elapsed = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs", elapsed)
    return model


def run_hunyuan_inference(
    model: Any,
    image_paths: List[str],
    prompt: str,
    *,
    seed: int = 42,
    diff_steps: int = 8,
    bot_task: str = "think_recaption",
) -> Tuple[Optional[str], Any]:
    """Run HunyuanImage instruct: image(s) + prompt -> (cot_text, output_image).

    Returns (cot_text_str_or_None, PIL.Image_or_None).
    """
    for p in image_paths:
        _validate_image_path(p)

    cot_text, samples = model.generate_image(
        prompt=prompt,
        image=image_paths,
        seed=seed,
        image_size="auto",
        use_system_prompt="en_unified",
        bot_task=bot_task,
        infer_align_image_size=True,
        diff_infer_steps=diff_steps,
        verbose=0,
    )

    cot_str = None
    if cot_text:
        cot_str = "\n".join(cot_text) if isinstance(cot_text, list) else str(cot_text)

    out_image = samples[0] if samples else None
    return cot_str, out_image


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_spatialiq_benchmark(
    views: List[Tuple[int, str]],
    model: Any,
    *,
    label: str = "",
    tasks: Optional[set] = None,
    override: bool = False,
    seed: int = 42,
    diff_steps: int = 8,
    bot_task: str = "think_recaption",
    task_prompts: Optional[dict] = None,
    main_task_prompt: Optional[str] = None,
) -> None:
    """Run SpatialIQ image tasks on the given views using HunyuanImage.

    Tasks 1-9: single view -> edited image.
    Tasks 10-11: two views (cross-view) -> edited image.
    Main: single view -> image with count + text prediction from CoT.
    """
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

    print(f"SpatialIQ benchmark (HunyuanImage): {label}")
    print(f"  Views: {view_indices} ({len(views)} total)")
    print(f"  Tasks: {sorted(run_tasks)}")
    print(f"  Diff steps: {diff_steps}, bot_task: {bot_task}\n")

    # Single-view tasks (task1-task9) and main per view
    for v_idx, img_path, out_dir in zip(view_indices, image_paths, output_dirs):
        print(f"--- View {v_idx} ---")

        for task_name in run_tasks_single:
            out_path = os.path.join(out_dir, f"{task_name}_pred_image_{MODEL_SLUG}.png")
            print(f"  {task_name} -> {os.path.basename(out_path)}")

            if not override and os.path.isfile(out_path):
                print(f"    (exists, skipped)")
                continue

            t0 = time.monotonic()
            try:
                cot, out_img = run_hunyuan_inference(
                    model, [img_path], task_prompts[task_name],
                    seed=seed, diff_steps=diff_steps, bot_task=bot_task,
                )
                elapsed = time.monotonic() - t0
                if out_img is not None:
                    out_img.save(out_path)
                    print(f"    Saved. ({elapsed:.1f}s)")
                    if cot:
                        logger.debug("CoT (%s): %s", task_name, cot[:200])
                else:
                    print(f"    No image generated. ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"    Error: {e} ({elapsed:.1f}s)")

        # Main task: image with count + text extraction from CoT
        if run_main:
            main_img_path = os.path.join(out_dir, f"task_main_pred_image_{MODEL_SLUG}.png")
            print(f"  main -> {os.path.basename(main_img_path)}")

            if not override and os.path.isfile(main_img_path):
                print(f"    (exists, skipped)")
            else:
                t0 = time.monotonic()
                try:
                    cot, out_img = run_hunyuan_inference(
                        model, [img_path], main_task_prompt,
                        seed=seed, diff_steps=diff_steps, bot_task=bot_task,
                    )
                    elapsed = time.monotonic() - t0

                    if out_img is not None:
                        out_img.save(main_img_path)
                        print(f"    Image saved. ({elapsed:.1f}s)")
                        if cot:
                            logger.debug("CoT (main): %s", cot[:200])
                    else:
                        print(f"    No image generated. ({elapsed:.1f}s)")

                except Exception as e:
                    elapsed = time.monotonic() - t0
                    print(f"    Error: {e} ({elapsed:.1f}s)")
        print()

    # Cross-view tasks: task10 (view n vs n+1) and task11 (view n vs n+2)
    for cv_task, step in [("task10", 1), ("task11", 2)]:
        if cv_task not in run_tasks:
            continue
        prompt_cv = task_prompts.get(cv_task, TASK_PROMPTS.get(cv_task, ""))
        if not prompt_cv:
            continue

        for v_idx, img_path, out_dir in zip(view_indices, image_paths, output_dirs):
            paired_path = _get_paired_view_path(img_path, step)
            out_path = os.path.join(out_dir, f"{cv_task}_pred_image_{MODEL_SLUG}.png")

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
                    cot, out_img = run_hunyuan_inference(
                        model, [img_path, paired_path], prompt_cv,
                        seed=seed, diff_steps=diff_steps, bot_task=bot_task,
                    )
                    elapsed = time.monotonic() - t0
                    if out_img is not None:
                        out_img.save(out_path)
                        print(f"  Saved. ({elapsed:.1f}s)")
                    else:
                        print(f"  No image generated. ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.monotonic() - t0
                    print(f"  Error: {e} ({elapsed:.1f}s)")
            print()

    print("SpatialIQ benchmark done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HunyuanImage 3.0 Instruct-Distil on SpatialIQ image tasks (local model)."
    )
    parser.add_argument(
        "folder",
        help="Sample folder, top-level folder, or a .txt file with one folder path per line.",
    )
    _default_model_path = os.environ.get(
        "HUNYUAN_MODEL_PATH",
        os.path.join(
            os.environ.get("SPATIALIQ_INFERENCE_DIR",
                           "spatialiq-inference"),
            "weights", "HunyuanImage-3-Instruct-Distil",
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=_default_model_path,
        metavar="DIR",
        help="Path to HunyuanImage-3-Instruct-Distil weights "
             "(default: spatialiq-inference/weights/HunyuanImage-3-Instruct-Distil "
             "or HUNYUAN_MODEL_PATH env var).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated tasks (default: all). E.g. 1,2,main or task1,task5,task10,main.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite existing prediction files.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a JSON file with task prompts (default: hardcoded prompts). "
             "E.g. inference/prompt_image_Type1.json.",
    )
    parser.add_argument(
        "--diff-steps",
        type=int,
        default=int(os.environ.get("HUNYUAN_STEPS", "8")),
        metavar="N",
        help="Diffusion inference steps (default: 8 for Distil, use 50 for full variant).",
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        default="think_recaption",
        choices=["think_recaption", "recaption", "image"],
        help="HunyuanImage bot_task mode (default: think_recaption = CoT + generate).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2"],
        help="Attention implementation (default: sdpa).",
    )
    parser.add_argument(
        "--moe-impl",
        type=str,
        default="eager",
        choices=["eager", "flashinfer"],
        help="MoE implementation (default: eager; flashinfer is ~3x faster if installed).",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        raise SystemExit(
            f"Model weights not found: {model_path}\n"
            "Download (~160 GB) from the spatialiq-inference repo:\n"
            "  cd spatialiq-inference\n"
            "  source env.sh\n"
            "  sbatch jobs/download_hunyuan_instruct.sh\n"
            "Or manually:\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('tencent/HunyuanImage-3.0-Instruct-Distil', "
            f"local_dir='{model_path}')\""
        )

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
        print(f"Loaded prompts from: {args.prompts}")

    samples = _scan_folders_cached(args.folder, folders)
    if not samples:
        raise SystemExit("No view.png files found in any of the specified folder(s).")

    print(f"Model: {model_path}")
    print(f"Diff steps: {args.diff_steps}, bot_task: {args.bot_task}")
    print(f"Attn: {args.attn_impl}, MoE: {args.moe_impl}")
    print(f"Seed: {args.seed}\n")

    model = load_hunyuan_model(
        model_path,
        attn_impl=args.attn_impl,
        moe_impl=args.moe_impl,
    )

    for i, (label, views) in enumerate(samples, 1):
        if len(samples) > 1:
            print(f"========== Sample {i}/{len(samples)}: {label} ==========")
        run_spatialiq_benchmark(
            views,
            model,
            label=label,
            tasks=tasks_set,
            override=args.override,
            seed=args.seed,
            diff_steps=args.diff_steps,
            bot_task=args.bot_task,
            task_prompts=task_prompts,
            main_task_prompt=main_task_prompt,
        )
        if len(samples) > 1:
            print()


if __name__ == "__main__":
    main()
