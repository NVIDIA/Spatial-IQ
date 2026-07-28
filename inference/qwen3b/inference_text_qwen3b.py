#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Text-output inference script for locally hosted Qwen2.5-VL-3B (via vLLM OpenAI-compatible server).
SpatialIQ tasks 1-11 and main return text (counts/answers).
Saves taskN_pred_text_<slug>.txt and task_main_pred_text_<slug>.txt per view (slug = parent folder, e.g. qwen3b).
"""

import argparse
import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from view_utils import find_and_group_views as _find_and_group_views, resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "qwen2.5-vl-3b"
MODEL_SLUG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFERENCE_RESULTS_DIR = os.path.join(REPO_ROOT, "inference_results")

### TASK PROMPTS ###

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
    "task1": CRITICAL_INSTRUCTION + """QUESTION: If the task is to count objects, what are those object(s) in the image?""",
    "task2": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many blocks are in the top-most layer? Your answer should be a single integer ONLY.""",
    "task3": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many columns are in the block structure? Your answer should be a single integer ONLY.""",
    "task4": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many layers are in the block structure? Your answer should be a single integer ONLY.""",
    "task5": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many visible blocks are there in the block structure? Your answer should be a single integer ONLY.""",
    "task6": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many VISIBLE blocks are directly supporting blocks of block(s) in the top-most layer? Your answer should be a single integer ONLY.""",
    "task7": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many VISIBLE blocks are in the column(s) supporting the block(s) in the top-most layer? Exclude the block(s) themselves that are in the top-most layer. Your answer should be a single integer ONLY.""",
    "task8": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many blocks are HIDDEN blocks that are directly supporting blocks of visible block(s)? Your answer should be a single integer ONLY.""",
    "task9": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many distinct clusters are there in the block structure? Your answer should be a single integer ONLY.""",
    "task10": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: You are given two images that are two different views of the same block structure. How many blocks are visible in the second view but are NOT visible in the first view? Your answer should be a single integer ONLY.""",
    "task11": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: You are given two images that are two different views of the same block structure. How many blocks are visible in the second view but are NOT visible in the first view? Your answer should be a single integer ONLY.""",
}

MAIN_TASK_PROMPT = CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + """QUESTION: How many TOTAL blocks are present in the block structure? Your answer should be a single integer ONLY."""


### UTILITY FUNCTIONS ###

def _get_output_dir(data_dir: str) -> str:
    """Map a dataset directory to the corresponding output directory under inference_results/."""
    m = re.search(r'.*/(?=dataset/)', data_dir)
    if m:
        rel = data_dir[m.end():]
        out = os.path.join(INFERENCE_RESULTS_DIR, rel)
        os.makedirs(out, exist_ok=True)
        return out
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def image_to_base64_data_url(path: str) -> str:
    """Read a PNG image and return a data URL. Refuses paths containing '_gt'."""
    if "_gt" in path:
        raise ValueError(f"Refusing to send ground-truth image to model: {path}")
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def call_api_image_and_text(
    image_path: str,
    prompt: str,
    client: OpenAI,
    model: str,
    sampling: Optional[Dict[str, Any]] = None,
) -> str:
    """Image + text -> text: send one image and the prompt, return the assistant message."""
    image_url = image_to_base64_data_url(image_path)
    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt},
    ]
    kw: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if sampling:
        kw.update(sampling)
    resp = client.chat.completions.create(**kw)
    text = resp.choices[0].message.content or ""
    return text.strip()


def call_api_two_images_and_text(
    image_path_1: str,
    image_path_2: str,
    prompt: str,
    client: OpenAI,
    model: str,
    sampling: Optional[Dict[str, Any]] = None,
) -> str:
    """Two images + text -> text: send both images and prompt, return the assistant message."""
    url_1 = image_to_base64_data_url(image_path_1)
    url_2 = image_to_base64_data_url(image_path_2)
    content = [
        {"type": "image_url", "image_url": {"url": url_1}},
        {"type": "image_url", "image_url": {"url": url_2}},
        {"type": "text", "text": prompt},
    ]
    kw: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if sampling:
        kw.update(sampling)
    resp = client.chat.completions.create(**kw)
    text = resp.choices[0].message.content or ""
    return text.strip()


NUM_VIEWS = 4
SPATIALIQ_TASK_NAMES = [f"task{i}" for i in range(1, 12)]  # task1 .. task11
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
    """Load task prompts from a JSON file."""
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
    return resolved, main_prompt


def run_spatialiq_benchmark(
    views: List[Tuple[int, str]],
    client: OpenAI,
    model: str,
    label: str = "",
    tasks: set | None = None,
    override: bool = False,
    sampling: Optional[Dict[str, Any]] = None,
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

    print(f"SpatialIQ benchmark (text out / Qwen2.5-3B): {label}")
    print(f"  Views: {view_indices} ({len(views)} total)")
    print(f"  Tasks: {sorted(run_tasks)}")
    print(f"  Model: {model}\n")

    for v_idx, img_path, out_dir in zip(view_indices, image_paths, output_dirs):
        print(f"--- View {v_idx} ---")
        for task_name in run_tasks_single:
            out_path = os.path.join(out_dir, f"{task_name}_pred_text_{MODEL_SLUG}.txt")
            print(f"  {task_name} -> {os.path.basename(out_path)}")
            if not override and os.path.isfile(out_path):
                print(f"    (exists, skipped)")
            else:
                t0 = time.monotonic()
                try:
                    answer = call_api_image_and_text(
                        img_path, task_prompts[task_name], client, model, sampling=sampling
                    )
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
                try:
                    with open(main_path) as f:
                        existing = f.read().strip() or "(empty)"
                except Exception:
                    existing = "?"
                print(f"    (exists: {existing}, skipped)")
            else:
                t0_main = time.monotonic()
                try:
                    answer = call_api_image_and_text(
                        img_path, main_task_prompt, client, model, sampling=sampling
                    )
                    elapsed = time.monotonic() - t0_main
                    if answer.strip():
                        with open(main_path, "w") as f:
                            f.write(answer.strip())
                        print(f"    {answer.strip()} ({elapsed:.1f}s)")
                    else:
                        print(f"    (empty response, not saved) ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.monotonic() - t0_main
                    print(f"    Error: {e} ({elapsed:.1f}s)")
        print()

    for cv_task, step in [("task10", 1), ("task11", 2)]:
        if cv_task not in run_tasks:
            continue
        prompt_cv = task_prompts.get(cv_task, TASK_PROMPTS.get(cv_task, ""))
        if not prompt_cv:
            continue
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
                t0_cv = time.monotonic()
                try:
                    answer = call_api_two_images_and_text(
                        img_path, paired_path, prompt_cv, client, model, sampling=sampling
                    )
                    elapsed = time.monotonic() - t0_cv
                    if answer.strip():
                        with open(out_path, "w") as f:
                            f.write(answer.strip())
                        print(f"  {answer.strip()} ({elapsed:.1f}s)")
                    else:
                        print(f"  (empty response, not saved) ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.monotonic() - t0_cv
                    print(f"  Error: {e} ({elapsed:.1f}s)")
            print()

    print("SpatialIQ benchmark done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-VL-3B via local vLLM server (text output: SpatialIQ benchmark)."
    )
    parser.add_argument(
        "folder",
        help="Sample folder, top-level folder, or a .txt file with one folder path per line.",
    )
    parser.add_argument("--tasks", type=str, default=None, metavar="LIST",
        help="Comma-separated tasks (default: all). E.g. 1,2,main or task1,task5,task10,main.")
    parser.add_argument("--override", action="store_true", help="Overwrite existing prediction files.")
    parser.add_argument("--prompts", type=str,
        default=os.path.join(REPO_ROOT, "inference", "prompt_text_Type1.json"), metavar="FILE",
        help="Path to a JSON file with task prompts (default: prompt_text_Type1.json).")
    parser.add_argument("--base-url", type=str, default=None, metavar="URL",
        help=f"vLLM server base URL (default: {DEFAULT_BASE_URL} or VLLM_BASE_URL env var).")
    parser.add_argument("--model", type=str, default=None, metavar="NAME",
        help=f"Served model name (default: {DEFAULT_MODEL} or QWEN_MODEL env var).")
    parser.add_argument("--temperature", type=float, default=None, metavar="T",
        help="Sampling temperature (default: 0).")
    parser.add_argument("--max-tokens", type=int, default=None, dest="max_tokens", metavar="N",
        help="Max new tokens per reply (default: 128).")
    parser.add_argument("--top-p", type=float, default=None, dest="top_p", metavar="P",
        help="Nucleus sampling; omit to skip.")
    args = parser.parse_args()

    base_url = (args.base_url or os.environ.get("VLLM_BASE_URL") or DEFAULT_BASE_URL).strip()
    model = (args.model or os.environ.get("QWEN_MODEL") or DEFAULT_MODEL).strip()

    temperature = (
        args.temperature if args.temperature is not None
        else float(os.environ.get("QWEN_TEMPERATURE", "0"))
    )
    max_tokens = (
        args.max_tokens if args.max_tokens is not None
        else int(os.environ.get("QWEN_MAX_TOKENS", "128"))
    )
    top_p = args.top_p
    if top_p is None and os.environ.get("QWEN_TOP_P", "").strip() != "":
        top_p = float(os.environ["QWEN_TOP_P"])
    sampling: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    if top_p is not None:
        sampling["top_p"] = top_p

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    print(f"vLLM base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Sampling: {sampling}\n")

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
    samples = _scan_folders_cached(args.folder, folders)
    if not samples:
        raise SystemExit("No view.png files found in any of the specified folder(s).")
    for i, (label, views) in enumerate(samples, 1):
        if len(samples) > 1:
            print(f"========== Sample {i}/{len(samples)}: {label} ==========")
        run_spatialiq_benchmark(
            views, client, model, label=label, tasks=tasks_set, override=args.override,
            sampling=sampling, task_prompts=task_prompts, main_task_prompt=main_task_prompt,
        )
        if len(samples) > 1:
            print()


if __name__ == "__main__":
    main()
