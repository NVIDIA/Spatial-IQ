#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
MCQ inference script for NVIDIA Inference API (GPT).
SpatialIQ tasks 1-10 and main: send reference image(s) + 5 MCQ choice images,
expect a single letter (A-E) answer.
Saves taskN_pred_mcq.txt and task_main_pred_mcq.txt per view.
"""

import argparse
import base64
import json
import os
import re
import requests
import time
from typing import List, Optional, Tuple
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from view_utils import find_and_group_views as _find_and_group_views, resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

API_BASE = os.environ.get("INFERENCE_API_BASE", "")
MODEL = "gpt-5.2"
MODEL_SLUG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
API_KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api_keys.txt")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFERENCE_RESULTS_DIR = os.path.join(REPO_ROOT, "inference_results")


def resolve_api_key(key_arg: str) -> str:
    """Resolve an API key from a line-number index or a literal key string.
    If key_arg parses as an integer N, reads line N (1-based) from api_keys.txt.
    Otherwise, treats key_arg as the literal API key.
    """
    try:
        n = int(key_arg)
    except ValueError:
        return key_arg.strip()
    try:
        with open(API_KEYS_FILE, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise SystemExit(f"api_keys.txt not found at: {API_KEYS_FILE}")
    if n < 1 or n > len(lines):
        raise SystemExit(f"API key index {n} out of range (api_keys.txt has {len(lines)} key(s)).")
    return lines[n - 1]

### TASK PROMPTS ###

DEFINITIONS_BLOCK = """
DEFINITIONS:
1. A COLUMN is a vertical stack of one or more blocks, where each block is directly supported by the block immediately beneath it. A single block also counts as a column.
2. A LAYER is a horizontal group of one or more blocks at the same vertical height. A single block also counts as a layer.
3. A VISIBLE BLOCK is a block with at least one face fully or partially visible.
4. A CLUSTER is a group of one or more blocks that are connected through face adjacency, possibly across multiple layers. Two blocks are considered connected if they share a face (i.e., are laterally adjacent in the 3D scene); diagonal contact does not count. A cluster includes all blocks that are connected directly or indirectly through such face contacts.
5. A SUPPORTING BLOCK is any block that is beneath a given block in the same column. A DIRECTLY SUPPORTING BLOCK is the supporting block in immediate contact beneath a given block. If a block is directly supported by the ground, then it has no supporting blocks.
6. A HIDDEN BLOCK is a block with no visible faces in the image. Any hidden block must be a supporting block of at least one visible block. Otherwise, such a block does not exist in a valid structure.
"""

CRITICAL_INSTRUCTION_MCQ = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. You will receive exactly 6 images in sequence:
- Image 1 is the REFERENCE structure.
- Images 2, 3, 4, 5, and 6 are the MULTIPLE-CHOICE OPTIONS corresponding to choices A, B, C, D, and E respectively.
Select the option image that correctly answers the question. Output ONLY a single capital letter (A, B, C, D, or E) corresponding to the correct option. Do not include full sentences, explanations, pleasantries, or context.
"""

CRITICAL_INSTRUCTION_TASK10 = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. You will receive exactly 7 images in sequence:
- Image 1 is VIEW A of a block structure.
- Image 2 is VIEW B of the exact same block structure.
- Images 3, 4, 5, 6, and 7 are the MULTIPLE-CHOICE OPTIONS corresponding to choices A, B, C, D, and E respectively.
Select the option image that correctly answers the question. Output ONLY a single capital letter (A, B, C, D, or E) corresponding to the correct option. Do not include full sentences, explanations, pleasantries, or context.
"""

CRITICAL_INSTRUCTION_MAIN = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. You will receive exactly 1 image of a block structure.
You are also given 5 MULTIPLE-CHOICE OPTIONS (A, B, C, D, E) as integer values below.
Select the option that correctly answers the question. Output ONLY a single capital letter (A, B, C, D, or E) corresponding to the correct option. Do not include full sentences, explanations, pleasantries, or context.
"""

TASK_PROMPTS = {
    "task1": CRITICAL_INSTRUCTION_MCQ + """QUESTION: Which option (A-E) correctly depicts the object(s) being counted in Image 1? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task2": CRITICAL_INSTRUCTION_MCQ + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of blocks in the top-most layer of Image 1? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task3": CRITICAL_INSTRUCTION_MCQ + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of columns in the block structure of Image 1? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task4": CRITICAL_INSTRUCTION_MCQ + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of layers in the block structure of Image 1? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task5": CRITICAL_INSTRUCTION_MCQ + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of visible blocks in the block structure of Image 1? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task6": CRITICAL_INSTRUCTION_MCQ + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of VISIBLE blocks directly supporting the block(s) in the top-most layer of Image 1? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task7": CRITICAL_INSTRUCTION_MCQ + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of VISIBLE blocks in the column(s) supporting the block(s) in the top-most layer of Image 1 (excluding the top-most blocks themselves)? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task8": CRITICAL_INSTRUCTION_MCQ + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of HIDDEN blocks that are directly supporting visible block(s) in Image 1? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task9": CRITICAL_INSTRUCTION_MCQ + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of distinct clusters in the block structure of Image 1? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task10": CRITICAL_INSTRUCTION_TASK10 + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of blocks that are visible in VIEW B (Image 2) but are NOT visible in VIEW A (Image 1)? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
    "task11": CRITICAL_INSTRUCTION_TASK10 + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the number of blocks that are visible in VIEW B (Image 2) but are NOT visible in VIEW A (Image 1)? Your output should be ONLY a single capital letter (A, B, C, D, or E).""",
}

MAIN_TASK_PROMPT = CRITICAL_INSTRUCTION_MAIN + DEFINITIONS_BLOCK + """QUESTION: Which option (A-E) correctly represents the TOTAL number of blocks present in the block structure of Image 1?

OPTIONS:
{choices}

Your output should be ONLY a single capital letter (A, B, C, D, or E)."""


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
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _mcq_choice_paths(view_dir: str, task_id: str) -> List[str]:
    """Return list of 5 MCQ choice image paths [A..E] for a task in a view dir."""
    return [
        os.path.join(view_dir, f"{task_id}_mcq_choice{L}.png")
        for L in ("A", "B", "C", "D", "E")
    ]


def _main_mcq_choice_values(view_dir: str) -> dict:
    """Read main task MCQ choices A-E from gt.json. Returns {letter: value_str}."""
    gt_path = os.path.join(view_dir, "gt.json")
    if os.path.isfile(gt_path):
        with open(gt_path, "r") as f:
            gt = json.load(f)
        raw = (gt.get("mcq") or {}).get("task_main", {}).get("choices", {})
        if raw:
            return {k: str(v) for k, v in raw.items()}
    return {}


def call_api_multi_image(image_paths: List[str], prompt: str, api_key: str) -> str:
    """Send multiple images + text prompt, return the assistant message."""
    content = []
    for ip in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_base64_data_url(ip)}})
    content.append({"type": "text", "text": prompt})
    resp = requests.post(
        API_BASE,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=120,
    )
    if resp.status_code == 401:
        raise SystemExit(
            "Error: API request returned HTTP 401 Unauthorized. "
            "The API key is invalid or has expired. Please check your key."
        )
    resp.raise_for_status()
    data = resp.json()
    choice = data.get("choices", [{}])[0]
    text = (choice.get("message") or {}).get("content", "")
    return text.strip() if isinstance(text, str) else ""


class _KeepMissing(dict):
    """format_map helper: leaves unknown {keys} intact (e.g. {choices} in main prompt)."""
    def __missing__(self, key):
        return "{" + key + "}"


def load_prompts(prompts_file: Optional[str]) -> Tuple[dict, str]:
    """Load task prompts from a JSON file.
    The JSON should have a "task_prompts" key containing task1..task10 and main.
    Other top-level keys (e.g. CRITICAL_INSTRUCTION_MCQ, DEFINITIONS_BLOCK) are variables
    that can be referenced inside task prompt strings as {VARIABLE_NAME}.
    The main prompt may contain a {choices} placeholder which is left intact for runtime use.
    Returns (task_prompts_dict, main_task_prompt).
    Falls back to hardcoded prompts if prompts_file is None.
    """
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


def run_spatialiq_mcq_benchmark(
    views: List[Tuple[int, str]],
    api_key: str,
    label: str = "",
    tasks: Optional[set] = None,
    override: bool = False,
    task_prompts: Optional[dict] = None,
    main_task_prompt: Optional[str] = None,
) -> None:
    """Run MCQ inference on the given views.
    Tasks 1-9: reference image + 5 choice images (6 total).
    Task 10: 2 reference images + 5 choice images (7 total).
    Main: reference image + text choices embedded in prompt.
    """
    if task_prompts is None:
        task_prompts = TASK_PROMPTS
    if main_task_prompt is None:
        main_task_prompt = MAIN_TASK_PROMPT

    view_indices = [v[0] for v in views]
    image_paths = [v[1] for v in views]
    data_dirs = [os.path.dirname(p) for p in image_paths]
    output_dirs = [_get_output_dir(d) for d in data_dirs]

    run_tasks = tasks if tasks is not None else SPATIALIQ_ALL_TASKS
    run_tasks_single = [t for t in SPATIALIQ_TASK_NAMES if t not in CROSS_VIEW_TASKS and t in run_tasks]
    run_main = "main" in run_tasks

    print(f"SpatialIQ benchmark (MCQ): {label}")
    print(f"  Views: {view_indices} ({len(views)} total)")
    print(f"  Tasks: {sorted(run_tasks)}\n")

    # Tasks 1-9 and Main per view
    for v_idx, img_path, data_dir, out_dir in zip(view_indices, image_paths, data_dirs, output_dirs):
        print(f"--- View {v_idx} ---")

        for task_name in run_tasks_single:
            out_path = os.path.join(out_dir, f"{task_name}_pred_mcq_{MODEL_SLUG}.txt")
            print(f"  {task_name} -> {os.path.basename(out_path)}")
            if not override and os.path.isfile(out_path):
                print(f"    (exists, skipped)")
                continue

            choice_imgs = _mcq_choice_paths(data_dir, task_name)
            missing = [p for p in choice_imgs if not os.path.isfile(p)]
            if missing:
                print(f"    (skipped: missing {len(missing)} choice image(s))")
                continue

            all_images = [img_path] + choice_imgs
            t0 = time.monotonic()
            try:
                answer = call_api_multi_image(all_images, task_prompts[task_name], api_key)
                elapsed = time.monotonic() - t0
                if answer.strip():
                    with open(out_path, "w") as f:
                        f.write(answer.strip())
                    print(f"    {answer.strip()} ({elapsed:.1f}s)")
                else:
                    print(f"    (empty response, not saved) ({elapsed:.1f}s)")
            except (requests.RequestException, OSError) as e:
                elapsed = time.monotonic() - t0
                print(f"    Error: {e} ({elapsed:.1f}s)")

        if run_main:
            main_path = os.path.join(out_dir, f"task_main_pred_mcq_{MODEL_SLUG}.txt")
            print(f"  main -> {os.path.basename(main_path)}")
            if not override and os.path.isfile(main_path):
                try:
                    with open(main_path) as f:
                        existing = f.read().strip() or "(empty)"
                except Exception:
                    existing = "?"
                print(f"    (exists: {existing}, skipped)")
            else:
                choice_vals = _main_mcq_choice_values(data_dir)
                if len(choice_vals) < 5:
                    print(f"    (skipped: only {len(choice_vals)}/5 choice text files found)")
                else:
                    choices_str = "\n".join(f"  {L}: {choice_vals[L]}" for L in ("A", "B", "C", "D", "E"))
                    prompt = main_task_prompt.format(choices=choices_str)
                    t0 = time.monotonic()
                    try:
                        answer = call_api_multi_image([img_path], prompt, api_key)
                        elapsed = time.monotonic() - t0
                        if answer.strip():
                            with open(main_path, "w") as f:
                                f.write(answer.strip())
                            print(f"    {answer.strip()} ({elapsed:.1f}s)")
                        else:
                            print(f"    (empty response, not saved) ({elapsed:.1f}s)")
                    except (requests.RequestException, OSError) as e:
                        elapsed = time.monotonic() - t0
                        print(f"    Error: {e} ({elapsed:.1f}s)")
        print()

    # Cross-view tasks (task10, task11): two reference images + 5 choice images
    for cv_task, step in [("task10", 1), ("task11", 2)]:
        if cv_task not in run_tasks:
            continue
        for v_idx, img_path, data_dir, out_dir in zip(view_indices, image_paths, data_dirs, output_dirs):
            paired_path = _get_paired_view_path(img_path, step)
            out_path = os.path.join(out_dir, f"{cv_task}_pred_mcq_{MODEL_SLUG}.txt")
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
                choice_imgs = _mcq_choice_paths(data_dir, cv_task)
                missing = [p for p in choice_imgs if not os.path.isfile(p)]
                if missing:
                    print(f"  (skipped: missing {len(missing)} choice image(s))")
                else:
                    all_images = [img_path, paired_path] + choice_imgs
                    t0 = time.monotonic()
                    try:
                        answer = call_api_multi_image(all_images, task_prompts.get(cv_task, TASK_PROMPTS.get(cv_task, "")), api_key)
                        elapsed = time.monotonic() - t0
                        if answer.strip():
                            with open(out_path, "w") as f:
                                f.write(answer.strip())
                            print(f"  {answer.strip()} ({elapsed:.1f}s)")
                        else:
                            print(f"  (empty response, not saved) ({elapsed:.1f}s)")
                    except (requests.RequestException, OSError) as e:
                        elapsed = time.monotonic() - t0
                        print(f"  Error: {e} ({elapsed:.1f}s)")
            print()

    print("SpatialIQ MCQ benchmark done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run GPT via NVIDIA Inference API (MCQ: send reference + choice images, get A-E answer)."
    )
    parser.add_argument(
        "folder",
        help="Sample folder, top-level folder, or a .txt file with one folder path per line.",
    )
    parser.add_argument(
        "key",
        nargs="?",
        default=None,
        metavar="N",
        help="1-based line index into inference/api_keys.txt, or a literal API key string (default: 1).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated tasks (default: all). E.g. 1,2,10,main or task1,task5,task10,main.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite existing prediction files.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=os.path.join(REPO_ROOT, "inference", "prompt_mcq_Type1.json"),
        metavar="FILE",
        help="Path to a JSON file with task prompts (default: prompt_mcq_Type1.json).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        metavar="KEY_OR_N",
        help=(
            "API key or 1-based line index into inference/api_keys.txt. "
            "Pass an integer N to use the key on line N of api_keys.txt, "
            "or pass a literal key string to use it directly. "
            "Falls back to NVIDIA_API_KEY / INFERENCE_API_KEY env vars."
        ),
    )
    args = parser.parse_args()

    key_arg = (args.key or args.api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("INFERENCE_API_KEY") or "1").strip()
    api_key = resolve_api_key(key_arg)
    if not api_key:
        raise SystemExit("Resolved API key is empty. Check api_keys.txt or the provided key.")

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
        run_spatialiq_mcq_benchmark(views, api_key, label=label, tasks=tasks_set, override=args.override, task_prompts=task_prompts, main_task_prompt=main_task_prompt)
        if len(samples) > 1:
            print()


if __name__ == "__main__":
    main()
