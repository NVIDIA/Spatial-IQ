#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Universal reduced-option MCQ inference script for all SpatialIQ models.

Drops incorrect options so the model sees only N choices (default 3).
The correct option is always among them.  The predicted letter is mapped
back to the ORIGINAL A-E space so the standard evaluation script works unchanged.

Supported backends:
  nvidia_api  — Claude, Gemini, GPT, Kimi (via NVIDIA Inference API)
  vllm        — Qwen, Qwen3B, GLM (via local vLLM OpenAI-compatible server)

Usage examples:
  # 3-option (default)
  python inference/inference_mcq_NCQ.py folders.txt --model claude --api-key 1
  python inference/inference_mcq_NCQ.py folders.txt --model qwen

  # 4-option
  python inference/inference_mcq_NCQ.py folders.txt --model claude --api-key 1 --num-options 4
  python inference/inference_mcq_NCQ.py folders.txt --model qwen --num-options 4

Evaluate with:
  python evaluation/evaluation_mcq.py folders.txt --pred-slug <model>3CQ   # 3-option
  python evaluation/evaluation_mcq.py folders.txt --pred-slug <model>4CQ   # 4-option
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_utils import resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_RESULTS_DIR = os.path.join(REPO_ROOT, "inference_results")
API_KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.txt")
PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_LETTERS = ("A", "B", "C", "D", "E")
LETTERS_BY_N = {3: ("A", "B", "C"), 4: ("A", "B", "C", "D")}

# ─── Model registry ──────────────────────────────────────────────────────────

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "claude": {
        "backend": "nvidia_api",
        "model_id": "claude-opus-4-6",
        "timeout": 120,
        "stream": False,
    },
    "gemini": {
        "backend": "nvidia_api",
        "model_id": "gemini-3-pro-image-preview",
        "timeout": 120,
        "stream": False,
    },
    "gpt": {
        "backend": "nvidia_api",
        "model_id": "gpt-5.2",
        "timeout": 120,
        "stream": False,
    },
    "kimi": {
        "backend": "nvidia_api",
        "model_id": "kimi-k2.5",
        "timeout": 300,
        "stream": True,
        "extra_json": {"chat_template_kwargs": {"thinking": False}},
    },
    "qwen": {
        "backend": "vllm",
        "model_id": "qwen3.5-27b",
        "model_env": "QWEN_MODEL",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "qwen3b": {
        "backend": "vllm",
        "model_id": "qwen2.5-vl-3b",
        "model_env": "QWEN_MODEL",
        "extra_body": {},
    },
    "glm": {
        "backend": "vllm",
        "model_id": "glm-edge-v",
        "model_env": "GLM_MODEL",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
}

# ─── Fallback prompts (default N=3 wording) ───────────────────────────────────

DEFINITIONS_BLOCK = """
DEFINITIONS:
1. A COLUMN is a vertical stack of one or more blocks, where each block is directly supported by the block immediately beneath it. A single block also counts as a column.
2. A LAYER is a horizontal group of one or more blocks at the same vertical height. A single block also counts as a layer.
3. A VISIBLE BLOCK is a block with at least one face fully or partially visible.
4. A CLUSTER is a group of one or more blocks that are connected through face adjacency, possibly across multiple layers. Two blocks are considered connected if they share a face (i.e., are laterally adjacent in the 3D scene); diagonal contact does not count. A cluster includes all blocks that are connected directly or indirectly through such face contacts.
5. A SUPPORTING BLOCK is any block that is beneath a given block in the same column. A DIRECTLY SUPPORTING BLOCK is the supporting block in immediate contact beneath a given block. If a block is directly supported by the ground, then it has no supporting blocks.
6. A HIDDEN BLOCK is a block with no visible faces in the image. Any hidden block must be a supporting block of at least one visible block. Otherwise, such a block does not exist in a valid structure.
"""

_CI_MCQ = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. You will receive exactly 4 images in sequence:
- Image 1 is the REFERENCE structure.
- Images 2, 3, and 4 are the MULTIPLE-CHOICE OPTIONS corresponding to choices A, B, and C respectively.
Select the option image that correctly answers the question. Output ONLY a single capital letter (A, B, or C) corresponding to the correct option. Do not include full sentences, explanations, pleasantries, or context. Do not describe your reasoning, steps, or analysis.
"""

_CI_T10 = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. You will receive exactly 5 images in sequence:
- Image 1 is VIEW A of a block structure.
- Image 2 is VIEW B of the exact same block structure.
- Images 3, 4, and 5 are the MULTIPLE-CHOICE OPTIONS corresponding to choices A, B, and C respectively.
Select the option image that correctly answers the question. Output ONLY a single capital letter (A, B, or C) corresponding to the correct option. Do not include full sentences, explanations, pleasantries, or context. Do not describe your reasoning, steps, or analysis.
"""

_CI_MAIN = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. You will receive exactly 1 image of a block structure.
You are also given 3 MULTIPLE-CHOICE OPTIONS (A, B, C) as integer values below.
Select the option that correctly answers the question. Output ONLY a single capital letter (A, B, or C) corresponding to the correct option. Do not include full sentences, explanations, pleasantries, or context. Do not describe your reasoning, steps, or analysis.
"""

_Q = "Your output should be ONLY a single capital letter (A, B, or C)."
TASK_PROMPTS = {
    "task1":  _CI_MCQ + f"QUESTION: Which option (A-C) correctly depicts the object(s) being counted in Image 1? {_Q}",
    "task2":  _CI_MCQ + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of blocks in the top-most layer of Image 1? {_Q}",
    "task3":  _CI_MCQ + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of columns in the block structure of Image 1? {_Q}",
    "task4":  _CI_MCQ + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of layers in the block structure of Image 1? {_Q}",
    "task5":  _CI_MCQ + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of visible blocks in the block structure of Image 1? {_Q}",
    "task6":  _CI_MCQ + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of VISIBLE blocks directly supporting the block(s) in the top-most layer of Image 1? {_Q}",
    "task7":  _CI_MCQ + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of VISIBLE blocks in the column(s) supporting the block(s) in the top-most layer of Image 1 (excluding the top-most blocks themselves)? {_Q}",
    "task8":  _CI_MCQ + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of HIDDEN blocks that are directly supporting visible block(s) in Image 1? {_Q}",
    "task9":  _CI_MCQ + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of distinct clusters in the block structure of Image 1? {_Q}",
    "task10": _CI_T10 + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of blocks that are visible in VIEW B (Image 2) but are NOT visible in VIEW A (Image 1)? {_Q}",
    "task11": _CI_T10 + DEFINITIONS_BLOCK + f"QUESTION: Which option (A-C) correctly represents the number of blocks that are visible in VIEW B (Image 2) but are NOT visible in VIEW A (Image 1)? {_Q}",
}
MAIN_TASK_PROMPT = _CI_MAIN + DEFINITIONS_BLOCK + f"""QUESTION: Which option (A-C) correctly represents the TOTAL number of blocks present in the block structure of Image 1?

OPTIONS:
{{choices}}

{_Q}"""


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _get_output_dir(data_dir: str) -> str:
    m = re.search(r'.*/(?=dataset/)', data_dir)
    if m:
        rel = data_dir[m.end():]
        out = os.path.join(INFERENCE_RESULTS_DIR, rel)
        os.makedirs(out, exist_ok=True)
        return out
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _image_to_base64_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _mcq_choice_paths(view_dir: str, task_id: str) -> Dict[str, str]:
    return {L: os.path.join(view_dir, f"{task_id}_mcq_choice{L}.png") for L in ALL_LETTERS}


def _load_gt_mcq(view_dir: str, task_key: str) -> Optional[Dict]:
    gt_path = os.path.join(view_dir, "gt.json")
    if not os.path.isfile(gt_path):
        return None
    try:
        with open(gt_path, "r") as f:
            gt = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    json_key = f"task_{task_key}" if task_key == "main" else task_key
    return (gt.get("mcq") or {}).get(json_key)


def _main_mcq_choice_values(view_dir: str) -> dict:
    gt_path = os.path.join(view_dir, "gt.json")
    if os.path.isfile(gt_path):
        with open(gt_path, "r") as f:
            gt = json.load(f)
        raw = (gt.get("mcq") or {}).get("task_main", {}).get("choices", {})
        if raw:
            return {k: str(v) for k, v in raw.items()}
    return {}


def _deterministic_seed(view_dir: str, task_name: str, num_options: int) -> int:
    h = hashlib.sha256(f"{view_dir}|{task_name}|mcq{num_options}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# Required wrong-answer types per task. When generating a reduced MCQ, the kept
# wrong options must include (one option of) each listed type when that type is
# actually present in the sample's choice_types. If more than (num_options - 1)
# required types are present, a random subset is chosen (e.g., 3CQ = 2 of 3).
# Tasks not listed (task7/8/9 and task_main) keep the original random behavior.
REQUIRED_WRONG_TYPES: Dict[str, Tuple[str, ...]] = {
    "task1":  ("phantom_block_above_top",),
    "task2":  ("phantom_block_above_top",),
    "task3":  ("merge_two_columns", "same_column_two_colors"),
    "task4":  ("missing_one_layer", "same_layer_two_colors", "far_layer_confusion"),
    "task5":  ("merge_adjacent_same_color", "drop_half_visible_blocks"),
    "task6":  ("empty_support",),
    "task10": ("random_single_block", "both_views_blocks", "random_wrong_highlight_blocks"),
    "task11": ("random_single_block", "both_views_blocks", "random_wrong_highlight_blocks"),
}

_FALLBACK_SUFFIX_RE = re.compile(r"_fallback_unique_\d+$")


def _normalize_choice_type(ctype: str) -> str:
    return _FALLBACK_SUFFIX_RE.sub("", ctype or "")


def _select_n_options(
    correct_letter: str,
    rng: random.Random,
    num_options: int,
    choice_types: Optional[Dict[str, str]] = None,
    required_wrong_types: Tuple[str, ...] = (),
) -> Tuple[List[str], Dict[str, str]]:
    reduced = LETTERS_BY_N[num_options]
    n_wrong_needed = num_options - 1
    all_wrong = [L for L in ALL_LETTERS if L != correct_letter]

    chosen_wrong: List[str] = []

    if choice_types and required_wrong_types:
        # Group available wrong options by normalized type.
        type_to_options: Dict[str, List[str]] = {}
        for L in all_wrong:
            base = _normalize_choice_type(choice_types.get(L, ""))
            if base and base != "correct":
                type_to_options.setdefault(base, []).append(L)

        available_required = [t for t in required_wrong_types if t in type_to_options]

        if len(available_required) > n_wrong_needed:
            selected_types = rng.sample(available_required, n_wrong_needed)
        else:
            selected_types = list(available_required)

        used: set = set()
        for t in selected_types:
            opts = [o for o in type_to_options[t] if o not in used]
            if not opts:
                continue
            pick = rng.choice(opts)
            chosen_wrong.append(pick)
            used.add(pick)

        remaining_pool = [L for L in all_wrong if L not in used]
        n_remaining = n_wrong_needed - len(chosen_wrong)
        if n_remaining > 0 and remaining_pool:
            n_take = min(n_remaining, len(remaining_pool))
            chosen_wrong.extend(rng.sample(remaining_pool, n_take))
    else:
        chosen_wrong = rng.sample(all_wrong, n_wrong_needed)

    kept = [correct_letter] + chosen_wrong
    rng.shuffle(kept)
    new_to_orig = {reduced[i]: kept[i] for i in range(num_options)}
    return kept, new_to_orig


def _map_answer_back(model_answer: str, new_to_orig: Dict[str, str], num_options: int) -> Optional[str]:
    max_letter = chr(ord("A") + num_options - 1)
    pattern_chars = f"A-{max_letter}a-{max_letter.lower()}"
    cleaned = re.sub(f"[^{pattern_chars}]", "", model_answer)
    if len(cleaned) == 1:
        return new_to_orig.get(cleaned.upper())
    matches = re.findall(rf"\b([A-{max_letter}])\b", model_answer)
    if matches:
        return new_to_orig.get(matches[-1])
    return None


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


def resolve_api_key(key_arg: str) -> str:
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


# ─── API backends ─────────────────────────────────────────────────────────────

def _nvidia_api_call(
    image_paths: List[str], prompt: str, *,
    api_key: str, model_id: str,
    timeout: int = 120, stream: bool = False,
    extra_json: Optional[Dict] = None,
) -> str:
    import requests as _requests

    content = []
    for ip in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_base64_data_url(ip)}})
    content.append({"type": "text", "text": prompt})

    body: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
    }
    if stream:
        body["stream"] = True
    if extra_json:
        body.update(extra_json)

    resp = _requests.post(
        os.environ.get("INFERENCE_API_BASE", ""),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
        stream=stream,
    )
    if resp.status_code == 401:
        raise SystemExit("Error: API returned HTTP 401 Unauthorized. Check your API key.")
    resp.raise_for_status()

    if stream:
        parts = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            token = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if token:
                parts.append(token)
        return "".join(parts).strip()
    else:
        data = resp.json()
        text = (data.get("choices", [{}])[0].get("message") or {}).get("content", "")
        return text.strip() if isinstance(text, str) else ""


def _vllm_api_call(
    image_paths: List[str], prompt: str, *,
    client, model_id: str,
    sampling: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict] = None,
) -> str:
    content = []
    for ip in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_base64_data_url(ip)}})
    content.append({"type": "text", "text": prompt})
    kw: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
    }
    if extra_body:
        kw["extra_body"] = extra_body
    if sampling:
        kw.update(sampling)
    resp = client.chat.completions.create(**kw)
    text = resp.choices[0].message.content or ""
    return text.strip()


# ─── Core benchmark logic ────────────────────────────────────────────────────

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


def _pred_filename(task_name: str, pred_slug: str) -> str:
    if task_name == "main":
        return f"task_main_pred_mcq_{pred_slug}.txt"
    return f"{task_name}_pred_mcq_{pred_slug}.txt"


def _run_image_task(
    task_name: str, img_path: str, data_dir: str, out_dir: str,
    api_fn, task_prompts: dict, override: bool, pred_slug: str,
    num_options: int = 3,
    extra_ref_images: Optional[List[str]] = None,
) -> None:
    out_path = os.path.join(out_dir, _pred_filename(task_name, pred_slug))
    print(f"  {task_name} -> {os.path.basename(out_path)}")

    if not override and os.path.isfile(out_path):
        print(f"    (exists, skipped)")
        return

    mcq_data = _load_gt_mcq(data_dir, task_name)
    if mcq_data is None:
        print(f"    (skipped: no mcq GT in gt.json for {task_name})")
        return
    correct_letter = mcq_data.get("correct_letter")
    if not correct_letter or correct_letter not in ALL_LETTERS:
        print(f"    (skipped: invalid correct_letter in gt.json)")
        return

    all_choice_paths = _mcq_choice_paths(data_dir, task_name)
    missing = [L for L, p in all_choice_paths.items() if not os.path.isfile(p)]
    if missing:
        print(f"    (skipped: missing choice image(s) for {', '.join(missing)})")
        return

    rng = random.Random(_deterministic_seed(data_dir, task_name, num_options))
    kept_originals, new_to_orig = _select_n_options(
        correct_letter, rng, num_options,
        choice_types=mcq_data.get("choice_types"),
        required_wrong_types=REQUIRED_WRONG_TYPES.get(task_name, ()),
    )

    ref_images = [img_path] + (extra_ref_images or [])
    choice_images = [all_choice_paths[orig_L] for orig_L in kept_originals]
    all_images = ref_images + choice_images

    prompt = task_prompts.get(task_name, TASK_PROMPTS.get(task_name, ""))
    t0 = time.monotonic()
    try:
        raw_answer = api_fn(all_images, prompt)
        elapsed = time.monotonic() - t0

        orig_letter = _map_answer_back(raw_answer, new_to_orig, num_options)
        if orig_letter:
            with open(out_path, "w") as f:
                f.write(orig_letter)
            mapping_str = " ".join(f"{n}->{o}" for n, o in sorted(new_to_orig.items()))
            print(f"    raw={raw_answer.strip()!r} -> {orig_letter} [{mapping_str}] ({elapsed:.1f}s)")
        else:
            fallback = random.choice(list(ALL_LETTERS))
            with open(out_path, "w") as f:
                f.write(fallback)
            print(f"    raw={raw_answer.strip()!r} -> (unparseable, random={fallback}) ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"    Error: {e} ({elapsed:.1f}s)")


def run_benchmark(
    views: List[Tuple[int, str]], api_fn, label: str,
    pred_slug: str, num_options: int = 3,
    tasks: set | None = None, override: bool = False,
    task_prompts: Optional[dict] = None,
    main_task_prompt: Optional[str] = None,
) -> None:
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

    reduced = LETTERS_BY_N[num_options]
    print(f"SpatialIQ benchmark (MCQ {num_options}-option): {label}")
    print(f"  Views: {view_indices} ({len(views)} total)")
    print(f"  Tasks: {sorted(run_tasks)}")
    print(f"  Pred slug: {pred_slug}\n")

    for v_idx, img_path, data_dir, out_dir in zip(view_indices, image_paths, data_dirs, output_dirs):
        print(f"--- View {v_idx} ---")

        for task_name in run_tasks_single:
            _run_image_task(
                task_name, img_path, data_dir, out_dir,
                api_fn, task_prompts, override, pred_slug,
                num_options=num_options,
            )

        if run_main:
            out_path = os.path.join(out_dir, _pred_filename("main", pred_slug))
            print(f"  main -> {os.path.basename(out_path)}")
            if not override and os.path.isfile(out_path):
                try:
                    with open(out_path) as f:
                        existing = f.read().strip() or "(empty)"
                except Exception:
                    existing = "?"
                print(f"    (exists: {existing}, skipped)")
            else:
                mcq_data = _load_gt_mcq(data_dir, "main")
                choice_vals = _main_mcq_choice_values(data_dir)
                correct_letter = (mcq_data or {}).get("correct_letter")

                if len(choice_vals) < 5 or not correct_letter or correct_letter not in ALL_LETTERS:
                    print(f"    (skipped: incomplete GT data for main)")
                else:
                    rng = random.Random(_deterministic_seed(data_dir, "main", num_options))
                    kept_originals, new_to_orig = _select_n_options(
                        correct_letter, rng, num_options,
                        choice_types=(mcq_data or {}).get("choice_types"),
                        required_wrong_types=REQUIRED_WRONG_TYPES.get("main", ()),
                    )
                    choices_str = "\n".join(
                        f"  {reduced[i]}: {choice_vals[kept_originals[i]]}"
                        for i in range(num_options)
                    )
                    prompt = main_task_prompt.format(choices=choices_str)
                    t0 = time.monotonic()
                    try:
                        raw_answer = api_fn([img_path], prompt)
                        elapsed = time.monotonic() - t0
                        orig_letter = _map_answer_back(raw_answer, new_to_orig, num_options)
                        if orig_letter:
                            with open(out_path, "w") as f:
                                f.write(orig_letter)
                            mapping_str = " ".join(f"{n}->{o}" for n, o in sorted(new_to_orig.items()))
                            print(f"    raw={raw_answer.strip()!r} -> {orig_letter} [{mapping_str}] ({elapsed:.1f}s)")
                        else:
                            fallback = random.choice(list(ALL_LETTERS))
                            with open(out_path, "w") as f:
                                f.write(fallback)
                            print(f"    raw={raw_answer.strip()!r} -> (unparseable, random={fallback}) ({elapsed:.1f}s)")
                    except Exception as e:
                        elapsed = time.monotonic() - t0
                        print(f"    Error: {e} ({elapsed:.1f}s)")
        print()

    for cv_task, step in [("task10", 1), ("task11", 2)]:
        if cv_task not in run_tasks:
            continue
        for v_idx, img_path, data_dir, out_dir in zip(view_indices, image_paths, data_dirs, output_dirs):
            paired_path = _get_paired_view_path(img_path, step)
            if paired_path is None:
                print(f"--- {cv_task} (view {v_idx}) -> (skipped: cannot parse view index) ---")
                continue
            if not os.path.isfile(paired_path):
                print(f"--- {cv_task} (view {v_idx}) -> (skipped: paired view not found) ---")
                continue
            paired_idx = ((v_idx - 1 + step) % NUM_VIEWS) + 1
            print(f"--- {cv_task} (view {v_idx} + view {paired_idx}) ---")
            _run_image_task(
                cv_task, img_path, data_dir, out_dir,
                api_fn, task_prompts, override, pred_slug,
                num_options=num_options,
                extra_ref_images=[paired_path],
            )
            print()

    print(f"SpatialIQ {num_options}-option MCQ benchmark done.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    models_str = ", ".join(sorted(MODEL_CONFIGS))
    parser = argparse.ArgumentParser(
        description="Universal reduced-option (NCQ) MCQ inference (all models).",
    )
    parser.add_argument("folder", help="Sample folder, top-level folder, or .txt file.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS), help=f"Model to use: {models_str}.")
    parser.add_argument("--num-options", type=int, default=3, choices=[3, 4], dest="num_options", help="Number of MCQ options (default: 3).")
    parser.add_argument("--tasks", type=str, default=None, metavar="LIST", help="Comma-separated tasks (default: all).")
    parser.add_argument("--override", action="store_true", help="Overwrite existing prediction files.")
    parser.add_argument("--prompts", type=str, default=None, metavar="FILE", help="Prompts JSON (default: auto-selected by --num-options).")
    parser.add_argument("--reverse", action="store_true", help="Process samples in reverse order.")
    parser.add_argument("--start", type=int, default=1, metavar="N", help="1-indexed sample to start from (default: 1).")

    # NVIDIA API options
    nvidia = parser.add_argument_group("NVIDIA API options (claude, gemini, gpt, kimi)")
    nvidia.add_argument("--api-key", type=str, default=None, metavar="KEY", help="API key or line number in api_keys.txt.")

    # vLLM options
    vllm = parser.add_argument_group("vLLM options (qwen, qwen3b, glm)")
    vllm.add_argument("--base-url", type=str, default=None, metavar="URL", help="vLLM base URL (default: localhost:8000).")
    vllm.add_argument("--temperature", type=float, default=None, metavar="T")
    vllm.add_argument("--max-tokens", type=int, default=None, dest="max_tokens", metavar="N")
    vllm.add_argument("--top-p", type=float, default=None, dest="top_p", metavar="P")

    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    backend = cfg["backend"]
    model_id = cfg["model_id"]
    num_options = args.num_options
    pred_slug = f"{args.model}{num_options}CQ"
    prompts_file = args.prompts or os.path.join(PROMPTS_DIR, f"prompt_mcq{num_options}options_Type1.json")

    if backend == "nvidia_api":
        raw_key = args.api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("INFERENCE_API_KEY")
        if not raw_key:
            raise SystemExit("--api-key required for NVIDIA API models (or set NVIDIA_API_KEY env var).")
        api_key = resolve_api_key(raw_key)

        def api_fn(image_paths: List[str], prompt: str) -> str:
            return _nvidia_api_call(
                image_paths, prompt,
                api_key=api_key, model_id=model_id,
                timeout=cfg.get("timeout", 120),
                stream=cfg.get("stream", False),
                extra_json=cfg.get("extra_json"),
            )

        print(f"Backend: NVIDIA Inference API")
        print(f"Model: {model_id}")

    elif backend == "vllm":
        from openai import OpenAI

        env_model = cfg.get("model_env", "")
        if env_model and os.environ.get(env_model):
            model_id = os.environ[env_model]

        base_url = (args.base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1").strip()
        temperature = args.temperature if args.temperature is not None else float(os.environ.get("QWEN_TEMPERATURE", os.environ.get("GLM_TEMPERATURE", "0")))
        max_tokens = args.max_tokens if args.max_tokens is not None else int(os.environ.get("QWEN_MAX_TOKENS", os.environ.get("GLM_MAX_TOKENS", "32")))
        sampling: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if args.top_p is not None:
            sampling["top_p"] = args.top_p

        client = OpenAI(base_url=base_url, api_key="EMPTY")
        extra_body = cfg.get("extra_body", {})

        def api_fn(image_paths: List[str], prompt: str) -> str:
            return _vllm_api_call(
                image_paths, prompt,
                client=client, model_id=model_id,
                sampling=sampling, extra_body=extra_body,
            )

        print(f"Backend: vLLM ({base_url})")
        print(f"Model: {model_id}")
        print(f"Sampling: {sampling}")

    else:
        raise SystemExit(f"Unknown backend: {backend}")

    print(f"Prediction slug: {pred_slug}  (evaluate with --pred-slug {pred_slug})\n")

    folders = _resolve_folders(args.folder)
    try:
        tasks_set = parse_tasks_arg(args.tasks or "")
    except ValueError as e:
        raise SystemExit(f"--tasks: {e}")
    try:
        task_prompts, main_task_prompt = load_prompts(prompts_file)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"--prompts: {e}")
    print(f"Loaded prompts from: {prompts_file}\n")

    samples = _scan_folders_cached(args.folder, folders)
    if not samples:
        raise SystemExit("No view.png files found in any of the specified folder(s).")
    if args.reverse:
        samples = list(reversed(samples))
    if args.start > 1:
        samples = samples[args.start - 1:]
    total = len(samples)
    qualifiers = []
    if args.reverse:
        qualifiers.append("reversed")
    if args.start > 1:
        qualifiers.append(f"starting at {args.start}")
    if qualifiers:
        print(f"  ({', '.join(qualifiers)}, {total} samples)\n")
    for i, (label, views) in enumerate(samples, 1):
        if total > 1:
            print(f"========== Sample {i}/{total}: {label} ==========")
        run_benchmark(
            views, api_fn, label=label, pred_slug=pred_slug,
            num_options=num_options,
            tasks=tasks_set, override=args.override,
            task_prompts=task_prompts, main_task_prompt=main_task_prompt,
        )
        if total > 1:
            print()


if __name__ == "__main__":
    main()
