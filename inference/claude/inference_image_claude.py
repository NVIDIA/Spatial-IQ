#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Image-output inference script for NVIDIA Inference API (Claude Opus 4.6 via Bedrock).
For text-output SpatialIQ tasks use inference_text_claude.py instead.
Requires a folder path: run SpatialIQ benchmark (image out) on that sample folder or discover all samples under it.
Saves task1_pred_image.png .. task10_pred_image.png and task_main_pred_text.txt per view.
"""

import argparse
import base64
import json
import os
import re
import requests
import time
from typing import Any, List, Optional, Tuple
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from view_utils import find_and_group_views as _find_and_group_views, resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

API_BASE = os.environ.get("INFERENCE_API_BASE", "")
MODEL = "claude-opus-4-6"
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
    """Read a PNG image and return a data URL (data:image/png;base64,...).
    Refuses to load paths containing '_gt' so ground-truth images are never sent to the model.
    """
    if "_gt" in path:
        raise ValueError(f"Refusing to send ground-truth image to API: {path}")
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def call_api_image_and_text(image_path: str, prompt: str, api_key: str) -> str:
    """Image + text: send one image (as base64 data URL) and the prompt, return the assistant message."""
    image_url = image_to_base64_data_url(image_path)
    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt},
    ]
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
    content = (choice.get("message") or {}).get("content", "")
    return content.strip() if isinstance(content, str) else ""


def _extract_image_base64_from_response(data: dict, debug: bool = False) -> Optional[bytes]:
    """Extract raw base64 image bytes from chat completions response. Returns None if no image found."""

    def from_parts(parts: Any) -> Optional[bytes]:
        if not isinstance(parts, list):
            return None
        for part in parts:
            if not isinstance(part, dict):
                continue
            # Gemini-style: inline_data with base64 data
            inline = part.get("inline_data")
            if isinstance(inline, dict):
                b64 = inline.get("data")
                if b64:
                    return base64.standard_b64decode(b64)
            # image_url with data URL
            image_url = (part.get("image_url") or {}).get("url") or part.get("url")
            if image_url and "base64," in image_url:
                b64 = image_url.split("base64,", 1)[1].strip()
                return base64.standard_b64decode(b64)
            # inline image blob
            b64 = part.get("image") or part.get("data")
            if b64:
                return base64.standard_b64decode(b64)
        return None

    # Gemini/native format: candidates[0].content.parts
    candidates = data.get("candidates") or []
    if candidates:
        content = (candidates[0] or {}).get("content") or {}
        out = from_parts(content.get("parts"))
        if out is not None:
            return out

    # OpenAI-style: choices[0].message.content (list of parts)
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    if content is not None and isinstance(content, list):
        out = from_parts(content)
        if out is not None:
            return out

    # NVIDIA/Gemini image-out: choices[0].message.images (array of base64 or data URLs)
    images = msg.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            # Try various keys used by different APIs
            b64 = (
                first.get("data")
                or first.get("b64_json")
                or first.get("base64")
                or first.get("content")
            )
            url = first.get("url") or (first.get("image_url") or {}).get("url")
            if isinstance(url, str) and "base64," in url:
                b64 = url.split("base64,", 1)[1].strip()
            if b64 and isinstance(b64, str):
                return base64.standard_b64decode(b64)
            if isinstance(b64, bytes):
                return b64
        if isinstance(first, str):
            if "base64," in first:
                b64 = first.split("base64,", 1)[1].strip()
                return base64.standard_b64decode(b64)
            return base64.standard_b64decode(first)
        if isinstance(first, bytes):
            return first
        if debug:
            print("      [debug] message.images[0] type:", type(first).__name__)
            if isinstance(first, dict):
                print("      [debug] message.images[0] keys:", list(first.keys()))
                for k in first:
                    v = first[k]
                    if isinstance(v, str) and len(v) > 60:
                        print("      [debug]   ", k, "=> str(len=%d) starts %r" % (len(v), v[:60]))
                    else:
                        print("      [debug]   ", k, "=>", type(v).__name__, repr(v)[:80])

    if debug:
        # Dump structure (keys only for nested dicts) so we can see actual format
        def _summary(obj: Any, depth: int = 0) -> Any:
            if depth > 4:
                return "..."
            if isinstance(obj, dict):
                return {k: _summary(v, depth + 1) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_summary(obj[0], depth + 1)] if obj else []
            if isinstance(obj, str) and len(obj) > 80:
                return obj[:80] + "..."
            return obj

        print("      [debug] Response structure:", json.dumps(_summary(data), indent=2))
    return None


def call_api_image_and_text_to_image(
    image_path: str, prompt: str, api_key: str, debug: bool = False
) -> Optional[bytes]:
    """Image + text → image: send image and prompt, return generated image bytes or None."""
    image_url = image_to_base64_data_url(image_path)
    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt},
    ]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
    }
    # Ask for image output when supported (Gemini-style)
    body.setdefault("generation_config", {})["response_modalities"] = ["TEXT", "IMAGE"]
    resp = requests.post(
        API_BASE,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120,
    )
    if resp.status_code == 401:
        raise SystemExit(
            "Error: API request returned HTTP 401 Unauthorized. "
            "The API key is invalid or has expired. Please check your key."
        )
    resp.raise_for_status()
    data = resp.json()
    return _extract_image_base64_from_response(data, debug=debug)


def call_api_two_images_and_text_to_image(
    image_path_1: str,
    image_path_2: str,
    prompt: str,
    api_key: str,
    debug: bool = False,
) -> Optional[bytes]:
    """Two images + text → image: send first image, second image, and prompt; return generated image bytes (annotated second image) or None."""
    url_1 = image_to_base64_data_url(image_path_1)
    url_2 = image_to_base64_data_url(image_path_2)
    content = [
        {"type": "image_url", "image_url": {"url": url_1}},
        {"type": "image_url", "image_url": {"url": url_2}},
        {"type": "text", "text": prompt},
    ]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
    }
    body.setdefault("generation_config", {})["response_modalities"] = ["TEXT", "IMAGE"]
    resp = requests.post(
        API_BASE,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120,
    )
    if resp.status_code == 401:
        raise SystemExit(
            "Error: API request returned HTTP 401 Unauthorized. "
            "The API key is invalid or has expired. Please check your key."
        )
    resp.raise_for_status()
    data = resp.json()
    return _extract_image_base64_from_response(data, debug=debug)


NUM_VIEWS = 4
SPATIALIQ_TASK_NAMES = [f"task{i}" for i in range(1, 12)]  # task1 .. task11
SPATIALIQ_ALL_TASKS = set(SPATIALIQ_TASK_NAMES) | {"main"}
CROSS_VIEW_TASKS = {"task10", "task11"}


def parse_tasks_arg(tasks_str: str) -> set:
    """Parse --tasks string (e.g. '1,2,10,main' or 'task1,task2,main') into set of task names."""
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


def run_spatialiq_benchmark(
    views: List[Tuple[int, str]],
    api_key: str,
    label: str = "",
    debug: bool = False,
    tasks: Optional[set] = None,
    override: bool = False,
) -> None:
    """Run selected SpatialIQ annotation tasks + Main Task on the given views.
    tasks: set of 'task1'..'task10' and/or 'main'. If None, run all.
    override: if True, run inference even when prediction file already exists (overwrite).
    Tasks 1-9 and Main: one image per view → save taskN_pred_image.png, task_main_pred_text.txt per view.
    Task 10: (view_0, view_N) for each N>=1 → save task10_pred_image.png in view_N's directory.
    """
    view_indices = [v[0] for v in views]
    image_paths = [v[1] for v in views]
    output_dirs = [_get_output_dir(os.path.dirname(p)) for p in image_paths]

    run_tasks = tasks if tasks is not None else SPATIALIQ_ALL_TASKS
    run_tasks_single = [t for t in SPATIALIQ_TASK_NAMES if t not in CROSS_VIEW_TASKS and t in run_tasks]
    run_main = "main" in run_tasks

    print(f"SpatialIQ benchmark: {label}")
    print(f"  Views: {view_indices} ({len(views)} total)")
    print(f"  Tasks: {sorted(run_tasks)}\n")

    # Single-view tasks and Main per view
    for v_idx, img_path, out_dir in zip(view_indices, image_paths, output_dirs):
        print(f"--- View {v_idx} ---")
        for task_name in run_tasks_single:
            out_path = os.path.join(out_dir, f"{task_name}_pred_image_{MODEL_SLUG}.png")
            print(f"  {task_name} -> {os.path.basename(out_path)}")
            if not override and os.path.isfile(out_path):
                print(f"    (exists, skipped)")
            else:
                t0 = time.monotonic()
                try:
                    img_bytes = call_api_image_and_text_to_image(
                        img_path, TASK_PROMPTS[task_name], api_key, debug=debug
                    )
                    elapsed = time.monotonic() - t0
                    if img_bytes:
                        with open(out_path, "wb") as f:
                            f.write(img_bytes)
                        print(f"    Saved. ({elapsed:.1f}s)")
                    else:
                        print(f"    No image in response. ({elapsed:.1f}s)")
                except (requests.RequestException, OSError) as e:
                    elapsed = time.monotonic() - t0
                    print(f"    Error: {e} ({elapsed:.1f}s)")
        # Main task: image → integer
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
                    answer = call_api_image_and_text(img_path, MAIN_TASK_PROMPT, api_key)
                    elapsed = time.monotonic() - t0_main
                    if answer.strip():
                        with open(main_path, "w") as f:
                            f.write(answer.strip())
                        print(f"    {answer.strip()} ({elapsed:.1f}s)")
                    else:
                        print(f"    (empty response, not saved) ({elapsed:.1f}s)")
                except (requests.RequestException, OSError) as e:
                    elapsed = time.monotonic() - t0_main
                    print(f"    Error: {e} ({elapsed:.1f}s)")
        print()

    # Cross-view tasks: task10 (view n vs n+1) and task11 (view n vs n+2)
    for cv_task, step in [("task10", 1), ("task11", 2)]:
        if cv_task not in run_tasks:
            continue
        prompt_cv = TASK_PROMPTS[cv_task]
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
                t0_cv = time.monotonic()
                try:
                    img_bytes = call_api_two_images_and_text_to_image(
                        img_path, paired_path, prompt_cv, api_key, debug=debug
                    )
                    elapsed = time.monotonic() - t0_cv
                    if img_bytes:
                        with open(out_path, "wb") as f:
                            f.write(img_bytes)
                        print(f"  Saved. ({elapsed:.1f}s)")
                    else:
                        print(f"  No image in response. ({elapsed:.1f}s)")
                except (requests.RequestException, OSError) as e:
                    elapsed = time.monotonic() - t0_cv
                    print(f"  Error: {e} ({elapsed:.1f}s)")
            print()

    print("SpatialIQ benchmark done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Claude Opus 4.6 via NVIDIA Inference API (image output: SpatialIQ benchmark)."
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
        help="Comma-separated tasks (default: all). E.g. 1,2,main or task1,task5,task10,main.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite existing prediction files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When no image in response, print response structure.",
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
    samples = _scan_folders_cached(args.folder, folders)
    if not samples:
        raise SystemExit("No view.png files found in any of the specified folder(s).")
    for i, (label, views) in enumerate(samples, 1):
        if len(samples) > 1:
            print(f"========== Sample {i}/{len(samples)}: {label} ==========")
        run_spatialiq_benchmark(views, api_key, label=label, debug=args.debug, tasks=tasks_set, override=args.override)
        if len(samples) > 1:
            print()


if __name__ == "__main__":
    main()
