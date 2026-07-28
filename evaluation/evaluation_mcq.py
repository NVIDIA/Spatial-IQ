#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate MCQ predictions vs. ground truth for SpatialIQ.

For each task, the prediction file should contain a single letter (A-E).
Ground truth comes from gt.json: mcq.<task>.correct_letter.

Input: a folder, tree root, or .txt file with one folder path per line.
       Discovers view.png files recursively (handles offset subfolders).

Pred files per view: taskN_pred_mcq_<slug>.txt, task_main_pred_mcq_<slug>.txt.
"""

from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

_MAX_WORKERS = 32


def _progress(iterable, **kwargs):
    """Wrap *iterable* with tqdm if available, otherwise iterate silently."""
    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return iterable

TASK_KEYS = [f"task{i}" for i in range(1, 12)]
MAIN_KEY = "main"
EVAL_KEYS = TASK_KEYS + [MAIN_KEY]

_GT_JSON_MCQ_KEY = {
    "main": "task_main",
    **{f"task{i}": f"task{i}" for i in range(1, 12)},
}


def _resolve_folders(path: str) -> List[str]:
    """If path is a .txt file, read folder paths from it. Otherwise treat as single folder."""
    path = os.path.abspath(path)
    if path.endswith(".txt") and os.path.isfile(path):
        with open(path) as f:
            folders = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return folders
    return [path]


def _find_view_dirs(root: str) -> List[str]:
    """Find directories containing view.png under root.

    Uses a fast shallow scan (depth <= 1) before falling back to a full
    recursive walk.  The shallow path minimises ``stat`` calls on networked
    filesystems.
    """
    root = os.path.abspath(root)

    # Fast path 1: root itself is a view dir.
    if os.path.isfile(os.path.join(root, "view.png")):
        return [root]

    # Fast path 2: immediate children contain view.png (offset dirs).
    try:
        children = sorted(os.listdir(root))
    except OSError:
        return []
    dirs: List[str] = []
    for name in children:
        if name.startswith("."):
            continue
        if os.path.isfile(os.path.join(root, name, "view.png")):
            dirs.append(os.path.join(root, name))
    if dirs:
        return dirs

    # Slow fallback: full recursive walk (needed for deep/unusual layouts).
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if "view.png" in filenames:
            dirs.append(dirpath)
    return sorted(dirs)


def _load_gt_json(view_dir: str) -> Optional[Dict]:
    """Load gt.json once for a view directory."""
    try:
        with open(os.path.join(view_dir, "gt.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _gt_letter_from_data(data: Dict, task_key: str) -> Optional[str]:
    """Extract GT letter for a task from an already-loaded gt.json dict."""
    json_key = _GT_JSON_MCQ_KEY.get(task_key, task_key)
    mcq = data.get("mcq", {}).get(json_key, {})
    letter = mcq.get("correct_letter")
    if isinstance(letter, str) and letter.strip().upper() in "ABCDE":
        return letter.strip().upper()
    return None


def _choice_type_from_data(data: Dict, task_key: str, letter: Optional[str]) -> Optional[str]:
    """Look up the choice_type for a given letter from gt.json's mcq section."""
    if letter is None or data is None:
        return None
    json_key = _GT_JSON_MCQ_KEY.get(task_key, task_key)
    choice_types = data.get("mcq", {}).get(json_key, {}).get("choice_types", {})
    return choice_types.get(letter)


def _load_gt_letter(view_dir: str, task_key: str) -> Optional[str]:
    """Load GT correct letter from gt.json (legacy single-task interface)."""
    data = _load_gt_json(view_dir)
    if data is None:
        return None
    return _gt_letter_from_data(data, task_key)


def _map_to_inference_dir(view_dir: str, inference_results_dir: str) -> str:
    """Map a dataset view_dir to the corresponding path under inference_results/."""
    m = re.search(r'(?:^|/)(dataset/.*)$', view_dir)
    if m:
        rel = m.group(1)
        return os.path.join(inference_results_dir, rel)
    return view_dir


_NCQ_RE = re.compile(r"^(.+?)(\d)CQ$")


def _pred_filenames(task_key: str, pred_slug: Optional[str] = None) -> List[str]:
    """Return candidate prediction filenames (preferred first).

    For slugs ending in NCQ (e.g. qwen3CQ), also accept the legacy format
    without CQ (e.g. qwen3) so old prediction files still evaluate correctly.
    """
    suf = f"_{pred_slug}" if pred_slug else ""
    if task_key == MAIN_KEY:
        primary = f"task_main_pred_mcq{suf}.txt"
    else:
        primary = f"{task_key}_pred_mcq{suf}.txt"
    candidates = [primary]
    if pred_slug:
        m = _NCQ_RE.match(pred_slug)
        if m:
            legacy_slug = m.group(1) + m.group(2)
            legacy_suf = f"_{legacy_slug}"
            if task_key == MAIN_KEY:
                candidates.append(f"task_main_pred_mcq{legacy_suf}.txt")
            else:
                candidates.append(f"{task_key}_pred_mcq{legacy_suf}.txt")
    return candidates


def _strip_model_tokens(text: str) -> str:
    """Strip common model special tokens (GLM box tags, think tags, etc.)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</think>", "", text)
    return text.strip()


def _load_pred_letter(path: str) -> tuple[Optional[str], bool]:
    """Load predicted letter from file. Returns (letter, was_random).

    Fallback chain:
      1. Exact single letter after stripping model tokens.
      2. Only one A-E letter in the entire text -> use it.
      3. Last standalone capital A-E letter in the text (models
         typically reason first, then state a final answer).
      4. Random pick from A-E.
    """
    if not os.path.isfile(path):
        return None, False
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read().strip()
    except OSError:
        return None, False
    return _parse_letter(raw)


def _read_file(path: str) -> Optional[str]:
    """Read a small text file. Returns None if missing/unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_letter(raw: str) -> tuple[Optional[str], bool]:
    """Parse a predicted letter from raw text (no file I/O).

    Returns (letter, was_random).  If no letter can be parsed,
    picks one uniformly at random from A-E.
    """
    raw = _strip_model_tokens(raw)
    cleaned = re.sub(r"[^A-Ea-e]", "", raw)
    if len(cleaned) == 1:
        return cleaned.upper(), False
    matches = re.findall(r"\b([A-E])\b", raw)
    if matches:
        return matches[-1], False
    return random.choice("ABCDE"), True


def evaluate_view_mcq(view_dir: str, pred_slug: Optional[str] = None, pred_dir: Optional[str] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    _pred_base = pred_dir if pred_dir is not None else view_dir

    # Load gt.json ONCE for all tasks in this view.
    gt_data = _load_gt_json(view_dir)

    # List pred directory ONCE to avoid per-file stat calls.
    try:
        pred_files_on_disk = set(os.listdir(_pred_base))
    except OSError:
        pred_files_on_disk = set()

    for key in EVAL_KEYS:
        gt = _gt_letter_from_data(gt_data, key) if gt_data else None

        pred_fname = None
        for candidate in _pred_filenames(key, pred_slug):
            if candidate in pred_files_on_disk:
                pred_fname = candidate
                break
        file_exists = pred_fname is not None
        pred: Optional[str] = None
        if file_exists:
            raw = _read_file(os.path.join(_pred_base, pred_fname))
            pred, _ = _parse_letter(raw if raw is not None else "")

        correct: Optional[int] = None
        if gt is not None and pred is not None:
            correct = 1 if pred == gt else 0
        choice_type = _choice_type_from_data(gt_data, key, pred) if gt_data else None
        row[key] = {
            "gt": gt,
            "pred": pred,
            "correct": correct,
            "pred_file_exists": file_exists,
            "choice_type": choice_type,
        }
    return row


def evaluate_from_view_dirs(
    view_dirs: List[str],
    pred_slug: Optional[str] = None,
    inference_results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate MCQ predictions across a flat list of view directories."""
    def _eval_one(view_dir: str) -> Dict[str, Any]:
        pred_dir = _map_to_inference_dir(view_dir, inference_results_dir) if inference_results_dir else None
        tasks = evaluate_view_mcq(view_dir, pred_slug=pred_slug, pred_dir=pred_dir)
        return {"view_dir": view_dir, "tasks": tasks}

    per_view: List[Dict[str, Any]] = [None] * len(view_dirs)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_eval_one, vd): i for i, vd in enumerate(view_dirs)}
        for fut in _progress(as_completed(futs), total=len(futs), desc="[mcq] evaluating views"):
            per_view[futs[fut]] = fut.result()

    aggregate: Dict[str, Any] = {}
    for key in EVAL_KEYS:
        n_evaluated = 0
        n_correct = 0
        n_missing_pred = 0
        for v in per_view:
            entry = v["tasks"].get(key)
            if entry is None:
                continue
            if not entry["pred_file_exists"]:
                n_missing_pred += 1
                continue
            if entry["correct"] is not None:
                n_evaluated += 1
                n_correct += entry["correct"]
        pct = (n_correct / n_evaluated * 100.0) if n_evaluated else None
        aggregate[key] = {
            "n_evaluated": n_evaluated,
            "n_correct": n_correct,
            "n_missing_pred": n_missing_pred,
            "pct_correct": round(pct, 2) if pct is not None else None,
        }

    total_eval = sum(a["n_evaluated"] for a in aggregate.values())
    total_correct = sum(a["n_correct"] for a in aggregate.values())
    overall_pct = (total_correct / total_eval * 100.0) if total_eval else None

    return {
        "type": "mcq",
        "pred_slug": pred_slug,
        "n_views": len(view_dirs),
        "aggregate": aggregate,
        "overall": {
            "n_evaluated": total_eval,
            "n_correct": total_correct,
            "pct_correct": round(overall_pct, 2) if overall_pct is not None else None,
        },
        "per_view": per_view,
    }


def evaluate_from_folders(
    folders: List[str],
    pred_slug: Optional[str] = None,
    inference_results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate MCQ predictions from a list of root folders."""
    all_view_dirs: List[str] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_find_view_dirs, f): i for i, f in enumerate(folders)}
        results: List[List[str]] = [[] for _ in folders]
        for fut in _progress(as_completed(futs), total=len(futs), desc="[mcq] discovering views"):
            results[futs[fut]] = fut.result()
    for r in results:
        all_view_dirs.extend(r)
    return evaluate_from_view_dirs(all_view_dirs, pred_slug=pred_slug, inference_results_dir=inference_results_dir)


def format_summary(result: Dict[str, Any]) -> str:
    """Format evaluation result as a human-readable summary table."""
    lines = []
    slug = result.get("pred_slug") or "default"
    lines.append(f"\n{'='*55}")
    lines.append(f"  MCQ Evaluation Summary  (pred_slug={slug})")
    lines.append(f"  Views: {result['n_views']}")
    lines.append(f"{'='*55}")
    lines.append(f"  {'Task':<12} {'Eval':>6} {'Correct':>8} {'Accuracy':>9} {'Missing':>8}")
    lines.append(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*9} {'-'*8}")
    for key in EVAL_KEYS:
        a = result["aggregate"].get(key, {})
        n_eval = a.get("n_evaluated", 0)
        n_ok = a.get("n_correct", 0)
        n_miss = a.get("n_missing_pred", 0)
        pct = a.get("pct_correct")
        pct_s = f"{pct:.1f}%" if pct is not None else "N/A"
        lines.append(f"  {key:<12} {n_eval:>6} {n_ok:>8} {pct_s:>9} {n_miss:>8}")
    ov = result.get("overall", {})
    lines.append(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*9} {'-'*8}")
    ov_pct = ov.get("pct_correct")
    ov_pct_s = f"{ov_pct:.1f}%" if ov_pct is not None else "N/A"
    lines.append(f"  {'OVERALL':<12} {ov.get('n_evaluated', 0):>6} {ov.get('n_correct', 0):>8} {ov_pct_s:>9}")
    lines.append(f"{'='*55}\n")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate MCQ predictions (task1–task11, main)."
    )
    parser.add_argument(
        "sample_or_root",
        help="Sample folder, tree root, or .txt file with folder paths.",
    )
    parser.add_argument(
        "--pred-slug",
        type=str,
        default=None,
        metavar="SLUG",
        help="Prediction filename suffix (e.g. glm, qwen): task1_pred_mcq_<slug>.txt.",
    )
    parser.add_argument(
        "--inference-results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory with inference results (maps dataset paths onto this local tree).",
    )
    args = parser.parse_args()

    pred_slug = (args.pred_slug or "").strip() or None
    folders = _resolve_folders(args.sample_or_root)
    result = evaluate_from_folders(folders, pred_slug=pred_slug,
                                   inference_results_dir=args.inference_results_dir)
    print(format_summary(result))

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.sample_or_root)), "mcq_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Full results: {out_path}")


if __name__ == "__main__":
    main()
