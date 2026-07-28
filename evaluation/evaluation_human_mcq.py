#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate human MCQ responses vs. ground truth for SpatialIQ.

Human responses are read from chunk files in:
    data/human-mcq/chunks/<task>_<NN>.json

Each chunk contains 100 answers keyed by integer index (0-based).
The index corresponds to position in the view-dir list discovered from
the input folder list (same ordering as AI model evaluation).

Ground truth comes from gt.json in each view directory.

Produces the same JSON/CSV output format as evaluation_mcq.py.

Usage:
    python evaluation_human_mcq.py inference_results/subset_eval.txt \\
        --chunks-dir data/human-mcq/chunks
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

_MAX_WORKERS = 32

TASK_KEYS = [f"task{i}" for i in range(1, 12)]
MAIN_KEY = "main"
EVAL_KEYS = TASK_KEYS + [MAIN_KEY]

_GT_JSON_MCQ_KEY = {
    "main": "task_main",
    **{f"task{i}": f"task{i}" for i in range(1, 12)},
}


def _progress(iterable, **kwargs):
    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return iterable


# ---------------------------------------------------------------------------
# Folder / view discovery (same as AI model eval)
# ---------------------------------------------------------------------------

def _resolve_folders(path: str) -> List[str]:
    path = os.path.abspath(path)
    if path.endswith(".txt") and os.path.isfile(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [path]


def _find_view_dirs(root: str) -> List[str]:
    root = os.path.abspath(root)
    if os.path.isfile(os.path.join(root, "view.png")):
        return [root]
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
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if "view.png" in filenames:
            dirs.append(dirpath)
    return sorted(dirs)


def _discover_view_dirs_cached(folders: List[str], cache_dir: str) -> List[str]:
    content_hash = hashlib.md5("\n".join(sorted(folders)).encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"eval_viewcache_{content_hash}.json")

    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            view_dirs = json.load(f)
        print(f"Loaded {len(view_dirs)} view dir(s) from cache: {cache_path}")
        return view_dirs

    print(f"Discovering view directories from {len(folders)} folder(s)...")
    all_view_dirs: List[str] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_find_view_dirs, f): i for i, f in enumerate(folders)}
        results: List[List[str]] = [[] for _ in folders]
        for fut in _progress(as_completed(futs), total=len(futs), desc="[human-mcq] discovering views"):
            results[futs[fut]] = fut.result()
    for r in results:
        all_view_dirs.extend(r)

    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump(all_view_dirs, f)
        print(f"Cached {len(all_view_dirs)} view dir(s) to: {cache_path}")
    except OSError:
        pass

    return all_view_dirs


# ---------------------------------------------------------------------------
# Chunk-index ↔ view-dir mapping (must match the JS annotation app)
# ---------------------------------------------------------------------------

CAMERA_DIRS = [
    "offset12.5_fov1_dist1",
    "offset12.5_fov3_dist0.3",
    "offset3_fov1_dist1",
    "offset3_fov3_dist0.3",
]


def _build_chunk_view_list(folders: List[str]) -> List[str]:
    """Build the view-dir list in the same order the JS annotation app uses.

    The JS app (samples.js) sorts the folder lines alphabetically, then
    expands each with 4 fixed camera dirs.  Global index =
    sorted_folder_idx * 4 + camera_idx.
    """
    # Extract relative suffixes so sort order matches the JS app's sort on
    # relative paths like "eval_set/data_.../sample_01/view_02".
    marker = "/dataset/"
    rel_to_abs: Dict[str, str] = {}
    for f in folders:
        idx = f.find(marker)
        rel = f[idx + len("/dataset/"):] if idx >= 0 else f
        rel_to_abs[rel] = f

    sorted_rels = sorted(rel_to_abs.keys())

    view_dirs: List[str] = []
    for rel in sorted_rels:
        abs_folder = rel_to_abs[rel]
        for cam in CAMERA_DIRS:
            view_dirs.append(os.path.join(abs_folder, cam))
    return view_dirs


# ---------------------------------------------------------------------------
# Load human answers from chunk files
# ---------------------------------------------------------------------------

def _load_human_answers(chunks_dir: str) -> Dict[str, Dict[int, str]]:
    """Load all human answers from chunk JSON files.

    Returns {task_key: {chunk_global_index: letter}}.
    """
    chunks_dir = os.path.abspath(chunks_dir)
    answers: Dict[str, Dict[int, str]] = {k: {} for k in EVAL_KEYS}

    for task_key in EVAL_KEYS:
        for chunk_num in range(30):
            fname = f"{task_key}_{chunk_num:02d}.json"
            path = os.path.join(chunks_dir, fname)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for idx_str, entry in data.get("answers", {}).items():
                letter = entry.get("letter", "").strip().upper()
                if len(letter) == 1 and letter in "ABCDE":
                    answers[task_key][int(idx_str)] = letter

    return answers


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------

def _load_gt_json(path: str) -> Optional[Dict]:
    try:
        with open(os.path.join(path, "gt.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _gt_letter_from_data(data: Dict, task_key: str) -> Optional[str]:
    json_key = _GT_JSON_MCQ_KEY.get(task_key, task_key)
    mcq = data.get("mcq", {}).get(json_key, {})
    letter = mcq.get("correct_letter")
    if isinstance(letter, str) and letter.strip().upper() in "ABCDE":
        return letter.strip().upper()
    return None


def _choice_type_from_data(data: Dict, task_key: str, letter: Optional[str]) -> Optional[str]:
    if letter is None or data is None:
        return None
    json_key = _GT_JSON_MCQ_KEY.get(task_key, task_key)
    choice_types = data.get("mcq", {}).get(json_key, {}).get("choice_types", {})
    return choice_types.get(letter)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_one_view(args: tuple) -> Dict[str, Any]:
    view_dir, view_idx, human_answers = args
    gt_data = _load_gt_json(view_dir)

    tasks: Dict[str, Any] = {}
    for key in EVAL_KEYS:
        gt = _gt_letter_from_data(gt_data, key) if gt_data else None
        pred = human_answers[key].get(view_idx)
        has_answer = pred is not None

        correct: Optional[int] = None
        if gt is not None and pred is not None:
            correct = 1 if pred == gt else 0
        choice_type = _choice_type_from_data(gt_data, key, pred) if gt_data else None

        tasks[key] = {
            "gt": gt,
            "pred": pred,
            "correct": correct,
            "pred_file_exists": has_answer,
            "choice_type": choice_type,
        }
    return {"view_dir": view_dir, "tasks": tasks}


def evaluate(
    view_dirs: List[str],
    human_answers: Dict[str, Dict[int, str]],
) -> Dict[str, Any]:
    per_view: List[Dict[str, Any]] = [None] * len(view_dirs)  # type: ignore[list-item]
    work = [(vd, i, human_answers) for i, vd in enumerate(view_dirs)]

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_evaluate_one_view, w): w[1] for w in work}
        for fut in _progress(as_completed(futs), total=len(futs), desc="[human-mcq] evaluating views"):
            idx = futs[fut]
            per_view[idx] = fut.result()

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
        "pred_slug": "human",
        "n_views": len(view_dirs),
        "aggregate": aggregate,
        "overall": {
            "n_evaluated": total_eval,
            "n_correct": total_correct,
            "pct_correct": round(overall_pct, 2) if overall_pct is not None else None,
        },
        "per_view": per_view,
    }


# ---------------------------------------------------------------------------
# Summary / CSV output
# ---------------------------------------------------------------------------

def format_summary(result: Dict[str, Any]) -> str:
    lines = []
    slug = result.get("pred_slug") or "human"
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


_GT_METADATA_FIELDS = ["object_type", "total_blocks", "num_hidden_blocks", "num_columns", "num_layers"]


def _shorten_view_path(view_dir: str) -> str:
    marker = "/dataset/"
    idx = view_dir.find(marker)
    if idx >= 0:
        return view_dir[idx + len(marker):]
    return view_dir


def _load_gt_metadata(view_dir: str) -> dict:
    gt_path = os.path.join(view_dir, "gt.json")
    try:
        with open(gt_path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: data.get(k) for k in _GT_METADATA_FIELDS}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_all_gt_metadata(view_dir_by_short: dict, cache_dir: str | None = None) -> dict:
    if not view_dir_by_short:
        return {}

    cache_path = None
    if cache_dir:
        hash_input = "\n".join(sorted(view_dir_by_short.values()))
        hash_input += "\n" + ",".join(_GT_METADATA_FIELDS)
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"eval_gt_metadata_{content_hash}.json")
        if os.path.isfile(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                print(f"  Loaded gt metadata from cache: {cache_path}")
                return cached
            except (OSError, json.JSONDecodeError):
                pass

    unique_dirs = set(view_dir_by_short.values())
    dir_to_meta: dict = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_load_gt_metadata, d): d for d in unique_dirs}
        for fut in as_completed(futs):
            dir_to_meta[futs[fut]] = fut.result()

    results: dict = {}
    for short_key, full_dir in view_dir_by_short.items():
        results[short_key] = dir_to_meta.get(full_dir, {})

    if cache_path:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(results, f)
            print(f"  Cached gt metadata to: {cache_path}")
        except OSError:
            pass

    return results


def _write_raw_csv(result: dict, csv_path: str, meta_by_view: dict) -> None:
    by_view: dict = {}
    for entry in result.get("per_view", []):
        key = _shorten_view_path(entry["view_dir"])
        by_view[key] = entry.get("tasks", {})

    all_views = sorted(by_view)
    columns = ["view"] + _GT_METADATA_FIELDS
    for t in EVAL_KEYS:
        columns.append(t)
        columns.append(f"{t}_pred")
        columns.append(f"{t}_gt")
        columns.append(f"{t}_choice_type")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for view in all_views:
            tasks_data = by_view.get(view, {})
            meta = meta_by_view.get(view, {})
            row = [view] + [meta.get(k, "") for k in _GT_METADATA_FIELDS]
            for t in EVAL_KEYS:
                info = tasks_data.get(t)
                if not info:
                    row.extend(["", "", "", ""])
                    continue
                val = info.get("correct")
                row.append(str(int(val)) if val is not None else "")
                row.append(info.get("pred") or "")
                row.append(info.get("gt") or "")
                row.append(info.get("choice_type") or "")
            writer.writerow(row)

    print(f"  Raw CSV: {len(all_views)} views x {len(EVAL_KEYS)} tasks -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate human MCQ responses from chunk files (task1-task11, main)."
    )
    parser.add_argument(
        "sample_or_root",
        help="Sample folder, tree root, or .txt file with folder paths (same as AI model eval).",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory with human MCQ chunk JSONs (default: data/human-mcq/chunks).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for output files (default: evaluation_results/human_eval/).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_dir = args.chunks_dir or os.path.join(repo_root, "data", "human-mcq", "chunks")
    out_dir = args.output_dir or os.path.join(repo_root, "evaluation_results", "human_eval")
    os.makedirs(out_dir, exist_ok=True)

    folders = _resolve_folders(args.sample_or_root)
    print(f"Loaded {len(folders)} folder(s) from {args.sample_or_root}")

    view_dirs = _build_chunk_view_list(folders)
    print(f"Evaluating {len(view_dirs)} view directories (sorted folder order × {len(CAMERA_DIRS)} cameras).")

    print(f"Loading human answers from {chunks_dir} ...")
    human_answers = _load_human_answers(chunks_dir)
    for task_key in EVAL_KEYS:
        print(f"  {task_key}: {len(human_answers[task_key])} answers")

    result = evaluate(view_dirs, human_answers)
    print(format_summary(result))

    json_path = os.path.join(out_dir, "eval_mcq_human.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Full results: {json_path}")

    view_dir_by_short: dict = {}
    for entry in result.get("per_view", []):
        key = _shorten_view_path(entry["view_dir"])
        if key not in view_dir_by_short:
            view_dir_by_short[key] = entry["view_dir"]

    print(f"  Loading gt metadata for {len(view_dir_by_short)} views...")
    meta_by_view = _load_all_gt_metadata(view_dir_by_short, cache_dir=out_dir)

    csv_path = os.path.join(out_dir, "eval_raw_human_mcq.csv")
    _write_raw_csv(result, csv_path, meta_by_view)


if __name__ == "__main__":
    main()
