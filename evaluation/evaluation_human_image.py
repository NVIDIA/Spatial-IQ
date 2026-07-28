#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate image-output models (Gemini image, Qwen-image-edit, Hunyuan) from
human verdicts.

There is no automatic ground truth for image outputs — humans graded each
generated image as "correct" or "incorrect". Verdicts are stored as chunk
files under:

    data/<model_dir>/chunks/<task>_<NN>.json

where <model_dir> is one of:
    gemini-image      -> image-eval
    hunyuan           -> hunyuan-eval
    qwen-image-edit   -> qwen_image_edit-eval

Each chunk JSON looks like:
    {
      "task": "task1", "chunk": 0,
      "answers": {
        "<global_idx>": {"verdict": "correct"|"incorrect", "time_ms": N},
        ...
      }
    }

Global index = JS-annotation-app order: sorted_folder_idx * 4 cameras + camera_idx
(same scheme as evaluation_human_mcq.py).

Tasks: task1..task11 (no main task).

Usage:
    python evaluation_human_image.py inference_results/subset_eval.txt
    python evaluation_human_image.py inference_results/subset_eval.txt --model gemini-image
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

_MAX_WORKERS = 32

EVAL_KEYS = [f"task{i}" for i in range(1, 12)]

MODEL_DIRS: Dict[str, str] = {
    "gemini-image":    "image-eval",
    "hunyuan":         "hunyuan-eval",
    "qwen-image-edit": "qwen-eval",
}

CAMERA_DIRS = [
    "offset12.5_fov1_dist1",
    "offset12.5_fov3_dist0.3",
    "offset3_fov1_dist1",
    "offset3_fov3_dist0.3",
]

_GT_METADATA_FIELDS = ["object_type", "total_blocks", "num_hidden_blocks", "num_columns", "num_layers"]


def _progress(iterable, **kwargs):
    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return iterable


# ---------------------------------------------------------------------------
# Folder / view discovery (same convention as evaluation_human_mcq.py)
# ---------------------------------------------------------------------------

def _resolve_folders(path: str) -> List[str]:
    path = os.path.abspath(path)
    if path.endswith(".txt") and os.path.isfile(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [path]


def _build_chunk_view_list(folders: List[str]) -> List[str]:
    """View-dir list in JS annotation-app order: sort by relative path under
    /dataset/, then expand each with the 4 camera dirs.
    Global index = sorted_folder_idx * 4 + camera_idx.
    """
    marker = "/dataset/"
    rel_to_abs: Dict[str, str] = {}
    for f in folders:
        idx = f.find(marker)
        rel = f[idx + len(marker):] if idx >= 0 else f
        rel_to_abs[rel] = f

    sorted_rels = sorted(rel_to_abs.keys())

    view_dirs: List[str] = []
    for rel in sorted_rels:
        abs_folder = rel_to_abs[rel]
        for cam in CAMERA_DIRS:
            view_dirs.append(os.path.join(abs_folder, cam))
    return view_dirs


# ---------------------------------------------------------------------------
# Load human verdicts from chunk files
# ---------------------------------------------------------------------------

def _load_image_verdicts(chunks_dir: str) -> Dict[str, Dict[int, str]]:
    """Returns {task_key: {global_index: "correct"|"incorrect"}}.

    Mirrors evaluation_human_mcq.py._load_human_answers — the chunk filename
    suffix _NN is metadata; the answer dict's keys are the global indices.
    """
    chunks_dir = os.path.abspath(chunks_dir)
    verdicts: Dict[str, Dict[int, str]] = {k: {} for k in EVAL_KEYS}
    failed: List[str] = []

    candidates = [(t, n) for t in EVAL_KEYS for n in range(30)]
    for task_key, chunk_num in _progress(candidates, desc="    loading chunks"):
        fname = f"{task_key}_{chunk_num:02d}.json"
        path = os.path.join(chunks_dir, fname)
        if not os.path.isfile(path):
            continue
        # Retry transient network filesystem I/O errors. If we still fail, record and
        # move on so one bad file doesn't kill the whole run.
        data = None
        for attempt in range(3):
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                data = json.loads(raw.decode("utf-8"))
                break
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                if attempt == 2:
                    failed.append(f"{fname}: {type(exc).__name__}: {exc}")
                    data = None
        if data is None:
            continue
        for idx_str, entry in data.get("answers", {}).items():
            v = (entry.get("verdict") or "").strip().lower()
            if v in ("correct", "incorrect"):
                verdicts[task_key][int(idx_str)] = v
    if failed:
        print(f"    WARN: {len(failed)} chunk file(s) unreadable after 3 attempts:")
        for line in failed[:5]:
            print(f"      - {line}")
        if len(failed) > 5:
            print(f"      ... and {len(failed) - 5} more")
    return verdicts


# ---------------------------------------------------------------------------
# GT metadata (no GT verdict to compare against, but useful for the CSV)
# ---------------------------------------------------------------------------

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


def _load_all_gt_metadata(view_dir_by_short: dict, cache_dir: Optional[str] = None) -> dict:
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
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

    unique_dirs = set(view_dir_by_short.values())
    dir_to_meta: dict = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_load_gt_metadata, d): d for d in unique_dirs}
        for fut in _progress(as_completed(futs), total=len(futs), desc="    loading gt.json"):
            dir_to_meta[futs[fut]] = fut.result()

    results: dict = {}
    for short_key, full_dir in view_dir_by_short.items():
        results[short_key] = dir_to_meta.get(full_dir, {})

    if cache_path:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(results, f)
        except OSError:
            pass

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_one_view(args: tuple) -> Dict[str, Any]:
    view_dir, view_idx, verdicts = args
    tasks: Dict[str, Any] = {}
    for key in EVAL_KEYS:
        v = verdicts[key].get(view_idx)
        has_verdict = v is not None
        correct = 1 if v == "correct" else (0 if v == "incorrect" else None)
        tasks[key] = {
            "verdict": v,
            "correct": correct,
            "verdict_exists": has_verdict,
        }
    return {"view_dir": view_dir, "tasks": tasks}


def evaluate(view_dirs: List[str], verdicts: Dict[str, Dict[int, str]], desc: str = "image") -> Dict[str, Any]:
    per_view: List[Dict[str, Any]] = [None] * len(view_dirs)  # type: ignore[list-item]
    work = [(vd, i, verdicts) for i, vd in enumerate(view_dirs)]

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_evaluate_one_view, w): w[1] for w in work}
        for fut in _progress(as_completed(futs), total=len(futs), desc=f"[{desc}] evaluating views"):
            idx = futs[fut]
            per_view[idx] = fut.result()

    aggregate: Dict[str, Any] = {}
    for key in EVAL_KEYS:
        n_eval = 0
        n_correct = 0
        n_missing = 0
        for v in per_view:
            entry = v["tasks"].get(key)
            if entry is None:
                continue
            if not entry["verdict_exists"]:
                n_missing += 1
                continue
            n_eval += 1
            n_correct += entry["correct"] or 0
        pct = (n_correct / n_eval * 100.0) if n_eval else None
        aggregate[key] = {
            "n_evaluated": n_eval,
            "n_correct": n_correct,
            "n_missing_verdict": n_missing,
            "pct_correct": round(pct, 2) if pct is not None else None,
        }

    total_eval = sum(a["n_evaluated"] for a in aggregate.values())
    total_correct = sum(a["n_correct"] for a in aggregate.values())
    overall_pct = (total_correct / total_eval * 100.0) if total_eval else None

    return {
        "type": "image",
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
# Output
# ---------------------------------------------------------------------------

def format_summary(result: Dict[str, Any], slug: str) -> str:
    lines = []
    lines.append(f"\n{'='*55}")
    lines.append(f"  IMAGE Evaluation Summary  (model={slug})")
    lines.append(f"  Views: {result['n_views']}")
    lines.append(f"{'='*55}")
    lines.append(f"  {'Task':<12} {'Eval':>6} {'Correct':>8} {'Accuracy':>9} {'Missing':>8}")
    lines.append(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*9} {'-'*8}")
    for key in EVAL_KEYS:
        a = result["aggregate"].get(key, {})
        n_eval = a.get("n_evaluated", 0)
        n_ok = a.get("n_correct", 0)
        n_miss = a.get("n_missing_verdict", 0)
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


def _write_raw_csv(result: dict, csv_path: str, meta_by_view: dict) -> None:
    by_view: dict = {}
    for entry in result.get("per_view", []):
        key = _shorten_view_path(entry["view_dir"])
        by_view[key] = entry.get("tasks", {})

    all_views = sorted(by_view)
    columns = ["view"] + _GT_METADATA_FIELDS
    for t in EVAL_KEYS:
        columns.append(t)
        columns.append(f"{t}_verdict")

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
                    row.extend(["", ""])
                    continue
                val = info.get("correct")
                row.append(str(int(val)) if val is not None else "")
                row.append(info.get("verdict") or "")
            writer.writerow(row)
    print(f"  Raw CSV: {len(all_views)} views x {len(EVAL_KEYS)} tasks -> {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_one_model(slug: str, chunks_dir: str, view_dirs: List[str], out_dir: str,
                   meta_by_view: dict) -> None:
    print(f"\n>>> Evaluating image model: {slug}")
    print(f"    chunks_dir: {chunks_dir}")

    if not os.path.isdir(chunks_dir):
        print(f"    SKIP: chunks dir not found: {chunks_dir}")
        return

    verdicts = _load_image_verdicts(chunks_dir)
    for task_key in EVAL_KEYS:
        print(f"    {task_key}: {len(verdicts[task_key])} verdicts")

    result = evaluate(view_dirs, verdicts, desc=slug)
    print(format_summary(result, slug))

    json_path = os.path.join(out_dir, f"eval_image_{slug}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Full results: {json_path}")

    csv_path = os.path.join(out_dir, f"eval_raw_image_{slug}.csv")
    _write_raw_csv(result, csv_path, meta_by_view)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate image-output models from human verdicts (gemini-image, hunyuan, qwen-image-edit)."
    )
    parser.add_argument(
        "sample_or_root",
        help="Sample folder, tree root, or .txt file with folder paths (same as AI/human eval).",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_DIRS.keys()) + ["all"],
        default="all",
        help="Which model to evaluate (default: all).",
    )
    parser.add_argument(
        "--chunks-root",
        type=str,
        default=None,
        metavar="DIR",
        help="Root containing the per-model dirs (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for output files (default: evaluation_results/image_eval/).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_root = args.chunks_root or os.path.join(repo_root, "data")
    out_dir = args.output_dir or os.path.join(repo_root, "evaluation_results", "image_eval")
    os.makedirs(out_dir, exist_ok=True)

    folders = _resolve_folders(args.sample_or_root)
    print(f"Loaded {len(folders)} folder(s) from {args.sample_or_root}")

    view_dirs = _build_chunk_view_list(folders)
    print(f"Evaluating {len(view_dirs)} view directories (sorted folder order x {len(CAMERA_DIRS)} cameras).")

    view_dir_by_short: dict = {}
    for vd in view_dirs:
        view_dir_by_short.setdefault(_shorten_view_path(vd), vd)
    print(f"Loading gt metadata for {len(view_dir_by_short)} unique views...")
    meta_by_view = _load_all_gt_metadata(view_dir_by_short, cache_dir=out_dir)

    selected = [args.model] if args.model != "all" else list(MODEL_DIRS.keys())
    for slug in selected:
        chunks_dir = os.path.join(chunks_root, MODEL_DIRS[slug], "chunks")
        _run_one_model(slug, chunks_dir, view_dirs, out_dir, meta_by_view)


if __name__ == "__main__":
    main()
