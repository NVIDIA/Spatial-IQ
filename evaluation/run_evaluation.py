#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Run text and/or MCQ evaluation on a folder list and write all output files.

This is the same logic that runs at the end of inference/run_inference.py;
running this script directly is equivalent to
``SKIP_INFERENCE=1 python inference/run_inference.py ...``
without needing a placeholder inference script argument.

Usage:
    python evaluation/run_evaluation.py inference_results/subset_eval.txt \\
        --pred-slug gpt --mode mcq3options
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

REPO_ROOT = os.path.dirname(EVAL_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation_results")

_GT_METADATA_FIELDS = ["object_type", "total_blocks", "num_hidden_blocks", "num_columns", "num_layers"]


def _shorten_view_path(view_dir: str) -> str:
    """Shorten an absolute view_dir to a relative path starting after 'dataset/'."""
    marker = "/dataset/"
    idx = view_dir.find(marker)
    if idx >= 0:
        return view_dir[idx + len(marker):]
    return view_dir


def _load_gt_metadata(view_dir: str) -> dict:
    """Load scene metadata from gt.json in a view directory."""
    gt_path = os.path.join(view_dir, "gt.json")
    try:
        with open(gt_path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: data.get(k) for k in _GT_METADATA_FIELDS}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_all_gt_metadata(
    view_dir_by_short: dict[str, str],
    cache_dir: Optional[str] = None,
) -> dict[str, dict]:
    """Load gt metadata for all views, with optional disk caching.

    The cache is keyed by a hash of the full directory paths so it
    auto-invalidates when the input view set changes.
    """
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
    dir_to_meta: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=32) as pool:
        futs = {pool.submit(_load_gt_metadata, d): d for d in unique_dirs}
        for fut in as_completed(futs):
            dir_to_meta[futs[fut]] = fut.result()

    results: dict[str, dict] = {}
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


def _get_correct(tasks: dict, task_name: str) -> str:
    info = tasks.get(task_name)
    if not info:
        return ""
    val = info.get("correct")
    if val is None:
        return ""
    return str(int(val))


def _get_score(tasks: dict, task_name: str) -> str:
    info = tasks.get(task_name)
    if not info:
        return ""
    val = info.get("score")
    if val is None:
        val = info.get("correct")
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _get_pred(tasks: dict, task_name: str) -> str:
    info = tasks.get(task_name)
    if not info:
        return ""
    val = info.get("pred")
    if val is None:
        return ""
    return str(val)


def _get_gt(tasks: dict, task_name: str) -> str:
    info = tasks.get(task_name)
    if not info:
        return ""
    val = info.get("gt")
    if val is None:
        return ""
    return str(val)


def _get_choice_type(tasks: dict, task_name: str) -> str:
    info = tasks.get(task_name)
    if not info:
        return ""
    return info.get("choice_type") or ""


def _write_modality_csv(
    result: dict, task_names: list, csv_path: str,
    meta_by_view: dict, value_fn=_get_correct,
    extra_fields: Optional[list] = None,
) -> None:
    """Write a raw CSV for a single modality (text, mcq, or image).

    extra_fields: optional list of (suffix, extractor_fn) pairs.  For each
    task column an additional column named ``<task>_<suffix>`` is appended
    with the value returned by ``extractor_fn(tasks_data, task_name)``.
    """
    by_view: dict[str, dict] = {}
    for entry in result.get("per_view", []):
        key = _shorten_view_path(entry["view_dir"])
        by_view[key] = entry.get("tasks", {})

    all_views = sorted(by_view)
    columns = ["view"] + _GT_METADATA_FIELDS
    for t in task_names:
        columns.append(t)
        for suffix, _ in (extra_fields or []):
            columns.append(f"{t}_{suffix}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for view in all_views:
            tasks_data = by_view.get(view, {})
            meta = meta_by_view.get(view, {})
            row = [view] + [meta.get(k, "") for k in _GT_METADATA_FIELDS]
            for t in task_names:
                row.append(value_fn(tasks_data, t))
                for _, fn in (extra_fields or []):
                    row.append(fn(tasks_data, t))
            writer.writerow(row)

    print(f"  Raw CSV: {len(all_views)} views x {len(task_names)} tasks -> {csv_path}")


def _discover_view_dirs_cached(folders: list, out_dir: str) -> list:
    """Discover view directories (containing view.png) with disk caching.

    The cache file is stored in out_dir and keyed by a hash of the folder list,
    so it auto-invalidates when the input changes.
    """
    from evaluation_text import _find_view_dirs

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    content_hash = hashlib.md5("\n".join(sorted(folders)).encode()).hexdigest()[:12]
    cache_path = os.path.join(out_dir, f"eval_viewcache_{content_hash}.json")

    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            view_dirs = json.load(f)
        print(f"Loaded {len(view_dirs)} view dir(s) from cache: {cache_path}")
        return view_dirs

    print(f"Discovering view directories from {len(folders)} folder(s)...")
    all_view_dirs: list = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futs = {pool.submit(_find_view_dirs, f): i for i, f in enumerate(folders)}
        results: list = [[] for _ in folders]
        completed = as_completed(futs)
        if _tqdm is not None:
            completed = _tqdm(completed, total=len(futs), desc="Discovering views", unit="folder")
        for fut in completed:
            results[futs[fut]] = fut.result()
    for r in results:
        all_view_dirs.extend(r)

    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump(all_view_dirs, f)
        print(f"Cached {len(all_view_dirs)} view dir(s) to: {cache_path}")
    except OSError:
        pass

    return all_view_dirs


def run_full_evaluation(
    folders: list, pred_slug: str, out_dir: str,
    inference_results_dir: Optional[str] = None, mode: str = "both",
) -> None:
    """Run text and/or MCQ evaluation, print summaries, and save JSON + raw CSV results.

    mode: "text", "mcq", "mcq3options", "mcq4options", or "both".
    """
    do_text = mode in ("text", "both")
    do_mcq = mode in ("mcq", "mcq3options", "mcq4options", "both")
    if mode in ("mcq3options", "mcq4options"):
        n = mode[3]  # "3" or "4"
        suffix = f"{n}CQ"
        mcq_slug = pred_slug + suffix if not pred_slug.endswith(suffix) else pred_slug
    else:
        mcq_slug = pred_slug

    print(f"\n{'='*60}")
    print(f"Running evaluation  (pred_slug={pred_slug}, mcq_slug={mcq_slug}, mode={mode})")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    view_dirs = _discover_view_dirs_cached(folders, out_dir)
    print(f"Evaluating {len(view_dirs)} views...")

    text_result = None
    mcq_result = None

    if do_text:
        from evaluation_text import evaluate_from_view_dirs as eval_text, format_summary as fmt_text
        text_result = eval_text(view_dirs, pred_slug=pred_slug, inference_results_dir=inference_results_dir)
        print(fmt_text(text_result))

    if do_mcq:
        from evaluation_mcq import evaluate_from_view_dirs as eval_mcq, format_summary as fmt_mcq
        mcq_result = eval_mcq(view_dirs, pred_slug=mcq_slug, inference_results_dir=inference_results_dir)
        print(fmt_mcq(mcq_result))

    saved = []

    if text_result is not None:
        text_path = os.path.join(out_dir, f"eval_text_{pred_slug}.json")
        with open(text_path, "w") as f:
            json.dump(text_result, f, indent=2)
        saved.append(f"  Text:    {text_path}")

    if mcq_result is not None:
        mcq_path = os.path.join(out_dir, f"eval_mcq_{mcq_slug}.json")
        with open(mcq_path, "w") as f:
            json.dump(mcq_result, f, indent=2)
        saved.append(f"  MCQ:     {mcq_path}")

    summary_slug = mcq_slug if mode in ("mcq3options", "mcq4options") else pred_slug
    summary: dict = {"pred_slug": summary_slug, "n_folders": len(folders)}
    if text_result is not None:
        summary["text"] = {
            "n_views": text_result["n_views"],
            "overall": text_result["overall"],
            "per_task": {k: v for k, v in text_result["aggregate"].items()},
        }
    if mcq_result is not None:
        summary["mcq"] = {
            "n_views": mcq_result["n_views"],
            "overall": mcq_result["overall"],
            "per_task": {k: v for k, v in mcq_result["aggregate"].items()},
        }
    summary_path = os.path.join(out_dir, f"eval_summary_{summary_slug}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    saved.append(f"  Summary: {summary_path}")

    view_dir_by_short: dict[str, str] = {}
    for res in (text_result, mcq_result):
        if res is None:
            continue
        for entry in res.get("per_view", []):
            key = _shorten_view_path(entry["view_dir"])
            if key not in view_dir_by_short:
                view_dir_by_short[key] = entry["view_dir"]

    text_tasks = ["task1", "task2", "task3", "task4", "task5", "task6", "task7",
                  "task8", "task9", "task10", "task11", "main"]
    mcq_tasks = ["task1", "task2", "task3", "task4", "task5", "task6", "task7",
                 "task8", "task9", "task10", "task11", "main"]

    if view_dir_by_short:
        print(f"  Loading gt metadata for {len(view_dir_by_short)} views...")
        meta_by_view = _load_all_gt_metadata(view_dir_by_short, cache_dir=out_dir)
    else:
        meta_by_view = {}

    if text_result is not None:
        csv_path = os.path.join(out_dir, f"eval_raw_{pred_slug}_text.csv")
        _write_modality_csv(
            text_result, text_tasks, csv_path, meta_by_view,
            extra_fields=[("pred", _get_pred), ("gt", _get_gt)],
        )
        saved.append(f"  Raw CSV: {csv_path}")

    if mcq_result is not None:
        csv_path = os.path.join(out_dir, f"eval_raw_{mcq_slug}_mcq.csv")
        _write_modality_csv(
            mcq_result, mcq_tasks, csv_path, meta_by_view,
            extra_fields=[("pred", _get_pred), ("gt", _get_gt), ("choice_type", _get_choice_type)],
        )
        saved.append(f"  Raw CSV: {csv_path}")

    print(f"Results saved to:")
    for line in saved:
        print(line)


def _read_folder_list(txt_path: str) -> list:
    with open(txt_path) as f:
        folders = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not folders:
        sys.exit(f"No folders found in {txt_path}")
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run text and/or MCQ evaluation on a folder list. "
            "Functionally equivalent to "
            "`SKIP_INFERENCE=1 python inference/run_inference.py FOLDER_LIST <any-script>`."
        ),
    )
    parser.add_argument("folder_list", help="Path to .txt file with one folder path per line")
    parser.add_argument(
        "--pred-slug", type=str, required=True, metavar="SLUG",
        help="Prediction filename slug (e.g. glm, qwen, gpt).",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["text", "mcq", "both", "mcq3options", "mcq4options"],
        help="Evaluation mode (default: both).",
    )
    parser.add_argument(
        "--eval-output-dir", type=str, default=None, metavar="DIR",
        help="Directory for evaluation output files (default: evaluation_results/<split_name>/).",
    )
    parser.add_argument(
        "--inference-results-dir", type=str, default=None, metavar="DIR",
        help="Directory with inference results (default: <repo>/inference_results).",
    )
    args = parser.parse_args()

    folders = _read_folder_list(args.folder_list)
    if args.eval_output_dir:
        out_dir = args.eval_output_dir
    else:
        split_name = os.path.splitext(os.path.basename(args.folder_list))[0]
        out_dir = os.path.join(RESULTS_DIR, split_name)
    inference_results_dir = args.inference_results_dir or os.path.join(REPO_ROOT, "inference_results")

    print(f"Loaded {len(folders)} folder(s) from {args.folder_list}")
    print(f"Pred slug: {args.pred_slug}")
    print(f"Mode: {args.mode}")
    print(f"Output dir: {out_dir}")

    run_full_evaluation(
        folders, args.pred_slug, out_dir,
        inference_results_dir=inference_results_dir, mode=args.mode,
    )


if __name__ == "__main__":
    main()
