#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate text predictions for the trained / baseline checkpoints.

Two evaluation modes are supported:

* ``standard`` — Per-task pred files (task{N}_pred_text_<slug>.txt and
  task_main_pred_text_<slug>.txt) are read and the last integer in each
  file is taken as the answer.  Identical to ``evaluation_text.py``.
  Used for: Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct,
  sft-7b-plain-100pct, sft-32b-plain-100pct.

* ``cot`` — Only ``task_main_pred_text_<slug>.txt`` is parsed.  Each
  sub-task answer is extracted from the corresponding numbered line of
  the chain-of-thought block, and ``task_main`` is taken from the
  ``<answer>...</answer>`` tag (with fallback to "The total is X+Y=Z" or
  the last integer).
  Used for: sft-7b-cot-100pct, sft-32b-cot-100pct, dapo-7b-tight,
  dapo-32b-tight.

task10 (cross-view: visible in view B but not view A) and task11
(hidden objects, also evaluated cross-view here) are dropped from the
evaluation for **all** checkpoints in this script.

Outputs (under ``--eval-output-dir``, default
``evaluation_results/<split_name>/trained_ckpts/``):

* ``eval_text_<slug>.json``        — full per-view results
* ``eval_summary_<slug>.json``     — aggregate summary
* ``eval_raw_<slug>_text.csv``     — flat CSV for spreadsheet inspection
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# Make sibling evaluation/ modules importable.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(EVAL_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation_results")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from evaluation_text import (  # noqa: E402
    MAIN_KEY,
    _correctness,
    _extract_integer,
    _find_view_dirs,
    _gt_int_from_data,
    _load_gt_json,
    _map_to_inference_dir,
    _read_file,
    evaluate_view_text,
)

# Local task set: drop task10 and task11 from evaluation for all checkpoints.
TASK_KEYS = [f"task{i}" for i in range(2, 10)]  # task2..task9
EVAL_KEYS = TASK_KEYS + [MAIN_KEY]

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


# --------------------------------------------------------------------------- #
# Checkpoint registry
# --------------------------------------------------------------------------- #

# slug -> mode ("standard" | "cot")
CHECKPOINTS: Dict[str, str] = {
    "Qwen2.5-VL-7B-Instruct":  "standard",
    "Qwen2.5-VL-32B-Instruct": "standard",
    "sft-7b-plain-100pct":     "standard",
    "sft-32b-plain-100pct":    "standard",
    "sft-7b-cot-100pct":       "cot",
    "sft-32b-cot-100pct":      "cot",
    "dapo-7b-tight":           "cot",
    "dapo-32b-tight":          "cot",
}

_MAX_WORKERS = 32
_GT_METADATA_FIELDS = [
    "object_type", "total_blocks", "num_hidden_blocks",
    "num_columns", "num_layers",
]


def _progress(it, **kw):
    return _tqdm(it, **kw) if _tqdm is not None else it


# --------------------------------------------------------------------------- #
# CoT parsing
# --------------------------------------------------------------------------- #

# Each sub-task is identified by a regex matching its line in the
# numbered CoT.  Patterns include the leading "N)" so that line 7 (which
# also contains "top-most") is not matched by the simpler line 4 pattern.
#
# Mapping derived from the model's CoT template:
#   1) blocks                                 -> task1 (skipped, not integer)
#   2) columns                                -> task3
#   3) layers                                 -> task4
#   4) objects in the top-most layer          -> task2
#   5) distinct clusters                      -> task9
#   6) the number of visible objects is       -> task5
#   7) columns containing the top-most ...    -> task7
#   8) directly supporting the top-most       -> task6
#   9) directly above a hidden object         -> task8
#   <answer>...</answer>  /  total is X       -> task_main
#
# Lines 1 and 10 (task1 "blocks" / task11 hidden-objects) are not part
# of EVAL_KEYS, so they have no patterns here.
_COT_LINE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("task3",  re.compile(r"^\s*2\)\s*The number of columns",                 re.IGNORECASE)),
    ("task4",  re.compile(r"^\s*3\)\s*The number of layers",                  re.IGNORECASE)),
    ("task2",  re.compile(r"^\s*4\)\s*The number of objects in the top-most", re.IGNORECASE)),
    ("task9",  re.compile(r"^\s*5\)\s*The number of distinct clusters",       re.IGNORECASE)),
    ("task5",  re.compile(r"^\s*6\)",                                         re.IGNORECASE)),
    ("task7",  re.compile(r"^\s*7\)",                                         re.IGNORECASE)),
    ("task6",  re.compile(r"^\s*8\)",                                         re.IGNORECASE)),
    ("task8",  re.compile(r"^\s*9\)",                                         re.IGNORECASE)),
]

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE)
_TOTAL_LINE_RE = re.compile(
    r"the\s+total\s+is\s+-?\d+\s*\+\s*-?\d+\s*=\s*(-?\d+)", re.IGNORECASE
)


def _last_int_in_line(line: str) -> Optional[int]:
    """Return the last integer that appears in the line, or None."""
    nums = re.findall(r"-?\d+", line)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def parse_cot_main_pred(raw: str) -> Dict[str, Optional[int]]:
    """Parse a CoT-style task_main pred file.

    Returns a dict mapping each EVAL_KEY to either an int answer or None
    (None == not found / needs_manual_review).
    """
    out: Dict[str, Optional[int]] = {k: None for k in EVAL_KEYS}

    if not raw:
        return out

    # Sub-tasks: scan each line.
    for line in raw.splitlines():
        for task_key, pat in _COT_LINE_PATTERNS:
            if out[task_key] is not None:
                continue
            if pat.search(line):
                v = _last_int_in_line(line)
                if v is not None:
                    out[task_key] = v
                break  # one task per line

    # Main answer: prefer <answer>X</answer>, then "total is A+B=Z", then
    # _extract_integer fallback (last integer in stripped text).
    m = _ANSWER_TAG_RE.search(raw)
    if m:
        try:
            out[MAIN_KEY] = int(m.group(1))
        except ValueError:
            pass
    if out[MAIN_KEY] is None:
        m = _TOTAL_LINE_RE.search(raw)
        if m:
            try:
                out[MAIN_KEY] = int(m.group(1))
            except ValueError:
                pass
    if out[MAIN_KEY] is None:
        v, manual = _extract_integer(raw)
        if not manual:
            out[MAIN_KEY] = v

    return out


# --------------------------------------------------------------------------- #
# Per-view evaluation
# --------------------------------------------------------------------------- #

def _evaluate_view_cot(view_dir: str, slug: str, pred_dir: str) -> Dict[str, Any]:
    """CoT-mode evaluation: derive every task's answer from
    ``task_main_pred_text_<slug>.txt`` only.
    """
    main_path = os.path.join(pred_dir, f"task_main_pred_text_{slug}.txt")
    raw = _read_file(main_path)
    parsed = parse_cot_main_pred(raw or "")

    gt_data = _load_gt_json(view_dir)

    row: Dict[str, Any] = {}
    for key in EVAL_KEYS:
        gt: Optional[int] = None
        if gt_data is not None:
            gt = _gt_int_from_data(gt_data, key)

        pred_val = parsed.get(key)
        # task10 is intentionally absent from the CoT template.
        needs_manual = pred_val is None
        cor = _correctness(gt, pred_val, needs_manual)
        correct_01 = "1" if cor == 1 else ("0" if cor == 0 else None)
        row[key] = {
            "gt": gt,
            "pred_raw": raw,            # same source for every key
            "pred": pred_val,
            "needs_manual_review": needs_manual,
            "correct": cor,
            "correct_01": correct_01,
        }
    return row


def _evaluate_one_view(
    view_dir: str, slug: str, mode: str, inference_results_dir: Optional[str],
) -> Dict[str, Any]:
    pred_dir = _map_to_inference_dir(view_dir, inference_results_dir) if inference_results_dir else view_dir
    if mode == "cot":
        tasks = _evaluate_view_cot(view_dir, slug=slug, pred_dir=pred_dir)
    else:  # "standard"
        full = evaluate_view_text(view_dir, pred_slug=slug, pred_dir=pred_dir)
        # Drop task10 / task11 — not evaluated for any checkpoint here.
        tasks = {k: full[k] for k in EVAL_KEYS if k in full}
    return {"view_dir": view_dir, "tasks": tasks}


# --------------------------------------------------------------------------- #
# Aggregation (mirrors evaluation_text.evaluate_from_view_dirs)
# --------------------------------------------------------------------------- #

def _aggregate(per_view: List[Dict[str, Any]], slug: str, mode: str) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for key in EVAL_KEYS:
        n_eval = 0
        n_correct = 0
        n_missing = 0
        for v in per_view:
            entry = v["tasks"].get(key)
            if entry is None:
                continue
            if entry["pred_raw"] is None:
                n_missing += 1
                continue
            if entry["correct"] is not None:
                n_eval += 1
                n_correct += entry["correct"]
        pct = (n_correct / n_eval * 100.0) if n_eval else None
        aggregate[key] = {
            "n_evaluated": n_eval,
            "n_correct": n_correct,
            "n_missing_pred": n_missing,
            "pct_correct": round(pct, 2) if pct is not None else None,
        }

    total_eval = sum(a["n_evaluated"] for a in aggregate.values())
    total_correct = sum(a["n_correct"] for a in aggregate.values())
    overall_pct = (total_correct / total_eval * 100.0) if total_eval else None

    return {
        "type": "text",
        "pred_slug": slug,
        "mode": mode,
        "n_views": len(per_view),
        "aggregate": aggregate,
        "overall": {
            "n_evaluated": total_eval,
            "n_correct": total_correct,
            "pct_correct": round(overall_pct, 2) if overall_pct is not None else None,
        },
        "per_view": per_view,
    }


def format_summary(result: Dict[str, Any]) -> str:
    """Human-readable summary table over EVAL_KEYS (task2..task9 + main)."""
    lines = []
    slug = result.get("pred_slug") or "default"
    mode = result.get("mode", "")
    lines.append(f"\n{'='*55}")
    lines.append(f"  TEXT Evaluation Summary  (pred_slug={slug}, mode={mode})")
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


def evaluate_checkpoint(
    view_dirs: List[str],
    slug: str,
    mode: str,
    inference_results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    per_view: List[Dict[str, Any]] = [None] * len(view_dirs)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {
            pool.submit(_evaluate_one_view, vd, slug, mode, inference_results_dir): i
            for i, vd in enumerate(view_dirs)
        }
        for fut in _progress(
            as_completed(futs), total=len(futs),
            desc=f"[{slug}] evaluating views ({mode})",
        ):
            per_view[futs[fut]] = fut.result()
    return _aggregate(per_view, slug=slug, mode=mode)


# --------------------------------------------------------------------------- #
# CSV output (mirrors run_evaluation._write_modality_csv)
# --------------------------------------------------------------------------- #

def _shorten_view_path(view_dir: str) -> str:
    marker = "/dataset/"
    idx = view_dir.find(marker)
    return view_dir[idx + len(marker):] if idx >= 0 else view_dir


def _load_gt_metadata(view_dir: str) -> dict:
    gt_path = os.path.join(view_dir, "gt.json")
    try:
        with open(gt_path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: data.get(k) for k in _GT_METADATA_FIELDS}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_all_gt_metadata(view_dirs: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_load_gt_metadata, d): d for d in set(view_dirs)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


def _csv_value_correct(tasks: dict, k: str) -> str:
    info = tasks.get(k) or {}
    v = info.get("correct")
    return "" if v is None else str(int(v))


def _csv_value_pred(tasks: dict, k: str) -> str:
    info = tasks.get(k) or {}
    v = info.get("pred")
    return "" if v is None else str(v)


def _csv_value_gt(tasks: dict, k: str) -> str:
    info = tasks.get(k) or {}
    v = info.get("gt")
    return "" if v is None else str(v)


def _write_csv(result: dict, csv_path: str, meta_by_view: Dict[str, dict]) -> None:
    rows_by_view: Dict[str, dict] = {}
    full_dir_by_short: Dict[str, str] = {}
    for entry in result.get("per_view", []):
        short = _shorten_view_path(entry["view_dir"])
        rows_by_view[short] = entry.get("tasks", {})
        full_dir_by_short[short] = entry["view_dir"]

    task_names = list(EVAL_KEYS)  # task2..task9, main

    columns = ["view"] + _GT_METADATA_FIELDS
    for t in task_names:
        columns += [t, f"{t}_pred", f"{t}_gt"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for short in sorted(rows_by_view):
            tasks = rows_by_view[short]
            full = full_dir_by_short[short]
            meta = meta_by_view.get(full, {})
            row = [short] + [meta.get(k, "") for k in _GT_METADATA_FIELDS]
            for t in task_names:
                row += [
                    _csv_value_correct(tasks, t),
                    _csv_value_pred(tasks, t),
                    _csv_value_gt(tasks, t),
                ]
            w.writerow(row)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _read_folder_list(path: str) -> List[str]:
    if not path.endswith(".txt"):
        return [path]
    with open(path) as f:
        folders = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not folders:
        sys.exit(f"No folders found in {path}")
    return folders


def _discover_view_dirs(folders: List[str], cache_dir: str) -> List[str]:
    h = hashlib.md5("\n".join(sorted(folders)).encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"trained_ckpts_viewcache_{h}.json")
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            view_dirs = json.load(f)
        print(f"Loaded {len(view_dirs)} view dir(s) from cache: {cache_path}")
        return view_dirs

    print(f"Discovering view directories from {len(folders)} folder(s)...")
    all_dirs: List[str] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_find_view_dirs, fld): i for i, fld in enumerate(folders)}
        results: List[List[str]] = [[] for _ in folders]
        for fut in _progress(as_completed(futs), total=len(futs), desc="Discovering views"):
            results[futs[fut]] = fut.result()
    for r in results:
        all_dirs.extend(r)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_dirs, f)
    print(f"Cached {len(all_dirs)} view dir(s) to: {cache_path}")
    return all_dirs


def run(
    folders: List[str],
    out_dir: str,
    inference_results_dir: Optional[str],
    ckpts: List[str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    view_dirs = _discover_view_dirs(folders, out_dir)
    print(f"Evaluating {len(view_dirs)} views across {len(ckpts)} checkpoint(s).\n")

    if not view_dirs:
        sys.exit("No view directories found. Aborting.")

    print("Loading gt metadata for CSV output...")
    meta_by_view = _load_all_gt_metadata(view_dirs)

    summary_rows: List[Dict[str, Any]] = []
    for slug in ckpts:
        mode = CHECKPOINTS[slug]
        print(f"\n{'='*60}")
        print(f"Checkpoint: {slug}  (mode={mode})")
        print(f"{'='*60}")

        result = evaluate_checkpoint(
            view_dirs, slug=slug, mode=mode,
            inference_results_dir=inference_results_dir,
        )
        print(format_summary(result))

        json_path = os.path.join(out_dir, f"eval_text_{slug}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        summary = {
            "pred_slug": slug,
            "mode": mode,
            "n_views": result["n_views"],
            "overall": result["overall"],
            "per_task": result["aggregate"],
        }
        summary_path = os.path.join(out_dir, f"eval_summary_{slug}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        csv_path = os.path.join(out_dir, f"eval_raw_{slug}_text.csv")
        _write_csv(result, csv_path, meta_by_view)

        print(f"  JSON:    {json_path}")
        print(f"  Summary: {summary_path}")
        print(f"  CSV:     {csv_path}")

        summary_rows.append({
            "slug": slug, "mode": mode,
            "overall_pct": result["overall"]["pct_correct"],
            "n_evaluated": result["overall"]["n_evaluated"],
            "n_correct": result["overall"]["n_correct"],
        })

    # Compact comparison table.
    print("\n" + "=" * 60)
    print("All-checkpoints comparison")
    print("=" * 60)
    print(f"  {'Slug':<28} {'Mode':<10} {'Overall':>9} {'N':>8}")
    for r in summary_rows:
        pct = f"{r['overall_pct']:.1f}%" if r["overall_pct"] is not None else "N/A"
        print(f"  {r['slug']:<28} {r['mode']:<10} {pct:>9} {r['n_evaluated']:>8}")

    combined_path = os.path.join(out_dir, "eval_summary_all.json")
    with open(combined_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nCombined summary: {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate text predictions for the trained / baseline checkpoints "
                    "(standard last-integer for plain models, CoT main_pred parsing for cot/dapo models).",
    )
    parser.add_argument(
        "folder_list",
        nargs="?",
        default=os.path.join(REPO_ROOT, "inference_results", "subset_eval.txt"),
        help="Path to .txt file with one dataset folder per line, or a single folder. "
             "Default: inference_results/subset_eval.txt",
    )
    parser.add_argument(
        "--ckpts", type=str, default=None,
        help=(f"Comma-separated subset of checkpoints to evaluate. "
              f"Default: all ({len(CHECKPOINTS)}). Available: "
              f"{','.join(CHECKPOINTS)}"),
    )
    parser.add_argument(
        "--eval-output-dir", type=str, default=None, metavar="DIR",
        help="Directory for evaluation output files "
             "(default: evaluation_results/<split_name>/trained_ckpts/).",
    )
    parser.add_argument(
        "--inference-results-dir", type=str, default=None, metavar="DIR",
        help="Directory with inference results (default: <repo>/inference_results).",
    )
    args = parser.parse_args()

    folders = _read_folder_list(args.folder_list)

    if args.ckpts:
        ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
        unknown = [c for c in ckpts if c not in CHECKPOINTS]
        if unknown:
            sys.exit(f"Unknown checkpoint(s): {unknown}. "
                     f"Available: {list(CHECKPOINTS)}")
    else:
        ckpts = list(CHECKPOINTS)

    if args.eval_output_dir:
        out_dir = args.eval_output_dir
    else:
        if args.folder_list.endswith(".txt"):
            split_name = os.path.splitext(os.path.basename(args.folder_list))[0]
        else:
            split_name = os.path.basename(os.path.abspath(args.folder_list))
        out_dir = os.path.join(RESULTS_DIR, split_name, "trained_ckpts")

    inference_results_dir = (
        args.inference_results_dir or os.path.join(REPO_ROOT, "inference_results")
    )

    print(f"Loaded {len(folders)} folder(s) from {args.folder_list}")
    print(f"Checkpoints: {ckpts}")
    print(f"Output dir:  {out_dir}")
    print(f"Inference:   {inference_results_dir}\n")

    run(folders, out_dir, inference_results_dir, ckpts)


if __name__ == "__main__":
    main()
