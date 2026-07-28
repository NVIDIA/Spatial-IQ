#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate text predictions vs. ground truth for SpatialIQ (integer tasks only).

Task 1 (word "blocks") is not evaluated — predictions are expected to be integers only.

For each of task2..task11 and main, the prediction file must contain a single integer
(optional surrounding whitespace). If not, ``needs_manual_review`` is set so a human
can judge the response; ``correct`` is null in that case (excluded from % correct and
from correlation).

Input: a sample folder, ancestor folder, or a .txt file with one folder path per line.
       Discovers view.png files recursively (handles offset subfolders).

GT source: gt.json in the same directory as view.png (fields: task2..task11, task_main).
           Falls back to taskN_gt_text.txt if gt.json is missing.

Pred files per view: taskN_pred_text_<slug>.txt, task_main_pred_text_<slug>.txt.

Output:
  - One ``text_evaluation.json`` per sample folder with per-view 1/0/null correctness.
  - One ``text_level_aggregate.json`` per level directory that contains samples,
    with per-sample matrices, % correct per task, and Pearson correlation between
    each sub-task (task2..task11) and main over sample-view pairs with both scores defined.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

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

# Integer tasks only; task1 skipped.
TASK_KEYS = [f"task{i}" for i in range(2, 12)]
MAIN_KEY = "main"
EVAL_KEYS = TASK_KEYS + [MAIN_KEY]

OUTPUT_SAMPLE = "text_evaluation.json"
OUTPUT_LEVEL = "text_level_aggregate.json"

# gt.json field name mapping
_GT_JSON_KEY = {
    "main": "task_main",
    **{f"task{i}": f"task{i}" for i in range(2, 12)},
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
    # One listdir + one isfile per child (no separate isdir needed).
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


def _get_sample_view_paths(sample_folder: str) -> List[Tuple[int, str]]:
    sample_folder = os.path.abspath(sample_folder)
    if not os.path.isdir(sample_folder):
        return []
    results: List[Tuple[int, str]] = []
    view_dir_pat = re.compile(r"^view_(\d+)$", re.IGNORECASE)
    for name in sorted(os.listdir(sample_folder)):
        m = view_dir_pat.match(name)
        if m:
            view_idx = int(m.group(1))
            view_dir = os.path.join(sample_folder, name)
            view_png = os.path.join(view_dir, "view.png")
            if os.path.isfile(view_png):
                results.append((view_idx, view_dir))
    if results:
        return sorted(results, key=lambda x: x[0])
    view_file_pat = re.compile(r"^view_(\d+)\.png$", re.IGNORECASE)
    for f in sorted(os.listdir(sample_folder)):
        m = view_file_pat.match(f)
        if m:
            view_idx = int(m.group(1))
            results.append((view_idx, sample_folder))
    return sorted(results, key=lambda x: x[0]) if results else []


def _discover_sample_folders(root_folder: str) -> List[str]:
    root_folder = os.path.abspath(root_folder)
    if not os.path.isdir(root_folder):
        return []
    samples: List[str] = []
    for dirpath, dirnames, _ in os.walk(root_folder, topdown=True):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if _get_sample_view_paths(dirpath):
            samples.append(dirpath)
    return sorted(samples)


def _strip_model_tokens(text: str) -> str:
    """Strip common model special tokens (GLM box tags, think tags, etc.)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>|</think>", "", text)
    return text.strip()


def _extract_integer(raw: str) -> Tuple[Optional[int], bool]:
    """
    Extract an integer from model output with progressive fallbacks.
    Returns (value, needs_manual_review).

    Fallback chain:
      1. Exact integer match after stripping model tokens.
      2. Exactly one integer found in the text -> use it.
      3. Multiple integers found -> use the last one (models typically
         reason first, then state a final answer).
      4. No integers found -> needs_manual_review.
    """
    if raw is None:
        return None, True
    t = _strip_model_tokens(raw).strip()
    if not t:
        return None, True
    if re.fullmatch(r"-?\d+", t):
        return int(t), False
    nums = re.findall(r"-?\d+", t)
    if len(nums) == 1:
        return int(nums[0]), False
    if len(nums) > 1:
        return int(nums[-1]), False
    return None, True


def _load_gt_int(path: str) -> Optional[int]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return None
    v, manual = _extract_integer(raw)
    if manual:
        t = raw.strip()
        try:
            return int(t)
        except ValueError:
            return None
    return v


def _load_gt_json(view_dir: str) -> Optional[Dict[str, Any]]:
    """Load and cache gt.json for a view directory (called once per view)."""
    gt_path = os.path.join(view_dir, "gt.json")
    try:
        with open(gt_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _load_gt_from_json(view_dir: str, task_key: str) -> Optional[int]:
    """Load GT integer from gt.json in view_dir."""
    gt_path = os.path.join(view_dir, "gt.json")
    if not os.path.isfile(gt_path):
        return None
    try:
        with open(gt_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    json_key = _GT_JSON_KEY.get(task_key, task_key)
    val = data.get(json_key)
    if isinstance(val, int):
        return val
    return None


def _gt_int_from_data(data: Dict[str, Any], task_key: str) -> Optional[int]:
    """Extract GT integer for a task from an already-loaded gt.json dict."""
    json_key = _GT_JSON_KEY.get(task_key, task_key)
    val = data.get(json_key)
    return val if isinstance(val, int) else None


def _load_pred_text(path: str) -> Tuple[Optional[str], Optional[int], bool]:
    """Returns (raw_text, parsed_int_or_none, needs_manual_review)."""
    if not os.path.isfile(path):
        return None, None, True
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return None, None, True
    v, manual = _extract_integer(raw)
    return raw, v, manual


def _read_file(path: str) -> Optional[str]:
    """Read a small text file. Returns None if missing/unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _correctness(
    gt: Optional[int], pred_val: Optional[int], needs_manual: bool
) -> Optional[int]:
    if needs_manual:
        return None
    if gt is None or pred_val is None:
        return None
    return 1 if pred_val == gt else 0


def _map_to_inference_dir(view_dir: str, inference_results_dir: str) -> str:
    """Map a dataset view_dir to the corresponding path under inference_results/."""
    m = re.search(r'(?:^|/)(dataset/.*)$', view_dir)
    if m:
        rel = m.group(1)
        return os.path.join(inference_results_dir, rel)
    return view_dir


def _pred_filename(task_key: str, pred_slug: Optional[str] = None) -> str:
    suf = f"_{pred_slug}" if pred_slug else ""
    if task_key == MAIN_KEY:
        return f"task_main_pred_text{suf}.txt"
    return f"{task_key}_pred_text{suf}.txt"


def _gt_filename(task_key: str) -> str:
    if task_key == MAIN_KEY:
        return "task_main_gt_text.txt"
    return f"{task_key}_gt_text.txt"


def evaluate_view_text(view_dir: str, pred_slug: Optional[str] = None, pred_dir: Optional[str] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    _pred_base = pred_dir if pred_dir is not None else view_dir

    # Load gt.json ONCE for all tasks in this view.
    gt_data = _load_gt_json(view_dir)

    # Batch-read all pred files: list the directory once, then only open
    # files that actually exist (avoids per-file stat calls).
    try:
        pred_files_on_disk = set(os.listdir(_pred_base))
    except OSError:
        pred_files_on_disk = set()

    for key in EVAL_KEYS:
        gt: Optional[int] = None
        if gt_data is not None:
            gt = _gt_int_from_data(gt_data, key)
        if gt is None:
            gt = _load_gt_int(os.path.join(view_dir, _gt_filename(key)))

        pred_fname = _pred_filename(key, pred_slug)
        if pred_fname in pred_files_on_disk:
            raw = _read_file(os.path.join(_pred_base, pred_fname))
            if raw is not None:
                pred_val, needs_manual = _extract_integer(raw)
            else:
                raw, pred_val, needs_manual = None, None, True
        else:
            raw, pred_val, needs_manual = None, None, True

        cor = _correctness(gt, pred_val, needs_manual)
        correct_01: Optional[str]
        if cor == 1:
            correct_01 = "1"
        elif cor == 0:
            correct_01 = "0"
        else:
            correct_01 = None
        row[key] = {
            "gt": gt,
            "pred_raw": raw,
            "pred": pred_val,
            "needs_manual_review": needs_manual,
            "correct": cor,
            "correct_01": correct_01,
        }
    return row


def _pearson_binary(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson r; None if undefined (constant or too few points)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return None
    aa = a[m].astype(np.float64)
    bb = b[m].astype(np.float64)
    if np.std(aa) < 1e-15 or np.std(bb) < 1e-15:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _collect_vectors(
    level_samples_data: List[Dict[str, Any]], task_key: str, main_key: str = MAIN_KEY
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Stack 0/1 correct flags for task_key and main across all views in level."""
    xs: List[float] = []
    ys: List[float] = []
    for sample_entry in level_samples_data:
        for view_entry in sample_entry["views"].values():
            tasks = view_entry["tasks"]
            t = tasks[task_key]
            m = tasks[main_key]
            tc = t.get("correct")
            mc = m.get("correct")
            if tc is None or mc is None:
                continue
            xs.append(float(tc))
            ys.append(float(mc))
    if not xs:
        return None, None
    return np.array(xs), np.array(ys)


def evaluate_sample(sample_folder: str, pred_slug: Optional[str] = None) -> Dict[str, Any]:
    sample_folder = os.path.abspath(sample_folder)
    views = _get_sample_view_paths(sample_folder)
    out: Dict[str, Any] = {
        "sample_folder": sample_folder,
        "task1": "skipped_not_integer_benchmark",
        "views": {},
    }
    for view_idx, view_dir in views:
        view_name = os.path.basename(view_dir)
        out["views"][view_name] = {
            "view_index": view_idx,
            "tasks": evaluate_view_text(view_dir, pred_slug=pred_slug),
        }
    return out


def _flatten_view_tasks(
    sample_entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """One dict per view with flat task2..main correct flags for level aggregation."""
    flat: List[Dict[str, Any]] = []
    sample_folder = sample_entry["sample_folder"]
    sample_id = os.path.basename(sample_folder)
    for view_name, vdata in sample_entry["views"].items():
        tasks = vdata["tasks"]
        rec: Dict[str, Any] = {
            "sample_id": sample_id,
            "sample_folder": sample_folder,
            "view_name": view_name,
            "view_index": vdata["view_index"],
        }
        for k in EVAL_KEYS:
            rec[k] = tasks[k]["correct_01"]
        flat.append(rec)
    return flat


def write_level_aggregate(
    level_dir: str,
    sample_results: List[Dict[str, Any]],
) -> str:
    """Build text_level_aggregate.json for one level directory."""
    level_dir = os.path.abspath(level_dir)
    rows: List[Dict[str, Any]] = []
    for se in sample_results:
        rows.extend(_flatten_view_tasks(se))

    aggregate: Dict[str, Any] = {}
    for key in EVAL_KEYS:
        vals = [r[key] for r in rows if r[key] is not None]
        n_def = len(vals)
        n_ok = sum(1 for v in vals if v == "1")
        pct = (n_ok / n_def * 100.0) if n_def else None
        aggregate[key] = {
            "n_evaluated": n_def,
            "n_correct": n_ok,
            "pct_correct": round(pct, 4) if pct is not None else None,
        }

    corr: Dict[str, Optional[float]] = {}
    # Sub-task vs main: Pearson r on paired 0/1 correctness (same sample-views).
    for tk in TASK_KEYS:
        a, b = _collect_vectors(sample_results, tk, MAIN_KEY)
        corr[tk] = _pearson_binary(a, b) if a is not None and b is not None else None

    payload = {
        "level_folder": level_dir,
        "n_sample_views": len(rows),
        "per_sample_view": rows,
        "aggregate_pct_correct": aggregate,
        "pearson_subtask_vs_main": corr,
    }
    out_path = os.path.join(level_dir, OUTPUT_LEVEL)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def run_evaluation(
    root_folder: str,
    sample_output: str = OUTPUT_SAMPLE,
    write_level_files: bool = True,
    pred_slug: Optional[str] = None,
) -> None:
    root_folder = os.path.abspath(root_folder)
    if not os.path.isdir(root_folder):
        raise SystemExit(f"Not a directory: {root_folder}")

    samples = _discover_sample_folders(root_folder)
    if not samples:
        # Single sample path
        if _get_sample_view_paths(root_folder):
            samples = [root_folder]
        else:
            raise SystemExit(
                f"No samples found under {root_folder}. "
                "Expect view_01/view.png under sample folders."
            )

    # Group samples by level (parent directory)
    by_level: Dict[str, List[str]] = {}
    for s in samples:
        parent = os.path.dirname(os.path.abspath(s))
        by_level.setdefault(parent, []).append(s)

    for i, sample in enumerate(samples, 1):
        print(f"========== Sample {i}/{len(samples)}: {sample} ==========")
        data = evaluate_sample(sample, pred_slug=pred_slug)
        out_path = os.path.join(sample, sample_output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  -> {out_path}\n")

    if write_level_files:
        for level_dir, sample_list in sorted(by_level.items()):
            level_results = []
            for s in sorted(sample_list):
                p = os.path.join(s, sample_output)
                if not os.path.isfile(p):
                    level_results.append(evaluate_sample(s, pred_slug=pred_slug))
                else:
                    with open(p, encoding="utf-8") as f:
                        level_results.append(json.load(f))
            path = write_level_aggregate(level_dir, level_results)
            print(f"Level aggregate: {path}")


def format_summary(result: Dict[str, Any]) -> str:
    """Format evaluation result as a human-readable summary table."""
    lines = []
    eval_type = result.get("type", "text").upper()
    slug = result.get("pred_slug") or "default"
    lines.append(f"\n{'='*55}")
    lines.append(f"  {eval_type} Evaluation Summary  (pred_slug={slug})")
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


def evaluate_from_view_dirs(
    view_dirs: List[str],
    pred_slug: Optional[str] = None,
    inference_results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate text predictions across a flat list of view directories (containing view.png + gt.json).
    Returns a summary dict with per-task accuracy and per-view results.
    """
    def _eval_one(view_dir: str) -> Dict[str, Any]:
        pred_dir = _map_to_inference_dir(view_dir, inference_results_dir) if inference_results_dir else None
        tasks = evaluate_view_text(view_dir, pred_slug=pred_slug, pred_dir=pred_dir)
        return {"view_dir": view_dir, "tasks": tasks}

    per_view: List[Dict[str, Any]] = [None] * len(view_dirs)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_eval_one, vd): i for i, vd in enumerate(view_dirs)}
        for fut in _progress(as_completed(futs), total=len(futs), desc="[text] evaluating views"):
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
            if entry["pred_raw"] is None:
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
        "type": "text",
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
    """Evaluate text predictions from a list of root folders (discovers view.png recursively)."""
    all_view_dirs: List[str] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_find_view_dirs, f): i for i, f in enumerate(folders)}
        results: List[List[str]] = [[] for _ in folders]
        for fut in _progress(as_completed(futs), total=len(futs), desc="[text] discovering views"):
            results[futs[fut]] = fut.result()
    for r in results:
        all_view_dirs.extend(r)
    return evaluate_from_view_dirs(all_view_dirs, pred_slug=pred_slug, inference_results_dir=inference_results_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate text predictions (task2–task11, main); task1 skipped."
    )
    parser.add_argument(
        "sample_or_root",
        help="Sample folder, level folder, tree root, or .txt file with folder paths.",
    )
    parser.add_argument(
        "--sample-output",
        default=OUTPUT_SAMPLE,
        metavar="FILE",
        help=f"Filename written in each sample folder (default: {OUTPUT_SAMPLE}).",
    )
    parser.add_argument(
        "--no-level-aggregate",
        action="store_true",
        help="Do not write per-level text_level_aggregate.json files.",
    )
    parser.add_argument(
        "--pred-slug",
        type=str,
        default=None,
        metavar="SLUG",
        help="Prediction filename suffix (e.g. qwen, gpt, gemini): task2_pred_text_<slug>.txt.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a quick accuracy summary table (uses folder-list mode).",
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
    inference_results_dir = args.inference_results_dir

    if args.summary or args.sample_or_root.endswith(".txt"):
        folders = _resolve_folders(args.sample_or_root)
        result = evaluate_from_folders(folders, pred_slug=pred_slug,
                                       inference_results_dir=inference_results_dir)
        print(format_summary(result))
        return

    run_evaluation(
        args.sample_or_root,
        sample_output=args.sample_output,
        write_level_files=not args.no_level_aggregate,
        pred_slug=pred_slug,
    )
    print("Done.")


if __name__ == "__main__":
    main()
