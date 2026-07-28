#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate model predictions vs. ground truth for the SpatialIQ benchmark.

Input: path to a sample folder (e.g. level_6/sample_00) or an ancestor folder (e.g. block_images).
       If the path contains no views, all descendant folders that contain view_01/view.png (or view_01.png) are run.
       Each sample folder should have view_XX/, taskN_pred.png, taskN_gt_image_mask.npy, scene.json (task prefix groups files).
Missing or unreadable files (task prediction PNGs, task*_gt_image_mask.npy, scene.json, main pred text) are
handled by recording null for that task; the script does not abort. Use --pred-image-stem-suffix for
slugged names (e.g. task1_pred_image_qwen_image_edit.png).
Output: one JSON file per view (view_XX/scores.json) with:
  - task1..task10: Silhouette score (higher = better). Uses GT mask to define blocks;
    for each pixel, s(i) = (b(i)-a(i))/max(a(i),b(i)) where a(i)=mean dist to same block,
    b(i)=mean dist to closest other block in prediction image. Background (mask 0) is
    not scored but is used as a cluster for b(i).
  - main: GT total_blocks from scene.json vs. predicted integer from task_main_pred.txt.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Max side for silhouette computation (downsample to keep runtime manageable)
SILHOUETTE_MAX_SIDE = 256

SCENE_JSON_FILENAME = "scene.json"
TASKS = [f"task{i}" for i in range(1, 12)]


def _get_sample_view_paths(sample_folder: str) -> List[Tuple[int, str]]:
    """Discover view directories. Returns list of (view_index, view_dir path)."""
    sample_folder = os.path.abspath(sample_folder)
    if not os.path.isdir(sample_folder):
        return []
    results = []
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
    """Find all sample folders under root_folder. A folder is a sample if it
    contains view_01/view.png or view_01.png (i.e. _get_sample_view_paths returns non-empty).
    Returns sorted list of absolute paths.
    """
    root_folder = os.path.abspath(root_folder)
    if not os.path.isdir(root_folder):
        return []
    samples = []
    for dirpath, dirnames, _ in os.walk(root_folder, topdown=True):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if _get_sample_view_paths(dirpath):
            samples.append(dirpath)
    return sorted(samples)


def _load_scene_json(sample_folder: str) -> Optional[Dict[str, Any]]:
    """Load scene.json. Returns None if missing or unreadable."""
    path = os.path.join(os.path.abspath(sample_folder), SCENE_JSON_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_image(path: str) -> Optional[np.ndarray]:
    """Load PNG as (H, W, 3) float in [0, 1]. Returns None if missing or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def _load_mask(path: str, expected_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Load .npy mask as (H, W) int. If expected_shape, resize with nearest. Returns None if missing or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        mask = np.load(path)
        if mask.ndim != 2:
            return None
        mask = np.asarray(mask, dtype=np.int32)
        if expected_shape and mask.shape != expected_shape:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(mask.astype(np.uint8))
            pil = pil.resize((expected_shape[1], expected_shape[0]), PILImage.NEAREST)
            mask = np.asarray(pil, dtype=np.int32)
        return mask
    except Exception:
        return None


def _resize_for_silhouette(
    pred: np.ndarray, mask: np.ndarray, max_side: int = SILHOUETTE_MAX_SIDE
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize pred (H,W,3) and mask (H,W) so max dimension is max_side. Returns (pred, mask)."""
    h, w = mask.shape
    if max(h, w) <= max_side:
        return pred, mask
    scale = max_side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    from PIL import Image as PILImage
    pred_pil = PILImage.fromarray((np.clip(pred * 255, 0, 255).astype(np.uint8)))
    pred_pil = pred_pil.resize((new_w, new_h), PILImage.BILINEAR)
    pred = np.asarray(pred_pil, dtype=np.float32) / 255.0
    mask_pil = PILImage.fromarray(mask.astype(np.uint8))
    mask_pil = mask_pil.resize((new_w, new_h), PILImage.NEAREST)
    mask = np.asarray(mask_pil, dtype=np.int32)
    return pred, mask


def _silhouette_score(pred: np.ndarray, mask: np.ndarray, score_only_foreground: bool = True) -> float:
    """
    Silhouette score using GT mask to define blocks and prediction image for colors.
    s(i) = (b(i) - a(i)) / max(a(i), b(i)):
      - a(i) = mean color distance from pixel i to other pixels in the same block
      - b(i) = mean color distance from pixel i to pixels of the closest other block
    Only pixels in non-background blocks (mask != 0) are included in the mean if score_only_foreground.
    """
    pred, mask = _resize_for_silhouette(pred, mask)
    h, w = mask.shape
    pred_flat = pred.reshape(-1, 3)  # (N, 3)
    mask_flat = mask.ravel()          # (N,)
    block_ids = np.unique(mask_flat)
    if len(block_ids) < 2:
        return float("nan")  # need at least 2 blocks for silhouette
    # Only score pixels in non-background blocks
    if score_only_foreground:
        score_mask = mask_flat != 0
    else:
        score_mask = np.ones_like(mask_flat, dtype=bool)
    if not np.any(score_mask):
        return float("nan")
    # Precompute block index lists
    blocks: Dict[int, np.ndarray] = {}
    for k in block_ids:
        blocks[k] = np.where(mask_flat == k)[0]
    # For each scored pixel, compute a(i) and b(i)
    scores_i = []
    for idx in np.where(score_mask)[0]:
        k = mask_flat[idx]
        same_block = blocks[k]
        if same_block.size == 0:
            continue
        # a(i): mean distance to same block (excluding self if needed)
        same_colors = pred_flat[same_block]
        diff_same = np.linalg.norm(same_colors - pred_flat[idx], axis=1)
        # Exclude self (distance 0) from mean
        if same_block.size == 1:
            a_i = 0.0
        else:
            self_in_same = np.where(same_block == idx)[0]
            if len(self_in_same) > 0:
                diff_same = np.delete(diff_same, self_in_same[0])
            a_i = float(np.mean(diff_same))
        # b(i): min over other blocks of mean distance to that block
        other_blocks = [b for bid, b in blocks.items() if bid != k and b.size > 0]
        if not other_blocks:
            b_i = float("inf")
        else:
            mean_dists = []
            for other_idx in other_blocks:
                other_colors = pred_flat[other_idx]
                mean_d = np.mean(np.linalg.norm(other_colors - pred_flat[idx], axis=1))
                mean_dists.append(mean_d)
            b_i = float(min(mean_dists))
        denom = max(a_i, b_i)
        if denom <= 0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / denom
        scores_i.append(s_i)
    if not scores_i:
        return float("nan")
    return float(np.mean(scores_i))


def _evaluate_main_task(
    view_dir: str,
    sample_folder: str,
    main_pred_filename: str = "task_main_pred.txt",
) -> Dict[str, Any]:
    """Compare task_main pred file to scene.json total_blocks."""
    scene = _load_scene_json(sample_folder)
    gt_total = None
    if scene and "total_blocks" in scene:
        gt_total = int(scene["total_blocks"])
    pred_path = os.path.join(view_dir, main_pred_filename)
    pred_value = None
    if os.path.isfile(pred_path):
        try:
            with open(pred_path) as f:
                raw = f.read().strip()
            pred_value = int(raw)
        except ValueError:
            m = re.search(r"-?\d+", raw)
            if m:
                pred_value = int(m.group(0))
        except Exception:
            pass
    result = {"main_gt": gt_total, "main_pred": pred_value}
    if gt_total is not None and pred_value is not None:
        result["main_correct"] = pred_value == gt_total
        result["main"] = 1.0 if pred_value == gt_total else 0.0
    else:
        result["main_correct"] = None
        result["main"] = None
    return result


def evaluate_view(
    view_dir: str,
    view_idx: int,
    sample_folder: str,
    pred_image_stem_suffix: str = "",
    main_pred_filename: str = "task_main_pred.txt",
) -> Dict[str, Any]:
    """Compute scores for one view. Returns dict with task1..task10 (silhouette) and main.
    pred_image_stem_suffix: e.g. '_image_qwen_image_edit' → task1_pred_image_qwen_image_edit.png
    """
    scores = {}
    # Tasks 1-10: silhouette
    for task_name in TASKS:
        pred_path = os.path.join(view_dir, f"{task_name}_pred{pred_image_stem_suffix}.png")
        gt_mask_path = os.path.join(view_dir, f"{task_name}_gt_image_mask.npy")
        pred_img = _load_image(pred_path)
        if pred_img is None:
            scores[task_name] = None
            print(f"    {task_name}: (missing)")
            continue
        mask = _load_mask(gt_mask_path, expected_shape=(pred_img.shape[0], pred_img.shape[1]))
        if mask is None:
            scores[task_name] = None
            print(f"    {task_name}: (missing)")
            continue
        if pred_img.shape[:2] != mask.shape:
            mask = _load_mask(gt_mask_path)  # no resize
            if mask is not None and pred_img.shape[:2] != mask.shape:
                # Resize mask to match pred
                from PIL import Image as PILImage
                pil = PILImage.fromarray(mask.astype(np.uint8))
                pil = pil.resize((pred_img.shape[1], pred_img.shape[0]), PILImage.NEAREST)
                mask = np.asarray(pil, dtype=np.int32)
        if mask is None:
            scores[task_name] = None
            print(f"    {task_name}: (missing)")
            continue
        try:
            sc = _silhouette_score(pred_img, mask, score_only_foreground=True)
            scores[task_name] = round(sc, 6)
            print(f"    {task_name}: {scores[task_name]}")
        except Exception as e:
            scores[task_name] = None
            print(f"    {task_name}: (error)")
    # Main task
    main_result = _evaluate_main_task(view_dir, sample_folder, main_pred_filename=main_pred_filename)
    scores["main_gt"] = main_result["main_gt"]
    scores["main_pred"] = main_result["main_pred"]
    scores["main_correct"] = main_result["main_correct"]
    scores["main"] = main_result["main"] if main_result["main"] is not None else None
    if scores["main"] is not None:
        print(f"    main: {scores['main']} (pred={scores['main_pred']}, gt={scores['main_gt']})")
    else:
        print(f"    main: (missing)")
    return scores


def run_evaluation(
    sample_folder: str,
    output_filename: str = "scores.json",
    pred_image_stem_suffix: str = "",
    main_pred_filename: str = "task_main_pred.txt",
) -> None:
    """Run evaluation for all views and write one scores file per view.
    If sample_folder contains no views, discovers all descendant sample folders
    and runs evaluation on each.
    """
    sample_folder = os.path.abspath(sample_folder)
    if not os.path.isdir(sample_folder):
        raise SystemExit(f"Not a directory: {sample_folder}")
    views = _get_sample_view_paths(sample_folder)
    if not views:
        samples = _discover_sample_folders(sample_folder)
        if not samples:
            raise SystemExit(
                f"No views found in {sample_folder} or in any subfolder. "
                "Expect view_01/view.png, view_02/view.png, ... (or view_01.png, ...) inside sample folders."
            )
        print(f"Found {len(samples)} sample(s) under {sample_folder}; running evaluation on each.\n")
        for i, sample in enumerate(samples, 1):
            print(f"========== Sample {i}/{len(samples)}: {sample} ==========")
            run_evaluation(
                sample,
                output_filename=output_filename,
                pred_image_stem_suffix=pred_image_stem_suffix,
                main_pred_filename=main_pred_filename,
            )
            print()
        return
    scene = _load_scene_json(sample_folder)
    if scene is None:
        print(f"Warning: no {SCENE_JSON_FILENAME} in {sample_folder}; main task GT will be missing.")
    print(f"Evaluation: {sample_folder}")
    print(f"  Views: {[v[0] for v in views]}\n")
    for view_idx, view_dir in views:
        print(f"--- View {view_idx} ---")
        scores = evaluate_view(
            view_dir,
            view_idx,
            sample_folder,
            pred_image_stem_suffix=pred_image_stem_suffix,
            main_pred_filename=main_pred_filename,
        )
        out_path = os.path.join(view_dir, output_filename)
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"  -> {out_path}")
        print()
    print("Evaluation done.")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate SpatialIQ predictions vs. ground truth (silhouette for task1-10, exact match for main)."
    )
    parser.add_argument(
        "sample_folder",
        help="Path to a sample folder (e.g. level_6/sample_01) or an ancestor folder containing multiple samples (e.g. block_images); all discovered samples are evaluated.",
    )
    parser.add_argument(
        "-o", "--output",
        default="scores.json",
        metavar="FILE",
        help="Output filename per view (default: scores.json).",
    )
    parser.add_argument(
        "--pred-image-stem-suffix",
        default="",
        metavar="SUFFIX",
        help="Prediction PNG stem: {task}_pred{SUFFIX}.png (default: {task}_pred.png). "
        "Example: _image_qwen_image_edit for task1_pred_image_qwen_image_edit.png.",
    )
    parser.add_argument(
        "--main-pred-filename",
        default="task_main_pred.txt",
        metavar="NAME",
        help="Main-task prediction filename inside each view dir (default: task_main_pred.txt).",
    )
    args = parser.parse_args()
    run_evaluation(
        args.sample_folder,
        output_filename=args.output,
        pred_image_stem_suffix=args.pred_image_stem_suffix,
        main_pred_filename=args.main_pred_filename,
    )


if __name__ == "__main__":
    main()
