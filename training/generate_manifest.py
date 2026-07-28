# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
generate_manifest.py — walk a SpatialIQ dataset directory and emit spatialiq_manifest.json.

Usage:
    python generate_manifest.py <dataset_dir> [--output <path>]

    dataset_dir : path to a dataset directory, e.g. data/data_20260406_120000
    --output    : where to write the manifest JSON (default: <dataset_dir>/spatialiq_manifest.json)

Example:
    python generate_manifest.py data/data_20260406_120000
"""

import argparse
import json
import os
import re
import sys

from schema import (
    GtAnswers,
    GtMasks,
    McqTask,
    SceneSample,
    SpatialIQDataset,
    ViewSample,
)

TASK_IDS = ["task1", "task2", "task3", "task4", "task5",
            "task6", "task7", "task8", "task9", "task10", "task11"]
MCQ_LETTERS = ("A", "B", "C", "D", "E")

# Directory name pattern for view+fov+dist subfolders: offsetO_fovF_distD
_FOV_DIR_RE = re.compile(r"^offset([\d.]+)_fov([\d.]+)_dist([\d.]+)$")


def rel(dataset_dir: str, abs_path: str) -> str:
    """Return abs_path relative to dataset_dir, with forward slashes."""
    return os.path.relpath(abs_path, dataset_dir).replace("\\", "/")


def build_gt_answers(gt: dict) -> GtAnswers:
    return GtAnswers(
        task1=gt.get("task1"),
        task_main=gt.get("task_main"),
        task2=gt.get("task2"),
        task3=gt.get("task3"),
        task4=gt.get("task4"),
        task5=gt.get("task5"),
        task6=gt.get("task6"),
        task7=gt.get("task7"),
        task8=gt.get("task8"),
        task9=gt.get("task9"),
        task10=gt.get("task10"),
        task11=gt.get("task11"),
        visible_blocks=gt.get("visible_blocks", []),
    )


def build_gt_masks(dataset_dir: str, view_dir: str) -> GtMasks:
    def _path(task_id: str) -> str | None:
        p = os.path.join(view_dir, f"{task_id}_gt_mask.png")
        return rel(dataset_dir, p) if os.path.isfile(p) else None

    return GtMasks(
        task1=_path("task1"),
        task2=_path("task2"),
        task3=_path("task3"),
        task4=_path("task4"),
        task5=_path("task5"),
        task6=_path("task6"),
        task7=_path("task7"),
        task8=_path("task8"),
        task9=_path("task9"),
        task10=_path("task10"),
        task11=_path("task11"),
    )


def build_mcq(dataset_dir: str, view_dir: str, gt: dict) -> dict[str, McqTask]:
    mcq_raw = gt.get("mcq", {})
    result = {}
    for task_id, task_data in mcq_raw.items():
        # Collect choice image paths — only if the files actually exist on disk
        choice_images = {}
        for letter in MCQ_LETTERS:
            p = os.path.join(view_dir, f"{task_id}_mcq_choice{letter}.png")
            if os.path.isfile(p):
                choice_images[letter] = rel(dataset_dir, p)

        result[task_id] = McqTask(
            correct_letter=task_data["correct_letter"],
            choices=task_data.get("choices"),           # present for task_main only
            choice_types=task_data["choice_types"],
            choice_images=choice_images if choice_images else None,
        )
    return result


def build_view(dataset_dir: str, view_dir: str, view_idx: int,
               offset: float, fov_mult: float, dist_mult: float) -> ViewSample | None:
    """Build a ViewSample from a view_XX/offsetO_fovF_distD/ leaf directory."""
    gt_path = os.path.join(view_dir, "gt.json")
    view_png = os.path.join(view_dir, "view.png")

    if not os.path.isfile(gt_path):
        print(f"  WARNING: missing gt.json in {view_dir}, skipping view", file=sys.stderr)
        return None
    if not os.path.isfile(view_png):
        print(f"  WARNING: missing view.png in {view_dir}, skipping view", file=sys.stderr)
        return None

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    return ViewSample(
        view_idx=view_idx,
        offset=offset,
        fov_mult=fov_mult,
        dist_mult=dist_mult,
        view_image=rel(dataset_dir, view_png),
        gt_answers=build_gt_answers(gt),
        gt_masks=build_gt_masks(dataset_dir, view_dir),
        mcq=build_mcq(dataset_dir, view_dir, gt),
    )


def build_sample(dataset_dir: str, sample_dir: str,
                 block_textures: list[str]) -> SceneSample | None:
    scene_path = os.path.join(sample_dir, "scene.json")
    if not os.path.isfile(scene_path):
        print(f"  WARNING: missing scene.json in {sample_dir}, skipping sample", file=sys.stderr)
        return None

    with open(scene_path, encoding="utf-8") as f:
        scene = json.load(f)

    # Walk view_XX/ subdirectories, then offsetO_fovF_distD/ within each
    view_dirs = sorted(
        [os.path.join(sample_dir, d) for d in os.listdir(sample_dir)
         if re.match(r"view_\d+", d) and os.path.isdir(os.path.join(sample_dir, d))],
        key=lambda p: int(re.search(r"view_(\d+)", os.path.basename(p)).group(1)),
    )

    views = []
    for vd in view_dirs:
        view_idx = int(re.search(r"view_(\d+)", os.path.basename(vd)).group(1))
        fov_dirs = sorted(
            d for d in os.listdir(vd)
            if _FOV_DIR_RE.match(d) and os.path.isdir(os.path.join(vd, d))
        )
        for fov_dir_name in fov_dirs:
            m = _FOV_DIR_RE.match(fov_dir_name)
            offset = float(m.group(1))
            fov_mult = float(m.group(2))
            dist_mult = float(m.group(3))
            full_fov_dir = os.path.join(vd, fov_dir_name)
            v = build_view(dataset_dir, full_fov_dir, view_idx, offset, fov_mult, dist_mult)
            if v is not None:
                views.append(v)

    if not views:
        print(f"  WARNING: no valid views in {sample_dir}, skipping sample", file=sys.stderr)
        return None

    sample_idx = int(re.search(r"sample_(\d+)", os.path.basename(sample_dir)).group(1))

    return SceneSample(
        sample_idx=sample_idx,
        height_map=scene["height_map"],
        rows=scene["rows"],
        cols=scene["cols"],
        total_blocks=scene["total_blocks"],
        block_textures=block_textures,
        views=views,
    )


def generate_manifest(dataset_dir: str, output_path: str) -> None:
    dataset_dir = os.path.abspath(dataset_dir)
    dataset_id = os.path.basename(dataset_dir)

    # Block textures live at the dataset root
    block_textures = sorted(
        rel(dataset_dir, os.path.join(dataset_dir, f))
        for f in os.listdir(dataset_dir)
        if re.match(r"block_edge_texture_\d+\.png", f)
    )
    print(f"Found {len(block_textures)} block texture(s)")

    samples = []

    # Walk sample_* subdirectories directly under the dataset root
    sample_dirs = sorted(
        [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
         if re.match(r"sample_\d+", d) and os.path.isdir(os.path.join(dataset_dir, d))],
        key=lambda p: int(re.search(r"sample_(\d+)", os.path.basename(p)).group(1)),
    )
    print(f"Found {len(sample_dirs)} sample(s)")
    for sd in sample_dirs:
        s = build_sample(dataset_dir, sd, block_textures)
        if s is not None:
            samples.append(s)

    dataset = SpatialIQDataset(dataset_id=dataset_id, samples=samples)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=2))

    print(f"\nWrote {len(samples)} samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SpatialIQ dataset manifest JSON")
    parser.add_argument("dataset_dir", help="Path to a dataset directory (e.g. data/data_20260406_120000)")
    parser.add_argument("--output", help="Output path for manifest JSON (default: <dataset_dir>/spatialiq_manifest.json)")
    args = parser.parse_args()

    output = args.output or os.path.join(args.dataset_dir, "spatialiq_manifest.json")
    generate_manifest(args.dataset_dir, output)


if __name__ == "__main__":
    main()
