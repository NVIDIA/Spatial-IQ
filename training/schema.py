# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
SpatialIQ dataset schema — Pydantic models with 1:1 mapping to the dataset manifest JSON.

The manifest JSON (spatialiq_manifest.json) is the single source of truth for all
training data, including all image paths. All paths are relative to the dataset root dir.

Hierarchy:
    SpatialIQDataset
    └── samples: list[SceneSample]
        ├── scene metadata (height_map, level, ...)
        ├── block_textures: list[str]          # shared block texture images
        └── views: list[ViewSample]
            ├── view_image: str                # rendered view.png
            ├── gt_answers: GtAnswers          # free-response ground truth
            ├── gt_masks: GtMasks              # segmentation mask PNGs
            └── mcq: dict[str, McqTask]        # MCQ metadata + choice images

Usage:
    dataset = SpatialIQDataset.model_validate_json(open("spatialiq_manifest.json").read())
    for sample in dataset.samples:
        for view in sample.views:
            img_path = view.view_image          # relative to dataset root
            answer   = view.gt_answers.task_main
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Ground-truth answers (text outputs for free-response training)
# ---------------------------------------------------------------------------

class GtAnswers(BaseModel):
    """Per-view ground-truth answers for all tasks.

    task1      : object category label (always "blocks")
    task_main  : total block count (visible + hidden structural supporters)
    task2      : number of top-layer visible blocks
    task3      : number of distinct color columns (visible)
    task4      : number of distinct height layers (visible)
    task5      : number of visible blocks
    task6      : number of visible support blocks
    task7      : number of visible support-column blocks
    task8      : number of visible blocks supported by a non-visible block
    task9      : number of connected block clusters
    task10     : number of blocks visible in this view but NOT in the next view (view_idx+1 mod 4)
    task11     : number of blocks visible in this view but NOT in the opposite view (view_idx+2 mod 4)
    visible_blocks : ordered list of [i, j, k] coords of all visible blocks from this view
    """
    task1: Optional[Literal["blocks"]] = None
    task_main: Optional[int] = None
    task2: Optional[int] = None
    task3: Optional[int] = None
    task4: Optional[int] = None
    task5: Optional[int] = None
    task6: Optional[int] = None
    task7: Optional[int] = None
    task8: Optional[int] = None
    task9: Optional[int] = None
    task10: Optional[int] = None
    task11: Optional[int] = None
    visible_blocks: list[list[int]] = []  # each entry is [i, j, k]


# ---------------------------------------------------------------------------
# GT segmentation masks (one PNG per task)
# ---------------------------------------------------------------------------

class GtMasks(BaseModel):
    """Paths to per-task GT segmentation mask PNGs (relative to dataset root).
    Each mask is a grayscale PNG where pixel value encodes the task-specific label.
    None if the mask was not generated (task disabled at generation time)."""
    task1: Optional[str] = None
    task2: Optional[str] = None
    task3: Optional[str] = None
    task4: Optional[str] = None
    task5: Optional[str] = None
    task6: Optional[str] = None
    task7: Optional[str] = None
    task8: Optional[str] = None
    task9: Optional[str] = None
    task10: Optional[str] = None
    task11: Optional[str] = None


# ---------------------------------------------------------------------------
# MCQ data per task
# ---------------------------------------------------------------------------

class McqTask(BaseModel):
    """MCQ data for one task within one view.

    correct_letter : which choice letter (A–E) is correct
    choices        : numeric answer per letter — present for task_main only
                     (task_main MCQ is text/number only, has no choice images)
    choice_types   : distractor type label per letter (e.g. "correct", "high_by_2")
    choice_images  : paths to the rendered MCQ choice PNGs (relative to dataset root)
                     None for task_main (numeric choices have no image)
    """
    correct_letter: str
    choices: Optional[dict[str, int]] = None      # task_main only
    choice_types: dict[str, str]
    choice_images: Optional[dict[str, str]] = None  # None for task_main


# ---------------------------------------------------------------------------
# Per-view data
# ---------------------------------------------------------------------------

class ViewSample(BaseModel):
    """All data for one rendered view of a scene (one view_XX/offsetO_fovF_distD/ subfolder).

    view_idx    : 1-based azimuth index (1–4), corresponding to base azimuths 0°, 90°, 180°, 270°
    offset      : azimuth offset in degrees added to the base azimuth for this render pass
    fov_mult    : FOV multiplier (1.0 = default lens; >1 = wider)
    dist_mult   : camera distance multiplier (1.0 = default distance)
    view_image  : path to view.png (relative to dataset root)
    gt_answers  : ground-truth text answers for all tasks
    gt_masks    : per-task segmentation mask PNG paths
    mcq         : MCQ metadata keyed by task id ("task_main", "task1" … "task11")
    """
    view_idx: int
    offset: float
    fov_mult: float
    dist_mult: float
    view_image: str  # e.g. "level_1/sample_01/view_01/offset3_fov1_dist1/view.png"
    gt_answers: GtAnswers
    gt_masks: GtMasks
    mcq: dict[str, McqTask]


# ---------------------------------------------------------------------------
# Per-scene (sample) data
# ---------------------------------------------------------------------------

class SceneSample(BaseModel):
    """All data for one generated block scene, across all rendered views.

    sample_idx    : 1-based sample index
    height_map    : 2-D list [row][col] of block stack heights
    rows, cols    : grid dimensions
    total_blocks  : total blocks in the scene (sum of height_map)
    block_textures: paths to the block edge texture PNGs used to render this scene
                    (relative to dataset root, e.g. "block_edge_texture_0.png")
    views         : list of rendered views (one entry per view_XX/offsetO_fovF_distD/ subfolder)
    """
    sample_idx: int
    height_map: list[list[int]]
    rows: int
    cols: int
    total_blocks: int
    block_textures: list[str]
    views: list[ViewSample]


# ---------------------------------------------------------------------------
# Top-level dataset manifest
# ---------------------------------------------------------------------------

class SpatialIQDataset(BaseModel):
    """Top-level manifest for a SpatialIQ dataset directory.

    dataset_id : timestamp-based identifier matching the output directory name
                 (e.g. "data_20260406_120000")
    samples    : all scene samples across all difficulty levels, sorted by level then sample_idx
    """
    dataset_id: str
    samples: list[SceneSample]
