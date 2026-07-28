# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Block Dataset Generator for Isaac Sim 4.5
==========================================
Generates 3D block structures and renders them from multiple camera angles.
Designed for benchmarking multimodal LLMs on 3D block counting from images.

TWO USAGE MODES:
  1. Script Editor (async)  — paste the entire file into the Isaac Sim Script Editor
  2. CLI / Standalone (sync) — run with Isaac Sim's Python (no UI paste):
       Windows:  cd %LOCALAPPDATA%\ov\pkg\isaac-sim-4.5.0  &&  python.bat path\to\blocks_render_image.py [--headless]
       Linux:    cd ~/.local/share/ov/pkg/isaac-sim-4.5.0  &&  ./python.sh path/to/blocks_render_image.py [--headless]
     Or use data_generation/run_blocks_render.bat (Windows) / run_blocks_render.sh (Linux). See README_CLI.md.

Tested against the Isaac Sim 4.5 "Useful Snippets" multi_camera.py pattern:
  - Cameras created with UsdGeom.Camera (pure USD)
  - Render products take the *prim path string* (not an OG node)
  - RGB annotators read via annotator.get_data()
  - Images saved with PIL Image.fromarray (Isaac 4.5 ships PIL)
"""

# ── Standalone bootstrap (must happen before any omni imports) ───────────────
import os
import sys
_standalone = "omni.isaac" not in sys.modules
if _standalone:
    _headless = "--headless" in sys.argv or os.environ.get("ISAAC_HEADLESS", "").lower() in ("1", "true", "yes")
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
    # Match GUI: Rendering → "RTX - Interactive (Path Tracing)" vs default real-time stack.
    # See BLOCK_RENDER_RTX_PATH_TRACING below; must be set before SimulationApp starts Kit.
    if os.environ.get("BLOCK_RENDER_RTX_PATH_TRACING", "").lower() in ("1", "true", "yes"):
        sys.argv.append("--/persistent/rtx/modes/pt/enabled=true")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": _headless})

# ── Imports ──────────────────────────────────────────────────────────────────
import gc
import hashlib
import io
import concurrent.futures
import json
import math
import threading
import time
from datetime import datetime
import random

import carb.settings
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from omni.replicator.core import AnnotatorRegistry
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

# =============================================================================
# IMPORTANT: PATHS TO SET IN YOUR SYSTEM
# =============================================================================
# Resolve default paths relative to this script. Falls back to the current
# working directory when pasted into the Isaac Sim Script Editor (no __file__).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# Directory of texture images; one is chosen at random per sample. If invalid or empty, no file is used (procedural texture).
BLOCK_TEXTURES_PATH = os.environ.get("BLOCK_TEXTURES_PATH", os.path.join(SCRIPT_DIR, "textures_cube"))

_OUTPUT_BASE = os.environ.get("SPATIALIQ_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "output"))
DATASET_SPLIT = "eval_set"
OUTPUT_DIR = os.path.join(_OUTPUT_BASE, DATASET_SPLIT, time.strftime("data_%Y%m%d_%H%M%S"))

# =============================================================================
# OPTIONAL S3 UPLOAD — set UPLOAD_TO_S3=1 to upload outputs to an S3-compatible
# object store instead of writing locally. The endpoint and credentials are read
# from environment variables; nothing is hardcoded.
# =============================================================================
UPLOAD_TO_S3          = os.environ.get("UPLOAD_TO_S3", "").lower() in ("1", "true", "yes")
S3_ENDPOINT           = os.environ.get("S3_ENDPOINT", "")
S3_REGION             = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID      = os.environ.get("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY  = os.environ.get("S3_SECRET_ACCESS_KEY", "")
S3_BUCKET             = os.environ.get("S3_BUCKET", "dataset")

# When ENVIRONMENT_USD_OVERRIDE is set, it is used exclusively (no CDN fetch).
# Set to a local .usd path to use your own background environment, or leave None
# to use the Isaac public content CDN (ENVIRONMENT_ASSETS_BASE_URL below).
ENVIRONMENT_USD_OVERRIDE = None
# Examples (point at your own local USD files):
# ENVIRONMENT_USD_OVERRIDE = "/path/to/environments/simple_room/simple_room.usd"
# ENVIRONMENT_USD_OVERRIDE = "/path/to/environments/warehouse/warehouse.usd"

# Fallback when ENVIRONMENT_USD_OVERRIDE is None: Isaac public content (requires network for HTTPS).
ENVIRONMENT_ASSETS_BASE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/"
)
# Path under ENVIRONMENT_ASSETS_BASE_URL (no leading slash).
ENVIRONMENT_USD_RELATIVE_PATH = "Hospital/hospital.usd"
# If True and ENVIRONMENT_USD_RELATIVE_PATH is non-empty, reference URL = BASE + RELATIVE (requires network for HTTPS).
LOAD_ENVIRONMENT_USD = True
# Prim path for the referencing Xform (cleared on each build_block_scene).
ENVIRONMENT_PRIM_PATH = "/World/Environment"

# =============================================================================
# CONFIGURATION
# =============================================================================
# # Grid dimensions: H = number of rows, W = number of columns (used when not using driver)
# H, W = 1, 1

# # Mode: "manual" = use HEIGHTS dict; "random" = uniform random 0..max_height per column
# MODE = "manual"
# MAX_HEIGHT = 1   # For random mode: each column height is randint(0, MAX_HEIGHT) inclusive

# # Height of each column (stack count per cell). Used when MODE == "manual".
# # Keys are (row, col) 1-based. Example: (1, 1) = top-left, (H, W) = bottom-right. Unspecified cells default to 0.
# HEIGHTS = {
#     (1, 1): 1, (1, 2): 0, (1, 3): 3, (1, 4): 2,
#     (2, 1): 1, (2, 2): 0, (2, 3): 0, (2, 4): 2,
#     (3, 1): 3, (3, 2): 2, (3, 3): 0, (3, 4): 1,
#     (4, 1): 1, (4, 2): 3, (4, 3): 2, (4, 4): 3,
# }

# Driver: 5 difficulty levels; each level has SCENES_PER_LEVEL scenes, NUM_VIEWS views per scene.
# Each level is (L, W, H) = rows x cols x max stack height. Levels with multiple shape variants
# distribute scenes evenly across variants (e.g. 10 scenes → 5 of each variant for level_2/3).
SCENES_PER_LEVEL = 100000

VIEW_AZIMUTHS_BASE = [0, 90, 180, 270]   # base azimuth angles; OFFSETS is added at render time
OFFSETS = [3.0, 12.5]  # list of azimuth offsets (degrees); one full render pass per offset value
NUM_VIEWS = len(VIEW_AZIMUTHS_BASE)

# FOV multipliers and distance multipliers — must be the same length; index i defines one (fov, dist) pair.
# Each pair produces one fov_XX subfolder under every view_XX folder.
# FOV: 1.0 = default lens. >1 = wider angle (zoom out); <1 = zoom in. Only lens changes, not camera position.
# Dist: 1.0 = current camera position. 0.0 = camera at scene center. Scales along the ray from center to camera.
FOV_MULTIPLIERS  = [1.0, 3.0]
DIST_MULTIPLIERS = [1.0, 0.3]  # must be same length as FOV_MULTIPLIERS

# (level_name, min_blocks, shape_variants). Each variant is (rows, cols, max_stack).
# Total blocks per scene = Uniform(min_blocks, max_possible_blocks) inclusive.
LEVELS = [
    ("level_1", 3,  [(4, 4, 4)]),   # UPDATE: USING ONE GLOBAL LEVEL
    # ("level_2", 2,  [(3, 1, 3), (1, 3, 3)]),                        # 3x1x3 or 1x3x3, max 9
    # ("level_3", 2,  [(3, 2, 2), (2, 3, 2)]),                        # 3x2x2 or 2x3x2, max 12
    # ("level_4", 4,  [(2, 2, 3), (3, 3, 2)]),                        # 3x3x2, max 18
    # ("level_5", 6,  [(3, 3, 3)]),                                    # 3x3x3, max 27

]

BLOCK_SIZE = 0.05           # metres (base size before SCALE)
SPACING = 0.0              # 0 = blocks flush (no gaps)
SCALE = 2.0                # Scale factor for block positions, sizes, and camera
IMAGE_RES = (512, 512)
RT_SUBFRAMES = 32
MASK_RT_SUBFRAMES = 4   # subframes for per-block visibility probe (segmentation only)
NUM_FRAMES = 1


# Camera margin multiplier — how much padding around the blocks (1.0 = tight fit)
CAMERA_MARGIN = 1.75

# How many times further the camera is from block center B (1.0 = original distance, 1.4 = 1.4x further).
CAMERA_DISTANCE_MULTIPLIER = 1.0

# Viewpoint rotation in the x-y plane (degrees). 0 = default viewing angle;
# 0–360 exclusive, rotating clockwise. Camera z-height and intrinsics unchanged.
# (Set per-view in level driver.)
CAMERA_AZIMUTH_DEG = 0.0

# If True, run_sync uses LEVELS driver (5 levels x 10 samples x 5 views). Else single-scene.
USE_LEVELS_DRIVER = True

# If set to a path (e.g. ".../level_1/sample_01/scene.json"), load that scene and re-render it
# (same NUM_VIEWS), saving views into that scene's folder. Overrides USE_LEVELS_DRIVER.
SCENE_JSON_PATH = None  # e.g. "/path/to/data/eval_set/data_.../level_1/sample_01/scene.json" to re-render a saved scene
# Camera z-height multiplier. 1 = default (diagonal offset). 0 = camera at z=0.
# Values in between scale the default height linearly.
CAMERA_Z_MULTIPLIER = 1.1

# USD default camera intrinsics (aperture in tenths of scene unit, focal length in same units).
# These match UsdGeom.Camera defaults and define what FOV_MULTIPLIER=1 produces.
_USD_CAM_H_APERTURE   = 20.955  # horizontal aperture (≈ 23.6° full horizontal FOV with default focal length)
_USD_CAM_V_APERTURE   = 20.955  # set equal to H for square (512×512) renders
_USD_CAM_FOCAL_LENGTH = 50.0    # focal length

# View0 visibility: depth-buffer bin size = block_size * this (smaller = finer, fewer false occlusions). 0.125 = ~8x8 bins/block.
VIEW0_PATCH_SCALE = 0.125

# Ground truth: re-renders with colored blocks for segmentation / spatial tasks.
# All GT filenames use task prefix so files group together: task1_gt.png, task1_gt_mask.npy, etc. in view_XX/.
GENERATE_GT_T1 = True   # Blocks vs background: all blocks one solid color (random per scene), save task1_gt.png
GENERATE_GT_T2 = True   # Top-most (and optionally left/right/bottom/nearest/furthest). See T2_TOP_ONLY.
T2_TOP_ONLY = True   # If True, only save top-most block(s) as task2_gt.png. If False, save 6 images: task2_gt_top.png, task2_gt_left.png, etc.
GENERATE_GT_T3 = True   # Per-column colors (task3_gt.png)
GENERATE_GT_T4 = True   # Per-layer colors (task4_gt.png)
GENERATE_GT_T5 = True   # Per visible-block colors (task5_gt.png)
GENERATE_GT_T6 = True   # Blocks that directly support the top-most block(s) (task6_gt.png)
GENERATE_GT_T7 = True   # Entire support column(s) of the top-most block(s), excluding top-most (task7_gt.png)
GENERATE_GT_T8 = True   # Visible blocks directly supported by a non-visible block (task8_gt.png)
GENERATE_GT_T9 = True   # Per-cluster colors: each connected cluster gets a unique color (task9_gt.png)
GENERATE_GT_T10 = True  # For view-N: blocks visible in this view but not in View (N+1)%4 (task10_gt.png)
GENERATE_GT_T11 = True  # For view-N: blocks visible in this view but not in View (N+2)%4 — 180° opposite (task11_gt.png)
# Random solid color per scene for GT and MCQ highlight tint (T1, T2, T6–T8, T10).
GT_HIGHLIGHT_MATERIAL_PATH = "/World/Looks/GTHighlightMaterial"
# Multiple-choice: 5 shuffled images per task (taskN_mcq_choiceA..E.png); answers consolidated in gt.json.
# When MCQ is enabled the correct choice image serves as the GT visual; no separate overlay is saved.
# When MCQ is disabled, taskN_gt_overlay.png is composited from view.png for human scoring instead.
GENERATE_MCQ = True
# Extra render step(s) after changing materials so the pipeline picks up new bindings before capture.
GT_EXTRA_RENDER_STEPS = 3

# Log wall time for each saved file (PNG/NPY/JSON/text) to stdout.
RECORD_FILE_WRITE_TIMINGS = True

# Deterministic color palette for T3/T4/T5/T9 (same order every run). RGB 0-255, normalized to 0-1 when creating materials.
# 64 entries so Task 5 can assign a unique color to every block in a fully-packed 4×4×4 scene.
COLORS_PALETTE_255 = [
    (0, 0, 255),     # 1.  Blue
    (230, 25, 75),   # 2.  Red
    (60, 180, 75),   # 3.  Green
    (255, 225, 25),  # 4.  Yellow
    (245, 130, 49),  # 5.  Orange
    (145, 30, 180),  # 6.  Purple
    (70, 240, 240),  # 7.  Cyan
    (154, 99, 36),   # 8.  Brown
    (255, 250, 200), # 9.  Beige
    (210, 245, 60),  # 10. Lime
    (250, 190, 212), # 11. Pink
    (0, 128, 128),   # 12. Teal
    (220, 190, 255), # 13. Lavender
    (128, 0, 0),     # 14. Maroon
    (170, 255, 195), # 15. Mint
    (128, 128, 0),   # 16. Olive
    (255, 216, 177), # 17. Apricot
    (128, 128, 128), # 18. Grey
    (240, 50, 230),  # 19. Magenta
    (0, 0, 0),       # 20. Black
    (0, 0, 128),     # 21. Navy
    (255, 0, 128),   # 22. Rose
    (0, 200, 100),   # 23. Emerald
    (200, 200, 0),   # 24. Gold
    (180, 90, 0),    # 25. Sienna
    (80, 0, 255),    # 26. Indigo
    (0, 255, 200),   # 27. Aquamarine
    (100, 60, 20),   # 28. Chocolate
    (255, 180, 50),  # 29. Amber
    (150, 255, 0),   # 30. Chartreuse
    (255, 100, 150), # 31. Salmon
    (0, 90, 90),     # 32. Dark Teal
    (180, 130, 255), # 33. Periwinkle
    (100, 0, 0),     # 34. Dark Maroon
    (100, 255, 160), # 35. Sea Green
    (100, 100, 0),   # 36. Dark Olive
    (255, 180, 130), # 37. Peach
    (80, 80, 80),    # 38. Dark Grey
    (200, 0, 200),   # 39. Violet
    (255, 255, 255), # 40. White
    (0, 150, 255),   # 41. Sky Blue
    (200, 0, 50),    # 42. Crimson
    (0, 130, 60),    # 43. Forest Green
    (200, 160, 0),   # 44. Dark Yellow
    (220, 60, 0),    # 45. Burnt Orange
    (100, 0, 150),   # 46. Dark Purple
    (0, 200, 220),   # 47. Turquoise
    (120, 70, 20),   # 48. Dark Brown
    (255, 230, 100), # 49. Pale Yellow
    (180, 230, 30),  # 50. Yellow-Green
    (255, 140, 180), # 51. Light Pink
    (0, 60, 80),     # 52. Dark Cyan
    (160, 100, 220), # 53. Lilac
    (80, 20, 0),     # 54. Dark Sienna
    (120, 230, 180), # 55. Light Mint
    (80, 80, 20),    # 56. Dark Khaki
    (230, 150, 80),  # 57. Sandy
    (50, 50, 50),    # 58. Charcoal
    (180, 0, 180),   # 59. Dark Magenta
    (200, 230, 255), # 60. Ice Blue
    (60, 100, 200),  # 61. Cornflower
    (180, 60, 80),   # 62. Dusty Rose
    (40, 160, 80),   # 63. Fern
    (255, 200, 60),  # 64. Sunflower
]
# Normalized to 0-1 for UsdPreviewSurface.
GT_COLOR_PALETTE = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in COLORS_PALETTE_255]

# Edge color for block faces (RGB 0–255). Default #e9d4c4 (light beige).
BLOCK_EDGE_COLOR = (193, 150, 112)
# Edge = average texture color scaled by this factor (1.0 = same, <1 = darker).
BLOCK_EDGE_DARKEN = 0.8

# Block geometry: "cube" (procedural UV cube), "factory_box" (textured cube from textures_box/), or "asset" (USD references).
BLOCK_SHAPE = "cube"
# When BLOCK_SHAPE == "asset", how each block picks its asset type.
# "random" = deterministic per (i,j,k) from ASSET_RANDOM_SEED; "alternate" = parity of i+j+k;
# or a specific key name (e.g. "factory_box") to use that asset for all blocks.
ASSET_BLOCK_MODE = "soup_can"
# ASSET_BLOCK_MODE = "spam"
# ASSET_BLOCK_MODE = "mug"
# ASSET_BLOCK_MODE = "bowl"

ASSET_RANDOM_SEED = 42
# If True, each asset block gets an extra rotation about +Z from that asset's `random_z_rotation_degrees`.
ASSET_RANDOM_Z_ROTATION = True
# Default lateral gap between grid cells in X/Y (metres before SCALE) when an asset omits `lateral_gap_xy`.
ASSET_LATERAL_GAP_XY = 0.003
# USD asset base directory and per-asset specs.
# Per-asset keys: usd (path), rotate_x/y/z_deg, random_z_rotation_degrees, scale_mult,
# spacing_xy / spacing_x / spacing_y (extra XY gap between cells, metres before SCALE),
# spacing_z (extra Z gap between layers, metres before SCALE), lateral_gap_xy (per-asset override),
# stack_z_extent_scale (multiply fitted Z height; <1 for tighter stacks on assets with loose AABBs).
ASSET_USD_BASE_DIR = os.environ.get("ASSET_USD_BASE_DIR", os.path.join(SCRIPT_DIR, "assets"))
# Textures directory for factory_box (procedural textured-cube mode).
FACTORY_BOX_TEXTURES_DIR = os.environ.get("FACTORY_BOX_TEXTURES_DIR", os.path.join(SCRIPT_DIR, "textures_box"))
# Random Z rotations applied to factory_box blocks (deterministic per block from ASSET_RANDOM_SEED).
FACTORY_BOX_RANDOM_Z_ROTATIONS = [0, 90, 180, 270]
ASSET_SPECS = {
    "soup_can": {
        "usd": os.path.join(ASSET_USD_BASE_DIR, "005_tomato_soup_can.usd"),
        "rotate_x_deg": 90.0,
        "rotate_y_deg": 0.0,
        "rotate_z_deg": 0.0,
        "random_z_rotation_degrees": [0, 45, 90, 135, 180, 225, 270, 315],
        "scale_mult": 1,
        "spacing_xy": -0.015,
        "spacing_z": 0.0,
        "lateral_gap_xy": 0.0,
        "stack_z_extent_scale": 1,
    },
    "mug": {
        "usd": os.path.join(ASSET_USD_BASE_DIR, "025_mug.usd"),
        "rotate_x_deg": 90.0,
        "rotate_y_deg": 0.0,
        "rotate_z_deg": 0.0,
        "random_z_rotation_degrees": [0, 90, 180, 270],
        "scale_mult": 1.0,
        "spacing_x": 0,
        "spacing_y": 0,
        "spacing_z": 0.0,
        "lateral_gap_xy": 0,
        "stack_z_extent_scale": 1,
    },
    "spam": {
        "usd": os.path.join(ASSET_USD_BASE_DIR, "010_potted_meat_can.usd"),
        "rotate_x_deg": 90.0,
        "rotate_y_deg": 0.0,
        "rotate_z_deg": 0.0,
        "random_z_rotation_degrees": [0, 180],
        "scale_mult": 1.0,
        "spacing_x": 0,
        "spacing_y": 0,
        "spacing_z": 0.0,
        "lateral_gap_xy": 0.0,
        "stack_z_extent_scale": 1,
    },
    "bowl": {
        "usd": os.path.join(ASSET_USD_BASE_DIR, "024_bowl.usd"),
        "rotate_x_deg": 90.0,
        "rotate_y_deg": 0.0,
        "rotate_z_deg": 0.0,
        "random_z_rotation_degrees": [0, 180],
        "scale_mult": 1.0,
        "spacing_x": 0.0,
        "spacing_y": 0.0,
        "spacing_z": 0.0,
        "lateral_gap_xy": 0.002,
        "stack_z_extent_scale": 1,
    },
}

# =============================================================================
# FUNCTION TIMING
# =============================================================================
# Set to True to print a wall-clock summary table after each run.
PROFILE_FN_TIMINGS = True

_fn_timings: dict = {}  # {fn_name: {"total_s": float, "calls": int}}


def _time_fn(fn):
    """Decorator: accumulate wall-clock time per function when PROFILE_FN_TIMINGS is True."""
    if not PROFILE_FN_TIMINGS:
        return fn
    import functools
    name = fn.__name__
    _fn_timings[name] = {"total_s": 0.0, "calls": 0}

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _fn_timings[name]["total_s"] += time.perf_counter() - t0
        _fn_timings[name]["calls"] += 1
        return result
    return _wrapper


def _print_fn_timing_summary():
    """Print a sorted table of accumulated function wall-clock timings."""
    if not PROFILE_FN_TIMINGS or not _fn_timings:
        return
    rows = sorted(_fn_timings.items(), key=lambda kv: kv[1]["total_s"], reverse=True)
    print("\n[Timing] Function timing summary:", flush=True)
    print(f"  {'Function':<48} {'Total (s)':>10} {'Calls':>7} {'Avg (ms)':>10}", flush=True)
    print(f"  {'-'*48} {'-'*10} {'-'*7} {'-'*10}", flush=True)
    for name, d in rows:
        avg_ms = d["total_s"] / d["calls"] * 1000 if d["calls"] else 0.0
        print(f"  {name:<48} {d['total_s']:>10.2f} {d['calls']:>7} {avg_ms:>10.1f}", flush=True)
    print("", flush=True)


# =============================================================================
# HELPERS
# =============================================================================
SCENE_JSON_FILENAME = "scene.json"

# Filled in build_block_scene for asset-block mode; None in cube mode.
# Used by _block_world_center, _block_world_extent, _camera_and_bounds_for_azimuth, etc.
BLOCK_ASSET_LAYOUT = None
_asset_bbox_cache = {}  # asset_path -> (center, extent); persists across scenes for speed


def _apply_rtx_path_tracing_if_env():
    """When BLOCK_RENDER_RTX_PATH_TRACING=1, enable Interactive Path Tracing (carb; Script Editor / redundancy)."""
    if os.environ.get("BLOCK_RENDER_RTX_PATH_TRACING", "").lower() not in ("1", "true", "yes"):
        return
    try:
        s = carb.settings.get_settings()
        s.set("/persistent/rtx/modes/pt/enabled", True)
        print("[BlockGen] RTX Interactive Path Tracing enabled (carb /persistent/rtx/modes/pt/enabled)", flush=True)
    except Exception as e:
        print(f"[BlockGen] WARN: could not set RTX path tracing mode: {e}", flush=True)


def _apply_rtx_dlss_exec_mode():
    """Apply rtx/post/dlss/execMode from RTX_DLSS_EXEC_MODE (env BLOCK_RENDER_DLSS_EXEC_MODE)."""
    try:
        carb.settings.get_settings().set("rtx/post/dlss/execMode", RTX_DLSS_EXEC_MODE)
        print(f"[BlockGen] rtx/post/dlss/execMode = {RTX_DLSS_EXEC_MODE}", flush=True)
    except Exception as e:
        print(f"[BlockGen] WARN: rtx/post/dlss/execMode: {e}", flush=True)


def _normalize_rgb_data(rgb_data, view_index=0):
    """Unwrap Replicator RGB annotator output to a single numpy array (H, W, C).
    get_data() may return a list (one per render product), a dict with 'data', or a raw array."""
    if rgb_data is None:
        return None
    if isinstance(rgb_data, (list, tuple)):
        if len(rgb_data) == 0:
            return None
        rgb_data = rgb_data[view_index] if view_index < len(rgb_data) else rgb_data[0]
    if isinstance(rgb_data, dict):
        if "renderProducts" in rgb_data:
            rps = list(rgb_data["renderProducts"].values())
            if not rps:
                return None
            rp = rps[min(view_index, len(rps) - 1)]
            if isinstance(rp, dict):
                rgb_data = rp.get("rgb")
                if isinstance(rgb_data, dict):
                    rgb_data = rgb_data.get("data")
                elif rgb_data is None:
                    rgb_data = rp.get("data")
            else:
                rgb_data = rp
        else:
            rgb_data = rgb_data.get("data") or rgb_data.get("rgb")
        if rgb_data is None:
            return None
    arr = np.asarray(rgb_data)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None
    return arr


def _record_file_timing(label, path, t0):
    """Print and optionally append JSONL for one file-write duration (seconds since t0 = perf_counter())."""
    if not RECORD_FILE_WRITE_TIMINGS:
        return
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  [timing] {label} {os.path.basename(path)} {dt_ms:.2f} ms", flush=True)


def _reset_file_timing_log():
    pass


def _log_sample_start(scene_dir):
    """Append a timestamped line to sample_timing.txt in OUTPUT_DIR (always written locally)."""
    log_path = os.path.join(OUTPUT_DIR, "sample_timing.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as _f:
        _f.write(f"{datetime.now().isoformat()}  {scene_dir}\n")


_s3_thread_local = threading.local()
_upload_executor = None
_pending_uploads: list = []


def _get_s3_client():
    """Return a thread-local boto3 S3 client (safe for concurrent uploads)."""
    if not hasattr(_s3_thread_local, "client"):
        import boto3
        import botocore.config
        _s3_thread_local.client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
            region_name=S3_REGION,
            config=botocore.config.Config(connect_timeout=10, max_pool_connections=16),
        )
    return _s3_thread_local.client


def _get_upload_executor():
    global _upload_executor
    if _upload_executor is None:
        _upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    return _upload_executor


def _do_s3_upload(data: bytes, key: str) -> float:
    """Blocking S3 upload; returns wall-clock seconds elapsed."""
    t0 = time.perf_counter()
    try:
        _get_s3_client().upload_fileobj(io.BytesIO(data), S3_BUCKET, key)
    except Exception as e:
        print(f"  [s3] WARN upload failed for {key}: {e}", flush=True)
    return time.perf_counter() - t0


@_time_fn
def _s3_upload(data: bytes, local_path: str) -> None:
    """Queue an async upload to S3, or write to local_path if UPLOAD_TO_S3 is False."""
    if UPLOAD_TO_S3:
        key = os.path.relpath(local_path, _OUTPUT_BASE).replace("\\", "/")
        _pending_uploads.append(_get_upload_executor().submit(_do_s3_upload, data, key))
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)


def _wait_uploads():
    """Block until all queued S3 uploads finish and print a timing summary."""
    if not _pending_uploads:
        return
    total_s = sum(f.result() for f in concurrent.futures.as_completed(_pending_uploads))
    n = len(_pending_uploads)
    _pending_uploads.clear()
    print(f"  [s3] flushed {n} uploads — {total_s:.2f}s total, {total_s/n*1000:.1f}ms avg", flush=True)


@_time_fn
def save_rgb(rgb_data, file_path, view_index=0):
    """Save an RGBA numpy array to a PNG file (Isaac 4.5 pattern).
    Unwraps list/dict from Replicator rgb annotator if needed."""
    arr = _normalize_rgb_data(rgb_data, view_index)
    if arr is None:
        return
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    t0 = time.perf_counter()
    img = Image.fromarray(arr).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _s3_upload(buf.getvalue(), file_path)
    _record_file_timing("save_rgb", file_path, t0)


def save_scene_json(scene_dir, height_map, level_name=None, sample_idx=None, num_visible=None):
    """Save scene structure to scene_dir/scene.json. If num_visible is set, include visible_blocks and non_visible_blocks (view 0)."""
    rows, cols = len(height_map), len(height_map[0])
    total_blocks = sum(cell for row in height_map for cell in row)
    data = {
        "height_map": height_map,
        "rows": rows,
        "cols": cols,
        "total_blocks": total_blocks,
    }
    if num_visible is not None:
        data["visible_blocks"] = num_visible
        data["non_visible_blocks"] = total_blocks - num_visible
    if level_name is not None:
        data["level"] = level_name
    if sample_idx is not None:
        data["sample_idx"] = sample_idx
    path = os.path.join(scene_dir, SCENE_JSON_FILENAME)
    t0 = time.perf_counter()
    _s3_upload(json.dumps(data, indent=2).encode("utf-8"), path)
    _record_file_timing("scene_json", path, t0)
    return path


def _view_dir(scene_dir, view_idx):
    """Path to view subfolder: scene_dir/view_XX (level > scene > view). View numbering is 1-based (view_01, view_02, ...)."""
    return os.path.join(scene_dir, f"view_{view_idx + 1:02d}")


def _view_fov_dir(scene_dir, view_idx, offset, fov_mult, dist_mult):
    """Path to view+offset+FOV+dist subfolder: scene_dir/view_XX/offsetO_fovF_distD."""
    return os.path.join(_view_dir(scene_dir, view_idx), f"offset{offset:g}_fov{fov_mult:g}_dist{dist_mult:g}")


def _gt_file_path(view_dir, task_id, filename_override=None):
    """Path for a GT mask: view_dir/taskN_gt_mask.png so files group by task. If filename_override is set (e.g. task2_gt_mask_top), use that base."""
    if filename_override is None:
        filename_override = f"{task_id}_gt_mask"
    return os.path.join(view_dir, f"{filename_override}.png")


def _gt_json_path(view_dir):
    """Path for the consolidated GT JSON file: view_dir/gt.json."""
    return os.path.join(view_dir, "gt.json")


def _update_gt_json(view_dir, data):
    """Merge data into the in-memory gt.json cache. Call _flush_gt_json once per view to upload."""
    path = _gt_json_path(view_dir)
    if path not in _GT_JSON_CACHE:
        _GT_JSON_CACHE[path] = {}
    current = _GT_JSON_CACHE[path]
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(current.get(k), dict):
            current[k].update(v)
        else:
            current[k] = v


def _flush_gt_json(view_dir):
    """Upload the accumulated gt.json for this view (call once per view after all updates)."""
    path = _gt_json_path(view_dir)
    if path in _GT_JSON_CACHE:
        _s3_upload(json.dumps(_GT_JSON_CACHE[path], indent=2).encode("utf-8"), path)


def _save_gt_text(view_dir, task_id, value):
    """Merge a single GT answer into gt.json under the task_id key."""
    t0 = time.perf_counter()
    _update_gt_json(view_dir, {task_id: value})
    path = _gt_json_path(view_dir)
    print(f"  -> {path} [{task_id}={value}] (GT text)")
    _record_file_timing("gt_text", path, t0)


def _get_binary_segmentation_from_annotator(seg_annotator):
    """Get semantic_segmentation from annotator, return 2D array with 1=block, 0=background, or None on failure."""
    if seg_annotator is None:
        return None
    data = seg_annotator.get_data()
    if data is None:
        return None
    if isinstance(data, dict):
        seg_array = data.get("data")
        info = data.get("info") or data.get("semantic_ids") or {}
        if seg_array is None:
            return None
        seg_array = np.asarray(seg_array)
        block_ids = set()
        if isinstance(info, dict):
            id_to_labels = info.get("idToLabels")
            if id_to_labels and isinstance(id_to_labels, dict):
                # New format: {'idToLabels': {'2': {'class': 'block'}, ...}}
                for sid, label_info in id_to_labels.items():
                    cls = label_info.get("class", "") if isinstance(label_info, dict) else str(label_info)
                    if cls.lower() == "block":
                        try:
                            block_ids.add(int(sid))
                        except (ValueError, TypeError):
                            block_ids.add(sid)
            else:
                # Old format: {id: "label_str"}
                for sid, label in info.items():
                    if isinstance(label, str) and label.lower() == "block":
                        block_ids.add(int(sid) if isinstance(sid, (int, float)) else sid)
                    if isinstance(sid, str) and sid.lower() == "block":
                        block_ids.add(int(label) if isinstance(label, (int, float)) else label)
        if block_ids:
            mask = np.isin(seg_array, list(block_ids))
        else:
            unique = np.unique(seg_array)
            if len(unique) <= 1:
                return None
            bg_id = 0 if 0 in unique else int(unique.min())
            mask = (seg_array != bg_id)
            if np.mean(mask) > 0.95:
                return None
    else:
        seg_array = np.asarray(data)
        if seg_array.ndim != 2:
            return None
        unique = np.unique(seg_array)
        if len(unique) <= 1:
            return None
        bg_id = 0 if 0 in unique else int(unique.min())
        mask = (seg_array != bg_id)
        if np.mean(mask) > 0.95:
            return None
    return mask.astype(np.int32)


@_time_fn
def _save_gt_mask_png(mask, path):
    """Save integer mask as mono-channel uint8 PNG (mode L)."""
    if mask is None:
        return
    t0 = time.perf_counter()
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), "L").save(buf, format="PNG")
    _s3_upload(buf.getvalue(), path)
    print(f"  -> {path} (GT mask)")
    _record_file_timing("gt_mask_png", path, t0)


def _save_visible_blocks(view_dir, visible_ordered):
    """Store visible block coordinates in gt.json under 'visible_blocks' for compositing."""
    _update_gt_json(view_dir, {"visible_blocks": [[i, j, k] for (i, j, k) in visible_ordered]})


def _get_column_blocks(height_map, column_index):
    """Return set of (i, j, k) for all blocks in the given column. Column index c = i * w + j (row-major)."""
    h, w = len(height_map), len(height_map[0])
    c = column_index
    if c < 0 or c >= h * w:
        return set()
    i, j = c // w, c % w
    return {(i, j, k) for k in range(height_map[i][j])}


def _get_column_order_back_to_front(height_map, azimuth_deg):
    """Return column indices (0 .. h*w-1) in back-to-front order for the given view.
    Depth = closest orthogonal distance from the image plane to the column (front face of top block).
    So when painting the T3 mask, front columns overwrite back columns at occluded pixels."""
    cam_pos, target, _, block_size, h, w, _, _ = _camera_and_bounds_for_azimuth(height_map, azimuth_deg)
    cam_pos_v = Gf.Vec3d(*cam_pos)
    target_v = Gf.Vec3d(*target)
    forward = (target_v - cam_pos_v).GetNormalized()
    depths = []
    for c in range(h * w):
        i, j = c // w, c % w
        # Column center in world space (midpoint in z)
        n_blocks = height_map[i][j]
        if n_blocks == 0:
            depths.append((c, -float("inf")))  # empty column: put first so it gets overwritten
            continue
        cx, cy, _ = _block_world_center(height_map, i, j, 0)
        cz = _column_stack_z_mid(height_map, i, j)
        to_center = Gf.Vec3d(cx - cam_pos_v[0], cy - cam_pos_v[1], cz - cam_pos_v[2])
        # Orthogonal distance from image plane to closest point on column (front face of top block)
        depth_center = to_center[0] * forward[0] + to_center[1] * forward[1] + to_center[2] * forward[2]
        depth = depth_center - block_size / 2.0
        depths.append((c, depth))
    # Sort by depth descending: back (large depth) first, front (small depth) last
    depths.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in depths]


def _get_block_order_back_to_front(height_map, azimuth_deg):
    """Return all block (i, j, k) coords in back-to-front order for the given view.
    Depth = closest orthogonal distance from the image plane to the block (front face).
    Used for T2 mask so front blocks overwrite back (occluded blue pixels become 0)."""
    cam_pos, target, _, _, h, w, _, _ = _camera_and_bounds_for_azimuth(height_map, azimuth_deg)
    cam_pos_v = Gf.Vec3d(*cam_pos)
    target_v = Gf.Vec3d(*target)
    forward = (target_v - cam_pos_v).GetNormalized()
    blocks_with_depth = []
    for i in range(h):
        for j in range(w):
            for k in range(height_map[i][j]):
                cx, cy, cz = _block_world_center(height_map, i, j, k)
                ex, ey, ez = _block_world_extent(height_map, i, j, k)
                half = (ex + ey + ez) / 3.0 / 2.0  # approximate front-face offset
                to_center = Gf.Vec3d(cx - cam_pos_v[0], cy - cam_pos_v[1], cz - cam_pos_v[2])
                # Orthogonal distance from image plane to closest point on block (front face center)
                depth_center = to_center[0] * forward[0] + to_center[1] * forward[1] + to_center[2] * forward[2]
                depth = depth_center - half
                blocks_with_depth.append(((i, j, k), depth))
    blocks_with_depth.sort(key=lambda x: x[1], reverse=True)
    return [b for b, _ in blocks_with_depth]





@_time_fn
def _build_mask_back_to_front(stage, height_map, azimuth, seg_annotator, get_value_for_block):
    """Paint mask back-to-front; returns mask array or None."""
    if seg_annotator is None:
        return None
    mask = None
    for (i, j, k) in _get_block_order_back_to_front(height_map, azimuth):
        _set_block_visibility_selective(stage, {(i, j, k)})
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=MASK_RT_SUBFRAMES)
        binary = _get_binary_segmentation_from_annotator(seg_annotator)
        if binary is not None:
            if mask is None:
                mask = np.zeros(binary.shape, dtype=np.int32)
            mask[binary == 1] = get_value_for_block(i, j, k)
        _set_all_blocks_visible(stage)
    return mask


@_time_fn
def _build_masks_combined(stage, height_map, azimuth, seg_annotator, value_fns):
    """Run the back-to-front block loop ONCE; apply multiple value functions simultaneously.
    value_fns: dict of {key: callable(i, j, k) -> int}
    Returns dict of {key: mask_array_or_None}.
    Produces bit-for-bit identical results to calling _build_mask_back_to_front
    separately for each key (same block order, same overwrite semantics).
    """
    if seg_annotator is None:
        return {key: None for key in value_fns}
    masks = {key: None for key in value_fns}
    for (i, j, k) in _get_block_order_back_to_front(height_map, azimuth):
        _set_block_visibility_selective(stage, {(i, j, k)})
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=MASK_RT_SUBFRAMES)
        binary = _get_binary_segmentation_from_annotator(seg_annotator)
        if binary is not None:
            for key, fn in value_fns.items():
                if masks[key] is None:
                    masks[key] = np.zeros(binary.shape, dtype=np.int32)
                masks[key][binary == 1] = fn(i, j, k)
        _set_all_blocks_visible(stage)
    return masks


# Default camera path used by create_usd_cameras (single perspective camera).
_CAMERA_PRIM_PATH = "/World/Cameras/cam_perspective"


def _set_camera_to_azimuth(stage, height_map, azimuth_deg, dist_mult=1.0):
    """Set the perspective camera prim to the given azimuth (same math as create_usd_cameras)."""
    cam_prim = stage.GetPrimAtPath(_CAMERA_PRIM_PATH)
    if not cam_prim or not cam_prim.IsValid():
        return
    cam_pos, target, _u, _bs, _h, _w, _cr, _cc = _camera_and_bounds_for_azimuth(height_map, azimuth_deg)
    if dist_mult != 1.0:
        cx, cy, cz = target
        px, py, pz = cam_pos
        cam_pos = (cx + dist_mult * (px - cx), cy + dist_mult * (py - cy), cz + dist_mult * (pz - cz))
    xf = UsdGeom.Xformable(cam_prim)
    if not xf:
        return
    mat_op = xf.GetOrderedXformOps()[0] if xf.GetOrderedXformOps() else None
    if not mat_op:
        return
    mat_op.Set(look_at_matrix(cam_pos, target))




@_time_fn
def get_visible_blocks_from_replicator(
    stage, height_map, azimuth, seg_annotator, dist_mult=1.0, restore_azimuth_after=None
):
    """Pixel-level visible set via Replicator back-to-front mask. Returns set of (i,j,k) blocks visible from azimuth."""
    if seg_annotator is None:
        return set()
    _set_camera_to_azimuth(stage, height_map, azimuth, dist_mult)
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
    blocks_ordered = _get_block_order_back_to_front(height_map, azimuth)
    block_to_id = {(i, j, k): idx + 1 for idx, (i, j, k) in enumerate(blocks_ordered)}
    id_to_block = {idx + 1: (i, j, k) for idx, (i, j, k) in enumerate(blocks_ordered)}
    get_value = lambda i, j, k: block_to_id.get((i, j, k), 0)
    mask = _build_mask_back_to_front(stage, height_map, azimuth, seg_annotator, get_value)
    if mask is None:
        if restore_azimuth_after is not None:
            _set_camera_to_azimuth(stage, height_map, restore_azimuth_after, dist_mult)
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
        return set()
    visible_ids = set(np.unique(mask)) - {0}
    visible_set = {id_to_block[vid] for vid in visible_ids if vid in id_to_block}
    if restore_azimuth_after is not None:
        _set_camera_to_azimuth(stage, height_map, restore_azimuth_after, dist_mult)
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
    return visible_set



def load_scene_json(path):
    """Load scene structure from a scene.json file. Returns height_map (list of lists)."""
    with open(path) as f:
        data = json.load(f)
    return data["height_map"]


def look_at_matrix(eye, target, up=(0, 0, 1)):
    """Compute a Gf.Matrix4d camera transform (world-space) that looks from
    *eye* toward *target* with the given *up* vector."""
    eye = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    up = Gf.Vec3d(*up)

    forward = (target - eye).GetNormalized()
    right = Gf.Cross(forward, up).GetNormalized()
    new_up = Gf.Cross(right, forward).GetNormalized()

    m = Gf.Matrix4d(1)
    m.SetRow(0, Gf.Vec4d(right[0],    right[1],    right[2],    0))
    m.SetRow(1, Gf.Vec4d(new_up[0],   new_up[1],   new_up[2],   0))
    m.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0))
    m.SetRow(3, Gf.Vec4d(eye[0],      eye[1],      eye[2],      1))
    return m


def _generate_edge_texture(file_path, size=128, border_pct=0.2):
    """Two triangles per face: top-left triangle #FFFFFF, bottom-right triangle #000000. Black borders."""
    border = int(size * border_pct)
    u = np.linspace(0, 1, size)
    v = np.linspace(0, 1, size)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    top_left_triangle = (uu + vv <= 1.0)
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    arr[top_left_triangle, :3] = 255
    arr[~top_left_triangle, :3] = 0
    edge_rgb = (np.mean(arr[:, :, :3], axis=(0, 1)) * BLOCK_EDGE_DARKEN).clip(0, 255).astype(np.uint8)
    arr[:border, :, :3] = edge_rgb
    arr[-border:, :, :3] = edge_rgb
    arr[:, :border, :3] = edge_rgb
    arr[:, -border:, :3] = edge_rgb
    t0 = time.perf_counter()
    Image.fromarray(arr).save(file_path)
    _record_file_timing("edge_texture", file_path, t0)


def _get_block_texture_image_list():
    """Return list of texture image paths from BLOCK_TEXTURES_PATH (may be empty).
    If path is invalid or folder empty, returns [] (procedural texture used).
    """
    if not BLOCK_TEXTURES_PATH or not os.path.isdir(BLOCK_TEXTURES_PATH):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tga", ".tif", ".tiff", ".webp")
    try:
        candidates = [
            os.path.join(BLOCK_TEXTURES_PATH, f)
            for f in sorted(os.listdir(BLOCK_TEXTURES_PATH))
            if os.path.isfile(os.path.join(BLOCK_TEXTURES_PATH, f)) and f.lower().endswith(exts)
        ]
    except OSError:
        return []
    return candidates


def _load_texture_with_black_edges(source_path, file_path, size=256, border_pct=0.05):
    """Load an image from source_path, resize to size×size, draw edges using darker average color of the texture, save to file_path (PNG)."""
    img = Image.open(source_path).convert("RGB")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    img = img.resize((size, size), resample)
    arr = np.array(img, dtype=np.uint8)
    arr = np.concatenate([arr, np.full((size, size, 1), 255, dtype=np.uint8)], axis=2)  # RGBA
    border = int(size * border_pct)
    edge_rgb = (np.mean(arr[:, :, :3], axis=(0, 1)) * BLOCK_EDGE_DARKEN).clip(0, 255).astype(np.uint8)
    arr[:border, :, :3] = edge_rgb
    arr[-border:, :, :3] = edge_rgb
    arr[:, :border, :3] = edge_rgb
    arr[:, -border:, :3] = edge_rgb
    t0 = time.perf_counter()
    Image.fromarray(arr).save(file_path)
    _record_file_timing("block_texture", file_path, t0)


def create_white_material_with_edges(stage, material_path, texture_path):
    """Create a white material with thick edges (average texture color) via textured UsdPreviewSurface."""
    material = UsdShade.Material.Define(stage, material_path)
    pbr = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.5)

    st_reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, f"{material_path}/diffuseTexture")
    tex.CreateIdAttr("UsdUVTexture")
    # Absolute path with forward slashes; wrap in Sdf.AssetPath for resolver
    abs_path = os.path.abspath(texture_path).replace("\\", "/")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(abs_path))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")

    material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return material


def _create_cube_mesh_with_uvs(stage, prim_path, size):
    """Create a cube mesh with UVs for texturing. Center at origin, extent ±size/2.
    Original: 6 faces, 4 vertices each (no rounding, no chamfer)."""
    h = size / 2.0
    # 6 faces, 4 vertices each. Order for correct outward normals (right-hand rule).
    points = [
        (h, -h, -h), (h, -h, h), (h, h, h), (h, h, -h),   # +X
        (-h, -h, -h), (-h, h, -h), (-h, h, h), (-h, -h, h),  # -X
        (-h, h, -h), (-h, h, h), (h, h, h), (h, h, -h),   # +Y
        (-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h),  # -Y
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),    # +Z
        (-h, -h, -h), (-h, h, -h), (h, h, -h), (h, -h, -h),  # -Z
    ]
    uv_coords = [(0, 0), (1, 0), (1, 1), (0, 1)] * 6

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    uv_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uv_primvar.Set(uv_coords)
    return mesh


# =============================================================================
# ASSET BLOCK HELPERS (USD references, per-asset spacing / rotation)
# =============================================================================

def _get_asset_bbox(asset_path):
    """Open asset USD, compute world bounding box, return (center, extent_xyz). Cached."""
    asset_path = os.path.abspath(asset_path)
    if asset_path in _asset_bbox_cache:
        return _asset_bbox_cache[asset_path]
    if not os.path.isfile(asset_path):
        raise FileNotFoundError(f"Asset USD not found: {asset_path}")
    tmp_stage = Usd.Stage.Open(asset_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    root = tmp_stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = tmp_stage.GetPseudoRoot()
    bbox = bbox_cache.ComputeWorldBound(root)
    aligned = bbox.ComputeAlignedBox()
    mn = aligned.GetMin()
    mx = aligned.GetMax()
    center = Gf.Vec3d((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)
    extent = Gf.Vec3d(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    _asset_bbox_cache[asset_path] = (center, extent)
    print(
        f"[AssetBlock] {os.path.basename(asset_path)}: "
        f"bbox center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}), "
        f"extent=({extent[0]:.3f},{extent[1]:.3f},{extent[2]:.3f})"
    )
    return center, extent


def _axis_aligned_aabb_after_z_rotation(center, extent, z_deg):
    """AABB of the 8 corners of center±extent/2 after rotation about +Z through center (degrees)."""
    rz = math.radians(float(z_deg))
    cos_z, sin_z = math.cos(rz), math.sin(rz)
    hx, hy, hz = extent[0] / 2.0, extent[1] / 2.0, extent[2] / 2.0
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                dx, dy, dz = sx * hx, sy * hy, sz * hz
                corners.append(Gf.Vec3d(
                    center[0] + cos_z * dx - sin_z * dy,
                    center[1] + sin_z * dx + cos_z * dy,
                    center[2] + dz,
                ))
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    zs = [p[2] for p in corners]
    mn = Gf.Vec3d(min(xs), min(ys), min(zs))
    mx = Gf.Vec3d(max(xs), max(ys), max(zs))
    return (mn + mx) / 2.0, mx - mn


def _rotate_offset_intrinsic_xyz(dx, dy, dz, rx_deg, ry_deg, rz_deg):
    """Apply rotations X first, Y second, Z last in world (degrees), matching USD op order RotateZ→RotateY→RotateX."""
    rx, ry, rz = map(math.radians, (float(rx_deg), float(ry_deg), float(rz_deg)))
    crx, srx = math.cos(rx), math.sin(rx)
    y1 = crx * dy - srx * dz
    z1 = srx * dy + crx * dz
    x1 = dx
    cry, sry = math.cos(ry), math.sin(ry)
    x2 = cry * x1 + sry * z1
    z2 = -sry * x1 + cry * z1
    y2 = y1
    crz, srz = math.cos(rz), math.sin(rz)
    x3 = crz * x2 - srz * y2
    y3 = srz * x2 + crz * y2
    z3 = z2
    return x3, y3, z3


def _axis_aligned_aabb_after_euler_xyz(center, extent, rx_deg, ry_deg, rz_deg):
    """AABB of the 8 corners of center±extent/2 after RotateX first, RotateZ last in world (USD op order ZYX)."""
    hx, hy, hz = extent[0] / 2.0, extent[1] / 2.0, extent[2] / 2.0
    corners = []
    for sxi in (-1, 1):
        for syi in (-1, 1):
            for szi in (-1, 1):
                dx, dy, dz = sxi * hx, syi * hy, szi * hz
                x2, y2, z2 = _rotate_offset_intrinsic_xyz(dx, dy, dz, rx_deg, ry_deg, rz_deg)
                corners.append(Gf.Vec3d(center[0] + x2, center[1] + y2, center[2] + z2))
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    zs = [p[2] for p in corners]
    mn = Gf.Vec3d(min(xs), min(ys), min(zs))
    mx = Gf.Vec3d(max(xs), max(ys), max(zs))
    return (mn + mx) / 2.0, mx - mn


def _spec_euler_deg(spec, z_random_deg):
    """Euler degrees (applied X first, Z last in world) from spec plus per-block random Z."""
    rx = float(spec.get("rotate_x_deg", 0.0))
    ry = float(spec.get("rotate_y_deg", 0.0))
    rz = float(spec.get("rotate_z_deg", 0.0)) + float(z_random_deg)
    return rx, ry, rz


def _rotated_bbox_for_spec(center, extent, spec, z_random_deg):
    """Axis-aligned bbox after spec orientation + random Z (matches USD placement)."""
    rx, ry, rz = _spec_euler_deg(spec, z_random_deg)
    if rx == 0.0 and ry == 0.0:
        return _axis_aligned_aabb_after_z_rotation(center, extent, rz)
    return _axis_aligned_aabb_after_euler_xyz(center, extent, rx, ry, rz)


def _spacing_xy_metres(spec):
    """Return (spacing_x, spacing_y) in metres before SCALE from spacing_x/y and/or spacing_xy."""
    if "spacing_x" in spec or "spacing_y" in spec:
        sx = float(spec.get("spacing_x", spec.get("spacing_xy", 0.0)))
        sy = float(spec.get("spacing_y", spec.get("spacing_xy", 0.0)))
        return sx, sy
    s = float(spec.get("spacing_xy", 0.0))
    return s, s


def _stack_z_extent_scale(spec):
    """Z-only scale after uniform fit (1.0 = default). <1 shortens the asset in Z for tighter stacks."""
    return float(spec.get("stack_z_extent_scale", 1.0))


def _lateral_gap_xy_metres(spec):
    """Lateral gap in metres before SCALE: per-asset lateral_gap_xy if set, else ASSET_LATERAL_GAP_XY."""
    if "lateral_gap_xy" in spec and spec["lateral_gap_xy"] is not None:
        return float(spec["lateral_gap_xy"])
    return float(ASSET_LATERAL_GAP_XY)


def _physical_scaled_extent(spec, target_size, z_random_deg=0.0):
    """Axis-aligned extent after RotateXYZ(spec + random Z) and uniform fit to cube edge target_size."""
    center, extent = _get_asset_bbox(spec["usd"])
    c2, e2 = _rotated_bbox_for_spec(center, extent, spec, z_random_deg)
    max_dim = max(e2[0], e2[1], e2[2])
    if max_dim <= 0:
        max_dim = 1.0
    norm_scale = (target_size / max_dim) * spec["scale_mult"]
    return Gf.Vec3d(e2[0] * norm_scale, e2[1] * norm_scale, e2[2] * norm_scale)


def _footprint_xy_world(spec, target_size, z_random_deg=0.0):
    """Per-axis grid pitch (fx, fy): asset extent + spacing*SCALE + lateral_gap*SCALE for each axis."""
    ex = _physical_scaled_extent(spec, target_size, z_random_deg)
    lateral = _lateral_gap_xy_metres(spec) * SCALE
    sx, sy = _spacing_xy_metres(spec)
    fx = ex[0] + sx * SCALE + lateral
    fy = ex[1] + sy * SCALE + lateral
    return fx, fy


def _pick_asset_key(i, j, k, available_keys):
    """Choose asset type for block (i,j,k) based on ASSET_BLOCK_MODE."""
    if not available_keys:
        return None
    mode = (ASSET_BLOCK_MODE or "random").strip().lower()
    if mode in ASSET_SPECS:
        return mode if mode in available_keys else available_keys[0]
    if mode == "alternate":
        return available_keys[(i + j + k) % len(available_keys)]
    rng = random.Random(ASSET_RANDOM_SEED + i * 100_000 + j * 300 + k * 7)
    return rng.choice(available_keys)


def _pick_asset_z_deg(i, j, k, spec):
    """Random extra rotation about +Z from spec random_z_rotation_degrees (deterministic from ASSET_RANDOM_SEED)."""
    if not ASSET_RANDOM_Z_ROTATION:
        return 0.0
    choices = spec.get("random_z_rotation_degrees") or [0, 90]
    rng = random.Random(ASSET_RANDOM_SEED + i * 91_000 + j * 17 + k * 12_345)
    return float(rng.choice(list(choices)))


def _create_asset_block_from_spec(stage, prim_path, spec, target_size, z_random_deg=0.0):
    """Reference a USD asset: RotateX first then RotateZ last in world (USD op order ZYX), then scale-to-fit, re-center."""
    center, extent = _get_asset_bbox(spec["usd"])
    rx, ry, rz_total = _spec_euler_deg(spec, z_random_deg)
    c2, e2 = _rotated_bbox_for_spec(center, extent, spec, z_random_deg)
    max_dim = max(e2[0], e2[1], e2[2])
    if max_dim <= 0:
        max_dim = 1.0
    norm_scale = (target_size / max_dim) * spec["scale_mult"]
    sz_scale = _stack_z_extent_scale(spec)

    outer = UsdGeom.Xform.Define(stage, prim_path)

    orient = UsdGeom.Xform.Define(stage, f"{prim_path}/orient")
    orient_xf = UsdGeom.Xformable(orient.GetPrim())
    orient_xf.ClearXformOpOrder()
    orient_xf.AddRotateZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(float(rz_total))
    orient_xf.AddRotateYOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(float(ry))
    orient_xf.AddRotateXOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(float(rx))

    norm = UsdGeom.Xform.Define(stage, f"{prim_path}/orient/norm")
    norm_xf = UsdGeom.Xformable(norm.GetPrim())
    norm_xf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(norm_scale, norm_scale, norm_scale * sz_scale)
    )
    norm_xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(-c2[0], -c2[1], -c2[2])
    )

    ref_xf = UsdGeom.Xform.Define(stage, f"{prim_path}/orient/norm/ref")
    ref_prim = ref_xf.GetPrim()
    ref_prim.GetReferences().AddReference(os.path.abspath(spec["usd"]).replace("\\", "/"))
    UsdGeom.Xformable(ref_prim).ClearXformOpOrder()

    return outer


# Face index → material name for factory_box per-face texturing.
# Face order from _create_cube_mesh_with_uvs: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z(top), 5=-Z(bottom)
_FACTORY_BOX_FACE_GROUPS = [
    ("top",    [4],    "tape_top"),
    ("bottom", [5],    "tape_bottom"),
    ("arrows", [0, 1], "arrows"),
    ("digi",   [2, 3], "digitaltwin"),
]


def _pick_factory_box_z_deg(i, j, k):
    """Deterministic random Z rotation for factory_box block (i,j,k)."""
    if not ASSET_RANDOM_Z_ROTATION:
        return 0.0
    rng = random.Random(ASSET_RANDOM_SEED + i * 91_000 + j * 17 + k * 12_345)
    return float(rng.choice(FACTORY_BOX_RANDOM_Z_ROTATIONS))


def _ensure_factory_box_materials(stage):
    """Create factory_box face materials under /World/Looks/FactoryBox/ if not already present."""
    tex_dir = FACTORY_BOX_TEXTURES_DIR
    for name, fname in [
        ("tape_top",    "tape_top.png"),
        ("tape_bottom", "tape_bottom.png"),
        ("arrows",      "arrows.png"),
        ("digitaltwin", "digitaltwin.png"),
    ]:
        mat_path = f"/World/Looks/FactoryBox/{name}"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            create_white_material_with_edges(stage, mat_path, os.path.join(tex_dir, fname))


def _create_factory_box_block(stage, prim_path, target_size, z_random_deg=0.0):
    """Create a factory_box block: outer Xform (caller adds Translate), orient child holds RotateZ,
    cube mesh with per-face textures inside orient.
    Face layout: tape_top on +Z, tape_bottom on -Z, arrows on ±X, digitaltwin on ±Y."""
    outer = UsdGeom.Xform.Define(stage, prim_path)
    # RotateZ on inner child so rotation is about block center BEFORE translation by caller
    orient = UsdGeom.Xform.Define(stage, f"{prim_path}/orient")
    UsdGeom.Xformable(orient.GetPrim()).AddRotateZOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(float(z_random_deg))
    mesh = _create_cube_mesh_with_uvs(stage, f"{prim_path}/orient/mesh", target_size)
    _ensure_factory_box_materials(stage)
    for group_name, face_indices, mat_name in _FACTORY_BOX_FACE_GROUPS:
        subset = UsdGeom.Subset.Define(stage, f"{prim_path}/orient/mesh/{group_name}")
        subset.CreateElementTypeAttr("face")
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(face_indices)
        mat = UsdShade.Material(stage.GetPrimAtPath(f"/World/Looks/FactoryBox/{mat_name}"))
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)
    return outer


def _compute_asset_layout_from_height_map(height_map):
    """Compute block centers, extents, and horizontal pitch for the given height_map in asset mode.
    Returns a dict with keys: unit_xy, centers, extents, assignment, z_rot_deg; or None if no valid assets."""
    h, w = len(height_map), len(height_map[0])
    center_row, center_col = (h - 1) / 2.0, (w - 1) / 2.0
    target_size = BLOCK_SIZE * SCALE
    available_keys = [k for k, v in ASSET_SPECS.items() if os.path.isfile(v.get("usd", ""))]
    if not available_keys:
        return None
    assignment = {}
    z_rot_deg = {}
    for i in range(h):
        for j in range(w):
            for k in range(height_map[i][j]):
                ak = _pick_asset_key(i, j, k, available_keys)
                assignment[(i, j, k)] = ak
                z_rot_deg[(i, j, k)] = _pick_asset_z_deg(i, j, k, ASSET_SPECS[ak])
    unit_x = 0.0
    unit_y = 0.0
    for (ii, jj, kk), key in assignment.items():
        zdeg = z_rot_deg[(ii, jj, kk)]
        fx, fy = _footprint_xy_world(ASSET_SPECS[key], target_size, zdeg)
        unit_x = max(unit_x, fx)
        unit_y = max(unit_y, fy)
    if unit_x <= 0:
        unit_x = target_size
    if unit_y <= 0:
        unit_y = target_size
    centers = {}
    extents = {}
    for i in range(h):
        for j in range(w):
            z_top = 0.0
            for k in range(height_map[i][j]):
                ak = assignment[(i, j, k)]
                spec = ASSET_SPECS[ak]
                phys = _physical_scaled_extent(spec, target_size, z_rot_deg[(i, j, k)])
                sz = _stack_z_extent_scale(spec)
                hz = phys[2] * sz
                gap_z = float(spec.get("spacing_z", 0.0)) * SCALE
                x = (i - center_row) * unit_x
                y = (j - center_col) * unit_y
                if k == 0:
                    z_c = hz / 2.0
                    z_top = hz
                else:
                    z_c = z_top + gap_z + hz / 2.0
                    z_top = z_top + gap_z + hz
                centers[(i, j, k)] = (x, y, z_c)
                extents[(i, j, k)] = (phys[0], phys[1], hz)
    return {
        "unit_xy": max(unit_x, unit_y),
        "centers": centers,
        "extents": extents,
        "assignment": assignment,
        "z_rot_deg": z_rot_deg,
    }


# =============================================================================
# BLOCK WORLD GEOMETRY HELPERS
# =============================================================================

def _block_world_center(height_map, i, j, k):
    """World-space center of block (i,j,k). Uses asset layout if available, else cube grid."""
    if BLOCK_ASSET_LAYOUT is not None:
        return BLOCK_ASSET_LAYOUT["centers"][(i, j, k)]
    unit = (BLOCK_SIZE + SPACING) * SCALE
    bs = BLOCK_SIZE * SCALE
    h, w = len(height_map), len(height_map[0])
    cr, cc = (h - 1) / 2.0, (w - 1) / 2.0
    return ((i - cr) * unit, (j - cc) * unit, k * unit + bs / 2.0)


def _block_world_extent(height_map, i, j, k):
    """World-space (sx, sy, sz) extent of block (i,j,k). Uses asset layout if available, else cube."""
    if BLOCK_ASSET_LAYOUT is not None:
        return BLOCK_ASSET_LAYOUT["extents"][(i, j, k)]
    s = BLOCK_SIZE * SCALE
    return (s, s, s)


def _column_stack_z_mid(height_map, i, j):
    """Vertical midpoint of the stack in column (i,j) for depth ordering (asset mode aware)."""
    n = height_map[i][j]
    if n <= 0:
        return 0.0
    zs_lo = []
    zs_hi = []
    for k in range(n):
        cx, cy, cz = _block_world_center(height_map, i, j, k)
        ex, ey, ez = _block_world_extent(height_map, i, j, k)
        zs_lo.append(cz - ez / 2.0)
        zs_hi.append(cz + ez / 2.0)
    return (min(zs_lo) + max(zs_hi)) / 2.0


# =============================================================================
# GROUND TRUTH (T1: blocks vs background, T2: spatial extrema, T3/T4/T5: per-column/layer/block colors, T6: blocks directly supporting top-most, T7: full support columns of top-most, T8: visible blocks above non-visible, T9: per-cluster colors, T10: visible in view-N but not view-0)
# =============================================================================

def create_solid_color_material(stage, material_path, r, g, b):
    """Create a solid RGB material (no texture). Used for GT foreground (blue) and background (black)."""
    material = UsdShade.Material.Define(stage, material_path)
    pbr = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(float(r), float(g), float(b)))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return material


_GT_HIGHLIGHT_COLOR_255 = (200, 100, 50)  # updated each scene by create_gt_highlight_material_random


def create_gt_highlight_material_random(stage, material_path, *, rng=None):
    """Solid RGB for GT/MCQ highlight tint. ``rng`` optional; defaults to global ``random``."""
    global _GT_HIGHLIGHT_COLOR_255
    r = rng.uniform(0.15, 1.0) if rng is not None else random.uniform(0.15, 1.0)
    g = rng.uniform(0.15, 1.0) if rng is not None else random.uniform(0.15, 1.0)
    b = rng.uniform(0.15, 1.0) if rng is not None else random.uniform(0.15, 1.0)
    _GT_HIGHLIGHT_COLOR_255 = (round(r * 255), round(g * 255), round(b * 255))
    return create_solid_color_material(stage, material_path, r, g, b)


def _camera_and_bounds_for_azimuth(height_map, azimuth_deg):
    """Same as compute_perspective_camera logic: (cam_pos, target, unit, block_size, h, w, center_row, center_col)."""
    h, w = len(height_map), len(height_map[0])
    center_row, center_col = (h - 1) / 2.0, (w - 1) / 2.0
    block_size = BLOCK_SIZE * SCALE

    if BLOCK_ASSET_LAYOUT is not None:
        centers = BLOCK_ASSET_LAYOUT["centers"]
        extents = BLOCK_ASSET_LAYOUT["extents"]
        unit = BLOCK_ASSET_LAYOUT["unit_xy"]
        all_blocks = list(centers.keys())
        min_x = min(centers[b][0] - extents[b][0] / 2.0 for b in all_blocks)
        max_x = max(centers[b][0] + extents[b][0] / 2.0 for b in all_blocks)
        min_y = min(centers[b][1] - extents[b][1] / 2.0 for b in all_blocks)
        max_y = max(centers[b][1] + extents[b][1] / 2.0 for b in all_blocks)
        min_z = min(centers[b][2] - extents[b][2] / 2.0 for b in all_blocks)
        max_z = max(centers[b][2] + extents[b][2] / 2.0 for b in all_blocks)
        block_size = max(max(e[0], e[1], e[2]) for e in extents.values())
    else:
        unit = (BLOCK_SIZE + SPACING) * SCALE
        min_x = -center_row * unit - block_size / 2.0
        max_x = (h - 1 - center_row) * unit + block_size / 2.0
        min_y = -center_col * unit - block_size / 2.0
        max_y = (w - 1 - center_col) * unit + block_size / 2.0
        max_z_blocks = max(max(row) for row in height_map) if height_map else 1
        min_z = 0.0
        max_z = max_z_blocks * unit + block_size

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    cz = (min_z + max_z) / 2.0
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    extent_z = max_z - min_z
    bbox_diagonal = math.sqrt(extent_x ** 2 + extent_y ** 2 + extent_z ** 2)
    effective_radius = bbox_diagonal / 2.0
    effective_radius = max(effective_radius, block_size * 2.0)
    half_fov_base_rad = math.radians(23.6)
    dist = (effective_radius / math.tan(half_fov_base_rad)) * CAMERA_MARGIN * 1.5
    offset_dir = Gf.Vec3d(-2.5, -2.3, 2.3)
    offset_dir = offset_dir.GetNormalized()
    dx, dy = offset_dir[0] * dist, offset_dir[1] * dist
    default_cam_z = cz + offset_dir[2] * dist
    cam_z = CAMERA_Z_MULTIPLIER * default_cam_z
    theta_rad = math.radians(azimuth_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    cam_x = cx + dx * cos_t + dy * sin_t
    cam_y = cy - dx * sin_t + dy * cos_t
    k = CAMERA_DISTANCE_MULTIPLIER - 1.0
    cam_x = cam_x + k * (cam_x - cx)
    cam_y = cam_y + k * (cam_y - cy)
    cam_z = cam_z + k * (cam_z - cz)
    cam_pos = (cam_x, cam_y, cam_z)
    target = (cx, cy, cz)
    return cam_pos, target, unit, block_size, h, w, center_row, center_col


def get_block_view_info(height_map, azimuth_deg):
    """Return list of ((i, j, k), u, v, depth) for each block in view space (same projection as render camera)."""
    cam_pos, target, _, _, h, w, _, _ = _camera_and_bounds_for_azimuth(height_map, azimuth_deg)
    cam_pos_v = Gf.Vec3d(*cam_pos)
    target_v = Gf.Vec3d(*target)
    forward = (target_v - cam_pos_v).GetNormalized()
    up_world = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(forward, up_world).GetNormalized()
    up_cam = Gf.Cross(right, forward).GetNormalized()
    out = []
    for i in range(h):
        for j in range(w):
            for k in range(height_map[i][j]):
                x, y, z = _block_world_center(height_map, i, j, k)
                pos = Gf.Vec3d(x, y, z)
                rel = pos - cam_pos_v
                depth = Gf.Dot(rel, forward)
                u = Gf.Dot(rel, right)
                v = Gf.Dot(rel, up_cam)
                out.append(((i, j, k), u, v, depth))
    return out


@_time_fn
def get_visible_blocks(height_map, azimuth_deg):
    """Return set of (i, j, k) blocks visible from the given view (perspective + front-face depth buffer, same as view0 logic)."""
    if not height_map or not any(cell for row in height_map for cell in row):
        return set()
    cam_pos, target, _, _, h, w, _, _ = _camera_and_bounds_for_azimuth(height_map, azimuth_deg)
    cam_pos_v = Gf.Vec3d(*cam_pos)
    target_v = Gf.Vec3d(*target)
    forward = (target_v - cam_pos_v).GetNormalized()
    up_world = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(forward, up_world).GetNormalized()
    up_cam = Gf.Cross(right, forward).GetNormalized()
    blocks = []
    for i in range(h):
        for j in range(w):
            for k in range(height_map[i][j]):
                x, y, z = _block_world_center(height_map, i, j, k)
                ex, ey, ez = _block_world_extent(height_map, i, j, k)
                half = (ex + ey + ez) / 3.0 / 2.0
                pos = Gf.Vec3d(x, y, z)
                rel = pos - cam_pos_v
                depth = Gf.Dot(rel, forward)
                u = Gf.Dot(rel, right)
                v = Gf.Dot(rel, up_cam)
                blocks.append((i, j, k, depth, u, v, half))
    depth_front_list = [(b[0], b[1], b[2], b[3] - b[6], b[3], b[4], b[5], b[6]) for b in blocks]
    depth_front_list.sort(key=lambda b: b[3])
    ref_depth = (target_v - cam_pos_v).GetLength()
    # Use the average block half-extent for the cell size normalization (cube mode: block_size/2)
    avg_half = sum(b[6] for b in blocks) / len(blocks) if blocks else (BLOCK_SIZE * SCALE / 2.0)
    cell_size_norm = (avg_half * 2.0 / ref_depth) * VIEW0_PATCH_SCALE
    depth_buf = {}
    visible = set()
    for (i, j, k, depth_front, depth_center, u, v, half) in depth_front_list:
        if depth_front <= 0:
            continue
        u_norm = u / depth_center
        v_norm = v / depth_center
        half_norm = half / depth_center
        u_lo = (u_norm - half_norm) / cell_size_norm
        u_hi = (u_norm + half_norm) / cell_size_norm
        v_lo = (v_norm - half_norm) / cell_size_norm
        v_hi = (v_norm + half_norm) / cell_size_norm
        bin_u_lo, bin_u_hi = int(math.floor(u_lo)), int(math.ceil(u_hi))
        bin_v_lo, bin_v_hi = int(math.floor(v_lo)), int(math.ceil(v_hi))
        is_visible = False
        for bu in range(bin_u_lo, bin_u_hi + 1):
            for bv in range(bin_v_lo, bin_v_hi + 1):
                key = (bu, bv)
                if key not in depth_buf or depth_front < depth_buf[key]:
                    is_visible = True
                    break
            if is_visible:
                break
        if is_visible:
            visible.add((i, j, k))
            for bu in range(bin_u_lo - 1, bin_u_hi + 2):
                for bv in range(bin_v_lo - 1, bin_v_hi + 2):
                    key = (bu, bv)
                    if key not in depth_buf or depth_front < depth_buf[key]:
                        depth_buf[key] = depth_front
    return visible


def get_t6_support_blocks(height_map):
    """Return set of (i, j, k) blocks that directly support the top-most block(s).
    Top-most = blocks at the highest layer in the scene (global_max_k). Support = directly below,
    i.e. (i, j, global_max_k - 1) for every column that has a block at global_max_k. Empty if only one layer."""
    h, w = len(height_map), len(height_map[0])
    if not h or not w:
        return set()
    global_max_k = max(height_map[i][j] for i in range(h) for j in range(w)) - 1
    if global_max_k < 1:
        return set()
    return {
        (i, j, global_max_k - 1)
        for i in range(h)
        for j in range(w)
        if height_map[i][j] >= global_max_k + 1
    }


def get_t7_support_column_blocks(height_map):
    """Return set of (i, j, k) blocks in the entire support column(s) of the top-most block(s), excluding the top-most.
    Top-most = blocks at the highest layer (global_max_k). For each column that has a top-most block, include all
    blocks in that column from the bottom up to but not including the top layer (k = 0 .. global_max_k - 1)."""
    h, w = len(height_map), len(height_map[0])
    if not h or not w:
        return set()
    global_max_k = max(height_map[i][j] for i in range(h) for j in range(w)) - 1
    if global_max_k < 1:
        return set()
    return {
        (i, j, k)
        for i in range(h)
        for j in range(w)
        for k in range(global_max_k)
        if height_map[i][j] >= global_max_k + 1
    }


def get_t8_visible_supported_by_non_visible(visible_set):
    """Return set of visible blocks (i, j, k) that are directly supported by a non-visible block.
    Directly supported = the block directly below (i, j, k-1) exists and is not in visible_set.
    So: visible (i,j,k) with k >= 1 and (i, j, k-1) not in visible_set."""
    if not visible_set:
        return set()
    return {
        (i, j, k)
        for (i, j, k) in visible_set
        if k >= 1 and (i, j, k - 1) not in visible_set
    }


def get_block_clusters(height_map, block_set=None):
    """Return list of clusters (list of sets of (i, j, k)). Adjacency = 6-connectivity (face-sharing only; diagonal does not count).
    If block_set is provided, cluster only those blocks (used for per-view cluster counts)."""
    h, w = len(height_map), len(height_map[0])
    if not h or not w:
        return []
    if block_set is not None:
        all_blocks = set(block_set)
    else:
        all_blocks = {
            (i, j, k)
            for i in range(h)
            for j in range(w)
            for k in range(height_map[i][j])
        }
    if not all_blocks:
        return []

    def neighbors(i, j, k):
        for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            n = (i + di, j + dj, k + dk)
            if n in all_blocks:
                yield n

    clusters = []
    remaining = set(all_blocks)
    while remaining:
        start = remaining.pop()
        cluster = {start}
        stack = [start]
        while stack:
            i, j, k = stack.pop()
            for n in neighbors(i, j, k):
                if n in remaining:
                    remaining.discard(n)
                    cluster.add(n)
                    stack.append(n)
        clusters.append(cluster)
    return clusters


def get_t2_extreme_block_sets(block_view_info, height_map, visible_set):
    """From block view info, return dict of 6 sets (only among visible blocks).
    top/bottom = by column layer (top of column = max k, bottom = k=0). left/right/nearest/furthest = screen/depth extrema."""
    empty = {k: set() for k in ("top", "left", "right", "bottom", "nearest", "furthest")}
    if not block_view_info or not visible_set:
        return empty
    visible_info = [b for b in block_view_info if b[0] in visible_set]
    if not visible_info:
        return empty
    h, w = len(height_map), len(height_map[0])
    global_max_k = max(height_map[i][j] for i in range(h) for j in range(w)) - 1  # highest layer in scene
    # top = visible blocks at the single highest layer in the entire scene (not top of each column)
    top = {entry[0] for entry in visible_info if entry[0][2] == global_max_k}
    # bottom = visible blocks at bottom of their column (layer k=0)
    bottom = {entry[0] for entry in visible_info if entry[0][2] == 0}
    u_vals = [b[1] for b in visible_info]
    d_vals = [b[3] for b in visible_info]
    u_min, u_max = min(u_vals), max(u_vals)
    d_min, d_max = min(d_vals), max(d_vals)
    left = {entry[0] for entry in visible_info if entry[1] == u_min}
    right = {entry[0] for entry in visible_info if entry[1] == u_max}
    nearest = {entry[0] for entry in visible_info if entry[3] == d_min}
    furthest = {entry[0] for entry in visible_info if entry[3] == d_max}
    return {"top": top, "left": left, "right": right, "bottom": bottom, "nearest": nearest, "furthest": furthest}


def _bind_block_material(prim, material):
    """Bind material to a block prim, using strongerThanDescendants in asset/factory_box mode so the
    binding overrides USD reference internal materials or per-face subset bindings."""
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if BLOCK_ASSET_LAYOUT is not None or BLOCK_SHAPE == "factory_box":
        api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
    else:
        api.Bind(material)


def set_blocks_material(stage, material_path):
    """Bind all /World/Blocks/b_* prims to the material at material_path (e.g. for GT T1)."""
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        return
    material = UsdShade.Material(material_prim)
    for prim in _BLOCK_PRIMS.values():
        _bind_block_material(prim, material)


def set_blocks_material_selective(stage, foreground_set, foreground_path, background_path):
    """Bind blocks in foreground_set to foreground_path, all others to background_path. For T2 GT."""
    fg_prim = stage.GetPrimAtPath(foreground_path)
    bg_prim = stage.GetPrimAtPath(background_path)
    if not fg_prim.IsValid() or not bg_prim.IsValid():
        return
    fg_mat = UsdShade.Material(fg_prim)
    bg_mat = UsdShade.Material(bg_prim)
    for key, prim in _BLOCK_PRIMS.items():
        _bind_block_material(prim, fg_mat if key in foreground_set else bg_mat)

        
# Filled in build_block_scene: (i,j,k) -> USD material path. Used to restore per-block textures after GT overwrites bindings.
BLOCK_ORIGINAL_MATERIAL_PATH = {}

# Filled in build_block_scene: (i,j,k) -> Usd.Prim. Eliminates GetChildren()+name-parsing overhead in every
# material/visibility/semantic function.  Cleared and repopulated on each build_block_scene call.
_BLOCK_PRIMS = {}

# Read-cache for _update_gt_json: path -> current dict.  Saves re-reading the file on every call within a view.
_GT_JSON_CACHE = {}


def set_blocks_material_selective_original_bg(stage, foreground_set, foreground_path):
    """Bind foreground_set to foreground_path; other blocks keep their original textured materials (see BLOCK_ORIGINAL_MATERIAL_PATH).
    Use this for T2/T6/T7/T8/T10 GT so non-highlighted blocks match view.png texture layout (not a single WhiteMaterial)."""
    fg_prim = stage.GetPrimAtPath(foreground_path)
    if not fg_prim.IsValid():
        return
    fg_mat = UsdShade.Material(fg_prim)
    _fb = stage.GetPrimAtPath("/World/Looks/WhiteMaterial")
    fallback_mat = UsdShade.Material(_fb) if _fb.IsValid() else None
    for key, prim in _BLOCK_PRIMS.items():
        if key in foreground_set:
            _bind_block_material(prim, fg_mat)
            continue
        if BLOCK_ASSET_LAYOUT is not None:
            # Asset mode: remove any strong GT override so USD reference's internal materials show through
            UsdShade.MaterialBindingAPI.Apply(prim).UnbindAllBindings()
            continue
        orig_path = BLOCK_ORIGINAL_MATERIAL_PATH.get(key)
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        if orig_path:
            op = stage.GetPrimAtPath(orig_path)
            if op.IsValid():
                binding_api.Bind(UsdShade.Material(op))
                continue
        if fallback_mat is not None:
            binding_api.Bind(fallback_mat)


def set_blocks_material_mapping(stage, block_to_material_path, default_path=None):
    """Bind each block (i,j,k) to the material at block_to_material_path[(i,j,k)]. If default_path is set, blocks not in the dict get that material."""
    default_mat = UsdShade.Material(stage.GetPrimAtPath(default_path)) if default_path else None
    _mat_cache = {}  # path string -> UsdShade.Material, avoids repeated GetPrimAtPath for shared paths
    for key, prim in _BLOCK_PRIMS.items():
        path = block_to_material_path.get(key)
        if path is None:
            if default_mat is not None:
                _bind_block_material(prim, default_mat)
            continue
        if path not in _mat_cache:
            mp = stage.GetPrimAtPath(path)
            if not mp.IsValid():
                continue
            _mat_cache[path] = UsdShade.Material(mp)
        _bind_block_material(prim, _mat_cache[path])


def restore_block_original_materials(stage):
    """Re-bind each block to its textured material from build_block_scene.
    GT tasks use BlueMaterial / WhiteMaterial / per-task colors and would otherwise leave all blocks on
    WhiteMaterial (first texture only), so later views would lose per-block texture variety."""
    if BLOCK_ASSET_LAYOUT is not None or BLOCK_SHAPE == "factory_box":
        # Asset/factory_box mode: clear the strong GT override bindings so internal/subset materials take effect
        for prim in _BLOCK_PRIMS.values():
            UsdShade.MaterialBindingAPI.Apply(prim).UnbindAllBindings()
        return
    if not BLOCK_ORIGINAL_MATERIAL_PATH:
        return
    set_blocks_material_mapping(stage, BLOCK_ORIGINAL_MATERIAL_PATH)


def _apply_gt_t3_colors(stage, height_map, palette=None):
    """Create per-column materials from palette and bind blocks. Returns number of materials created."""
    h, w = len(height_map), len(height_map[0])
    n_colors = h * w
    palette = palette or GT_COLOR_PALETTE
    for idx in range(n_colors):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T3_{idx}", r, g, b)
    block_to_path = {}
    for i in range(h):
        for j in range(w):
            path = f"/World/Looks/GT_T3_{i * w + j}"
            for k in range(height_map[i][j]):
                block_to_path[(i, j, k)] = path
    set_blocks_material_mapping(stage, block_to_path)
    return n_colors


def _apply_gt_t4_colors(stage, height_map, palette=None):
    """Color by layer: all blocks at same height (same k) get same color. Height 1 (k=0) = first color, height 2 (k=1) = second, etc."""
    h, w = len(height_map), len(height_map[0])
    max_k = max(height_map[i][j] for i in range(h) for j in range(w)) - 1
    if max_k < 0:
        return 0
    palette = palette or GT_COLOR_PALETTE
    n_pal = len(palette)
    t4_mats = {}
    for k in range(max_k + 1):
        r, g, b = palette[k % n_pal]
        t4_mats[k] = create_solid_color_material(stage, f"/World/Looks/GT_T4_{k}", r, g, b)
    for (i, j, k), prim in _BLOCK_PRIMS.items():
        if k in t4_mats:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(t4_mats[k])
    return max_k + 1


def _apply_gt_t5_colors(stage, height_map, visible_set, palette=None):
    """Create per-visible-block materials from palette; non-visible blocks stay white. Returns number of materials created."""
    visible_ordered = sorted(visible_set)
    if not visible_ordered:
        return 0
    palette = palette or GT_COLOR_PALETTE
    for idx in range(len(visible_ordered)):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T5_{idx}", r, g, b)
    block_to_path = {block: f"/World/Looks/GT_T5_{idx}" for idx, block in enumerate(visible_ordered)}
    set_blocks_material_mapping(stage, block_to_path, default_path="/World/Looks/WhiteMaterial")
    return len(visible_ordered)


def _apply_gt_t9_colors(stage, height_map, palette=None):
    """Color each cluster of blocks (6-adjacency, no diagonal) with a unique color from the palette. Returns number of clusters."""
    clusters = get_block_clusters(height_map)
    if not clusters:
        return 0
    palette = palette or GT_COLOR_PALETTE
    for idx in range(len(clusters)):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T9_{idx}", r, g, b)
    block_to_path = {}
    for idx, cluster in enumerate(clusters):
        path = f"/World/Looks/GT_T9_{idx}"
        for block in cluster:
            block_to_path[block] = path
    set_blocks_material_mapping(stage, block_to_path)
    return len(clusters)


# =============================================================================
# MULTIPLE CHOICE (MCQ) — images per task + integer choices for main
# =============================================================================

def _mcq_seed_int(view_dir, task_id):
    s = f"{os.path.abspath(view_dir)}|{task_id}".encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:12], 16)


# Memoization cache for _shuffled_palette_for_task: (view_dir, task_id) -> list.
# Safe because the function is fully deterministic (fixed seed from view_dir+task_id).
_PALETTE_CACHE = {}


def _shuffled_palette_for_task(view_dir, task_id):
    """Return a shuffled copy of GT_COLOR_PALETTE, deterministic per view+task."""
    key = (view_dir, task_id)
    if key not in _PALETTE_CACHE:
        rng = random.Random(_mcq_seed_int(view_dir, task_id + "_palette"))
        pal = list(GT_COLOR_PALETTE)
        rng.shuffle(pal)
        _PALETTE_CACHE[key] = pal
    return _PALETTE_CACHE[key]


def _all_blocks_set(height_map):
    h, w = len(height_map), len(height_map[0])
    return {(i, j, k) for i in range(h) for j in range(w) for k in range(height_map[i][j])}


def _nonempty_column_indices(height_map):
    h, w = len(height_map), len(height_map[0])
    return [i * w + j for i in range(h) for j in range(w) if height_map[i][j] > 0]


def _mcq_choice_png_path(view_dir, task_id, letter):
    return os.path.join(view_dir, f"{task_id}_mcq_choice{letter}.png")


def _write_mcq_image_shuffle(view_dir, task_id, pil_rgba_list, type_labels, rng):
    """pil_rgba_list and type_labels length 5; one label must be 'correct'.
    Saves shuffled PNG files and records correct letter + choice types in gt.json."""
    letters = ("A", "B", "C", "D", "E")
    pack = list(zip(pil_rgba_list, type_labels))
    rng.shuffle(pack)
    correct_letter = None
    for letter, (img, lbl) in zip(letters, pack):
        if lbl == "correct":
            correct_letter = letter
        out = _mcq_choice_png_path(view_dir, task_id, letter)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _s3_upload(buf.getvalue(), out)
        print(f"  -> {out} (MCQ {task_id})")
    _update_gt_json(view_dir, {"mcq": {
        task_id: {
            "correct_letter": correct_letter or "?",
            "choice_types": {L: t for L, (_, t) in zip(letters, pack)},
        }
    }})
    print(f"  -> gt.json [mcq.{task_id}] correct={correct_letter}")


def _write_task_main_mcq_text(view_dir, total_blocks, rng):
    """Five unique positive integers; one equals total_blocks.

    Options are centered around a randomly offset pivot (not always the correct answer),
    with non-uniform step sizes to avoid a predictable ±1,±2 pattern.
    """
    # Pick a pivot offset: anywhere from -3 to +3 relative to correct answer
    pivot_offset = rng.choice([-3, -2, -1, 0, 0, 1, 2, 3])  # 0 weighted 2x
    pivot = total_blocks + pivot_offset

    # Pick 4 non-uniform offsets from pivot (avoid always using ±1,±2)
    spread_choices = [1, 2, 3, 4, 5]
    rng.shuffle(spread_choices)
    spreads = sorted(spread_choices[:4])  # 4 distinct positive step sizes

    # Build candidates: pivot ± spreads[i], deduplicated
    seen = set()
    pool = []
    # Always include the correct answer
    for v in [total_blocks]:
        if v > 0 and v not in seen:
            seen.add(v)
            pool.append(v)
    # Add candidates around pivot
    for s in spreads:
        for v in [pivot - s, pivot + s, pivot]:
            if v > 0 and v not in seen and len(pool) < 5:
                seen.add(v)
                pool.append(v)
    # Fill remaining slots with random values further away
    for _ in range(40):
        if len(pool) >= 5:
            break
        v = total_blocks + rng.randint(-8, 8)
        if v > 0 and v not in seen:
            seen.add(v)
            pool.append(v)
    if total_blocks not in pool:
        pool[0] = total_blocks
    pool = pool[:5]
    labels = []
    for v in pool:
        if v == total_blocks:
            labels.append("correct")
        elif v > total_blocks:
            labels.append(f"high_by_{v - total_blocks}")
        else:
            labels.append(f"low_by_{total_blocks - v}")
    letters = ("A", "B", "C", "D", "E")
    pack = list(zip(pool, labels))
    rng.shuffle(pack)
    correct_letter = None
    for letter, (val, lbl) in zip(letters, pack):
        if lbl == "correct":
            correct_letter = letter
    _update_gt_json(view_dir, {"mcq": {
        "task_main": {
            "correct_letter": correct_letter or "?",
            "choices": {L: v for L, (v, _) in zip(letters, pack)},
            "choice_types": {L: t for L, (_, t) in zip(letters, pack)},
        }
    }})
    print(f"  -> gt.json [mcq.task_main] correct={correct_letter}")


def _apply_gt_t3_merge_columns(stage, height_map, c_keep, c_merge_into, palette=None):
    """Column(s) c_merge_into use the same color index as c_keep (wrong grouping).
    c_merge_into may be a single int or a list of ints."""
    h, w = len(height_map), len(height_map[0])
    palette = palette or GT_COLOR_PALETTE
    n = h * w
    merge_set = {c_merge_into} if isinstance(c_merge_into, int) else set(c_merge_into)
    for idx in range(n):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T3_{idx}", r, g, b)
    block_to_path = {}
    for i in range(h):
        for j in range(w):
            c = i * w + j
            col_idx = c_keep if c in merge_set else c
            path = f"/World/Looks/GT_T3_{col_idx}"
            for k in range(height_map[i][j]):
                block_to_path[(i, j, k)] = path
    set_blocks_material_mapping(stage, block_to_path)


def _apply_gt_t3_drop_column(stage, height_map, c_drop, palette=None):
    h, w = len(height_map), len(height_map[0])
    palette = palette or GT_COLOR_PALETTE
    n = h * w
    for idx in range(n):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T3_{idx}", r, g, b)
    block_to_path = {}
    for i in range(h):
        for j in range(w):
            c = i * w + j
            if c == c_drop:
                continue
            path = f"/World/Looks/GT_T3_{c}"
            for k in range(height_map[i][j]):
                block_to_path[(i, j, k)] = path
    set_blocks_material_mapping(stage, block_to_path, default_path="/World/Looks/WhiteMaterial")


def _apply_gt_t4_merge_layers(stage, height_map, k_take_color, k_mislabeled, palette=None):
    """Blocks at layer k_mislabeled are colored as layer k_take_color."""
    max_k = max(height_map[i][j] for i in range(len(height_map)) for j in range(len(height_map[0]))) - 1
    if max_k < 0:
        return
    palette = palette or GT_COLOR_PALETTE
    n_pal = len(palette)
    t4_mats = {}
    for k in range(max_k + 1):
        r, g, b = palette[k % n_pal]
        t4_mats[k] = create_solid_color_material(stage, f"/World/Looks/GT_T4_{k}", r, g, b)
    for (i, j, k), prim in _BLOCK_PRIMS.items():
        use_k = k_take_color if k == k_mislabeled else k
        if use_k in t4_mats:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(t4_mats[use_k])


def _apply_gt_t4_drop_layer(stage, height_map, k_drop, palette=None):
    max_k = max(height_map[i][j] for i in range(len(height_map)) for j in range(len(height_map[0]))) - 1
    palette = palette or GT_COLOR_PALETTE
    n_pal = len(palette)
    t4_mats = {}
    for k in range(max_k + 1):
        r, g, b = palette[k % n_pal]
        t4_mats[k] = create_solid_color_material(stage, f"/World/Looks/GT_T4_{k}", r, g, b)
    wm_prim = stage.GetPrimAtPath("/World/Looks/WhiteMaterial")
    white_mat = UsdShade.Material(wm_prim) if wm_prim.IsValid() else None
    for (i, j, k), prim in _BLOCK_PRIMS.items():
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        if k == k_drop:
            if white_mat is not None:
                binding_api.Bind(white_mat)
        elif k in t4_mats:
            binding_api.Bind(t4_mats[k])


def _t2_other_column_top_block(height_map, visible_set, top_set, rng):
    h, w = len(height_map), len(height_map[0])
    opts = []
    for i in range(h):
        for j in range(w):
            if height_map[i][j] == 0:
                continue
            b = (i, j, height_map[i][j] - 1)
            if b in visible_set and b not in top_set:
                opts.append(b)
    if opts:
        return {rng.choice(opts)}
    if top_set:
        b = rng.choice(list(top_set))
        if b[2] > 0:
            return {(b[0], b[1], b[2] - 1)}
    return set(list(visible_set)[:1]) if visible_set else set()




@_time_fn
def _mcq_capture_pil(rgb_annotator):
    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
    for _ in range(GT_EXTRA_RENDER_STEPS):
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
    data = rgb_annotator.get_data()
    arr = _normalize_rgb_data(data, 0)
    if arr is None:
        return None
    return Image.fromarray(arr).convert("RGBA")


def _mcq_remove_extra_blocks(stage):
    """Remove temporary MCQ cubes (names start with b_mcqextra)."""
    blocks_prim = stage.GetPrimAtPath("/World/Blocks")
    if not blocks_prim.IsValid():
        return
    for child in list(blocks_prim.GetChildren()):
        if child.GetName().startswith("b_mcqextra"):
            stage.RemovePrim(child.GetPath())


def _mcq_block_world_xyz(i, j, k, height_map):
    return _block_world_center(height_map, i, j, k)


def _mcq_spawn_extra_block(stage, height_map, suffix, i, j, k):
    """Spawn a block at grid (i,j,k); k may equal height_map[i][j] (stack above top).
    In asset mode spawns a USD asset; in cube mode spawns a textured cube."""
    block_size = BLOCK_SIZE * SCALE
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(suffix))
    pp = f"/World/Blocks/b_mcqextra_{safe}"
    if stage.GetPrimAtPath(pp).IsValid():
        stage.RemovePrim(pp)

    if BLOCK_ASSET_LAYOUT is not None:
        # Compute world position for the extra block; k may be above the layout range.
        if (i, j, k) in BLOCK_ASSET_LAYOUT["centers"]:
            cx, cy, cz = BLOCK_ASSET_LAYOUT["centers"][(i, j, k)]
        else:
            # One level above the top of the stack: estimate from top block extent.
            top_k = k - 1
            if (i, j, top_k) in BLOCK_ASSET_LAYOUT["centers"]:
                tcx, tcy, tcz = BLOCK_ASSET_LAYOUT["centers"][(i, j, top_k)]
                tex, tey, tez = BLOCK_ASSET_LAYOUT["extents"][(i, j, top_k)]
                cx, cy, cz = tcx, tcy, tcz + tez
            else:
                cx, cy, cz = _block_world_center(height_map, i, j, max(k - 1, 0))
                cz += block_size
        if ASSET_BLOCK_MODE == "factory_box":
            z_deg = _pick_factory_box_z_deg(i, j, k)
            _create_factory_box_block(stage, pp, block_size, z_deg)
        else:
            asset_key = _pick_asset_key(i, j, k, [key for key, v in ASSET_SPECS.items() if os.path.isfile(v.get("usd", ""))])
            if asset_key is None:
                asset_key = ASSET_BLOCK_MODE
            spec = ASSET_SPECS[asset_key]
            z_deg = _pick_asset_z_deg(i, j, k, spec)
            _create_asset_block_from_spec(stage, pp, spec, block_size, z_deg)
        _prim = stage.GetPrimAtPath(pp)
        UsdGeom.Xformable(_prim).AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
        prim_for_semantics = _prim
    elif BLOCK_SHAPE == "factory_box":
        x, y, z_pos = _mcq_block_world_xyz(i, j, k, height_map)
        z_deg = _pick_factory_box_z_deg(i, j, k)
        _create_factory_box_block(stage, pp, block_size, z_deg)
        _prim = stage.GetPrimAtPath(pp)
        xf = UsdGeom.Xformable(_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z_pos))
        prim_for_semantics = _prim
    else:
        mesh = _create_cube_mesh_with_uvs(stage, pp, block_size)
        x, y, z_pos = _mcq_block_world_xyz(i, j, k, height_map)
        xf = UsdGeom.Xformable(mesh.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z_pos))
        wm = stage.GetPrimAtPath("/World/Looks/WhiteMaterial")
        if wm.IsValid():
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(wm))
        prim_for_semantics = mesh.GetPrim()

    try:
        from omni.isaac.core.utils.semantics import add_update_semantics
        add_update_semantics(prim_for_semantics, "block")
    except Exception:
        pass


def _mcq_bind_gt_highlight_prim(stage, prim_path):
    pr = stage.GetPrimAtPath(prim_path)
    bg = stage.GetPrimAtPath(GT_HIGHLIGHT_MATERIAL_PATH)
    if pr.IsValid() and bg.IsValid():
        UsdShade.MaterialBindingAPI.Apply(pr).Bind(UsdShade.Material(bg))


@_time_fn
def _get_phantom_block_pixels(stage, seg_annotator):
    """Return binary pixel mask (1=phantom, 0=other) for the current phantom block.

    Works by hiding all regular blocks (names b_{i}_{j}_{k} — phantom names b_mcqextra_* are
    skipped by the length/format check in _set_block_visibility_selective) and rendering two
    steps so the seg annotator sees only the phantom block.
    """
    _set_block_visibility_selective(stage, set())  # hides regular blocks; phantom stays visible
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
    binary = _get_binary_segmentation_from_annotator(seg_annotator)
    _set_all_blocks_visible(stage)
    return binary  # H×W uint8 array with 1=phantom pixels, or None


def _apply_mcq_t1_phantom_block_above_top(stage, height_map, rng):
    """Distractor: extra cube stacked on top of a random non-empty column.
    Spawns with original texture only; returns (i, j, k) for downstream compositing, or None."""
    _mcq_remove_extra_blocks(stage)
    h, w = len(height_map), len(height_map[0])
    max_h = max(height_map[i][j] for i in range(h) for j in range(w))
    if max_h == 0:
        return None
    tallest = [(i, j) for i in range(h) for j in range(w) if height_map[i][j] == max_h]
    i, j = rng.choice(tallest)
    k = height_map[i][j]
    _mcq_spawn_extra_block(stage, height_map, "t1add", i, j, k)
    return (i, j, k)


def _apply_mcq_t2_phantom_top_block(stage, height_map, t2_top, rng):
    """Distractor: extra cube on top of a top-layer column.
    Spawns with original texture only; returns (i, j, k) for downstream compositing, or None."""
    _mcq_remove_extra_blocks(stage)
    if not t2_top:
        return None
    i, j, _k = rng.choice(list(t2_top))
    k_new = height_map[i][j]
    _mcq_spawn_extra_block(stage, height_map, "t2top", i, j, k_new)
    return (i, j, k_new)


def _apply_gt_t3_column_two_colors(stage, height_map, col_indices, palette=None):
    """Split each column in col_indices into two halves with different colors.
    col_indices may be a single int or a list of ints."""
    h, w = len(height_map), len(height_map[0])
    if isinstance(col_indices, int):
        col_indices = [col_indices]
    n = h * w
    # Build split map for columns that are tall enough
    split = {}
    for col_idx in col_indices:
        i0, j0 = col_idx // w, col_idx % w
        if height_map[i0][j0] >= 2:
            split[col_idx] = (col_idx + 1) % n
    if not split:
        _apply_gt_t3_colors(stage, height_map, palette=palette)
        return
    palette = palette or GT_COLOR_PALETTE
    for idx in range(n):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T3_{idx}", r, g, b)
    block_to_path = {}
    for i in range(h):
        for j in range(w):
            c = i * w + j
            nh = height_map[i][j]
            if c in split:
                alt_c = split[c]
                for k in range(nh):
                    use_idx = c if k < nh // 2 else alt_c
                    block_to_path[(i, j, k)] = f"/World/Looks/GT_T3_{use_idx}"
            else:
                path = f"/World/Looks/GT_T3_{c}"
                for k in range(nh):
                    block_to_path[(i, j, k)] = path
    set_blocks_material_mapping(stage, block_to_path)


def _apply_gt_t4_layer_two_colors(stage, height_map, layer_k, palette=None):
    """Same layer: checkerboard two palette colors by (i+j) parity."""
    max_k = max(height_map[i][j] for i in range(len(height_map)) for j in range(len(height_map[0]))) - 1
    if max_k < 0:
        return
    palette = palette or GT_COLOR_PALETTE
    n_pal = len(palette)
    t4_mats = {}
    for k in range(max_k + 1):
        r, g, b = palette[k % n_pal]
        t4_mats[k] = create_solid_color_material(stage, f"/World/Looks/GT_T4_{k}", r, g, b)
    alt = (layer_k + 1) % (max_k + 1)
    for (i, j, k), prim in _BLOCK_PRIMS.items():
        use_k = (layer_k if (i + j) % 2 == 0 else alt) if k == layer_k else k
        if use_k in t4_mats:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(t4_mats[use_k])


def _apply_gt_t5_adjacent_same_color_triple(stage, height_map, visible_set, rng, palette=None):
    """Merge spatially adjacent pairs and triplets of visible blocks to the same color."""
    vis = sorted(visible_set)
    if len(vis) < 2:
        _apply_gt_t5_colors(stage, height_map, visible_set, palette=palette)
        return
    palette = palette or GT_COLOR_PALETTE
    groups = _find_adjacent_merge_groups(visible_set, rng)
    block_to_idx = {block: idx for idx, block in enumerate(vis)}
    for group in groups:
        leader_idx = min(block_to_idx[b] for b in group)
        for b in group:
            block_to_idx[b] = leader_idx
    for idx in range(len(vis)):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T5_{idx}", r, g, b)
    block_to_path = {block: f"/World/Looks/GT_T5_{idx}" for block, idx in block_to_idx.items()}
    set_blocks_material_mapping(stage, block_to_path, default_path="/World/Looks/WhiteMaterial")


def _apply_gt_t9_fake_split_cluster(stage, height_map, rng, palette=None):
    """Split one true cluster into two colors (wrong cluster count)."""
    clusters = get_block_clusters(height_map)
    if not clusters:
        return
    target = max(clusters, key=len)
    if len(target) < 2:
        _apply_gt_t9_colors(stage, height_map, palette=palette)
        return
    bl = list(target)
    rng.shuffle(bl)
    mid = max(1, len(bl) // 2)
    a_set, b_set = set(bl[:mid]), set(bl[mid:])
    palette = palette or GT_COLOR_PALETTE
    nc = len(clusters)
    for idx in range(max(nc, 2)):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T9_{idx}", r, g, b)
    block_to_path = {}
    for idx, cluster in enumerate(clusters):
        if cluster is target:
            for block in cluster:
                block_to_path[block] = "/World/Looks/GT_T9_0" if block in a_set else "/World/Looks/GT_T9_1"
        else:
            for block in cluster:
                block_to_path[block] = f"/World/Looks/GT_T9_{idx}"
    set_blocks_material_mapping(stage, block_to_path)


def _apply_gt_t9_single_cluster_wrong_multicolor(stage, height_map, rng, palette=None):
    """Truth is one cluster; paint blocks with multiple colors (incorrect)."""
    clusters = get_block_clusters(height_map)
    if len(clusters) != 1:
        _apply_gt_t9_fake_split_cluster(stage, height_map, rng, palette=palette)
        return
    cl = clusters[0]
    bl = sorted(cl, key=lambda x: (x[0], x[1], x[2]))
    palette = palette or GT_COLOR_PALETTE
    for idx in range(len(bl)):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T9_{idx}", r, g, b)
    block_to_path = {bl[i]: f"/World/Looks/GT_T9_{i % len(palette)}" for i in range(len(bl))}
    set_blocks_material_mapping(stage, block_to_path)


def _apply_mcq_t10_random_wrong_highlights(stage, height_map, visible_set, t10_set, rng):
    """Highlight random visible block(s) that are not part of the true T10 set (scene highlight tint)."""
    _mcq_remove_extra_blocks(stage)
    pool = [b for b in visible_set if b not in t10_set]
    if not pool:
        pool = list(visible_set)
    if not pool:
        return
    n = rng.randint(1, min(3, len(pool)))
    wrong = set(rng.sample(pool, n))
    restore_block_original_materials(stage)
    set_blocks_material_selective_original_bg(stage, wrong, GT_HIGHLIGHT_MATERIAL_PATH)


def _pil_load_png(path):
    if not os.path.isfile(path):
        return None
    return Image.open(path).convert("RGBA")


def _mcq_rgba_fingerprint(pil_img):
    """Stable hash for comparing renders when visible_set is empty."""
    if pil_img is None:
        return None
    try:
        s = pil_img.resize((48, 48), Image.Resampling.LANCZOS).convert("RGB")
        return hashlib.md5(s.tobytes()).hexdigest()
    except Exception:
        return None


def _mcq_phantom_prims_suffix(stage):
    """Material + world position for b_mcqextra_* prims (Task 1/2 phantoms are not in visible_set coords)."""
    blocks_prim = stage.GetPrimAtPath("/World/Blocks")
    if not blocks_prim.IsValid():
        return ()
    parts = []
    for child in blocks_prim.GetChildren():
        name = child.GetName()
        if not name.startswith("b_mcqextra"):
            continue
        api = UsdShade.MaterialBindingAPI(child)
        p = None
        rel = api.GetDirectBindingRel()
        if rel and rel.IsValid():
            tgts = rel.GetTargets()
            if tgts:
                p = str(tgts[0])
        pos = (0.0, 0.0, 0.0)
        try:
            xf = UsdGeom.Xformable(child)
            m = xf.ComputeLocalToWorldTransform(Usd.Time.Default())
            t = m.ExtractTranslation()
            pos = (round(float(t[0]), 5), round(float(t[1]), 5), round(float(t[2]), 5))
        except Exception:
            pass
        parts.append((name, p or "__unbound__", pos))
    parts.sort(key=lambda x: x[0])
    return tuple(parts)


def _mcq_color_partition_signature(stage, blocks):
    """Multiset of same-color groups (frozensets of block coords), canonical order.

    Invariant under permuting which material/color is assigned to which group — e.g. Task 9 two clusters
    with colors A/B vs B/A yields the same signature. Applies to any MCQ task where only the partition into
    same-colored groups matters for equivalence.
    """
    if not blocks:
        return None
    path_to_blocks = {}
    for block in sorted(blocks):
        i, j, k = block
        prim = stage.GetPrimAtPath(f"/World/Blocks/b_{i}_{j}_{k}")
        if not prim.IsValid():
            p = "__missing__"
        else:
            api = UsdShade.MaterialBindingAPI(prim)
            p = None
            rel = api.GetDirectBindingRel()
            if rel and rel.IsValid():
                tgts = rel.GetTargets()
                if tgts:
                    p = str(tgts[0])
            p = p or "__unbound__"
        path_to_blocks.setdefault(p, []).append(block)
    groups = [frozenset(bs) for bs in path_to_blocks.values()]
    groups.sort(key=lambda g: sorted(g))
    return tuple(groups)


def _mcq_visible_block_material_signature(stage, visible_set):
    """Color-partition signature for visible blocks, plus any MCQ phantom cubes.

    Partition is color-permutation invariant (swap cluster/column/etc. colors without changing the answer).
    Without the phantom suffix, Task 1 ``phantom_block_above_top`` could match GT (all real blocks same tint) and was
    incorrectly treated as a duplicate of the correct answer.
    """
    base = None
    if visible_set:
        base = _mcq_color_partition_signature(stage, visible_set)

    extra = _mcq_phantom_prims_suffix(stage)
    if extra:
        if base is not None:
            return base + ("__mcq_extra__",) + extra
        return ("__mcq_extra_only__",) + extra
    return base


MCQ_FALLBACK_HIGHLIGHT_SUBSET_TASKS = frozenset(("task6", "task7", "task8", "task10", "task11"))


def _apply_mcq_fallback_random_block_coloring(stage, height_map, visible_set, fb_index, rng):
    """Random solid colors on visible blocks (or all blocks); unique material paths per attempt."""
    _mcq_remove_extra_blocks(stage)
    restore_block_original_materials(stage)
    pool = sorted(visible_set) if visible_set else sorted(_all_blocks_set(height_map))
    if not pool:
        return
    n_mat = min(16, max(4, len(pool)))
    sid = (fb_index * 486187739 + rng.randint(0, 2**24)) & 0x7FFFFFFF
    paths = []
    for i in range(n_mat):
        r, g, b = rng.random(), rng.random(), rng.random()
        p = f"/World/Looks/MCQ_FB_{sid}_{i}"
        create_solid_color_material(stage, p, r, g, b)
        paths.append(p)
    rng2 = random.Random(rng.randint(0, 2**30) ^ (fb_index * 0x9E3779B9))
    block_to_path = {}
    for j, block in enumerate(pool):
        slot = (fb_index * 1315423911 + j * 2654435761 + rng2.randint(0, 2**20)) % n_mat
        block_to_path[block] = paths[slot]
    set_blocks_material_mapping(stage, block_to_path, default_path="/World/Looks/WhiteMaterial")


def _apply_mcq_fallback_random_highlight_subset(stage, height_map, visible_set, fb_index, rng):
    """1–3 random blocks with scene highlight tint; others original (T6/T7/T8/T10 MCQ fallback)."""
    _mcq_remove_extra_blocks(stage)
    restore_block_original_materials(stage)
    pool = sorted(visible_set) if visible_set else sorted(_all_blocks_set(height_map))
    if not pool:
        return
    r2 = random.Random(rng.randint(0, 2**30) ^ (fb_index * 0x9E3779B9))
    n_pick = r2.randint(1, min(3, len(pool)))
    picked = set(r2.sample(pool, n_pick))
    set_blocks_material_selective_original_bg(stage, picked, GT_HIGHLIGHT_MATERIAL_PATH)


def _mcq_apply_fallback_distractor(stage, height_map, visible_set, task_id, fb_index, rng):
    if task_id in MCQ_FALLBACK_HIGHLIGHT_SUBSET_TASKS:
        _apply_mcq_fallback_random_highlight_subset(stage, height_map, visible_set, fb_index, rng)
    else:
        _apply_mcq_fallback_random_block_coloring(stage, height_map, visible_set, fb_index, rng)


def _mcq_apply_correct_gt_for_task(stage, task_id, *, height_map, visible_set, t2_sets, t6, t7, t8, t10_set, t11_set, view_dir=None):
    """Apply GT materials for task_id (same coloring as when taskN_gt_mask was saved)."""
    _mcq_remove_extra_blocks(stage)
    restore_block_original_materials(stage)
    if task_id == "task1":
        set_blocks_material(stage, GT_HIGHLIGHT_MATERIAL_PATH)
    elif task_id == "task2":
        if T2_TOP_ONLY:
            set_blocks_material_selective_original_bg(stage, t2_sets["top"], GT_HIGHLIGHT_MATERIAL_PATH)
    elif task_id == "task3":
        pal = _shuffled_palette_for_task(view_dir, "task3") if view_dir else None
        _apply_gt_t3_colors(stage, height_map, palette=pal)
    elif task_id == "task4":
        pal = _shuffled_palette_for_task(view_dir, "task4") if view_dir else None
        _apply_gt_t4_colors(stage, height_map, palette=pal)
    elif task_id == "task5":
        pal = _shuffled_palette_for_task(view_dir, "task5") if view_dir else None
        _apply_gt_t5_colors(stage, height_map, visible_set, palette=pal)
    elif task_id == "task6":
        set_blocks_material_selective_original_bg(stage, t6, GT_HIGHLIGHT_MATERIAL_PATH)
    elif task_id == "task7":
        set_blocks_material_selective_original_bg(stage, t7, GT_HIGHLIGHT_MATERIAL_PATH)
    elif task_id == "task8":
        set_blocks_material_selective_original_bg(stage, t8, GT_HIGHLIGHT_MATERIAL_PATH)
    elif task_id == "task9":
        pal = _shuffled_palette_for_task(view_dir, "task9") if view_dir else None
        _apply_gt_t9_colors(stage, height_map, palette=pal)
    elif task_id == "task10":
        set_blocks_material_selective_original_bg(stage, t10_set, GT_HIGHLIGHT_MATERIAL_PATH)
    elif task_id == "task11":
        set_blocks_material_selective_original_bg(stage, t11_set, GT_HIGHLIGHT_MATERIAL_PATH)



def _p255(r, g, b):
    """Convert 0-1 float RGB to 0-255 uint8 tuple."""
    return (round(r * 255), round(g * 255), round(b * 255))


@_time_fn
def _mcq_composite_pil(view_arr, pixel_map, block_ids, color_map):
    """Composite color_map onto view_arr using pixel_map for block→pixel lookups.

    view_arr:  RGBA uint8 H×W×4 numpy array (from view.png)
    pixel_map: uint8 H×W numpy array; value v>0 → block_ids[v-1]; 0=background
    block_ids: list of (i, j, k) tuples (sorted(visible_set) at GT-mask save time)
    color_map: {(i,j,k): (R,G,B)} uint8 — only painted blocks need entries
    Returns PIL RGBA image.
    """
    arr = view_arr.copy()
    block_to_label = {b: idx + 1 for idx, b in enumerate(block_ids)}
    for block, rgb in color_map.items():
        label = block_to_label.get(block)
        if label is None:
            continue
        m = pixel_map == label
        arr[m, 0] = rgb[0]
        arr[m, 1] = rgb[1]
        arr[m, 2] = rgb[2]
        arr[m, 3] = 255
    return Image.fromarray(arr)


# ─── Pure-Python color-map compute functions ─────────────────────────────────

def _compute_cm_highlight(visible_set):
    """All visible blocks → scene highlight color."""
    return {b: _GT_HIGHLIGHT_COLOR_255 for b in visible_set}


def _compute_cm_selective_highlight(fg_set, visible_set):
    """Blocks in fg_set ∩ visible_set → highlight; rest keeps view pixels."""
    vs = frozenset(visible_set)
    return {b: _GT_HIGHLIGHT_COLOR_255 for b in fg_set if b in vs}


def _compute_cm_t3(height_map, palette=None):
    pal = palette or GT_COLOR_PALETTE
    h, w = len(height_map), len(height_map[0])
    cm = {}
    for i in range(h):
        for j in range(w):
            rgb = _p255(*pal[(i * w + j) % len(pal)])
            for k in range(height_map[i][j]):
                cm[(i, j, k)] = rgb
    return cm


def _compute_cm_t3_column_two_colors(height_map, col_indices, palette=None):
    """Split each column in col_indices into two halves with different colors.
    col_indices may be a single int or a list of ints."""
    pal = palette or GT_COLOR_PALETTE
    h, w = len(height_map), len(height_map[0])
    n = h * w
    if isinstance(col_indices, int):
        col_indices = [col_indices]
    # Build split map: col_idx -> alt_c (the second color index for that column)
    split = {}
    for col_idx in col_indices:
        i0, j0 = col_idx // w, col_idx % w
        if height_map[i0][j0] >= 2:
            split[col_idx] = (col_idx + 1) % n
    if not split:
        return _compute_cm_t3(height_map, palette)
    cm = {}
    for i in range(h):
        for j in range(w):
            c = i * w + j
            nh = height_map[i][j]
            if c in split:
                alt_c = split[c]
                for k in range(nh):
                    use_idx = c if k < nh // 2 else alt_c
                    cm[(i, j, k)] = _p255(*pal[use_idx % len(pal)])
            else:
                rgb = _p255(*pal[c % len(pal)])
                for k in range(nh):
                    cm[(i, j, k)] = rgb
    return cm


def _compute_cm_t3_merge_columns(height_map, c_keep, c_merge, palette=None):
    """Merge one or more columns into c_keep's color.
    c_merge may be a single int or a list of ints."""
    return _compute_cm_t3_merge_column_groups(height_map, [(c_keep, c_merge)], palette)


def _compute_cm_t3_merge_column_groups(height_map, groups, palette=None):
    """Merge multiple independent groups of adjacent columns.
    groups: list of (c_keep, c_merge) where c_merge is an int or list of ints.
    Each group's merged columns take c_keep's color; groups must be disjoint."""
    pal = palette or GT_COLOR_PALETTE
    h, w = len(height_map), len(height_map[0])
    # Build remap: col_index -> color_index
    remap = {}
    for c_keep, c_merge in groups:
        merge_set = {c_merge} if isinstance(c_merge, int) else set(c_merge)
        for c in merge_set:
            remap[c] = c_keep
    cm = {}
    for i in range(h):
        for j in range(w):
            c = i * w + j
            rgb = _p255(*pal[remap.get(c, c) % len(pal)])
            for k in range(height_map[i][j]):
                cm[(i, j, k)] = rgb
    return cm


def _compute_cm_t3_drop_column(height_map, c_drop, palette=None):
    pal = palette or GT_COLOR_PALETTE
    h, w = len(height_map), len(height_map[0])
    cm = {}
    for i in range(h):
        for j in range(w):
            c = i * w + j
            if c == c_drop:
                continue  # dropped column: keep view pixels
            rgb = _p255(*pal[c % len(pal)])
            for k in range(height_map[i][j]):
                cm[(i, j, k)] = rgb
    return cm


def _compute_cm_t4(height_map, palette=None):
    pal = palette or GT_COLOR_PALETTE
    h, w = len(height_map), len(height_map[0])
    cm = {}
    for i in range(h):
        for j in range(w):
            for k in range(height_map[i][j]):
                cm[(i, j, k)] = _p255(*pal[k % len(pal)])
    return cm


def _compute_cm_t4_layer_two_colors(height_map, layer_k, palette=None):
    pal = palette or GT_COLOR_PALETTE
    h, w = len(height_map), len(height_map[0])
    max_k = max(height_map[i][j] for i in range(h) for j in range(w)) - 1
    if max_k < 0:
        return {}
    alt = (layer_k + 1) % (max_k + 1)
    cm = {}
    for i in range(h):
        for j in range(w):
            for k in range(height_map[i][j]):
                if k == layer_k:
                    use_k = layer_k if (i + j) % 2 == 0 else alt
                else:
                    use_k = k
                cm[(i, j, k)] = _p255(*pal[use_k % len(pal)])
    return cm


def _compute_cm_t4_merge_layers(height_map, k_take_color, k_mislabeled, palette=None):
    pal = palette or GT_COLOR_PALETTE
    h, w = len(height_map), len(height_map[0])
    cm = {}
    for i in range(h):
        for j in range(w):
            for k in range(height_map[i][j]):
                use_k = k_take_color if k == k_mislabeled else k
                cm[(i, j, k)] = _p255(*pal[use_k % len(pal)])
    return cm


def _compute_cm_t4_drop_layer(height_map, k_drop, palette=None):
    pal = palette or GT_COLOR_PALETTE
    h, w = len(height_map), len(height_map[0])
    cm = {}
    for i in range(h):
        for j in range(w):
            for k in range(height_map[i][j]):
                if k == k_drop:
                    continue  # dropped layer: keep view pixels
                cm[(i, j, k)] = _p255(*pal[k % len(pal)])
    return cm


def _compute_cm_t5(height_map, visible_set, palette=None):
    pal = palette or GT_COLOR_PALETTE
    vis = sorted(visible_set)
    return {block: _p255(*pal[idx % len(pal)]) for idx, block in enumerate(vis)}


def _find_adjacent_merge_groups(visible_set, rng):
    """Return non-overlapping groups (each 2-3 blocks) of spatially adjacent visible blocks.
    Groups alternate pair/triplet. Count = max(1, len(vis) // 5)."""
    vis_set = set(visible_set)
    neighbors = {}
    for (i, j, k) in vis_set:
        nbrs = []
        for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nb = (i + di, j + dj, k + dk)
            if nb in vis_set:
                nbrs.append(nb)
        neighbors[(i, j, k)] = nbrs
    candidates = [b for b in vis_set if neighbors[b]]
    rng.shuffle(candidates)
    max_groups = max(1, len(vis_set) // 5)
    groups = []
    used = set()
    want_triplet = True
    for start in candidates:
        if len(groups) >= max_groups:
            break
        if start in used:
            continue
        avail = [n for n in neighbors[start] if n not in used]
        if not avail:
            continue
        second = rng.choice(avail)
        if want_triplet:
            third_cands = [n for n in neighbors[second] if n not in used and n != start]
            if third_cands:
                groups.append([start, second, rng.choice(third_cands)])
                used |= set(groups[-1])
                want_triplet = False
                continue
        groups.append([start, second])
        used |= {start, second}
        want_triplet = True
    return groups


def _compute_cm_t5_adjacent_same_color_triple(height_map, visible_set, rng, palette=None):
    vis = sorted(visible_set)
    if len(vis) < 2:
        return _compute_cm_t5(height_map, visible_set, palette)
    pal = palette or GT_COLOR_PALETTE
    groups = _find_adjacent_merge_groups(visible_set, rng)
    block_to_idx = {block: idx for idx, block in enumerate(vis)}
    for group in groups:
        leader_idx = min(block_to_idx[b] for b in group)
        for b in group:
            block_to_idx[b] = leader_idx
    return {block: _p255(*pal[idx % len(pal)]) for block, idx in block_to_idx.items()}


def _compute_cm_t9(height_map, palette=None, block_set=None):
    clusters = get_block_clusters(height_map, block_set=block_set)
    pal = palette or GT_COLOR_PALETTE
    cm = {}
    for idx, cluster in enumerate(clusters):
        rgb = _p255(*pal[idx % len(pal)])
        for block in cluster:
            cm[block] = rgb
    return cm


def _compute_cm_t9_merge_two_clusters(height_map, color_src_idx, cluster_dup_idx, palette=None):
    clusters = get_block_clusters(height_map)
    if len(clusters) < 2:
        return _compute_cm_t9(height_map, palette)
    pal = palette or GT_COLOR_PALETTE
    cm = {}
    for idx, cluster in enumerate(clusters):
        use_idx = color_src_idx if idx == cluster_dup_idx else idx
        rgb = _p255(*pal[use_idx % len(pal)])
        for block in cluster:
            cm[block] = rgb
    return cm


def _compute_cm_t9_single_cluster_color(height_map, color_idx, palette=None):
    clusters = get_block_clusters(height_map)
    if not clusters:
        return {}
    pal = palette or GT_COLOR_PALETTE
    rgb = _p255(*pal[(color_idx % len(clusters)) % len(pal)])
    cm = {}
    for cluster in clusters:
        for block in cluster:
            cm[block] = rgb
    return cm


def _compute_cm_t9_fake_split_cluster(height_map, rng, palette=None):
    clusters = get_block_clusters(height_map)
    if not clusters:
        return {}
    pal = palette or GT_COLOR_PALETTE
    target = max(clusters, key=len)
    if len(target) < 2:
        return _compute_cm_t9(height_map, palette)
    bl = list(target)
    rng.shuffle(bl)
    mid = max(1, len(bl) // 2)
    a_set = set(bl[:mid])
    rgb0 = _p255(*pal[0 % len(pal)])
    rgb1 = _p255(*pal[1 % len(pal)])
    cm = {}
    for idx, cluster in enumerate(clusters):
        if cluster is target:
            for block in cluster:
                cm[block] = rgb0 if block in a_set else rgb1
        else:
            rgb = _p255(*pal[idx % len(pal)])
            for block in cluster:
                cm[block] = rgb
    return cm


def _compute_cm_t9_single_cluster_wrong_multicolor(height_map, rng, palette=None):
    clusters = get_block_clusters(height_map)
    if len(clusters) != 1:
        return _compute_cm_t9_fake_split_cluster(height_map, rng, palette)
    pal = palette or GT_COLOR_PALETTE
    bl = sorted(clusters[0], key=lambda x: (x[0], x[1], x[2]))
    return {block: _p255(*pal[i % len(pal)]) for i, block in enumerate(bl)}


def _compute_cm_correct_for_task(task_id, *, height_map, visible_set, t2_sets, t6, t7, t8, t10_set, t11_set,
                                  palette_t3=None, palette_t4=None, palette_t5=None, palette_t9=None):
    """Pure-Python color map for the correct GT answer of task_id."""
    vs = frozenset(visible_set)
    if task_id == "task1":
        return _compute_cm_highlight(vs)
    elif task_id == "task2":
        return _compute_cm_selective_highlight(t2_sets.get("top", set()), vs) if T2_TOP_ONLY else {}
    elif task_id == "task3":
        return _compute_cm_t3(height_map, palette_t3)
    elif task_id == "task4":
        return _compute_cm_t4(height_map, palette_t4)
    elif task_id == "task5":
        return _compute_cm_t5(height_map, vs, palette_t5)
    elif task_id == "task6":
        return _compute_cm_selective_highlight(t6, vs)
    elif task_id == "task7":
        return _compute_cm_selective_highlight(t7, vs)
    elif task_id == "task8":
        return _compute_cm_selective_highlight(t8, vs)
    elif task_id == "task9":
        return _compute_cm_t9(height_map, palette_t9, block_set=_needed_from_visible(height_map, visible_set))
    elif task_id == "task10":
        return _compute_cm_selective_highlight(t10_set, vs)
    elif task_id == "task11":
        return _compute_cm_selective_highlight(t11_set, vs)
    return {}


def _compute_cm_fallback_random_block_coloring(visible_set, fb_index, rng):
    """Random solid colors on visible blocks (fallback dedup, compositing version)."""
    pool = sorted(visible_set) if visible_set else []
    if not pool:
        return {}
    n_mat = min(16, max(4, len(pool)))
    colors = [(round(rng.random() * 255), round(rng.random() * 255), round(rng.random() * 255))
              for _ in range(n_mat)]
    rng2 = random.Random(rng.randint(0, 2 ** 30) ^ (fb_index * 0x9E3779B9))
    cm = {}
    for j, block in enumerate(pool):
        slot = (fb_index * 1315423911 + j * 2654435761 + rng2.randint(0, 2 ** 20)) % n_mat
        cm[block] = colors[slot]
    return cm


def _compute_cm_fallback_selective_highlight(visible_set, fb_index, rng):
    """1–3 random highlighted blocks (fallback dedup for T6/T7/T8/T10, compositing version)."""
    pool = sorted(visible_set) if visible_set else []
    if not pool:
        return {}
    r2 = random.Random(rng.randint(0, 2 ** 30) ^ (fb_index * 0x9E3779B9))
    n_pick = r2.randint(1, min(3, len(pool)))
    picked = set(r2.sample(pool, n_pick))
    return {b: _GT_HIGHLIGHT_COLOR_255 for b in picked}


@_time_fn
def _mcq_dedupe_wrong_entries(
    stage,
    rgb_annotator,
    wrong_entries,
    visible_set,
    height_map,
    task_id,
    rng,
    fp_seen,
    view_arr,
    pixel_map,
    block_ids,
    compositing_ok,
):
    """Deduplication using image fingerprints. Uses compositing fallbacks when available."""
    fb_idx = 0
    for i in range(len(wrong_entries)):
        pil, lbl, fp, _is_phantom = wrong_entries[i]
        is_dup = fp is not None and fp in fp_seen
        if not is_dup:
            if fp is not None:
                fp_seen.add(fp)
            continue
        replaced = False
        for attempt in range(8000):
            fb_idx += 1
            fr = random.Random(rng.randint(0, 2**30) ^ (fb_idx * 0x9E3779B9) ^ (i * 7919))
            if compositing_ok:
                if task_id in MCQ_FALLBACK_HIGHLIGHT_SUBSET_TASKS:
                    cm = _compute_cm_fallback_selective_highlight(visible_set, fb_idx, fr)
                else:
                    cm = _compute_cm_fallback_random_block_coloring(visible_set, fb_idx, fr)
                pil2 = _mcq_composite_pil(view_arr, pixel_map, block_ids, cm)
            else:
                _mcq_remove_extra_blocks(stage)
                restore_block_original_materials(stage)
                _mcq_apply_fallback_distractor(stage, height_map, visible_set, task_id, fb_idx, fr)
                pil2 = _mcq_capture_pil(rgb_annotator)
                restore_block_original_materials(stage)
            if pil2 is None:
                continue
            nfp = _mcq_rgba_fingerprint(pil2)
            if nfp is not None and nfp not in fp_seen:
                fp_seen.add(nfp)
                wrong_entries[i] = (pil2, f"{lbl}_fallback_unique_{attempt}", nfp, False)
                replaced = True
                break
        if not replaced:
            print(f"  !! MCQ {task_id}: could not replace duplicate choice {i} with unique fallback — inserting black dummy")
            dummy = Image.new("RGBA", IMAGE_RES, (0, 0, 0, 255))
            wrong_entries[i] = (dummy, f"{lbl}_dummy_black", None, False)


def _build_mcq_wrong_specs_t1(height_map, visible_set, rng):
    all_b = _all_blocks_set(height_map)
    if not all_b:
        return []
    bl = list(all_b)
    rng.shuffle(bl)
    w1 = all_b - {bl[0]}
    w2 = all_b - set(bl[: min(2, len(bl))])
    cols = _nonempty_column_indices(height_map)
    c_drop = rng.choice(cols)
    w3 = all_b - _get_column_blocks(height_map, c_drop)
    vs = frozenset(visible_set)
    return [
        (lambda st, hm=height_map, r=rng: _apply_mcq_t1_phantom_block_above_top(st, hm, r), "phantom_block_above_top", True),
        (lambda fg=w1, vs=vs: _compute_cm_selective_highlight(fg, vs), "omit_one_block", False),
        (lambda fg=w2, vs=vs: _compute_cm_selective_highlight(fg, vs), "omit_two_blocks", False),
        (lambda fg=w3, vs=vs: _compute_cm_selective_highlight(fg, vs), "drop_one_column_highlight", False),
    ]


def _build_mcq_wrong_specs_t2(height_map, visible_set, t2_sets, rng):
    top = t2_sets["top"]
    oth = _t2_other_column_top_block(height_map, visible_set, top, rng)
    w1 = top - {rng.choice(list(top))} if len(top) > 1 else top
    b = next(iter(top)) if top else None
    w2 = {(b[0], b[1], b[2] - 1)} if b and b[2] > 0 else top
    w3 = oth
    vs = frozenset(visible_set)
    return [
        (lambda st, hm=height_map, tp=top, r=rng: _apply_mcq_t2_phantom_top_block(st, hm, tp, r), "phantom_block_above_top", True),
        (lambda fg=w1, vs=vs: _compute_cm_selective_highlight(fg, vs), "missing_one_top_block", False),
        (lambda fg=w2, vs=vs: _compute_cm_selective_highlight(fg, vs), "one_below_top", False),
        (lambda fg=w3, vs=vs: _compute_cm_selective_highlight(fg, vs), "other_column_top", False),
    ]


def _pick_adjacent_column_group(cols, h, w, target_size, rng):
    """Return a connected group of target_size non-empty columns (4-connectivity in the grid).
    Grows from a random seed column by repeatedly adding a random adjacent neighbour.
    Falls back to smaller groups if not enough adjacent columns exist."""
    cols_set = set(cols)

    def col_neighbours(c):
        ci, cj = c // w, c % w
        nbrs = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < h and 0 <= nj < w:
                n = ni * w + nj
                if n in cols_set:
                    nbrs.append(n)
        return nbrs

    shuffled = list(cols)
    rng.shuffle(shuffled)
    for seed in shuffled:
        group = [seed]
        frontier = [n for n in col_neighbours(seed) if n not in group]
        rng.shuffle(frontier)
        while len(group) < target_size and frontier:
            pick = frontier.pop(0)
            if pick not in group:
                group.append(pick)
                new_nbrs = [n for n in col_neighbours(pick) if n not in group and n not in frontier]
                frontier.extend(new_nbrs)
                rng.shuffle(frontier)
        if len(group) >= 2:
            return group
    # Fallback: just return first two cols
    return list(cols[:2])


def _build_mcq_wrong_specs_t3(height_map, rng, palette=None):
    """Swap-style column permutations are invalid wrong answers (GT is permutation-invariant). Reuse the same distractor *types* with different columns / merge direction."""
    cols = _nonempty_column_indices(height_map)
    if not cols:
        return []
    h, w = len(height_map), len(height_map[0])
    tall = [c for c in cols if height_map[c // w][c % w] >= 2]
    c_drop = rng.choice(cols)

    # same_column_two_colors: pick 2-3 tall columns to split (or fall back to fewer)
    rng.shuffle(tall)
    n_split = min(len(tall), rng.choice([2, 2, 3]))
    split_cols_a = tall[:n_split] if n_split >= 1 else ([cols[0]] if cols else [])
    rng.shuffle(tall)
    n_split_b = min(len(tall), rng.choice([2, 2, 3]))
    split_cols_b = tall[:n_split_b] if n_split_b >= 1 else split_cols_a

    # merge_two_columns: pick 1-2 independent groups of 2-4 adjacent columns each
    def _pick_merge_groups(cols, h, w, rng):
        """Return 1 or 2 disjoint adjacent column groups as a list of (c_keep, c_merge_list)."""
        n_groups = 1 if len(cols) < 4 else rng.choice([1, 1, 2])
        used = set()
        groups = []
        for _ in range(n_groups):
            available = [c for c in cols if c not in used]
            if len(available) < 2:
                break
            size = min(len(available), rng.randint(2, 4))
            grp = _pick_adjacent_column_group(available, h, w, size, rng)
            if len(grp) < 2:
                break
            groups.append((grp[0], grp[1:]))
            used |= set(grp)
        return groups if groups else [([cols[0]], [])]

    groups_a = _pick_merge_groups(cols, h, w, rng)
    groups_b = _pick_merge_groups(cols, h, w, rng)

    out = [
        (lambda cs=split_cols_a: _compute_cm_t3_column_two_colors(height_map, cs, palette), "same_column_two_colors", False),
        (lambda cs=split_cols_b: _compute_cm_t3_column_two_colors(height_map, cs, palette), "same_column_two_colors", False),
        (lambda g=groups_a: _compute_cm_t3_merge_column_groups(height_map, g, palette), "merge_two_columns", False),
        (lambda g=groups_b: _compute_cm_t3_merge_column_groups(height_map, g, palette), "merge_two_columns", False),
        (lambda cd=c_drop: _compute_cm_t3_drop_column(height_map, cd, palette), "missing_one_column", False),
    ]
    return out


def _build_mcq_wrong_specs_t4(height_map, rng, palette=None):
    hm = height_map
    max_k = max(hm[i][j] for i in range(len(hm)) for j in range(len(hm[0]))) - 1
    if max_k < 0:
        return []
    layer_counts = {}
    h, w = len(hm), len(hm[0])
    for i in range(h):
        for j in range(w):
            for k in range(hm[i][j]):
                layer_counts[k] = layer_counts.get(k, 0) + 1
    multi = [k for k, v in layer_counts.items() if v >= 2]
    lk = rng.choice(multi) if multi else 0
    k0, k1 = (0, min(1, max_k)) if max_k >= 1 else (0, 0)
    return [
        (lambda lkk=lk: _compute_cm_t4_layer_two_colors(hm, lkk, palette), "same_layer_two_colors", False),
        (lambda a=k0, b=k1: _compute_cm_t4_merge_layers(hm, a, b, palette), "merge_two_layers", False),
        (lambda kd=k0: _compute_cm_t4_drop_layer(hm, kd, palette), "missing_one_layer", False),
        (lambda a=k1, b=k0: _compute_cm_t4_merge_layers(hm, a, b, palette), "far_layer_confusion", False),
    ]


def _build_mcq_wrong_specs_t5(height_map, visible_set, rng, palette=None):
    """Scrambling color labels is not a wrong answer (permutation-invariant). The same distractor *type* may appear more than once with different RNG or dropped subsets."""
    vis = sorted(visible_set)
    ra = random.Random(rng.randint(0, 2**30))
    rb = random.Random(rng.randint(0, 2**30))
    if len(vis) > 1:
        half = max(1, len(vis) // 2)
        shuffled_a = list(vis); ra.shuffle(shuffled_a)
        shuffled_b = list(vis); rb.shuffle(shuffled_b)
        drop_a = set(shuffled_a[:half])
        drop_b = set(shuffled_b[:half])
    else:
        drop_a = set()
        drop_b = set()
    return [
        (lambda r=ra: _compute_cm_t5_adjacent_same_color_triple(height_map, visible_set, r, palette), "merge_adjacent_same_color", False),
        (lambda r=rb: _compute_cm_t5_adjacent_same_color_triple(height_map, visible_set, r, palette), "merge_adjacent_same_color", False),
        (lambda vs=drop_a: _compute_cm_t5(height_map, vs, palette), "drop_half_visible_blocks", False),
        (lambda vs=drop_b: _compute_cm_t5(height_map, vs, palette), "drop_half_visible_blocks", False),
    ]


def _build_mcq_wrong_specs_t6(height_map, visible_set, t6, rng):
    w1 = t6 - {rng.choice(list(t6))} if len(t6) > 1 else t6
    alt = get_t6_support_blocks(height_map)
    w2 = alt - {rng.choice(list(alt))} if len(alt) > 1 else alt
    non_s = _all_blocks_set(height_map) - t6
    w4 = {rng.choice(list(non_s))} if non_s else (t6 - {rng.choice(list(t6))} if len(t6) > 1 else t6)
    vs = frozenset(visible_set)
    return [
        (lambda fg=w1, vs=vs: _compute_cm_selective_highlight(fg, vs), "miss_one_support", False),
        (lambda fg=w2, vs=vs: _compute_cm_selective_highlight(fg, vs), "wrong_support_subset", False),
        (lambda fg=frozenset(), vs=vs: _compute_cm_selective_highlight(fg, vs), "empty_support", False),
        (lambda fg=w4, vs=vs: _compute_cm_selective_highlight(fg, vs), "non_support_highlight", False),
    ]


def _build_mcq_wrong_specs_t7(height_map, visible_set, t7, rng):
    w1 = t7 - {rng.choice(list(t7))} if len(t7) > 1 else t7
    w3 = t7 - set(rng.sample(list(t7), min(2, len(t7)))) if len(t7) > 2 else w1
    t6b = get_t6_support_blocks(height_map)
    w4 = t6b - t7 if t6b - t7 else w1
    vs = frozenset(visible_set)
    return [
        (lambda fg=w1, vs=vs: _compute_cm_selective_highlight(fg, vs), "remove_support_column_blocks", False),
        (lambda fg=frozenset(), vs=vs: _compute_cm_selective_highlight(fg, vs), "wrong_column_empty", False),
        (lambda fg=w3, vs=vs: _compute_cm_selective_highlight(fg, vs), "remove_two_column_blocks", False),
        (lambda fg=w4, vs=vs: _compute_cm_selective_highlight(fg, vs), "support_blocks_not_column", False),
    ]


def _build_mcq_wrong_specs_t8(height_map, t8, visible_set, rng):
    w1 = t8 - {rng.choice(list(t8))} if len(t8) > 1 else t8
    outside = visible_set - t8
    w4 = {rng.choice(list(outside))} if outside else w1
    vs = frozenset(visible_set)
    return [
        (lambda fg=w1, vs=vs: _compute_cm_selective_highlight(fg, vs), "miss_one_supported", False),
        (lambda fg=frozenset(), vs=vs: _compute_cm_selective_highlight(fg, vs), "empty_t8", False),
        (lambda fg=w4, vs=vs: _compute_cm_selective_highlight(fg, vs), "wrong_visible_block", False),
        (lambda fg=w1, vs=vs: _compute_cm_selective_highlight(fg, vs), "alt_incomplete_subset", False),
    ]


def _apply_gt_t9_merge_two_clusters(stage, height_map, color_src_idx, cluster_dup_idx, palette=None):
    """Cluster ``cluster_dup_idx`` is drawn with the color of ``color_src_idx``."""
    clusters = get_block_clusters(height_map)
    if len(clusters) < 2:
        _apply_gt_t9_colors(stage, height_map, palette=palette)
        return
    palette = palette or GT_COLOR_PALETTE
    for idx in range(len(clusters)):
        r, g, b = palette[idx % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T9_{idx}", r, g, b)
    block_to_path = {}
    for idx, cluster in enumerate(clusters):
        use_idx = color_src_idx if idx == cluster_dup_idx else idx
        path = f"/World/Looks/GT_T9_{use_idx}"
        for block in cluster:
            block_to_path[block] = path
    set_blocks_material_mapping(stage, block_to_path)


def _apply_gt_t9_single_cluster_color(stage, height_map, color_idx, palette=None):
    """Wrong: every block uses the same palette slot (collapse clusters)."""
    clusters = get_block_clusters(height_map)
    if not clusters:
        return
    palette = palette or GT_COLOR_PALETTE
    for i in range(len(clusters)):
        r, g, b = palette[i % len(palette)]
        create_solid_color_material(stage, f"/World/Looks/GT_T9_{i}", r, g, b)
    path = f"/World/Looks/GT_T9_{color_idx % len(clusters)}"
    block_to_path = {}
    for cluster in clusters:
        for block in cluster:
            block_to_path[block] = path
    set_blocks_material_mapping(stage, block_to_path)


def _build_mcq_wrong_specs_t9(height_map, rng, palette=None, block_set=None):
    clusters = get_block_clusters(height_map, block_set=block_set)
    nc = len(clusters)
    if nc == 0:
        return []
    if nc == 1:
        return [
            (lambda r=rng: _compute_cm_t9_single_cluster_wrong_multicolor(height_map, r, palette), "multicolor_single_cluster", False),
            (lambda: _compute_cm_t9_single_cluster_color(height_map, 0, palette), "single_cluster_one_color_a", False),
            (lambda: _compute_cm_t9_single_cluster_color(height_map, 1, palette), "single_cluster_one_color_b", False),
            (lambda r=rng: _compute_cm_t9_fake_split_cluster(height_map, r, palette), "fake_split_into_two_colors", False),
        ]
    if nc == 2:
        return [
            (lambda r=rng: _compute_cm_t9_fake_split_cluster(height_map, r, palette), "fake_split_one_cluster", False),
            (lambda: _compute_cm_t9_merge_two_clusters(height_map, 0, 1, palette), "merge_c1_to_c0", False),
            (lambda: _compute_cm_t9_merge_two_clusters(height_map, 1, 0, palette), "merge_c0_to_c1", False),
            (lambda: _compute_cm_t9_single_cluster_color(height_map, 0, palette), "collapse_to_one_cluster_color", False),
        ]
    return [
        (lambda r=rng: _compute_cm_t9_fake_split_cluster(height_map, r, palette), "fake_split_largest_cluster", False),
        (lambda: _compute_cm_t9_merge_two_clusters(height_map, 0, 1, palette), "merge_clusters_color0_to_1", False),
        (lambda n=nc: _compute_cm_t9_merge_two_clusters(height_map, 0, min(2, n - 1), palette), "merge_clusters_color0_to_2", False),
        (lambda: _compute_cm_t9_merge_two_clusters(height_map, 1, 0, palette), "merge_clusters_color1_to_0", False),
    ]


def _build_mcq_wrong_specs_t10(height_map, visible_set, visible_set_0, t10_set, rng):
    vs = visible_set
    both = visible_set & visible_set_0
    only_new = t10_set
    w1 = set(rng.sample(list(vs), min(1, len(vs)))) if vs else set()
    w2 = both if both else w1
    w3 = (only_new - {rng.choice(list(only_new))}) if len(only_new) > 1 else only_new
    # Pre-compute random wrong highlight set using a derived RNG
    _rng_w = random.Random(rng.randint(0, 2 ** 30))
    _pool_w = [b for b in vs if b not in t10_set] or list(vs)
    _n_w = _rng_w.randint(1, min(3, len(_pool_w))) if _pool_w else 0
    wrong_t10 = frozenset(_rng_w.sample(_pool_w, _n_w)) if _n_w > 0 else frozenset()
    fvs = frozenset(vs)
    return [
        (lambda fg=wrong_t10, vs=fvs: _compute_cm_selective_highlight(fg, vs), "random_wrong_highlight_blocks", False),
        (lambda fg=w1, vs=fvs: _compute_cm_selective_highlight(fg, vs), "random_single_block", False),
        (lambda fg=w2, vs=fvs: _compute_cm_selective_highlight(fg, vs), "both_views_blocks", False),
        (lambda fg=w3, vs=fvs: _compute_cm_selective_highlight(fg, vs), "missing_nonmatch", False),
    ]


def _build_mcq_wrong_specs_t11(height_map, visible_set, visible_opp, t11_set, rng):
    vs = visible_set
    both = visible_set & visible_opp
    only_new = t11_set
    w1 = set(rng.sample(list(vs), min(1, len(vs)))) if vs else set()
    w2 = both if both else w1
    w3 = (only_new - {rng.choice(list(only_new))}) if len(only_new) > 1 else only_new
    _rng_w = random.Random(rng.randint(0, 2 ** 30))
    _pool_w = [b for b in vs if b not in t11_set] or list(vs)
    _n_w = _rng_w.randint(1, min(3, len(_pool_w))) if _pool_w else 0
    wrong_t11 = frozenset(_rng_w.sample(_pool_w, _n_w)) if _n_w > 0 else frozenset()
    fvs = frozenset(vs)
    return [
        (lambda fg=wrong_t11, vs=fvs: _compute_cm_selective_highlight(fg, vs), "random_wrong_highlight_blocks", False),
        (lambda fg=w1, vs=fvs: _compute_cm_selective_highlight(fg, vs), "random_single_block", False),
        (lambda fg=w2, vs=fvs: _compute_cm_selective_highlight(fg, vs), "both_views_blocks", False),
        (lambda fg=w3, vs=fvs: _compute_cm_selective_highlight(fg, vs), "missing_nonmatch", False),
    ]


@_time_fn
def _save_gt_overlays_for_view(view_dir, view_arr, pixel_map, block_ids, *,
                                height_map, visible_set, t2_sets, t6, t7, t8, t10_set, t11_set):
    """Save taskN_gt_overlay.png for each enabled task using PIL compositing over view.png.

    Uses the same pixel_map/block_ids as MCQ compositing (task5_gt_mask.png + gt.json[visible_blocks]),
    so no extra ray tracing is required. Intended for human scoring of image-output models.
    """
    tasks_and_cms = []
    if GENERATE_GT_T1:
        tasks_and_cms.append(("task1", _compute_cm_highlight(visible_set)))
    if GENERATE_GT_T2 and T2_TOP_ONLY:
        tasks_and_cms.append(("task2", _compute_cm_selective_highlight(t2_sets.get("top", set()), visible_set)))
    if GENERATE_GT_T3:
        tasks_and_cms.append(("task3", _compute_cm_t3(height_map, _shuffled_palette_for_task(view_dir, "task3"))))
    if GENERATE_GT_T4:
        tasks_and_cms.append(("task4", _compute_cm_t4(height_map, _shuffled_palette_for_task(view_dir, "task4"))))
    if GENERATE_GT_T5:
        tasks_and_cms.append(("task5", _compute_cm_t5(height_map, frozenset(visible_set), _shuffled_palette_for_task(view_dir, "task5"))))
    if GENERATE_GT_T6:
        tasks_and_cms.append(("task6", _compute_cm_selective_highlight(t6, visible_set)))
    if GENERATE_GT_T7:
        tasks_and_cms.append(("task7", _compute_cm_selective_highlight(t7, visible_set)))
    if GENERATE_GT_T8:
        tasks_and_cms.append(("task8", _compute_cm_selective_highlight(t8, visible_set)))
    if GENERATE_GT_T9:
        tasks_and_cms.append(("task9", _compute_cm_t9(height_map, _shuffled_palette_for_task(view_dir, "task9"), block_set=_needed_from_visible(height_map, visible_set))))
    if GENERATE_GT_T10:
        tasks_and_cms.append(("task10", _compute_cm_selective_highlight(t10_set, visible_set)))
    if GENERATE_GT_T11:
        tasks_and_cms.append(("task11", _compute_cm_selective_highlight(t11_set, visible_set)))
    for task_id, cm in tasks_and_cms:
        if not cm:
            continue
        path = os.path.join(view_dir, f"{task_id}_gt_overlay.png")
        pil = _mcq_composite_pil(view_arr, pixel_map, block_ids, cm)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        _s3_upload(buf.getvalue(), path)
        print(f"  -> {path} (GT overlay)")


@_time_fn
def _run_mcq_for_view(
    stage,
    height_map,
    azimuth,
    view_dir,
    view_idx,
    rgb_annotator,
    visible_set,
    block_view_info,
    t2_sets,
    views_azimuth,
    seg_annotator_opt,
    *,
    visible_next=None,
    visible_opp=None,
    t6=None,
    t7=None,
    dist_mult=1.0,
    _view_arr=None,
    _pixel_map=None,
):
    if t6 is None:
        t6 = get_t6_support_blocks(height_map)
    if t7 is None:
        t7 = get_t7_support_column_blocks(height_map)
    t8 = get_t8_visible_supported_by_non_visible(visible_set)
    nv = len(views_azimuth)
    if seg_annotator_opt is not None:
        if visible_next is None:
            visible_next = get_visible_blocks_from_replicator(
                stage, height_map, views_azimuth[(view_idx + 1) % nv], seg_annotator_opt, dist_mult=dist_mult, restore_azimuth_after=azimuth
            )
        if visible_opp is None:
            visible_opp = get_visible_blocks_from_replicator(
                stage, height_map, views_azimuth[(view_idx + 2) % nv], seg_annotator_opt, dist_mult=dist_mult, restore_azimuth_after=azimuth
            )
        t10_set = visible_set - visible_next
        t11_set = visible_set - visible_opp
    else:
        visible_next = visible_next or set()
        visible_opp = visible_opp or set()
        t10_set = set()
        t11_set = set()

    # Load compositing assets once (view.png + task5_gt_mask.png + gt.json[visible_blocks])
    view_path = os.path.join(view_dir, "view.png")
    mask_path = os.path.join(view_dir, "task5_gt_mask.png")
    gt_json_path = _gt_json_path(view_dir)
    view_arr = pixel_map = block_ids = None
    if _view_arr is not None and _pixel_map is not None:
        # In-memory path: used when UPLOAD_TO_S3=True and files aren't on disk
        view_arr = _view_arr
        pixel_map = _pixel_map
        _cached = _GT_JSON_CACHE.get(gt_json_path, {})
        block_ids = [tuple(x) for x in _cached.get("visible_blocks", sorted(visible_set))] or None
    elif os.path.isfile(view_path) and os.path.isfile(mask_path) and os.path.isfile(gt_json_path):
        try:
            view_arr = np.array(Image.open(view_path).convert("RGBA"))
            pixel_map = np.array(Image.open(mask_path))  # uint8 H×W
            with open(gt_json_path, encoding="utf-8") as _f:
                _gt = json.load(_f)
            block_ids = [tuple(x) for x in _gt.get("visible_blocks", [])] or None
        except Exception as _e:
            print(f"  !! Compositing assets load failed: {_e}")
            view_arr = pixel_map = block_ids = None
    compositing_ok = view_arr is not None and pixel_map is not None and block_ids is not None

    if not GENERATE_MCQ:
        # No MCQ: save standalone GT overlay PNGs for human scoring (correct MCQ choice would serve
        # this role when MCQ is enabled, so no separate file is needed in that case).
        if compositing_ok:
            _save_gt_overlays_for_view(
                view_dir, view_arr, pixel_map, block_ids,
                height_map=height_map, visible_set=visible_set, t2_sets=t2_sets,
                t6=t6, t7=t7, t8=t8, t10_set=t10_set, t11_set=t11_set,
            )
        return

    total_blocks = len(_needed_from_visible(height_map, visible_set))
    rng_main = random.Random(_mcq_seed_int(view_dir, "task_main"))
    _write_task_main_mcq_text(view_dir, total_blocks, rng_main)

    tasks = []
    if GENERATE_GT_T1:
        tasks.append((1, "task1", lambda: _build_mcq_wrong_specs_t1(height_map, visible_set, random.Random(_mcq_seed_int(view_dir, "task1")))))
    if GENERATE_GT_T2 and T2_TOP_ONLY:
        tasks.append((2, "task2", lambda: _build_mcq_wrong_specs_t2(height_map, visible_set, t2_sets, random.Random(_mcq_seed_int(view_dir, "task2")))))
    if GENERATE_GT_T3:
        _pal3 = _shuffled_palette_for_task(view_dir, "task3")
        tasks.append((3, "task3", lambda p=_pal3: _build_mcq_wrong_specs_t3(height_map, random.Random(_mcq_seed_int(view_dir, "task3")), palette=p)))
    if GENERATE_GT_T4:
        _pal4 = _shuffled_palette_for_task(view_dir, "task4")
        tasks.append((4, "task4", lambda p=_pal4: _build_mcq_wrong_specs_t4(height_map, random.Random(_mcq_seed_int(view_dir, "task4")), palette=p)))
    if GENERATE_GT_T5:
        _pal5 = _shuffled_palette_for_task(view_dir, "task5")
        tasks.append(
            (
                5,
                "task5",
                lambda p=_pal5: _build_mcq_wrong_specs_t5(height_map, visible_set, random.Random(_mcq_seed_int(view_dir, "task5")), palette=p),
            )
        )
    if GENERATE_GT_T6:
        tasks.append((6, "task6", lambda vs=visible_set: _build_mcq_wrong_specs_t6(height_map, vs, t6, random.Random(_mcq_seed_int(view_dir, "task6")))))
    if GENERATE_GT_T7:
        tasks.append((7, "task7", lambda vs=visible_set: _build_mcq_wrong_specs_t7(height_map, vs, t7, random.Random(_mcq_seed_int(view_dir, "task7")))))
    if GENERATE_GT_T8:
        tasks.append(
            (
                8,
                "task8",
                lambda vs=visible_set: _build_mcq_wrong_specs_t8(height_map, t8, vs, random.Random(_mcq_seed_int(view_dir, "task8"))),
            )
        )
    if GENERATE_GT_T9:
        _pal9 = _shuffled_palette_for_task(view_dir, "task9")
        _bs9 = _needed_from_visible(height_map, visible_set)
        tasks.append((9, "task9", lambda p=_pal9, bs=_bs9: _build_mcq_wrong_specs_t9(height_map, random.Random(_mcq_seed_int(view_dir, "task9")), palette=p, block_set=bs)))
    if GENERATE_GT_T10:
        tasks.append(
            (
                10,
                "task10",
                lambda: _build_mcq_wrong_specs_t10(
                    height_map, visible_set, visible_next, t10_set, random.Random(_mcq_seed_int(view_dir, "task10"))
                ),
            )
        )
    if GENERATE_GT_T11:
        tasks.append(
            (
                11,
                "task11",
                lambda: _build_mcq_wrong_specs_t11(
                    height_map, visible_set, visible_opp, t11_set, random.Random(_mcq_seed_int(view_dir, "task11"))
                ),
            )
        )

    for _tn, task_id, spec_fn in tasks:
        rng = random.Random(_mcq_seed_int(view_dir, task_id))
        wrong_specs = spec_fn()
        if len(wrong_specs) < 4:
            print(f"  !! MCQ skip {task_id}: not enough distractors")
            continue

        # Compute correct image via compositing (fast, no ray trace)
        pil_ok = None
        fp_seen = set()
        if compositing_ok and visible_set:
            _pal_t3 = _shuffled_palette_for_task(view_dir, "task3") if task_id == "task3" else None
            _pal_t4 = _shuffled_palette_for_task(view_dir, "task4") if task_id == "task4" else None
            _pal_t5 = _shuffled_palette_for_task(view_dir, "task5") if task_id == "task5" else None
            _pal_t9 = _shuffled_palette_for_task(view_dir, "task9") if task_id == "task9" else None
            cm_ok = _compute_cm_correct_for_task(
                task_id,
                height_map=height_map,
                visible_set=visible_set,
                t2_sets=t2_sets,
                t6=t6,
                t7=t7,
                t8=t8,
                t10_set=t10_set,
                t11_set=t11_set,
                palette_t3=_pal_t3,
                palette_t4=_pal_t4,
                palette_t5=_pal_t5,
                palette_t9=_pal_t9,
            )
            pil_ok = _mcq_composite_pil(view_arr, pixel_map, block_ids, cm_ok)
        elif visible_set:
            # Fallback: ray-trace the correct image
            _mcq_remove_extra_blocks(stage)
            restore_block_original_materials(stage)
            _mcq_apply_correct_gt_for_task(
                stage,
                task_id,
                height_map=height_map,
                visible_set=visible_set,
                t2_sets=t2_sets,
                t6=t6,
                t7=t7,
                t8=t8,
                t10_set=t10_set,
                t11_set=t11_set,
                view_dir=view_dir,
            )
            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
            pil_ok = _mcq_capture_pil(rgb_annotator)
            restore_block_original_materials(stage)
        if pil_ok is not None:
            _fpo = _mcq_rgba_fingerprint(pil_ok)
            if _fpo is not None:
                fp_seen.add(_fpo)

        # Generate wrong-option images
        wrong_entries = []
        for fn_or_cm, lbl, is_phantom in wrong_specs[:4]:
            if is_phantom:
                # Phantom path: spawn raw block → ray-trace → composite colors for visual consistency
                _mcq_remove_extra_blocks(stage)
                restore_block_original_materials(stage)
                phantom_ijk = fn_or_cm(stage)  # spawns block with original texture; returns (i,j,k)
                pil_raw = _mcq_capture_pil(rgb_annotator)
                phantom_binary = None
                if compositing_ok and seg_annotator_opt is not None and phantom_ijk is not None:
                    try:
                        phantom_binary = _get_phantom_block_pixels(stage, seg_annotator_opt)
                    except Exception as _pe:
                        print(f"  !! phantom pixel mask failed: {_pe}")
                restore_block_original_materials(stage)
                if pil_raw is None:
                    pil = pil_ok
                    lbl = lbl + "_capture_fail"
                elif compositing_ok and visible_set and phantom_binary is not None:
                    raw_arr = np.array(pil_raw)
                    if task_id == "task2":
                        # T2 phantom: only color the phantom block, not the original top blocks
                        arr = raw_arr.copy()
                    else:
                        # Composite: paint original block colors onto raw image, then highlight phantom pixels
                        pil = _mcq_composite_pil(raw_arr, pixel_map, block_ids, cm_ok)
                        arr = np.array(pil)
                    r, g, b = _GT_HIGHLIGHT_COLOR_255
                    m = phantom_binary == 1
                    arr[m, 0] = r; arr[m, 1] = g; arr[m, 2] = b; arr[m, 3] = 255
                    pil = Image.fromarray(arr)
                else:
                    pil = pil_raw  # fallback: raw ray-trace without color overlay
                wrong_entries.append([pil, lbl, _mcq_rgba_fingerprint(pil), is_phantom])
            elif not compositing_ok:
                # compositing unavailable: use random fallback via ray trace
                _mcq_remove_extra_blocks(stage)
                restore_block_original_materials(stage)
                _fr = random.Random(_mcq_seed_int(view_dir, f"{task_id}_{lbl}_ncomp"))
                _mcq_apply_fallback_distractor(stage, height_map, visible_set, task_id, 0, _fr)
                pil = _mcq_capture_pil(rgb_annotator)
                restore_block_original_materials(stage)
                if pil is None:
                    pil = pil_ok
                    lbl = lbl + "_capture_fail"
                wrong_entries.append([pil, lbl, _mcq_rgba_fingerprint(pil), is_phantom])
            else:
                # Compositing path (fast, no ray trace)
                cm = fn_or_cm()
                pil = _mcq_composite_pil(view_arr, pixel_map, block_ids, cm)
                wrong_entries.append([pil, lbl, _mcq_rgba_fingerprint(pil), is_phantom])

        _mcq_dedupe_wrong_entries(
            stage, rgb_annotator, wrong_entries, visible_set, height_map, task_id, rng,
            fp_seen, view_arr, pixel_map, block_ids, compositing_ok,
        )

        if pil_ok is None:
            print(f"  !! MCQ skip {task_id}: pil_ok=None "
                  f"compositing_ok={compositing_ok} visible_set={len(visible_set)} "
                  f"view_png={os.path.isfile(os.path.join(view_dir, 'view.png'))} "
                  f"t5mask={os.path.isfile(os.path.join(view_dir, 'task5_gt_mask.png'))} "
                  f"gtjson={os.path.isfile(os.path.join(view_dir, 'gt.json'))} "
                  f"block_ids={'None' if block_ids is None else len(block_ids)}", flush=True)
            continue
        imgs = [pil_ok] + [e[0] for e in wrong_entries]
        labels = ["correct"] + [e[1] for e in wrong_entries]
        _write_mcq_image_shuffle(view_dir, task_id, imgs, labels, rng)
    _mcq_remove_extra_blocks(stage)
    restore_block_original_materials(stage)

# =============================================================================
# HEIGHT MAP FROM USER CONFIG
# =============================================================================
def build_height_map():
    """Build 0-based height map [row][col]. Uses HEIGHTS if MODE=='manual', else random 0..MAX_HEIGHT."""
    if MODE == "random":
        return [
            [random.randint(0, MAX_HEIGHT) for j in range(W)]
            for i in range(H)
        ]
    return [
        [HEIGHTS.get((i + 1, j + 1), 0) for j in range(W)]
        for i in range(H)
    ]


def random_height_map(h, w, max_height):
    """Return a random height map of shape h x w with each cell in [0, max_height]."""
    return [[random.randint(0, max_height) for j in range(w)] for i in range(h)]


# -----------------------------------------------------------------------------
# View-0 ambiguity removal: keep only blocks that are visible from view 0 or
# support a visible block (so from view 0 there is no "hidden useless" block).
# Per-view GT block counts are computed independently during rendering using
# _needed_from_visible() applied to each view's ray-traced visible_set.
# -----------------------------------------------------------------------------
def _camera_and_bounds_for_azimuth0(height_map):
    """Same math as compute_perspective_camera for azimuth 0. Returns (cam_pos, target, unit, block_size, h, w, center_row, center_col)."""
    unit = (BLOCK_SIZE + SPACING) * SCALE
    block_size = BLOCK_SIZE * SCALE
    h, w = len(height_map), len(height_map[0])
    center_row, center_col = (h - 1) / 2.0, (w - 1) / 2.0
    min_x = -center_row * unit - block_size / 2.0
    max_x = (h - 1 - center_row) * unit + block_size / 2.0
    min_y = -center_col * unit - block_size / 2.0
    max_y = (w - 1 - center_col) * unit + block_size / 2.0
    max_z_blocks = max(max(row) for row in height_map) if height_map else 1
    max_z = max_z_blocks * unit + block_size
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    cz = max_z / 2.0
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    extent_z = max_z
    bbox_diagonal = math.sqrt(extent_x ** 2 + extent_y ** 2 + extent_z ** 2)
    effective_radius = bbox_diagonal / 2.0
    effective_radius = max(effective_radius, block_size * 2.0)
    half_fov_base_rad = math.radians(23.6)
    dist = (effective_radius / math.tan(half_fov_base_rad)) * CAMERA_MARGIN * 1.5
    offset_dir = Gf.Vec3d(-2.5, -2.3, 2.3)
    offset_dir = offset_dir.GetNormalized()
    dx, dy = offset_dir[0] * dist, offset_dir[1] * dist
    default_cam_z = cz + offset_dir[2] * dist
    cam_z = CAMERA_Z_MULTIPLIER * default_cam_z
    # Azimuth 0: no rotation
    cam_x = cx + dx
    cam_y = cy + dy
    # Same as compute_perspective_camera: C + (C - B) * (mult - 1)
    k = CAMERA_DISTANCE_MULTIPLIER - 1.0
    cam_x = cam_x + k * (cam_x - cx)
    cam_y = cam_y + k * (cam_y - cy)
    cam_z = cam_z + k * (cam_z - cz)
    cam_pos = (cam_x, cam_y, cam_z)
    target = (cx, cy, cz)
    return cam_pos, target, unit, block_size, h, w, center_row, center_col


def filter_height_map_for_view0(height_map):
    """Return (height_map, num_visible): height_map with only blocks visible from view 0 or needed to support them,
    and the count of blocks visible from view 0 (non-visible = total_blocks - num_visible).
    In asset mode, temporarily computes BLOCK_ASSET_LAYOUT to get correct block geometry."""
    if not height_map or not any(cell for row in height_map for cell in row):
        return height_map, 0

    global BLOCK_ASSET_LAYOUT
    _saved_layout = BLOCK_ASSET_LAYOUT  # may be stale from a previous sample's build_block_scene
    try:
        if BLOCK_SHAPE == "asset":
            # Always recompute from current height_map; stale layout from prior sample has wrong keys
            BLOCK_ASSET_LAYOUT = _compute_asset_layout_from_height_map(height_map)
        visible = get_visible_blocks(height_map, 0)
        needed = _needed_from_visible(height_map, visible)
        h, w = len(height_map), len(height_map[0])
        new_height_map = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                new_height_map[i][j] = max(
                    (k + 1 for k in range(height_map[i][j]) if (i, j, k) in needed), default=0
                )
        return new_height_map, len(visible)
    finally:
        BLOCK_ASSET_LAYOUT = _saved_layout  # restore to whatever was active before this call


def _needed_from_visible(height_map, visible):
    """Expand a visible-block set to include all structural supporters (downward transitive closure).

    A block at (i, j, k) is 'needed' if it is visible OR if the block directly above it (k+1)
    is needed — i.e. it must exist to support what the camera can see.
    Returns the full needed set: visible blocks + all their supporters.
    """
    h, w = len(height_map), len(height_map[0])
    needed = set(visible)
    max_k = max((height_map[i][j] for i in range(h) for j in range(w)), default=0) - 1
    for k in range(max_k, -1, -1):
        for i in range(h):
            for j in range(w):
                if (i, j, k + 1) in needed:
                    needed.add((i, j, k))
    return needed


def _random_height_map_with_exact_count(h, w, max_height, target_blocks):
    """Generate a random height map with exactly target_blocks blocks (no filter)."""
    max_blocks = h * w * max_height
    target_blocks = max(0, min(target_blocks, max_blocks))
    chosen_slots = random.sample(range(max_blocks), target_blocks)
    height_map = [[0] * w for _ in range(h)]
    for k in chosen_slots:
        cell_linear = k // max_height
        i, j = cell_linear // w, cell_linear % w
        height_map[i][j] += 1
    return height_map


_W3, _W11 = 0.017, 1.6       # linear rise endpoints (n=3..11)
_W12, _W30 = 0.8, 0.20       # linear fall endpoints (n=12..30)
_EXP_START, _EXP_K = 0.225, 0.2  # exponential decay (n=31+)


def _block_count_weight(n):
    if n < 3:
        return 0.0
    if n <= 11:    # linear rise 3 → 11
        return _W3 + (_W11 - _W3) * (n - 3) / 8
    if n <= 30:    # slow linear fall 12 → 30
        return _W12 - (_W12 - _W30) * (n - 12) / 18
    return _EXP_START * math.exp(-_EXP_K * (n - 31))


def random_height_map_with_block_count(h, w, max_height, min_blocks_param):
    """Return (height_map, final_blocks, pre_cull, num_visible). Final count is sampled with piecewise weight.
    num_visible = blocks visible from view 0; non_visible = final_blocks - num_visible."""
    max_blocks = h * w * max_height
    min_blocks = max(0, min(min_blocks_param, max_blocks))
    counts = list(range(min_blocks, max_blocks + 1))
    weights = [_block_count_weight(n) for n in counts]
    desired_final = random.choices(counts, weights=weights, k=1)[0]
    max_attempts = 150
    best_height_map = None
    best_final = None
    best_pre_cull = None
    best_num_visible = 0
    for _ in range(max_attempts):
        pre_cull = random.randint(desired_final, max_blocks)
        height_map = _random_height_map_with_exact_count(h, w, max_height, pre_cull)
        filtered, num_visible = filter_height_map_for_view0(height_map)
        final_blocks = sum(cell for row in filtered for cell in row)
        if final_blocks == desired_final:
            return filtered, final_blocks, pre_cull, num_visible
        if best_height_map is None or abs(final_blocks - desired_final) < abs(best_final - desired_final):
            best_height_map = filtered
            best_final = final_blocks
            best_pre_cull = pre_cull
            best_num_visible = num_visible
    return best_height_map, best_final, best_pre_cull, best_num_visible


# =============================================================================
# SCENE BUILDING (pure USD)
# =============================================================================
def resolve_environment_usd_reference():
    """Return USD reference string for the configured environment, or None to skip.
    Priority: ENVIRONMENT_USD_OVERRIDE, else ENVIRONMENT_ASSETS_BASE_URL + ENVIRONMENT_USD_RELATIVE_PATH if LOAD_ENVIRONMENT_USD."""
    if ENVIRONMENT_USD_OVERRIDE:
        s = str(ENVIRONMENT_USD_OVERRIDE).strip()
        return s if s else None
    if not LOAD_ENVIRONMENT_USD:
        return None
    rel = (ENVIRONMENT_USD_RELATIVE_PATH or "").strip()
    if not rel:
        return None
    base = ENVIRONMENT_ASSETS_BASE_URL.rstrip("/") + "/"
    return base + rel.lstrip("/")


def add_environment_reference(stage, usd_path, prim_path=None):
    """Reference a USD asset into the current stage (does not replace the stage).
    usd_path: absolute .usd path on disk, or https:// / omniverse:// URL (Kit resolves remote assets).
    prim_path: defaults to ENVIRONMENT_PRIM_PATH."""
    if prim_path is None:
        prim_path = ENVIRONMENT_PRIM_PATH
    if not usd_path:
        return False
    usd_path = str(usd_path).strip()
    if not usd_path:
        return False
    is_url = (
        usd_path.startswith("omniverse://")
        or usd_path.startswith("http://")
        or usd_path.startswith("https://")
    )
    if not is_url and not os.path.isfile(usd_path):
        print(f"[BlockGen] Environment USD not found: {usd_path}")
        return False
    ref_path = usd_path if is_url else os.path.abspath(usd_path).replace("\\", "/")
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    prim.GetReferences().AddReference(ref_path)
    print(f"[BlockGen] Referenced environment -> {prim_path} ({ref_path})")
    return True


@_time_fn
def build_block_scene(stage, height_map=None):
    """Place blocks + lights. Returns (total_blocks, height_map).
    If height_map is None, use global H, W and build_height_map(). Else use provided height_map (H, W from shape)."""
    # Clean previous runs and remove Isaac Sim default ground plane(s)
    to_remove = [
        "/World/Blocks",
        "/World/Lights",
        "/World/Cameras",
        "/World/Looks",
        ENVIRONMENT_PRIM_PATH,  # clear prior environment reference on rebuild
        "/World/defaultGroundPlane",  # Isaac Sim's default ground plane
        "/World/GroundPlane",
        "/GroundPlane",
    ]
    for p in to_remove:
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            stage.RemovePrim(p)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    UsdGeom.Xform.Define(stage, "/World/Blocks")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    BLOCK_ORIGINAL_MATERIAL_PATH.clear()
    _BLOCK_PRIMS.clear()

    _env_ref = resolve_environment_usd_reference()
    if _env_ref:
        add_environment_reference(stage, _env_ref)

    texture_images = _get_block_texture_image_list()
    if texture_images:
        # One material per texture; each block will get a random one. Also create /World/Looks/WhiteMaterial
        # (using first texture) so GT code (set_blocks_material, set_blocks_material_mapping default_path) can use it.
        block_materials = []
        for i, src in enumerate(texture_images):
            if not os.path.isfile(src):
                continue
            tex_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"block_edge_texture_{i}.png"))
            _load_texture_with_black_edges(src, tex_path, size=256, border_pct=0.05)
            if i == 0:
                create_white_material_with_edges(stage, "/World/Looks/WhiteMaterial", tex_path)
            mat = create_white_material_with_edges(stage, f"/World/Looks/WhiteMaterial_{i}", tex_path)
            block_materials.append(mat)
        if not block_materials:
            texture_images = []
    if not texture_images:
        # Procedural texture for all blocks
        texture_path = os.path.abspath(os.path.join(OUTPUT_DIR, "block_edge_texture.png"))
        _generate_edge_texture(texture_path, size=128, border_pct=0.05)
        single_material = create_white_material_with_edges(stage, "/World/Looks/WhiteMaterial", texture_path)
        block_materials = [single_material]

    create_solid_color_material(stage, "/World/Looks/BlueMaterial", 0.0, 0.0, 1.0)
    create_solid_color_material(stage, "/World/Looks/BackgroundMaterial", 0.0, 0.0, 0.0)

    block_size = BLOCK_SIZE * SCALE
    unit = (BLOCK_SIZE + SPACING) * SCALE
    total_blocks = 0

    if height_map is None:
        height_map = build_height_map()
    h, w = len(height_map), len(height_map[0])
    center_row, center_col = (h - 1) / 2.0, (w - 1) / 2.0

    global BLOCK_ASSET_LAYOUT
    if BLOCK_SHAPE == "asset":
        BLOCK_ASSET_LAYOUT = _compute_asset_layout_from_height_map(height_map)
        target_size = BLOCK_SIZE * SCALE
        for x in range(h):
            for y in range(w):
                for z in range(height_map[x][y]):
                    total_blocks += 1
                    pp = f"/World/Blocks/b_{x}_{y}_{z}"
                    z_deg = BLOCK_ASSET_LAYOUT["z_rot_deg"].get((x, y, z), 0.0)
                    asset_key = BLOCK_ASSET_LAYOUT["assignment"].get((x, y, z), ASSET_BLOCK_MODE)
                    spec = ASSET_SPECS[asset_key]
                    _create_asset_block_from_spec(stage, pp, spec, target_size, z_deg)
                    _prim = stage.GetPrimAtPath(pp)
                    cx, cy, cz = BLOCK_ASSET_LAYOUT["centers"][(x, y, z)]
                    UsdGeom.Xformable(_prim).AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
                    _BLOCK_PRIMS[(x, y, z)] = _prim
    elif BLOCK_SHAPE == "factory_box":
        BLOCK_ASSET_LAYOUT = None
        for x in range(h):
            for y in range(w):
                for z in range(height_map[x][y]):
                    total_blocks += 1
                    pp = f"/World/Blocks/b_{x}_{y}_{z}"
                    z_deg = _pick_factory_box_z_deg(x, y, z)
                    _create_factory_box_block(stage, pp, block_size, z_deg)
                    _prim = stage.GetPrimAtPath(pp)
                    xf = UsdGeom.Xformable(_prim)
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(
                        (x - center_row) * unit, (y - center_col) * unit, z * unit + block_size / 2.0))
                    _BLOCK_PRIMS[(x, y, z)] = _prim
    else:
        BLOCK_ASSET_LAYOUT = None
        for x in range(h):
            for y in range(w):
                for z in range(height_map[x][y]):
                    total_blocks += 1
                    pp = f"/World/Blocks/b_{x}_{y}_{z}"
                    mesh = _create_cube_mesh_with_uvs(stage, pp, block_size)

                    # Center grid at world origin
                    xf = UsdGeom.Xformable(mesh.GetPrim())
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(
                        (x - center_row) * unit, (y - center_col) * unit, z * unit + block_size / 2.0))

                    _prim = mesh.GetPrim()
                    _BLOCK_PRIMS[(x, y, z)] = _prim
                    binding_api = UsdShade.MaterialBindingAPI(_prim)
                    mat = random.choice(block_materials)
                    BLOCK_ORIGINAL_MATERIAL_PATH[(x, y, z)] = str(mat.GetPrim().GetPath())
                    binding_api.Bind(mat)

    # Improved lighting setup
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.GetIntensityAttr().Set(400.0)
    dome.GetColorAttr().Set(Gf.Vec3f(0.9, 0.95, 1.0))  # Slight blue tint
    
    # Key light from diagonal
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.GetIntensityAttr().Set(2500.0)
    key.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))  # Warm white
    xf = UsdGeom.Xformable(key.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-40, 45, 0))
    
    # Fill light from opposite side (softer)
    fill = UsdLux.DistantLight.Define(stage, "/World/Lights/Fill")
    fill.GetIntensityAttr().Set(800.0)
    fill.GetColorAttr().Set(Gf.Vec3f(0.95, 0.97, 1.0))  # Cool white
    xf_fill = UsdGeom.Xformable(fill.GetPrim())
    xf_fill.ClearXformOpOrder()
    xf_fill.AddRotateXYZOp().Set(Gf.Vec3f(-30, -135, 0))

    _add_block_semantics(stage)
    print(f"[BlockGen] Placed {total_blocks} blocks on {h}x{w} grid")
    return total_blocks, height_map


def _add_block_semantics(stage):
    """Add semantic label 'block' to all /World/Blocks/b_* prims for Replicator semantic_segmentation annotator.
    This is render-time per-pixel: the annotator labels each pixel with the semantic of the 3D object rendered there,
    so blocks should always show up as 'block' when visible. Prefer Isaac Sim's add_update_semantics when available."""
    blocks_prim = stage.GetPrimAtPath("/World/Blocks")
    if not blocks_prim.IsValid():
        return
    # Prefer Isaac Sim semantics API so the annotator (same stack) sees the labels.
    try:
        from omni.isaac.core.utils.semantics import add_update_semantics
        use_isaac_semantics = True
    except Exception:
        use_isaac_semantics = False
    for child in _BLOCK_PRIMS.values():
        # For asset blocks, the outer Xform has no geometry; label all imageable descendants so
        # the segmentation annotator can see them through the USD reference hierarchy.
        prims_to_label = [child]
        if BLOCK_ASSET_LAYOUT is not None:
            prims_to_label = [
                p for p in Usd.PrimRange(child)
                if UsdGeom.Imageable(p)
            ]
        for prim in prims_to_label:
            applied = False
            if use_isaac_semantics:
                try:
                    add_update_semantics(prim, "block")
                    applied = True
                except Exception:
                    use_isaac_semantics = False
            if not applied:
                # Fallback: raw USD attributes (Replicator may not read these in all builds).
                a1 = prim.CreateAttribute("semantic:class", Sdf.ValueTypeNames.String)
                if a1:
                    a1.Set("block")
                a2 = prim.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray)
                if a2:
                    a2.Set(["block"])
                a3 = prim.CreateAttribute("semantics:class", Sdf.ValueTypeNames.Token)
                if a3:
                    a3.Set("block")


def _set_block_semantics_selective(stage, block_set, label_in="block", label_out="background"):
    """Set semantic label_in on blocks in block_set (set of (i,j,k)), label_out on all others.
    Used for T2 masks so only the 'extreme' blocks for the current key get semantic 'block'.
    We always use USD attribute overwrite (not Isaac add_update_semantics) so the annotator
    sees the updated class per prim; Isaac can cache the initial 'block' and not refresh."""
    for key, child in _BLOCK_PRIMS.items():
        label = label_in if key in block_set else label_out
        # Overwrite in place so Replicator sees updated class (Isaac add_update_semantics can lag).
        a1 = child.CreateAttribute("semantic:class", Sdf.ValueTypeNames.String)
        if a1:
            a1.Set(label)
        a2 = child.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray)
        if a2:
            a2.Set([label])
        a3 = child.CreateAttribute("semantics:class", Sdf.ValueTypeNames.Token)
        if a3:
            a3.Set(label)


def _set_block_visibility_selective(stage, visible_set):
    """Set visibility to inherited for blocks in visible_set (set of (i,j,k)), invisible for all others.
    Used for T2 mask: render segmentation with only blue blocks visible so mask is 1 only on those pixels."""
    for key, child in _BLOCK_PRIMS.items():
        imageable = UsdGeom.Imageable(child)
        if imageable:
            vis_attr = imageable.GetVisibilityAttr()
            if not vis_attr:
                vis_attr = imageable.CreateVisibilityAttr()
            vis_attr.Set(UsdGeom.Tokens.inherited if key in visible_set else UsdGeom.Tokens.invisible)


def _set_all_blocks_visible(stage):
    """Restore visibility to inherited for all /World/Blocks/b_* prims."""
    for child in _BLOCK_PRIMS.values():
        imageable = UsdGeom.Imageable(child)
        if imageable:
            vis_attr = imageable.GetVisibilityAttr()
            if not vis_attr:
                vis_attr = imageable.CreateVisibilityAttr()
            vis_attr.Set(UsdGeom.Tokens.inherited)


# =============================================================================
# CAMERA CREATION (pure USD — Isaac 4.5 compatible)
# Computes position adaptively so the entire block structure fits in frame.
# =============================================================================
def compute_perspective_camera(height_map):
    """Compute camera pos + target that guarantees all blocks fit in frame.

    The camera is placed at a 45-ish degree diagonal looking at the scene
    centre.  Distance is derived from the scene's bounding sphere so the
    full structure is visible with CAMERA_MARGIN padding.
    """
    unit = (BLOCK_SIZE + SPACING) * SCALE
    block_size = BLOCK_SIZE * SCALE
    h, w = len(height_map), len(height_map[0])
    center_row, center_col = (h - 1) / 2.0, (w - 1) / 2.0

    # Scene bounding box — grid centered at origin
    min_x = -center_row * unit - block_size / 2.0
    max_x = (h - 1 - center_row) * unit + block_size / 2.0
    min_y = -center_col * unit - block_size / 2.0
    max_y = (w - 1 - center_col) * unit + block_size / 2.0
    max_z_blocks = max(max(row) for row in height_map) if height_map else 1
    max_z = max_z_blocks * unit + block_size

    # Centre of the bounding box (now at origin in x,y for 3x3)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    cz = max_z / 2.0

    # Bounding box dimensions for diagonal
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    extent_z = max_z

    # For a diagonal view, use the full bounding box diagonal as effective radius
    bbox_diagonal = math.sqrt(extent_x ** 2 + extent_y ** 2 + extent_z ** 2)
    effective_radius = bbox_diagonal / 2.0
    # Ensure minimum radius (and thus camera distance) so 1x1x1 and tiny scenes render correctly
    effective_radius = max(effective_radius, block_size * 2.0)

    # Original distance (no multiplier) so framing matches the working script
    half_fov_base_rad = math.radians(23.6)
    dist = (effective_radius / math.tan(half_fov_base_rad)) * CAMERA_MARGIN * 1.5

    # Place camera at a diagonal offset (roughly equal parts x, y, z)
    offset_dir = Gf.Vec3d(-2.5, -2.3, 2.3)
    offset_dir = offset_dir.GetNormalized()
    dx = offset_dir[0] * dist
    dy = offset_dir[1] * dist
    default_cam_z = cz + offset_dir[2] * dist
    cam_z = CAMERA_Z_MULTIPLIER * default_cam_z

    # Apply azimuth rotation in x-y plane (clockwise, 0° = current view)
    theta_rad = math.radians(CAMERA_AZIMUTH_DEG)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    # Clockwise rotation: (dx, dy) -> (dx*cos + dy*sin, -dx*sin + dy*cos)
    cam_x = cx + dx * cos_t + dy * sin_t
    cam_y = cy - dx * sin_t + dy * cos_t

    # B = block center, C = camera. New position: C + (C - B) * (mult - 1) → CAMERA_DISTANCE_MULTIPLIER x further from B
    k = CAMERA_DISTANCE_MULTIPLIER - 1.0
    cam_x = cam_x + k * (cam_x - cx)
    cam_y = cam_y + k * (cam_y - cy)
    cam_z = cam_z + k * (cam_z - cz)

    cam_pos = (cam_x, cam_y, cam_z)
    cam_target = (cx, cy, cz)

    return cam_pos, cam_target


@_time_fn
def create_usd_cameras(stage, height_map, fov_multiplier=None, dist_multiplier=None):
    """Create a single adaptive perspective camera. Returns list of prim-path strings."""
    UsdGeom.Xform.Define(stage, "/World/Cameras")

    cam_pos, cam_target, _, _, _, _, _, _ = _camera_and_bounds_for_azimuth(height_map, CAMERA_AZIMUTH_DEG)
    # Apply per-iteration distance scaling along the ray from target to cam_pos.
    # dist_multiplier=1.0 → unchanged; 0.0 → camera at scene center.
    _dist = dist_multiplier if dist_multiplier is not None else (DIST_MULTIPLIERS[0] if DIST_MULTIPLIERS else 1.0)
    if _dist != 1.0:
        cx, cy, cz = cam_target
        px, py, pz = cam_pos
        cam_pos = (cx + _dist * (px - cx), cy + _dist * (py - cy), cz + _dist * (pz - cz))
    print(f"[BlockGen] Camera pos={tuple(f'{v:.3f}' for v in cam_pos)}  "
          f"target={tuple(f'{v:.3f}' for v in cam_target)}"
          + (f"  dist_mult={_dist}" if _dist != 1.0 else ""))

    path = "/World/Cameras/cam_perspective"
    cam = UsdGeom.Camera.Define(stage, path)
    # Lens: always set aperture + focal length explicitly so that switching between FOV multipliers
    # within a loop correctly resets the previous value (USD prim retains old attrs otherwise).
    # When fov_multiplier=1, new_focal_length == _USD_CAM_FOCAL_LENGTH (no-op numerically).
    _fov = fov_multiplier if fov_multiplier is not None else (FOV_MULTIPLIERS[0] if FOV_MULTIPLIERS else 1.0)
    block_size = BLOCK_SIZE * SCALE
    near_clip = min(0.1, block_size * 0.5)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(near_clip, 50000.0))
    base_half_fov = math.atan(_USD_CAM_H_APERTURE / (2.0 * _USD_CAM_FOCAL_LENGTH))
    new_focal_length = _USD_CAM_H_APERTURE / (2.0 * math.tan(base_half_fov * _fov))
    cam.GetHorizontalApertureAttr().Set(_USD_CAM_H_APERTURE)
    cam.GetVerticalApertureAttr().Set(_USD_CAM_V_APERTURE)
    cam.GetFocalLengthAttr().Set(new_focal_length)
    if _fov != 1.0:
        print(f"[BlockGen] fov_multiplier={_fov}: focal_length={new_focal_length:.2f} "
              f"(full H-FOV={math.degrees(2 * base_half_fov * _fov):.1f}°)", flush=True)

    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    mat_op = xf.AddTransformOp()
    mat_op.Set(look_at_matrix(cam_pos, cam_target))

    return [path]


# =============================================================================
# RENDER PRODUCTS + ANNOTATORS
# =============================================================================
@_time_fn
def setup_render(cam_paths):
    """Create render products, rgb annotators, and optionally semantic_segmentation annotator for each camera path.
    Returns (render_products, rgb_annotators, seg_annotators or None).
    seg_annotators is a list of semantic_segmentation annotators (one per rp).
    """
    render_products = []
    rgb_annotators = []
    seg_annotators = []

    for path in cam_paths:
        rp = rep.create.render_product(path, IMAGE_RES)
        render_products.append(rp)

        rgb = AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(rp)
        rgb_annotators.append(rgb)

        try:
            seg = AnnotatorRegistry.get_annotator("semantic_segmentation")
            seg.attach(rp)
            seg_annotators.append(seg)
        except Exception:
            seg_annotators.append(None)

    return render_products, rgb_annotators, seg_annotators




# =============================================================================
# ENTRY POINT
# =============================================================================
def run_sync():
    global CAMERA_AZIMUTH_DEG
    if len(FOV_MULTIPLIERS) != len(DIST_MULTIPLIERS):
        raise ValueError(f"FOV_MULTIPLIERS ({len(FOV_MULTIPLIERS)}) and DIST_MULTIPLIERS ({len(DIST_MULTIPLIERS)}) must be the same length")
    print("[BlockGen] run_sync() starting — dataset generation (after Kit 'app ready').", flush=True)
    rep.orchestrator.set_capture_on_play(False)
    _apply_rtx_path_tracing_if_env()
    _apply_rtx_dlss_exec_mode()
    stage = omni.usd.get_context().get_stage()
    _reset_file_timing_log()

    if SCENE_JSON_PATH:
        # Recreate a single scene from a saved scene.json and render NUM_VIEWS into its folder
        scene_dir = os.path.dirname(os.path.abspath(SCENE_JSON_PATH))
        _log_sample_start(scene_dir)
        height_map = load_scene_json(SCENE_JSON_PATH)
        build_block_scene(stage, height_map)
        for fov_idx, (fov_mult, dist_mult) in enumerate(zip(FOV_MULTIPLIERS, DIST_MULTIPLIERS)):
            CAMERA_AZIMUTH_DEG = VIEW_AZIMUTHS_BASE[0] + OFFSETS[0]
            cam_paths = create_usd_cameras(stage, height_map, fov_mult, dist_mult)
            render_products, rgb_annotators, seg_annotators = setup_render(cam_paths)
            for view_idx, _base_azimuth in enumerate(VIEW_AZIMUTHS_BASE):
                for offset in OFFSETS:
                    azimuth = _base_azimuth + offset
                    views_azimuth = [b + offset for b in VIEW_AZIMUTHS_BASE]
                    CAMERA_AZIMUTH_DEG = azimuth
                    _set_camera_to_azimuth(stage, height_map, azimuth, dist_mult)
                    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                    for _ in range(2):
                        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                    data = rgb_annotators[0].get_data()
                    if UPLOAD_TO_S3:
                        _snap = _normalize_rgb_data(data)
                        _mcq_scene_snap = _snap.copy() if _snap is not None else None
                    else:
                        _mcq_scene_snap = None
                    view_dir = _view_fov_dir(scene_dir, view_idx, offset, fov_mult, dist_mult)
                    _update_gt_json(view_dir, {
                        "object_type": BLOCK_SHAPE if BLOCK_SHAPE != "asset" else ASSET_BLOCK_MODE,
                        "view_idx": view_idx + 1,
                        "offset": offset,
                        "fov": fov_mult,
                        "dist": dist_mult,
                    })
                    fp = os.path.join(view_dir, "view.png")
                    if data is not None:
                        save_rgb(data, fp)
                        print(f"  -> {fp}")
                    else:
                        print(f"  !! no data: {fp}")
                    if GENERATE_GT_T1:
                        set_blocks_material(stage, "/World/Looks/BlueMaterial")
                        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                        for _ in range(GT_EXTRA_RENDER_STEPS):
                            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                        fp_gt = _gt_file_path(view_dir, "task1")
                        if seg_annotators and seg_annotators[0] is not None:
                            _save_gt_mask_png(_get_binary_segmentation_from_annotator(seg_annotators[0]), fp_gt)
                        set_blocks_material(stage, "/World/Looks/WhiteMaterial")
                    _save_gt_text(view_dir, "task1", "blocks")
                    block_view_info = get_block_view_info(height_map, azimuth)
                    visible_set = get_visible_blocks_from_replicator(stage, height_map, azimuth, seg_annotators[0], dist_mult=dist_mult) if seg_annotators and seg_annotators[0] else set()
                    _vis_for_count = visible_set if visible_set else get_visible_blocks(height_map, azimuth)
                    total_blocks = len(_needed_from_visible(height_map, _vis_for_count))
                    _save_gt_text(view_dir, "task_main", total_blocks)
                    t2_sets = get_t2_extreme_block_sets(block_view_info, height_map, visible_set)
                    t8_set = get_t8_visible_supported_by_non_visible(visible_set)
                    _add_block_semantics(stage)  # restore semantic labels for next view's T1
                    _save_gt_text(view_dir, "task2", len(t2_sets["top"]))
                    h, w = len(height_map), len(height_map[0])
                    num_columns = sum(1 for i in range(h) for j in range(w) if height_map[i][j] > 0)
                    _save_gt_text(view_dir, "task3", num_columns)
                    num_layers = max((height_map[i][j] for i in range(h) for j in range(w)), default=0)
                    _save_gt_text(view_dir, "task4", num_layers)
                    _vis_needed = _needed_from_visible(height_map, _vis_for_count)
                    clusters = get_block_clusters(height_map, block_set=_vis_needed)
                    _update_gt_json(view_dir, {
                        "total_blocks": total_blocks,
                        "num_hidden_blocks": total_blocks - len(_vis_for_count),
                        "num_columns": num_columns,
                        "num_layers": num_layers,
                        "num_clusters": len(clusters),
                    })
                    _save_gt_text(view_dir, "task5", len(visible_set))
                    t6_support_set = get_t6_support_blocks(height_map)
                    _save_gt_text(view_dir, "task6", len(t6_support_set & visible_set))
                    t7_support_column_set = get_t7_support_column_blocks(height_map)
                    _save_gt_text(view_dir, "task7", len(t7_support_column_set & visible_set))
                    _save_gt_text(view_dir, "task8", len(t8_set))
                    _save_gt_text(view_dir, "task9", len(clusters))
                    _nv = len(views_azimuth)
                    if seg_annotators and seg_annotators[0]:
                        visible_next = get_visible_blocks_from_replicator(stage, height_map, views_azimuth[(view_idx + 1) % _nv], seg_annotators[0], dist_mult=dist_mult, restore_azimuth_after=azimuth)
                        t10_set = visible_set - visible_next
                        visible_opp = get_visible_blocks_from_replicator(stage, height_map, views_azimuth[(view_idx + 2) % _nv], seg_annotators[0], dist_mult=dist_mult, restore_azimuth_after=azimuth)
                        t11_set = visible_set - visible_opp
                    else:
                        visible_next = set()
                        visible_opp = set()
                        t10_set = set()
                        t11_set = set()
                    _save_gt_text(view_dir, "task10", len(t10_set))
                    _save_gt_text(view_dir, "task11", len(t11_set))
                    # --- Combined mask pass: single back-to-front loop for all tasks ---
                    _seg = seg_annotators[0] if seg_annotators else None
                    _need_t5_mask = (GENERATE_MCQ or GENERATE_GT_OVERLAY) and bool(visible_set) and _seg is not None
                    visible_ordered = sorted(visible_set)
                    block_to_label = {b: idx + 1 for idx, b in enumerate(visible_ordered)}
                    block_to_cid = {b: cid + 1 for cid, cl in enumerate(clusters) for b in cl}
                    _value_fns = {}
                    if _seg is not None:
                        if GENERATE_GT_T2 and T2_TOP_ONLY:
                            _t2s = frozenset(t2_sets["top"])
                            _value_fns["t2"] = lambda i, j, k, _s=_t2s: 1 if (i, j, k) in _s else 0
                        if GENERATE_GT_T3:
                            _w = w
                            _value_fns["t3"] = lambda i, j, k, _w=_w: i * _w + j + 1
                        if GENERATE_GT_T4:
                            _value_fns["t4"] = lambda i, j, k: k + 1
                        if GENERATE_GT_T5 or _need_t5_mask:
                            _btl = block_to_label
                            _value_fns["t5"] = lambda i, j, k, _m=_btl: _m.get((i, j, k), 0)
                        if GENERATE_GT_T6:
                            _t6s = t6_support_set
                            _value_fns["t6"] = lambda i, j, k, _s=_t6s: 1 if (i, j, k) in _s else 0
                        if GENERATE_GT_T7:
                            _t7s = t7_support_column_set
                            _value_fns["t7"] = lambda i, j, k, _s=_t7s: 1 if (i, j, k) in _s else 0
                        if GENERATE_GT_T8:
                            _t8s = t8_set
                            _value_fns["t8"] = lambda i, j, k, _s=_t8s: 1 if (i, j, k) in _s else 0
                        if GENERATE_GT_T9:
                            _btc = block_to_cid
                            _value_fns["t9"] = lambda i, j, k, _m=_btc: _m.get((i, j, k), 0)
                        if GENERATE_GT_T10:
                            _t10s = t10_set
                            _value_fns["t10"] = lambda i, j, k, _s=_t10s: 1 if (i, j, k) in _s else 0
                        if GENERATE_GT_T11:
                            _t11s = t11_set
                            _value_fns["t11"] = lambda i, j, k, _s=_t11s: 1 if (i, j, k) in _s else 0
                    _combined = _build_masks_combined(stage, height_map, azimuth, _seg, _value_fns)
                    if GENERATE_GT_T2 and T2_TOP_ONLY and _seg is not None:
                        _save_gt_mask_png(_combined.get("t2"), _gt_file_path(view_dir, "task2"))
                    if GENERATE_GT_T3 and _seg is not None:
                        _save_gt_mask_png(_combined.get("t3"), _gt_file_path(view_dir, "task3"))
                    if GENERATE_GT_T4 and _seg is not None:
                        _save_gt_mask_png(_combined.get("t4"), _gt_file_path(view_dir, "task4"))
                    if (GENERATE_GT_T5 or _need_t5_mask) and _seg is not None:
                        _save_gt_mask_png(_combined.get("t5"), _gt_file_path(view_dir, "task5"))
                    if GENERATE_GT_T6 and _seg is not None:
                        _save_gt_mask_png(_combined.get("t6"), _gt_file_path(view_dir, "task6"))
                    if GENERATE_GT_T7 and _seg is not None:
                        _save_gt_mask_png(_combined.get("t7"), _gt_file_path(view_dir, "task7"))
                    if GENERATE_GT_T8 and _seg is not None:
                        _save_gt_mask_png(_combined.get("t8"), _gt_file_path(view_dir, "task8"))
                    if GENERATE_GT_T9 and _seg is not None:
                        _save_gt_mask_png(_combined.get("t9"), _gt_file_path(view_dir, "task9"))
                    if GENERATE_GT_T10 and _seg is not None:
                        _save_gt_mask_png(_combined.get("t10"), _gt_file_path(view_dir, "task10"))
                    if GENERATE_GT_T11 and _seg is not None:
                        _save_gt_mask_png(_combined.get("t11"), _gt_file_path(view_dir, "task11"))
                    if _need_t5_mask:
                        _save_visible_blocks(view_dir, visible_ordered)
                    if UPLOAD_TO_S3 and _mcq_scene_snap is not None:
                        _arr = _mcq_scene_snap
                        if _arr.dtype != np.uint8:
                            _arr = np.clip(_arr, 0, 255).astype(np.uint8)
                        _va = np.array(Image.fromarray(_arr).convert("RGBA"))
                        _pm = _combined.get("t5") if _combined is not None else None
                    else:
                        _va = _pm = None
                    restore_block_original_materials(stage)
                    _run_mcq_for_view(
                        stage,
                        height_map,
                        azimuth,
                        view_dir,
                        view_idx,
                        rgb_annotators[0],
                        visible_set,
                        block_view_info,
                        t2_sets,
                        views_azimuth,
                        seg_annotators[0] if seg_annotators else None,
                        visible_next=visible_next,
                        visible_opp=visible_opp,
                        t6=t6_support_set,
                        t7=t7_support_column_set,
                        dist_mult=dist_mult,
                        _view_arr=_va,
                        _pixel_map=_pm,
                    )
                    restore_block_original_materials(stage)
                    _flush_gt_json(view_dir)
            _wait_uploads()
            for a in rgb_annotators:
                a.detach()
            if seg_annotators:
                for a in seg_annotators:
                    if a is not None:
                        a.detach()
            for rp in render_products:
                rp.destroy()
        print(f"[BlockGen] Recreated scene from {SCENE_JSON_PATH} — views in {scene_dir}")
        return

    if USE_LEVELS_DRIVER:
        for level_name, min_blocks, shape_variants in LEVELS:
            if level_name == "level_5":
                num_to_generate = round(3 * SCENES_PER_LEVEL)
                sample_count = 0
                (l, w, max_h) = shape_variants[0]
                level_dir = os.path.join(OUTPUT_DIR, "level_5")
                for _ in range(num_to_generate):
                    height_map, final_blocks, target_blocks, num_visible = random_height_map_with_block_count(l, w, max_h, min_blocks)
                    sample_idx = sample_count
                    sample_count += 1
                    scene_dir = os.path.join(level_dir, f"sample_{sample_idx + 1:02d}")
                    _log_sample_start(scene_dir)
                    print(f"[BlockGen] level_5 sample_{sample_idx + 1:02d} target_blocks={target_blocks} -> {final_blocks}")
                    total_blocks, _ = build_block_scene(stage, height_map)
                    save_scene_json(scene_dir, height_map, level_name="level_5", sample_idx=sample_idx + 1, num_visible=num_visible)
                    for fov_idx, (fov_mult, dist_mult) in enumerate(zip(FOV_MULTIPLIERS, DIST_MULTIPLIERS)):
                        CAMERA_AZIMUTH_DEG = VIEW_AZIMUTHS_BASE[0] + OFFSETS[0]
                        cam_paths = create_usd_cameras(stage, height_map, fov_mult, dist_mult)
                        render_products, rgb_annotators, seg_annotators = setup_render(cam_paths)
                        for view_idx, _base_azimuth in enumerate(VIEW_AZIMUTHS_BASE):
                            for offset in OFFSETS:
                                azimuth = _base_azimuth + offset
                                views_azimuth = [b + offset for b in VIEW_AZIMUTHS_BASE]
                                CAMERA_AZIMUTH_DEG = azimuth
                                _set_camera_to_azimuth(stage, height_map, azimuth, dist_mult)
                                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                for _ in range(2):
                                    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                data = rgb_annotators[0].get_data()
                                if UPLOAD_TO_S3:
                                    _snap = _normalize_rgb_data(data)
                                    _mcq_scene_snap = _snap.copy() if _snap is not None else None
                                else:
                                    _mcq_scene_snap = None
                                view_dir = _view_fov_dir(scene_dir, view_idx, offset, fov_mult, dist_mult)
                                _update_gt_json(view_dir, {
                                    "object_type": BLOCK_SHAPE if BLOCK_SHAPE != "asset" else ASSET_BLOCK_MODE,
                                    "sample_idx": sample_idx + 1,
                                    "view_idx": view_idx + 1,
                                    "offset": offset,
                                    "fov": fov_mult,
                                    "dist": dist_mult,
                                })
                                fp = os.path.join(view_dir, "view.png")
                                if data is not None:
                                    save_rgb(data, fp)
                                    print(f"  -> {fp}")
                                else:
                                    print(f"  !! no data: {fp}")
                                if GENERATE_GT_T1:
                                    set_blocks_material(stage, "/World/Looks/BlueMaterial")
                                    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                    for _ in range(GT_EXTRA_RENDER_STEPS):
                                        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                    fp_gt = _gt_file_path(view_dir, "task1")
                                    if seg_annotators and seg_annotators[0] is not None:
                                        _save_gt_mask_png(_get_binary_segmentation_from_annotator(seg_annotators[0]), fp_gt)
                                    set_blocks_material(stage, "/World/Looks/WhiteMaterial")
                                _save_gt_text(view_dir, "task1", "blocks")
                                block_view_info = get_block_view_info(height_map, azimuth)
                                visible_set = get_visible_blocks_from_replicator(stage, height_map, azimuth, seg_annotators[0], dist_mult=dist_mult) if seg_annotators and seg_annotators[0] else set()
                                _vis_for_count = visible_set if visible_set else get_visible_blocks(height_map, azimuth)
                                total_blocks = len(_needed_from_visible(height_map, _vis_for_count))
                                _save_gt_text(view_dir, "task_main", total_blocks)
                                _view_non_visible = total_blocks - len(_vis_for_count)
                                if _view_non_visible <= 1:
                                    _view_level = "level_5"
                                elif _view_non_visible <= 3:
                                    _view_level = "level_6"
                                else:
                                    _view_level = "level_7"
                                _update_gt_json(view_dir, {"level": _view_level})
                                t2_sets = get_t2_extreme_block_sets(block_view_info, height_map, visible_set)
                                t8_set = get_t8_visible_supported_by_non_visible(visible_set)
                                _add_block_semantics(stage)  # restore semantic labels for next view's T1
                                _save_gt_text(view_dir, "task2", len(t2_sets["top"]))
                                h, w = len(height_map), len(height_map[0])
                                num_columns = sum(1 for i in range(h) for j in range(w) if height_map[i][j] > 0)
                                _save_gt_text(view_dir, "task3", num_columns)
                                num_layers = max((height_map[i][j] for i in range(h) for j in range(w)), default=0)
                                _save_gt_text(view_dir, "task4", num_layers)
                                _update_gt_json(view_dir, {
                                    "total_blocks": total_blocks,
                                    "num_hidden_blocks": total_blocks - len(_vis_for_count),
                                    "num_columns": num_columns,
                                    "num_layers": num_layers,
                                })
                                _save_gt_text(view_dir, "task5", len(visible_set))
                                t6_support_set = get_t6_support_blocks(height_map)
                                _save_gt_text(view_dir, "task6", len(t6_support_set & visible_set))
                                t7_support_column_set = get_t7_support_column_blocks(height_map)
                                _save_gt_text(view_dir, "task7", len(t7_support_column_set & visible_set))
                                _save_gt_text(view_dir, "task8", len(t8_set))
                                clusters = get_block_clusters(height_map)
                                _save_gt_text(view_dir, "task9", len(clusters))
                                _nv = len(views_azimuth)
                                if seg_annotators and seg_annotators[0]:
                                    visible_next = get_visible_blocks_from_replicator(stage, height_map, views_azimuth[(view_idx + 1) % _nv], seg_annotators[0], dist_mult=dist_mult, restore_azimuth_after=azimuth)
                                    t10_set = visible_set - visible_next
                                    visible_opp = get_visible_blocks_from_replicator(stage, height_map, views_azimuth[(view_idx + 2) % _nv], seg_annotators[0], dist_mult=dist_mult, restore_azimuth_after=azimuth)
                                    t11_set = visible_set - visible_opp
                                else:
                                    visible_next = set()
                                    visible_opp = set()
                                    t10_set = set()
                                    t11_set = set()
                                _save_gt_text(view_dir, "task10", len(t10_set))
                                _save_gt_text(view_dir, "task11", len(t11_set))
                                # --- Combined mask pass: single back-to-front loop for all tasks ---
                                _seg = seg_annotators[0] if seg_annotators else None
                                _need_t5_mask = (GENERATE_MCQ or GENERATE_GT_OVERLAY) and bool(visible_set) and _seg is not None
                                visible_ordered = sorted(visible_set)
                                block_to_label = {b: idx + 1 for idx, b in enumerate(visible_ordered)}
                                block_to_cid = {b: cid + 1 for cid, cl in enumerate(clusters) for b in cl}
                                _value_fns = {}
                                if _seg is not None:
                                    if GENERATE_GT_T2 and T2_TOP_ONLY:
                                        _t2s = frozenset(t2_sets["top"])
                                        _value_fns["t2"] = lambda i, j, k, _s=_t2s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T3:
                                        _w = w
                                        _value_fns["t3"] = lambda i, j, k, _w=_w: i * _w + j + 1
                                    if GENERATE_GT_T4:
                                        _value_fns["t4"] = lambda i, j, k: k + 1
                                    if GENERATE_GT_T5 or _need_t5_mask:
                                        _btl = block_to_label
                                        _value_fns["t5"] = lambda i, j, k, _m=_btl: _m.get((i, j, k), 0)
                                    if GENERATE_GT_T6:
                                        _t6s = t6_support_set
                                        _value_fns["t6"] = lambda i, j, k, _s=_t6s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T7:
                                        _t7s = t7_support_column_set
                                        _value_fns["t7"] = lambda i, j, k, _s=_t7s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T8:
                                        _t8s = t8_set
                                        _value_fns["t8"] = lambda i, j, k, _s=_t8s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T9:
                                        _btc = block_to_cid
                                        _value_fns["t9"] = lambda i, j, k, _m=_btc: _m.get((i, j, k), 0)
                                    if GENERATE_GT_T10:
                                        _t10s = t10_set
                                        _value_fns["t10"] = lambda i, j, k, _s=_t10s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T11:
                                        _t11s = t11_set
                                        _value_fns["t11"] = lambda i, j, k, _s=_t11s: 1 if (i, j, k) in _s else 0
                                _combined = _build_masks_combined(stage, height_map, azimuth, _seg, _value_fns)
                                if GENERATE_GT_T2 and T2_TOP_ONLY and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t2"), _gt_file_path(view_dir, "task2"))
                                if GENERATE_GT_T3 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t3"), _gt_file_path(view_dir, "task3"))
                                if GENERATE_GT_T4 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t4"), _gt_file_path(view_dir, "task4"))
                                if (GENERATE_GT_T5 or _need_t5_mask) and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t5"), _gt_file_path(view_dir, "task5"))
                                if GENERATE_GT_T6 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t6"), _gt_file_path(view_dir, "task6"))
                                if GENERATE_GT_T7 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t7"), _gt_file_path(view_dir, "task7"))
                                if GENERATE_GT_T8 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t8"), _gt_file_path(view_dir, "task8"))
                                if GENERATE_GT_T9 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t9"), _gt_file_path(view_dir, "task9"))
                                if GENERATE_GT_T10 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t10"), _gt_file_path(view_dir, "task10"))
                                if GENERATE_GT_T11 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t11"), _gt_file_path(view_dir, "task11"))
                                if _need_t5_mask:
                                    _save_visible_blocks(view_dir, visible_ordered)
                                if UPLOAD_TO_S3 and _mcq_scene_snap is not None:
                                    _arr = _mcq_scene_snap
                                    if _arr.dtype != np.uint8:
                                        _arr = np.clip(_arr, 0, 255).astype(np.uint8)
                                    _va = np.array(Image.fromarray(_arr).convert("RGBA"))
                                    _pm = _combined.get("t5") if _combined is not None else None
                                else:
                                    _va = _pm = None
                                restore_block_original_materials(stage)
                                _run_mcq_for_view(
                                    stage,
                                    height_map,
                                    azimuth,
                                    view_dir,
                                    view_idx,
                                    rgb_annotators[0],
                                    visible_set,
                                    block_view_info,
                                    t2_sets,
                                    views_azimuth,
                                    seg_annotators[0] if seg_annotators else None,
                                    visible_next=visible_next,
                                    visible_opp=visible_opp,
                                    t6=t6_support_set,
                                    t7=t7_support_column_set,
                                    dist_mult=dist_mult,
                                    _view_arr=_va,
                                    _pixel_map=_pm,
                                )
                                restore_block_original_materials(stage)
                                _flush_gt_json(view_dir)
                        _wait_uploads()
                        for a in rgb_annotators:
                            a.detach()
                        if seg_annotators:
                            for a in seg_annotators:
                                if a is not None:
                                    a.detach()
                        for rp in render_products:
                            rp.destroy()
                print(f"[BlockGen] level_5 done: {sample_count} samples (level per-view in gt.json)")
            else:
                configs = [shape_variants[i % len(shape_variants)] for i in range(round(SCENES_PER_LEVEL))]
                level_dir = os.path.join(OUTPUT_DIR, level_name)
                print(f"[BlockGen] Level {level_name} ({SCENES_PER_LEVEL} samples)")
                for sample_idx, (l, w, max_h) in enumerate(configs):
                    scene_dir = os.path.join(level_dir, f"sample_{sample_idx + 1:02d}")
                    _log_sample_start(scene_dir)
                    height_map, final_blocks, target_blocks, num_visible = random_height_map_with_block_count(l, w, max_h, min_blocks)
                    print(f"[BlockGen] {level_name} sample_{sample_idx + 1:02d} target_blocks={target_blocks} -> {final_blocks} (after view0 cull)")
                    total_blocks, _ = build_block_scene(stage, height_map)
                    save_scene_json(scene_dir, height_map, level_name=level_name, sample_idx=sample_idx + 1, num_visible=num_visible)
                    for fov_idx, (fov_mult, dist_mult) in enumerate(zip(FOV_MULTIPLIERS, DIST_MULTIPLIERS)):
                        CAMERA_AZIMUTH_DEG = VIEW_AZIMUTHS_BASE[0] + OFFSETS[0]
                        cam_paths = create_usd_cameras(stage, height_map, fov_mult, dist_mult)
                        render_products, rgb_annotators, seg_annotators = setup_render(cam_paths)
                        for view_idx, _base_azimuth in enumerate(VIEW_AZIMUTHS_BASE):
                            for offset in OFFSETS:
                                azimuth = _base_azimuth + offset
                                views_azimuth = [b + offset for b in VIEW_AZIMUTHS_BASE]
                                CAMERA_AZIMUTH_DEG = azimuth
                                _set_camera_to_azimuth(stage, height_map, azimuth, dist_mult)
                                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                for _ in range(2):
                                    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                data = rgb_annotators[0].get_data()
                                if UPLOAD_TO_S3:
                                    _snap = _normalize_rgb_data(data)
                                    _mcq_scene_snap = _snap.copy() if _snap is not None else None
                                else:
                                    _mcq_scene_snap = None
                                view_dir = _view_fov_dir(scene_dir, view_idx, offset, fov_mult, dist_mult)
                                _update_gt_json(view_dir, {
                                    "object_type": BLOCK_SHAPE if BLOCK_SHAPE != "asset" else ASSET_BLOCK_MODE,
                                    "sample_idx": sample_idx + 1,
                                    "view_idx": view_idx + 1,
                                    "offset": offset,
                                    "fov": fov_mult,
                                    "dist": dist_mult,
                                })
                                fp = os.path.join(view_dir, "view.png")
                                if data is not None:
                                    save_rgb(data, fp)
                                    print(f"  -> {fp}")
                                else:
                                    print(f"  !! no data: {fp}")
                                if GENERATE_GT_T1:
                                    set_blocks_material(stage, "/World/Looks/BlueMaterial")
                                    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                    for _ in range(GT_EXTRA_RENDER_STEPS):
                                        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                                    fp_gt = _gt_file_path(view_dir, "task1")
                                    if seg_annotators and seg_annotators[0] is not None:
                                        _save_gt_mask_png(_get_binary_segmentation_from_annotator(seg_annotators[0]), fp_gt)
                                    set_blocks_material(stage, "/World/Looks/WhiteMaterial")
                                _save_gt_text(view_dir, "task1", "blocks")
                                block_view_info = get_block_view_info(height_map, azimuth)
                                visible_set = get_visible_blocks_from_replicator(stage, height_map, azimuth, seg_annotators[0], dist_mult=dist_mult) if seg_annotators and seg_annotators[0] else set()
                                _vis_for_count = visible_set if visible_set else get_visible_blocks(height_map, azimuth)
                                total_blocks = len(_needed_from_visible(height_map, _vis_for_count))
                                _save_gt_text(view_dir, "task_main", total_blocks)
                                t2_sets = get_t2_extreme_block_sets(block_view_info, height_map, visible_set)
                                t8_set = get_t8_visible_supported_by_non_visible(visible_set)
                                _add_block_semantics(stage)  # restore semantic labels for next view's T1
                                _save_gt_text(view_dir, "task2", len(t2_sets["top"]))
                                h, w = len(height_map), len(height_map[0])
                                num_columns = sum(1 for i in range(h) for j in range(w) if height_map[i][j] > 0)
                                _save_gt_text(view_dir, "task3", num_columns)
                                num_layers = max((height_map[i][j] for i in range(h) for j in range(w)), default=0)
                                _save_gt_text(view_dir, "task4", num_layers)
                                _update_gt_json(view_dir, {
                                    "total_blocks": total_blocks,
                                    "num_hidden_blocks": total_blocks - len(_vis_for_count),
                                    "num_columns": num_columns,
                                    "num_layers": num_layers,
                                })
                                _save_gt_text(view_dir, "task5", len(visible_set))
                                t6_support_set = get_t6_support_blocks(height_map)
                                _save_gt_text(view_dir, "task6", len(t6_support_set & visible_set))
                                t7_support_column_set = get_t7_support_column_blocks(height_map)
                                _save_gt_text(view_dir, "task7", len(t7_support_column_set & visible_set))
                                _save_gt_text(view_dir, "task8", len(t8_set))
                                clusters = get_block_clusters(height_map)
                                _save_gt_text(view_dir, "task9", len(clusters))
                                _nv = len(views_azimuth)
                                if seg_annotators and seg_annotators[0]:
                                    visible_next = get_visible_blocks_from_replicator(stage, height_map, views_azimuth[(view_idx + 1) % _nv], seg_annotators[0], dist_mult=dist_mult, restore_azimuth_after=azimuth)
                                    t10_set = visible_set - visible_next
                                    visible_opp = get_visible_blocks_from_replicator(stage, height_map, views_azimuth[(view_idx + 2) % _nv], seg_annotators[0], dist_mult=dist_mult, restore_azimuth_after=azimuth)
                                    t11_set = visible_set - visible_opp
                                else:
                                    visible_next = set()
                                    visible_opp = set()
                                    t10_set = set()
                                    t11_set = set()
                                _save_gt_text(view_dir, "task10", len(t10_set))
                                _save_gt_text(view_dir, "task11", len(t11_set))
                                # --- Combined mask pass: single back-to-front loop for all tasks ---
                                _seg = seg_annotators[0] if seg_annotators else None
                                _need_t5_mask = (GENERATE_MCQ or GENERATE_GT_OVERLAY) and bool(visible_set) and _seg is not None
                                visible_ordered = sorted(visible_set)
                                block_to_label = {b: idx + 1 for idx, b in enumerate(visible_ordered)}
                                block_to_cid = {b: cid + 1 for cid, cl in enumerate(clusters) for b in cl}
                                _value_fns = {}
                                if _seg is not None:
                                    if GENERATE_GT_T2 and T2_TOP_ONLY:
                                        _t2s = frozenset(t2_sets["top"])
                                        _value_fns["t2"] = lambda i, j, k, _s=_t2s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T3:
                                        _w = w
                                        _value_fns["t3"] = lambda i, j, k, _w=_w: i * _w + j + 1
                                    if GENERATE_GT_T4:
                                        _value_fns["t4"] = lambda i, j, k: k + 1
                                    if GENERATE_GT_T5 or _need_t5_mask:
                                        _btl = block_to_label
                                        _value_fns["t5"] = lambda i, j, k, _m=_btl: _m.get((i, j, k), 0)
                                    if GENERATE_GT_T6:
                                        _t6s = t6_support_set
                                        _value_fns["t6"] = lambda i, j, k, _s=_t6s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T7:
                                        _t7s = t7_support_column_set
                                        _value_fns["t7"] = lambda i, j, k, _s=_t7s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T8:
                                        _t8s = t8_set
                                        _value_fns["t8"] = lambda i, j, k, _s=_t8s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T9:
                                        _btc = block_to_cid
                                        _value_fns["t9"] = lambda i, j, k, _m=_btc: _m.get((i, j, k), 0)
                                    if GENERATE_GT_T10:
                                        _t10s = t10_set
                                        _value_fns["t10"] = lambda i, j, k, _s=_t10s: 1 if (i, j, k) in _s else 0
                                    if GENERATE_GT_T11:
                                        _t11s = t11_set
                                        _value_fns["t11"] = lambda i, j, k, _s=_t11s: 1 if (i, j, k) in _s else 0
                                _combined = _build_masks_combined(stage, height_map, azimuth, _seg, _value_fns)
                                if GENERATE_GT_T2 and T2_TOP_ONLY and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t2"), _gt_file_path(view_dir, "task2"))
                                if GENERATE_GT_T3 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t3"), _gt_file_path(view_dir, "task3"))
                                if GENERATE_GT_T4 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t4"), _gt_file_path(view_dir, "task4"))
                                if (GENERATE_GT_T5 or _need_t5_mask) and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t5"), _gt_file_path(view_dir, "task5"))
                                if GENERATE_GT_T6 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t6"), _gt_file_path(view_dir, "task6"))
                                if GENERATE_GT_T7 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t7"), _gt_file_path(view_dir, "task7"))
                                if GENERATE_GT_T8 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t8"), _gt_file_path(view_dir, "task8"))
                                if GENERATE_GT_T9 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t9"), _gt_file_path(view_dir, "task9"))
                                if GENERATE_GT_T10 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t10"), _gt_file_path(view_dir, "task10"))
                                if GENERATE_GT_T11 and _seg is not None:
                                    _save_gt_mask_png(_combined.get("t11"), _gt_file_path(view_dir, "task11"))
                                if _need_t5_mask:
                                    _save_visible_blocks(view_dir, visible_ordered)
                                if UPLOAD_TO_S3 and _mcq_scene_snap is not None:
                                    _arr = _mcq_scene_snap
                                    if _arr.dtype != np.uint8:
                                        _arr = np.clip(_arr, 0, 255).astype(np.uint8)
                                    _va = np.array(Image.fromarray(_arr).convert("RGBA"))
                                    _pm = _combined.get("t5") if _combined is not None else None
                                else:
                                    _va = _pm = None
                                restore_block_original_materials(stage)
                                _run_mcq_for_view(
                                    stage,
                                    height_map,
                                    azimuth,
                                    view_dir,
                                    view_idx,
                                    rgb_annotators[0],
                                    visible_set,
                                    block_view_info,
                                    t2_sets,
                                    views_azimuth,
                                    seg_annotators[0] if seg_annotators else None,
                                    visible_next=visible_next,
                                    visible_opp=visible_opp,
                                    t6=t6_support_set,
                                    t7=t7_support_column_set,
                                    dist_mult=dist_mult,
                                    _view_arr=_va,
                                    _pixel_map=_pm,
                                )
                                restore_block_original_materials(stage)
                                _flush_gt_json(view_dir)
                        _wait_uploads()
                        for a in rgb_annotators:
                            a.detach()
                        if seg_annotators:
                            for a in seg_annotators:
                                if a is not None:
                                    a.detach()
                        for rp in render_products:
                            rp.destroy()
        print(f"[BlockGen] Done — images in {OUTPUT_DIR}")
        return

    # Single-scene mode (original behavior)
    total_blocks, height_map = build_block_scene(stage)
    cam_paths = create_usd_cameras(stage, height_map)
    render_products, rgb_annotators, seg_annotators = setup_render(cam_paths)

    for i in range(NUM_FRAMES):
        print(f"[BlockGen] Capturing frame {i} ...")
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)

        data = rgb_annotators[0].get_data()
        if data is not None:
            fp = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
            save_rgb(data, fp)
            print(f"  -> {fp}")
        else:
            print(f"  !! cam_perspective: no data")

    rep.orchestrator.wait_until_complete()
    for a in rgb_annotators:
        a.detach()
    if seg_annotators:
        for a in seg_annotators:
            if a is not None:
                a.detach()
    for rp in render_products:
        rp.destroy()

    meta = {
        "total_blocks": total_blocks,
        "grid": [H, W],
        "mode": MODE,
        "max_height": MAX_HEIGHT if MODE == "random" else max(cell for row in height_map for cell in row),
        "height_map": height_map,
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    t0 = time.perf_counter()
    _s3_upload(json.dumps(meta, indent=2).encode("utf-8"), meta_path)
    print(f"[BlockGen] Done — {total_blocks} blocks, images in {OUTPUT_DIR}")
    _record_file_timing("metadata_json", meta_path, t0)


# =============================================================================
# DISPATCH
# =============================================================================
run_sync()
_print_fn_timing_summary()
if _standalone:
    simulation_app.close()
