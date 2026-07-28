<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# blocks_render_image.py

Isaac Sim 4.5 dataset generator for the SpatialIQ benchmark. Builds randomized 3D block structures, renders them from multiple camera angles, and outputs annotated images with ground-truth masks, answers, and multiple-choice questions.

---

## Prerequisites

- [NVIDIA Isaac Sim 4.5](https://developer.nvidia.com/isaac-sim)
- `boto3` + `botocore` (only when `UPLOAD_TO_S3=1`): install into Isaac Sim's Python environment

---

## How to run

### Option A — Script Editor (GUI)

Paste the entire file into the Isaac Sim Script Editor and click **Run**.

### Option B — CLI / standalone

```bat
# Windows
cd %LOCALAPPDATA%\ov\pkg\isaac-sim-4.5.0
python.bat C:\path\to\blocks_render_image.py [--headless]
```

```bash
# Linux
cd ~/.local/share/ov/pkg/isaac-sim-4.5.0
./python.sh /path/to/blocks_render_image.py [--headless]
```

`--headless` can also be set via the environment variable `ISAAC_HEADLESS=1`.

### Environment variables

| Variable | Effect |
|---|---|
| `ISAAC_HEADLESS` | `1` / `true` / `yes` → headless mode |
| `BLOCK_RENDER_RTX_PATH_TRACING` | `1` → enable RTX Interactive Path Tracing |

---

## Key configuration

All settings are at the top of the file.

### Paths

| Variable | Description |
|---|---|
| `_OUTPUT_BASE` | Root output directory on disk |
| `DATASET_SPLIT` | Subdirectory name (e.g. `eval_set`) |
| `OUTPUT_DIR` | Full output path: `_OUTPUT_BASE/DATASET_SPLIT/data_YYYYMMDD_HHMMSS/` |
| `BLOCK_TEXTURES_PATH` | Directory of texture images for cube blocks (optional — falls back to procedural textures when empty; see [Block textures](#block-textures)) |
| `ENVIRONMENT_USD_OVERRIDE` | Path to a `.usd` background environment, or a directory of environments (one chosen randomly per sample), or `None` to use the Isaac public CDN |

### Block textures

Block textures are **optional**. If `BLOCK_TEXTURES_PATH` is unset or points to
an empty/nonexistent directory, the renderer falls back to a procedural
per-block texture and everything still works.

To use textured blocks, drop any square-ish image files (`.jpg`, `.png`, or
`.webp`) into the directory pointed to by `BLOCK_TEXTURES_PATH` (default:
`data_generation/textures_cube/`). One image is chosen at random per sample.
The generator is texture-agnostic — bring your own textures under whatever
license you have the right to use. Freely redistributable options such as
[Poly Haven](https://polyhaven.com/textures) and
[ambientCG](https://ambientcg.com/) work well.

### Optional S3 upload

Outputs are written locally by default. To instead upload to an S3-compatible
object store, set `UPLOAD_TO_S3=1` and provide the endpoint and credentials via
environment variables — nothing is hardcoded in the script.

| Environment variable | Description |
|---|---|
| `UPLOAD_TO_S3` | `1`/`true` → upload to S3; unset/`0` → write locally (default) |
| `S3_ENDPOINT` | S3-compatible endpoint URL |
| `S3_BUCKET` | Bucket name (default `dataset`) |
| `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | Credentials |
| `S3_REGION` | Region string (default `us-east-1`) |

When `UPLOAD_TO_S3=1`, S3 keys are derived from the local path relative to `_OUTPUT_BASE`. `sample_timing.txt` is always written locally regardless of this flag.

### Scene layout

| Variable | Default | Description |
|---|---|---|
| `LEVELS` | `[("level_1", 3, [(4,4,4)])]` | List of `(name, min_blocks, shape_variants)`. Each variant is `(rows, cols, max_stack)` |
| `SCENES_PER_LEVEL` | `100000` | How many scenes to generate per level |
| `BLOCK_SHAPE` | `"cube"` | `"cube"`, `"factory_box"`, or `"asset"` |
| `ASSET_BLOCK_MODE` | `"soup_can"` | Asset type when `BLOCK_SHAPE = "asset"` (`"soup_can"`, `"mug"`, `"spam"`, `"bowl"`) |

### Camera / rendering

| Variable | Default | Description |
|---|---|---|
| `NUM_VIEWS` | `4` | Number of base azimuths (derived from `VIEW_AZIMUTHS_BASE`) |
| `VIEW_AZIMUTHS_BASE` | `[0, 90, 180, 270]` | Base camera azimuths in degrees |
| `OFFSETS` | `[3.0, 12.5]` | Per-offset render passes added to each base azimuth |
| `FOV_MULTIPLIERS` | `[1.0, 3.0]` | Lens FOV multipliers (paired with `DIST_MULTIPLIERS`) |
| `DIST_MULTIPLIERS` | `[1.0, 0.3]` | Camera distance multipliers (same length as `FOV_MULTIPLIERS`) |
| `IMAGE_RES` | `(512, 512)` | Render resolution |
| `RT_SUBFRAMES` | `32` | RTX subframes per render step |
| `CAMERA_MARGIN` | `1.75` | Padding around blocks |
| `CAMERA_Z_MULTIPLIER` | `1.1` | Vertical height of camera |

### Ground truth

| Variable | Default | Description |
|---|---|---|
| `GENERATE_GT_T1`–`T11` | `True` | Enable/disable each GT task mask |
| `T2_TOP_ONLY` | `True` | `True` = single topmost mask; `False` = six directional masks |
| `GENERATE_MCQ` | `True` | Generate 5-choice MCQ images per task |
| `GT_EXTRA_RENDER_STEPS` | `3` | Extra RTX steps after material swap before capture |

### Diagnostics

| Variable | Default | Description |
|---|---|---|
| `PROFILE_FN_TIMINGS` | `True` | Print wall-clock timing table per function after each run |
| `RECORD_FILE_WRITE_TIMINGS` | `True` | Print per-file write duration to stdout |

---

## Output file structure

```
<OUTPUT_DIR>/                            # e.g. data/eval_set/data_20260406_120000/
  sample_timing.txt                      # LOCAL ONLY — ISO timestamp per sample start

  level_1/                               # one folder per entry in LEVELS
    sample_01/                           # 1-indexed
      scene.json                         # scene structure and GT counts
      view_01/                           # one per base azimuth (1-indexed)
        offset3_fov1_dist1/              # one per (offset × fov × dist) combination
          view.png                       # 512×512 RGBA rendered scene image
          gt.json                        # all GT answers for this view+fov+dist
          task1_gt_mask.png              # T1: blocks vs background (binary)
          task2_gt_mask.png              # T2: topmost block(s)
          task3_gt_mask.png              # T3: per-column integer label
          task4_gt_mask.png              # T4: per-layer integer label
          task5_gt_mask.png              # T5: per-visible-block integer label
          task6_gt_mask.png              # T6: blocks directly supporting topmost
          task7_gt_mask.png              # T7: full support column(s) of topmost
          task8_gt_mask.png              # T8: visible blocks supported by non-visible
          task9_gt_mask.png              # T9: per-cluster integer label
          task10_gt_mask.png             # T10: visible here but not in next view
          task11_gt_mask.png             # T11: visible here but not in opposite view
          task1_mcq_choiceA.png          # MCQ choice images A–E per task
          task1_mcq_choiceB.png          #   (correct answer recorded in gt.json)
          task1_mcq_choiceC.png
          task1_mcq_choiceD.png
          task1_mcq_choiceE.png
          task2_mcq_choice{A-E}.png
          ...
          task11_mcq_choice{A-E}.png
          task_main_mcq_choice{A-E}.png  # MCQ for total block count
        offset12.5_fov1_dist1/           # second azimuth offset pass
          ...
        offset3_fov3_dist0.3/            # second (fov, dist) pair
          ...
        offset12.5_fov3_dist0.3/
          ...
      view_02/                           # azimuth base 90°
        ...
      view_03/                           # azimuth base 180°
        ...
      view_04/                           # azimuth base 270°
        ...
    sample_02/
      ...
```

With the default config (`OFFSETS = [3.0, 12.5]`, `FOV_MULTIPLIERS = [1.0, 3.0]`, `VIEW_AZIMUTHS_BASE = [0, 90, 180, 270]`) each sample has **4 views × 4 subfolders = 16 `view_dir` folders**, each containing `view.png`, `gt.json`, 11 GT masks, and up to 60 MCQ images.

### scene.json

```json
{
  "height_map": [[3, 2, 0, 1], [0, 2, 1, 0], ...],
  "rows": 4,
  "cols": 4,
  "total_blocks": 17,
  "visible_blocks": 10,
  "non_visible_blocks": 7,
  "level": "level_1",
  "sample_idx": 1
}
```

### gt.json

One file per `view_dir` (uploaded once after all GT passes for that view).

```json
{
  "object_type": "cube",
  "view_idx": 1,
  "offset": 3.0,
  "fov": 1.0,
  "dist": 1.0,
  "task1": "blocks",
  "task_main": 12,
  "task2": 2,
  "task3": 5,
  "task4": 3,
  "task5": 7,
  "task6": 2,
  "task7": 4,
  "task8": 1,
  "task9": 3,
  "task10": 2,
  "task11": 1,
  "total_blocks": 12,
  "num_hidden_blocks": 5,
  "num_columns": 5,
  "num_layers": 3,
  "num_clusters": 3,
  "visible_blocks": [[0, 0, 0], [0, 1, 0], ...],
  "mcq": {
    "task1":      {"correct_letter": "C", "choices": {"A": "...", "B": "...", ...}},
    "task_main":  {"correct_letter": "A", "choices": {...}},
    "task2":      {"correct_letter": "D", "choices": {...}},
    ...
  }
}
```

### sample_timing.txt

Written locally to `OUTPUT_DIR/sample_timing.txt` regardless of `UPLOAD_TO_S3`. One line per sample:

```
2026-04-06T12:00:01.234567  output/eval_set/data_20260406_120000/level_1/sample_01
2026-04-06T12:02:45.891234  output/eval_set/data_20260406_120000/level_1/sample_02
```

---

## Task definitions

| Task | GT mask type | Answer in gt.json | Description |
|---|---|---|---|
| `task_main` | — | integer | Total block count (gravity rule: infer hidden if needed to support visible) |
| `task1` | binary | `"blocks"` | Blocks vs background |
| `task2` | binary | integer | Topmost block(s) count |
| `task3` | per-column label | integer | Number of non-empty columns |
| `task4` | per-layer label | integer | Number of layers (max stack height) |
| `task5` | per-block label | integer | Number of visible blocks |
| `task6` | binary | integer | Blocks directly supporting the topmost |
| `task7` | binary | integer | Full support column(s) of the topmost, excluding topmost |
| `task8` | binary | integer | Visible blocks that sit directly on a non-visible block |
| `task9` | per-cluster label | integer | Number of connected clusters |
| `task10` | binary | integer | Blocks visible in this view but not in the next (clockwise) view |
| `task11` | binary | integer | Blocks visible in this view but not in the opposite (180°) view |

---

## Re-rendering a saved scene

Set `SCENE_JSON_PATH` to the path of an existing `scene.json` to skip scene generation and re-render that exact configuration into its original folder:

```python
SCENE_JSON_PATH = "/path/to/output/eval_set/data_20260406_120000/level_1/sample_01/scene.json"
```

This overrides `USE_LEVELS_DRIVER`.

---

## Performance notes

- Uploads to S3 use a background thread pool (`ThreadPoolExecutor(max_workers=8)`). Each S3 client is thread-local to avoid contention. `_wait_uploads()` is called once per sample to drain the queue before destroying render products.
- `gt.json` writes are batched in memory (`_GT_JSON_CACHE`) and uploaded once per `view_dir` via `_flush_gt_json()`, reducing upload calls by ~97% vs. per-field writes.
- Set `PROFILE_FN_TIMINGS = True` to print a per-function wall-clock table at the end of each run, useful for identifying bottlenecks in rendering vs. upload vs. compositing.
