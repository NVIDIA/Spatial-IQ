# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
summarize_gt_stats.py
Summarizes statistics across all gt.json files under a given folder.

Usage:
    python summarize_gt_stats.py <input_folder>

Fields summarized (only from files that contain them):
    total_blocks, num_hidden_blocks, num_columns, num_layers

Scene-level fields (total_blocks, num_columns, num_layers) are the same across
all views/fovs of a single sample. To avoid inflating counts, the script
deduplicates by sample directory (the grandparent of each gt.json, e.g.
.../level_1/sample_01/).
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont


FIELDS = ["total_blocks", "num_hidden_blocks", "num_columns", "num_layers"]
# Fields that are scene-level (deduplicated per sample); the rest are view-level.
SCENE_LEVEL_FIELDS = {"total_blocks", "num_columns", "num_layers"}


TILE_SIZE = 128   # each view.png is resized to this before tiling
TILE_LABEL_H = 14  # pixels reserved below each tile for a small label
TILE_PAD = 2       # gap between tiles


def find_view_pngs(root):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname == "view.png":
                yield os.path.join(dirpath, fname)



def build_tile_grid(view_paths):
    """
    Returns a 2-D list (list of rows, each row is a list of paths).
    Rows = unique sample directories; columns = views+fovs within that sample.
    """
    by_sample = defaultdict(list)
    for path in sorted(view_paths):
        sample = sample_dir_for(path)
        by_sample[sample].append(path)
    # Sort samples, then within each sample paths are already sorted by os.walk order
    rows = [by_sample[s] for s in sorted(by_sample)]
    return rows


def make_tile_label(path):
    """Short label: 'view_02 / fov3_dist0.3'."""
    parts = path.replace("\\", "/").split("/")
    view = next((p for p in parts if p.startswith("view_")), "")
    # fov dir is the immediate parent of view.png
    fov = parts[-2] if len(parts) >= 2 else ""
    if fov == view:
        return view
    return f"{view}/{fov}"


ROW_LABEL_W = 60  # pixels reserved on the left for the sample label


def save_tile_grid(rows, out_path):
    if not rows or not any(rows):
        print("  [tile] no view.png files found -- skipping tile image")
        return

    cols = max(len(row) for row in rows)
    cell_w = TILE_SIZE + TILE_PAD
    cell_h = TILE_SIZE + TILE_LABEL_H + TILE_PAD

    img_w = ROW_LABEL_W + cols * cell_w + TILE_PAD
    img_h = len(rows) * cell_h + TILE_PAD

    canvas = Image.new("RGB", (img_w, img_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 9)
    except Exception:
        font = ImageFont.load_default()

    for r, row in enumerate(rows):
        # Sample label on the left, vertically centred in the row
        sample_name = os.path.basename(sample_dir_for(row[0])) if row else ""
        label_y = TILE_PAD + r * cell_h + TILE_SIZE // 2
        draw.text((TILE_PAD, label_y), sample_name, fill=(240, 200, 80), font=font)

        for c, path in enumerate(row):
            x = ROW_LABEL_W + c * cell_w
            y = TILE_PAD + r * cell_h
            try:
                img = Image.open(path).convert("RGB").resize(
                    (TILE_SIZE, TILE_SIZE), Image.LANCZOS
                )
                canvas.paste(img, (x, y))
            except Exception as e:
                draw.rectangle([x, y, x + TILE_SIZE, y + TILE_SIZE], fill=(180, 40, 40))
                draw.text((x + 2, y + 2), str(e)[:20], fill=(255, 255, 255), font=font)

            label = make_tile_label(path)
            draw.text((x, y + TILE_SIZE + 1), label, fill=(200, 200, 200), font=font)

    canvas.save(out_path)
    print(f"  [tile] saved {len(rows)} rows x {cols} cols -> {out_path}")


def find_gt_jsons(root):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname == "gt.json":
                yield os.path.join(dirpath, fname)


def sample_dir_for(path):
    """Return the sample directory (two levels above gt.json's immediate parent view/fov dir)."""
    # gt.json lives in .../sample_XX/view_XX/[fov_dir/]gt.json
    # Walk up until we find a directory whose name starts with "sample"
    d = os.path.dirname(path)
    for _ in range(6):
        if os.path.basename(d).startswith("sample"):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(path)


def stats(values):
    n = len(values)
    if n == 0:
        return {"count": 0}
    mn = min(values)
    mx = max(values)
    mean = sum(values) / n
    sorted_v = sorted(values)
    mid = n // 2
    median = sorted_v[mid] if n % 2 else (sorted_v[mid - 1] + sorted_v[mid]) / 2
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return {"count": n, "min": mn, "max": mx, "mean": mean, "median": median, "std": std}


def histogram(values, bins=10):
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mn == mx:
        return [(mn, mx, len(values))]
    width = (mx - mn) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - mn) / width), bins - 1)
        counts[idx] += 1
    return [(mn + i * width, mn + (i + 1) * width, counts[i]) for i in range(bins)]


def bar(count, total, width=30):
    filled = round(count / total * width) if total else 0
    return "#" * filled + "." * (width - filled)


def print_field_summary(name, values):
    s = stats(values)
    print(f"\n  {name}  (n={s['count']})")
    if s["count"] == 0:
        print("    no data")
        return
    print(f"    min={s['min']}  max={s['max']}  mean={s['mean']:.2f}  median={s['median']:.1f}  std={s['std']:.2f}")

    # Frequency distribution (integer values: use Counter; float: use histogram)
    all_int = all(isinstance(v, int) for v in values)
    if all_int and s["max"] - s["min"] <= 40:
        freq = Counter(values)
        total = len(values)
        print(f"    {'value':>6}  {'count':>6}  {'%':>6}  distribution")
        for v in range(int(s["min"]), int(s["max"]) + 1):
            c = freq.get(v, 0)
            pct = 100 * c / total
            print(f"    {v:>6}  {c:>6}  {pct:>5.1f}%  {bar(c, total)}")
    else:
        bins = histogram(values)
        total = len(values)
        print(f"    {'range':>20}  {'count':>6}  {'%':>6}  distribution")
        for lo, hi, c in bins:
            pct = 100 * c / total
            label = f"[{lo:.1f}, {hi:.1f})"
            print(f"    {label:>20}  {c:>6}  {pct:>5.1f}%  {bar(c, total)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_gt_stats.py <input_folder>")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Error: {root!r} is not a directory")
        sys.exit(1)

    all_files = list(find_gt_jsons(root))
    print(f"Found {len(all_files)} gt.json files under {root}")

    # Collect raw records
    records = []       # all gt.json entries that have at least one target field
    skipped = 0
    for path in all_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [warn] could not read {path}: {e}")
            skipped += 1
            continue
        row = {field: data[field] for field in FIELDS if field in data}
        if row:
            row["_path"] = path
            row["_sample"] = sample_dir_for(path)
            records.append(row)
        else:
            skipped += 1

    print(f"  {len(records)} files have at least one target field  ({skipped} skipped)")

    # Per-view stats (all records)
    view_data = {f: [r[f] for r in records if f in r] for f in FIELDS}

    # Deduplicated scene-level stats (one entry per sample dir, first occurrence wins)
    seen_samples = set()
    scene_records = []
    for r in records:
        if r["_sample"] not in seen_samples:
            seen_samples.add(r["_sample"])
            scene_records.append(r)
    print(f"  {len(seen_samples)} unique sample directories")

    scene_data = {f: [r[f] for r in scene_records if f in r] for f in SCENE_LEVEL_FIELDS}

    print("\n" + "=" * 60)
    print("SCENE-LEVEL STATISTICS  (deduplicated per sample)")
    print("=" * 60)
    for field in ["total_blocks", "num_columns", "num_layers"]:
        print_field_summary(field, scene_data[field])

    print("\n" + "=" * 60)
    print("VIEW-LEVEL STATISTICS  (all views / all fov configs)")
    print("=" * 60)
    print_field_summary("num_hidden_blocks", view_data["num_hidden_blocks"])

    print()

    # ── Tile all view.png files ───────────────────────────────────────────────
    view_paths = list(find_view_pngs(root))
    print(f"Found {len(view_paths)} view.png files")
    rows = build_tile_grid(view_paths)
    tile_out = os.path.join(root, "views_tile.png")
    save_tile_grid(rows, tile_out)


if __name__ == "__main__":
    main()
