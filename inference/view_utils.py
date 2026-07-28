# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for discovering view.png files and grouping by sample.

All inference scripts import from here so the fast-path logic and caching
are maintained in one place.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


def _find_view_dirs(root: str) -> List[str]:
    """Find directories containing view.png under *root*.

    Uses a fast shallow scan (depth <= 1) before falling back to a full
    recursive walk.  The shallow path minimises ``stat`` calls on networked
    filesystems like Lustre.
    """
    root = os.path.abspath(root)

    if os.path.isfile(os.path.join(root, "view.png")):
        return [root]

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

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if "view.png" in filenames:
            dirs.append(dirpath)
    return sorted(dirs)


def find_and_group_views(root: str) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """Find all directories containing view.png, group by sample.

    Returns ``[(sample_dir, [(view_idx, view_png_path), ...]), ...]``, sorted.

    Directory layout: ``sample_NN/view_NN/offset_X_fov_Y_dist_Z/view.png``
    """
    by_sample: dict = {}
    for dirpath in _find_view_dirs(os.path.abspath(root)):
        view_dir = os.path.dirname(dirpath)
        m = re.search(r"(\d+)$", os.path.basename(view_dir))
        idx = int(m.group(1)) if m else 0
        sample_dir = os.path.dirname(view_dir)
        by_sample.setdefault(sample_dir, []).append(
            (idx, os.path.join(dirpath, "view.png"))
        )
    return [(p, sorted(v)) for p, v in sorted(by_sample.items())]


def resolve_folders(path: str) -> List[str]:
    """If *path* is a .txt file, read folder paths from it.  Otherwise treat as single folder."""
    path = os.path.abspath(path)
    if path.endswith(".txt") and os.path.isfile(path):
        with open(path) as f:
            folders = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"Read {len(folders)} folder(s) from {path}")
        return folders
    if not os.path.isdir(path):
        raise SystemExit(f"Not a directory or .txt file: {path}")
    return [path]


_DATASET_MARKER = "/dataset/"


def _normalize_path(p: str) -> str:
    """Strip the mount-specific prefix, keeping only the stable ``dataset/...`` suffix."""
    idx = p.find(_DATASET_MARKER)
    return p[idx + 1:] if idx >= 0 else p


def _detect_prefix(p: str) -> str:
    """Return everything up to and including the mount root before ``dataset/``."""
    idx = p.find(_DATASET_MARKER)
    return p[:idx + 1] if idx >= 0 else ""


def _rewrite_prefix(cached_path: str, new_prefix: str, old_prefix: str) -> str:
    """Swap the mount prefix in a cached path to match the current environment."""
    if old_prefix and cached_path.startswith(old_prefix):
        return new_prefix + cached_path[len(old_prefix):]
    return cached_path


def scan_folders_cached(
    input_path: str,
    folders: List[str],
    cache_dir: str | None = None,
) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """Scan *folders* for view.png, caching results to a JSON file.

    The cache hash is computed from **normalized** paths (stripped to ``dataset/...``)
    so the same view set always produces the same hash regardless of mount prefix.

    *cache_dir* controls where the cache file is stored.  Resolution order:
    1. Explicit *cache_dir* argument
    2. ``SPATIALIQ_CACHE_DIR`` environment variable
    3. Directory containing *input_path*

    On cache load, stored paths are re-prefixed to match the current mount so
    the cache works across jobs with different mount points.
    """
    input_path = os.path.abspath(input_path)

    # Resolve cache directory
    if cache_dir is None:
        cache_dir = os.environ.get("SPATIALIQ_CACHE_DIR", "").strip() or None
    if cache_dir is None:
        cache_dir = os.path.dirname(input_path)
    cache_dir = os.path.abspath(cache_dir)

    cache_path = None
    if input_path.endswith(".txt"):
        normalized = [_normalize_path(f) for f in folders]
        content_hash = hashlib.md5("\n".join(normalized).encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"viewcache_{content_hash}.json")

        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                raw = json.load(f)
            samples = [(s["label"], [(v[0], v[1]) for v in s["views"]]) for s in raw]

            # Rewrite cached prefixes to match the current mount
            if samples and folders:
                cur_prefix = _detect_prefix(folders[0])
                first_label = samples[0][0]
                old_prefix = _detect_prefix(first_label)
                if cur_prefix and old_prefix and cur_prefix != old_prefix:
                    samples = [
                        (
                            _rewrite_prefix(label, cur_prefix, old_prefix),
                            [(idx, _rewrite_prefix(vp, cur_prefix, old_prefix)) for idx, vp in views],
                        )
                        for label, views in samples
                    ]
                    print(f"Loaded {len(samples)} sample(s) from cache (prefix-rewritten): {cache_path}")
                else:
                    print(f"Loaded {len(samples)} sample(s) from cache: {cache_path}")
            else:
                print(f"Loaded {len(samples)} sample(s) from cache: {cache_path}")
            return samples

    samples: list = []
    for _folder in tqdm(folders, desc="Scanning folders", unit="folder"):
        samples.extend(find_and_group_views(_folder))

    if cache_path is not None and samples:
        raw = [{"label": label, "views": views} for label, views in samples]
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(raw, f)
            print(f"Cached {len(samples)} sample(s) to: {cache_path}")
        except OSError:
            pass

    return samples
