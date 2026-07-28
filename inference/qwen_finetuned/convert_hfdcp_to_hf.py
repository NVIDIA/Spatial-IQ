#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert NeMo-RL HF-shard DCP checkpoints to standard HF safetensors layout.

Input layout (NeMo-RL):
    <src>/
      shard-NNNNN-model-00001-of-00001.safetensors   (1..N)
      .hf_metadata/
        config.json
        generation_config.json
        fqn_to_file_index_mapping.json   (FQN -> 1-indexed shard idx; not load-bearing)

Sharding: every tensor is FSDP-row-sharded across all N shard files along
dim 0 with even division. Reconstruct by concatenating each tensor's slices
across all shards along dim 0, then renaming NeMo-RL FQNs to HF Qwen2.5-VL
keys:
    model.language_model.X  ->  model.X         (LLM)
    model.visual.X          ->  visual.X        (vision tower)
    other top-level keys    ->  unchanged       (e.g. lm_head.weight)

Output layout (HF, vLLM-loadable):
    <dst>/
      model-NNNNN-of-MMMMM.safetensors            (~5 GB chunks)
      model.safetensors.index.json
      config.json, generation_config.json         (from .hf_metadata)
      <tokenizer/processor files copied from --base-model-dir>

Usage:
    python convert_hfdcp_to_hf.py \\
        --src .../weights/dapo-7b-tight/model \\
        --dst .../weights/dapo-7b-tight \\
        --base-model-dir .../weights/Qwen2.5-VL-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, List

# Roughly 5 GB per HF shard (HF convention for large models).
_TARGET_SHARD_BYTES = 5 * (1024 ** 3)

# Files under base model dir that are tokenizer/processor metadata (not weights).
_BASE_NONWEIGHT_FILES = [
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "video_preprocessor_config.json",
]

_SHARD_RE = re.compile(r"^shard-(\d+)-model-\d+-of-\d+\.safetensors$")


def rename_fqn(k: str) -> str:
    """Map NeMo-RL FQN to HF Qwen2.5-VL key."""
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    if k.startswith("model.visual."):
        return "visual." + k[len("model.visual."):]
    return k


def list_shards(src: str) -> List[str]:
    """Return shard filenames sorted by their 1-based index."""
    indexed: List[tuple[int, str]] = []
    for fname in os.listdir(src):
        m = _SHARD_RE.match(fname)
        if m:
            indexed.append((int(m.group(1)), fname))
    indexed.sort(key=lambda x: x[0])
    return [f for _, f in indexed]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="NeMo-RL DCP-HF dir (contains shard-*.safetensors)")
    ap.add_argument("--dst", required=True, help="Output HF dir")
    ap.add_argument("--base-model-dir", required=True,
                    help="Local HF base model dir (for tokenizer/processor files)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output dir")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    base = os.path.abspath(args.base_model_dir)

    if not os.path.isdir(src):
        sys.exit(f"--src not a directory: {src}")
    if not os.path.isdir(base):
        sys.exit(f"--base-model-dir not a directory: {base}")
    if (
        os.path.exists(dst)
        and not args.overwrite
        and os.path.isfile(os.path.join(dst, "model.safetensors.index.json"))
    ):
        sys.exit(f"--dst already converted (model.safetensors.index.json exists): {dst}\n"
                 f"Pass --overwrite to redo.")

    os.makedirs(dst, exist_ok=True)

    # Lazy imports
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    shard_names = list_shards(src)
    if not shard_names:
        sys.exit(f"No shard-*.safetensors found under {src}")
    n_shards = len(shard_names)
    print(f"Found {n_shards} input shard(s).")

    # Open every input shard once for the duration of the conversion
    shard_paths = [os.path.join(src, n) for n in shard_names]
    handles = [safe_open(p, framework="pt", device="cpu") for p in shard_paths]
    try:
        # Take the key set from shard 0 (verified identical across shards by
        # caller; if a future ckpt diverges we'll error on missing keys below).
        all_keys = sorted(handles[0].keys())
        print(f"Total tensor keys per shard: {len(all_keys)}")

        # Stream tensors into output shards, flushing whenever we hit ~5 GB.
        weight_map: Dict[str, str] = {}
        total_bytes = 0
        out_idx = 1            # 1-based output shard index
        out_buf: Dict[str, "torch.Tensor"] = {}
        out_buf_bytes = 0
        out_files: List[str] = []  # collect filenames; finalize names at end

        # We don't know n_out_shards up-front, so write to temp names and
        # rename at the end.
        tmp_out_paths: List[str] = []

        def flush_buffer():
            nonlocal out_idx, out_buf, out_buf_bytes
            if not out_buf:
                return
            tmp_name = f"_tmp-{out_idx:05d}.safetensors"
            tmp_path = os.path.join(dst, tmp_name)
            print(f"  flushing output shard {out_idx} -> {tmp_name} "
                  f"({len(out_buf)} tensors, {out_buf_bytes / 1e9:.2f} GB)")
            save_file(out_buf, tmp_path, metadata={"format": "pt"})
            tmp_out_paths.append(tmp_path)
            for k in out_buf:
                weight_map[k] = tmp_name  # final filename patched up below
            out_buf = {}
            out_buf_bytes = 0
            out_idx += 1

        for i, key in enumerate(all_keys, 1):
            slices = [h.get_tensor(key) for h in handles]
            # Sanity: all slices must share dims 1+, only dim 0 differs
            ref = slices[0]
            for j, s in enumerate(slices[1:], 1):
                if s.shape[1:] != ref.shape[1:] or s.dtype != ref.dtype:
                    raise RuntimeError(
                        f"Inconsistent slices for {key!r}: "
                        f"shard 0 shape={tuple(ref.shape)} dtype={ref.dtype}, "
                        f"shard {j} shape={tuple(s.shape)} dtype={s.dtype}"
                    )
            full = torch.cat(slices, dim=0).contiguous()
            new_key = rename_fqn(key)
            nbytes = full.numel() * full.element_size()
            total_bytes += nbytes

            if out_buf and out_buf_bytes + nbytes > _TARGET_SHARD_BYTES:
                flush_buffer()

            out_buf[new_key] = full
            out_buf_bytes += nbytes

            if i % 50 == 0 or i == len(all_keys):
                print(f"  [{i}/{len(all_keys)}] {key} -> {new_key} "
                      f"shape={tuple(full.shape)}")

            # Free the slice tensors aggressively
            del slices
            del full

        flush_buffer()
    finally:
        # safe_open handles are exited via context-manager protocol on
        # garbage collection; explicit close not provided. Rely on GC.
        del handles

    # Now rename _tmp-N.safetensors -> model-N-of-TOTAL.safetensors and
    # patch weight_map.
    n_out = len(tmp_out_paths)
    print(f"\nWriting final {n_out} output shard(s) and index.")
    rename_map: Dict[str, str] = {}
    for i, tmp in enumerate(tmp_out_paths, 1):
        final_name = f"model-{i:05d}-of-{n_out:05d}.safetensors"
        final_path = os.path.join(dst, final_name)
        os.replace(tmp, final_path)
        rename_map[os.path.basename(tmp)] = final_name
    weight_map = {k: rename_map[v] for k, v in weight_map.items()}

    index = {
        "metadata": {"total_size": int(total_bytes)},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"  index: {len(weight_map)} tensors, {total_bytes / 1e9:.2f} GB total")

    # Copy config + generation_config from .hf_metadata
    hfmeta = os.path.join(src, ".hf_metadata")
    for fname in ("config.json", "generation_config.json"):
        s = os.path.join(hfmeta, fname)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, fname))
            print(f"  copied {fname}")
        else:
            print(f"  WARN: {fname} not found in .hf_metadata")

    # Copy tokenizer / processor files from base model
    print(f"Copying tokenizer/processor files from {base}")
    for fname in _BASE_NONWEIGHT_FILES:
        s = os.path.join(base, fname)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, fname))
            print(f"  + {fname}")

    print(f"\nDone. HF checkpoint at: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
