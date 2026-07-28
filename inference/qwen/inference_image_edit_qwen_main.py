#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
SpatialIQ MAIN task only, using Qwen/Qwen-Image-Edit (Diffusers).

Why a separate script:
The default `main` prompt asks the model to discard the scene and synthesize a
fresh image of just an integer on white. Image-edit diffusion models are
heavily biased toward preserving their input, so Qwen Image Edit usually
ignores that instruction and just regenerates the scene.

Workaround used here:
We feed Qwen a SIDE-BY-SIDE composite as the input image:
    [ scene | blank white canvas ]
The model still gets the scene to count from (left half), and the blank right
half is a much weaker prior — overwriting white with a digit is in-distribution
for an edit model. The prompt explicitly tells the model to leave the left
half untouched and only write the digit on the right half.

After inference, we crop the right half and save that as the answer image. The
existing main-task OCR pipeline can then read the digit as it would for the
Gemini outputs. Pass --save-composite to additionally dump the full uncropped
output for debugging.

Note on "true" two-image input:
The single-image `QwenImageEditPipeline` weights checked into
weights_image_edit/ accept one image. For literal multi-image conditioning,
Qwen-Image-Edit-2509 (`QwenImageEditPlusPipeline`) is the right pipeline, but
that needs separate weights. The composite trick gives us the same effect with
the weights already on disk.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Any, List, Optional, Tuple

from PIL import Image
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from view_utils import resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

MODEL_SLUG = "qwen_image_edit_canvas"
_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_image_edit")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFERENCE_RESULTS_DIR = os.path.join(REPO_ROOT, "inference_results")

# ---------------------------------------------------------------------------
# Prompt for the MAIN task. Tuned for the side-by-side composite input.
# ---------------------------------------------------------------------------
# Notes on phrasing:
# - We do NOT include the original CRITICAL_INSTRUCTION block — its rule
#   "Only coloring of the objects in the original image is allowed. Do NOT
#   change, add, or move the objects" directly contradicts "write a digit",
#   and edit models defer to the strong constraint.
# - We keep the HIDDEN OBJECT clause from DEFINITIONS because the count must
#   include hidden support objects to match the ground truth.
# - "LEFT must remain pixel-identical" is repeated to discourage the prior
#   from leaking edits into the scene half.

MAIN_TASK_PROMPT = """The image is a SIDE-BY-SIDE composite split exactly down the middle:
- The LEFT half contains a 3D scene of stacked objects (boxes, cubes, cylinders, cans, or mugs).
- The RIGHT half is a completely blank, solid white area provided as a writing canvas.

DEFINITIONS:
- A VISIBLE OBJECT is an object with at least one face fully or partially visible in the LEFT half.
- A HIDDEN OBJECT is an object with no visible faces, but which must exist to support a visible object directly above it.
- The TOTAL count includes BOTH visible and hidden objects.

YOUR TASK:
1. Count the TOTAL number of objects in the 3D scene shown on the LEFT half (including hidden support objects).
2. The LEFT half MUST remain pixel-identical to the input. Do not recolor, move, add, or remove anything on the left.
3. In the RIGHT half, render the integer count as a SINGLE large bold black digit (e.g., "7" or "12"), centered on the existing white background. The right half must contain ONLY that number — no scene, no extra text, no decoration. The rest of the right half stays solid white.

Output the same composite image with the LEFT half unchanged and the RIGHT half showing only the centered black integer on white.
"""

NUM_VIEWS = 4


def _get_output_dir(data_dir: str) -> str:
    """Map a dataset directory to the corresponding output directory under inference_results/."""
    data_prefix = os.environ.get("SPATIALIQ_DATA_PREFIX", "").rstrip("/")
    if data_prefix and data_dir.startswith(data_prefix + "/"):
        rel = data_dir[len(data_prefix) + 1:]
        out = os.path.join(INFERENCE_RESULTS_DIR, rel)
        os.makedirs(out, exist_ok=True)
        return out
    m = re.search(r'.*/(?=dataset/)', data_dir)
    if m:
        rel = data_dir[m.end():]
        out = os.path.join(INFERENCE_RESULTS_DIR, rel)
        os.makedirs(out, exist_ok=True)
        return out
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _load_rgb(path: str) -> Image.Image:
    if "_gt" in path:
        raise ValueError(f"Refusing to use ground-truth image as model input: {path}")
    return Image.open(path).convert("RGB")


def _make_composite(scene: Image.Image) -> Image.Image:
    """Concatenate the scene image with a same-size blank white canvas on its right."""
    w, h = scene.size
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    composite = Image.new("RGB", (w * 2, h), (255, 255, 255))
    composite.paste(scene, (0, 0))
    composite.paste(canvas, (w, 0))
    return composite


def _crop_right_half(composite: Image.Image, scene_w: int) -> Image.Image:
    """Crop the right half (the answer canvas) out of the composite output."""
    w, h = composite.size
    # Right half starts where the scene ended — but the model may have rescaled,
    # so split based on the actual output width and the original scene width.
    # If output width ≈ 2 * scene_w we use scene_w; otherwise split at the midpoint.
    if abs(w - 2 * scene_w) <= 4:
        return composite.crop((scene_w, 0, w, h))
    return composite.crop((w // 2, 0, w, h))


def _collect_work_items(
    samples: List[Tuple[str, List[Tuple[int, str]]]],
    override: bool,
    pred_stem_suffix: str,
    seed: int,
) -> List[dict]:
    """Build a flat list of independent main-task work items (one per view)."""
    items: List[dict] = []
    for label, views in samples:
        for v_idx, img_path in views:
            out_dir = _get_output_dir(os.path.dirname(img_path))
            out_path = os.path.join(out_dir, f"task_main_pred{pred_stem_suffix}.png")
            if not override and os.path.isfile(out_path):
                continue
            items.append({
                "img_path": img_path,
                "out_path": out_path,
                "label": f"{label} view_{v_idx} main",
                "seed": seed,
            })
    return items


def _gpu_worker(
    gpu_id: int,
    work_items: List[dict],
    ckpt: str,
    torch_dtype: Any,
    num_inference_steps: int,
    true_cfg_scale: float,
    negative_prompt: str,
    cpu_offload: bool,
    save_composite: bool,
) -> None:
    import torch
    from diffusers import QwenImageEditPipeline

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading pipeline ({len(work_items)} items) -> {device}")
    t0 = time.monotonic()
    pipe = QwenImageEditPipeline.from_pretrained(ckpt, torch_dtype=torch_dtype)
    if cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
    else:
        pipe.to(device)
    print(f"[GPU {gpu_id}] Pipeline ready ({time.monotonic() - t0:.1f}s)")

    for i, item in enumerate(work_items, 1):
        t1 = time.monotonic()
        try:
            scene = _load_rgb(item["img_path"])
            composite = _make_composite(scene)
            gen = torch.Generator(device=device).manual_seed(item["seed"])
            out = pipe(
                image=composite,
                prompt=MAIN_TASK_PROMPT,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=gen,
            )
            out_img = out.images[0]
            answer_img = _crop_right_half(out_img, scene.width)
            answer_img.save(item["out_path"])
            if save_composite:
                base, ext = os.path.splitext(item["out_path"])
                out_img.save(f"{base}_composite{ext}")
            elapsed = time.monotonic() - t1
            print(f"[GPU {gpu_id}] ({i}/{len(work_items)}) {item['label']} -> {os.path.basename(item['out_path'])} ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.monotonic() - t1
            print(f"[GPU {gpu_id}] ({i}/{len(work_items)}) {item['label']} ERROR: {e} ({elapsed:.1f}s)")

    print(f"[GPU {gpu_id}] Done.")


def run_main_task(
    samples: List[Tuple[str, List[Tuple[int, str]]]],
    ckpt: str,
    torch_dtype: Any,
    override: bool,
    *,
    num_inference_steps: int,
    true_cfg_scale: float,
    negative_prompt: str,
    seed: int,
    pred_stem_suffix: str,
    num_gpus: int,
    cpu_offload: bool,
    save_composite: bool,
) -> None:
    import torch.multiprocessing as mp

    items = _collect_work_items(samples, override, pred_stem_suffix, seed)
    if not items:
        print("All outputs already exist. Nothing to do.")
        return

    num_gpus = min(num_gpus, len(items))
    print(f"Work items: {len(items)}, GPUs: {num_gpus}")
    print(f"  Steps: {num_inference_steps}, CFG: {true_cfg_scale}, suffix: {pred_stem_suffix}\n")

    if num_gpus <= 1:
        _gpu_worker(0, items, ckpt, torch_dtype, num_inference_steps,
                    true_cfg_scale, negative_prompt, cpu_offload, save_composite)
        return

    shards: List[List[dict]] = [[] for _ in range(num_gpus)]
    for i, item in enumerate(items):
        shards[i % num_gpus].append(item)

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(
            target=_gpu_worker,
            args=(gpu_id, shards[gpu_id], ckpt, torch_dtype,
                  num_inference_steps, true_cfg_scale, negative_prompt,
                  cpu_offload, save_composite),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed = [p for p in procs if p.exitcode != 0]
    if failed:
        print(f"\nWARNING: {len(failed)} GPU worker(s) exited with errors.")
    else:
        print("\nAll GPU workers finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen-Image-Edit on the SpatialIQ MAIN task using a "
                    "scene+blank-canvas composite as input. Saves only the "
                    "right (answer canvas) half as the prediction PNG."
    )
    parser.add_argument(
        "folder",
        help="Sample folder, tree root, or a .txt file with one folder path per line.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("QWEN_IMAGE_EDIT_CHECKPOINT", _DEFAULT_CKPT),
        metavar="DIR",
        help=f"Local HF folder (default: {_DEFAULT_CKPT} or QWEN_IMAGE_EDIT_CHECKPOINT).",
    )
    parser.add_argument("--override", action="store_true", help="Overwrite existing PNGs.")
    parser.add_argument(
        "--pred-stem-suffix",
        type=str,
        default=f"_image_{MODEL_SLUG}",
        metavar="SUFFIX",
        help=f'Stem suffix after task_main_pred (default: _image_{MODEL_SLUG}).',
    )
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps (default 50).")
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="true_cfg_scale (default 4.0, per Qwen-Image-Edit README).",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="modify the left half, change the scene, recolor objects, add objects, remove objects, write text on the left half",
        help="Negative prompt; defaults to discouraging edits to the scene half.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="enable_sequential_cpu_offload() for lower VRAM (slower).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Pipeline torch dtype (default bfloat16).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        dest="num_gpus",
        metavar="N",
        help="Number of GPUs for data-parallel inference (default: all visible GPUs).",
    )
    parser.add_argument(
        "--save-composite",
        action="store_true",
        help="Also save the full uncropped composite output alongside the cropped answer "
             "(suffix _composite). Useful for debugging whether the model is leaking "
             "edits into the scene half.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "Missing dependencies for image edit. Diffusers needs Python >= 3.10.\n"
            "  cd inference/qwen\n"
            "  uv venv --python 3.12 .venv-image-edit\n"
            "  uv pip install --python .venv-image-edit/bin/python -r requirements_image_edit.txt\n"
            f"Original error: {e}"
        ) from e

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isdir(ckpt):
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() or 1
    print(f"Checkpoint: {ckpt}")
    print(f"GPUs available: {torch.cuda.device_count()}, using: {num_gpus}")

    folders = _resolve_folders(args.folder)
    samples = _scan_folders_cached(args.folder, folders)
    if not samples:
        raise SystemExit("No view.png files found in any of the specified folder(s).")

    run_main_task(
        samples,
        ckpt,
        torch_dtype,
        override=args.override,
        num_inference_steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        pred_stem_suffix=args.pred_stem_suffix,
        num_gpus=num_gpus,
        cpu_offload=args.cpu_offload,
        save_composite=args.save_composite,
    )


if __name__ == "__main__":
    main()
