#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
SpatialIQ MAIN task only, using HunyuanImage 3.0 Instruct-Distil.

Why a separate script:
The default `main` prompt asks the model to discard the scene and synthesize
a fresh image of just an integer on white. Image-edit/diffusion models bias
hard toward preserving their input, so HunyuanImage usually ignores that
and just regenerates the scene.

Two workaround modes are supported:

(1) Two-image input (DEFAULT):
    HunyuanImage's generate_image() natively accepts a LIST of input images
    (used by tasks 10/11 for cross-view comparison). We exploit this by
    passing TWO images:
      IMAGE 1 = the scene to count
      IMAGE 2 = a blank white canvas of the same size (written to a temp file)
    The prompt mirrors the cross-view task framing ("return the second image
    with modifications") so HunyuanImage is being asked to do something it
    already knows how to do: edit one of the inputs (the canvas) while
    ignoring the other (the scene).
    Output slug: hunyuan_canvas
    File: task_main_pred_image_hunyuan_canvas.png

(2) Composite input (--composite flag):
    Same trick used by inference_image_edit_qwen_main.py — concatenate the
    scene and a blank white canvas side-by-side into ONE input image:
        [ scene | blank white canvas ]
    Pass that as a single image and crop the right half of the output.
    Useful for apples-to-apples comparison with the Qwen Image Edit canvas
    run, since it uses the exact same input shape.
    Output slug: hunyuan_composite
    File: task_main_pred_image_hunyuan_composite.png

Both slugs are distinct from the existing `hunyuan` slug produced by
inference_image_hunyuan.py, so none of these clobber each other.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
import time
from typing import Any, List, Optional, Tuple

from PIL import Image

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from view_utils import resolve_folders as _resolve_folders, scan_folders_cached as _scan_folders_cached

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference_image_hunyuan_main")

MODEL_SLUG_TWOIMAGE = "hunyuan_canvas"
MODEL_SLUG_COMPOSITE = "hunyuan_composite"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFERENCE_RESULTS_DIR = os.path.join(REPO_ROOT, "inference_results")

# ---------------------------------------------------------------------------
# Prompts for the MAIN task — one per input mode.
# ---------------------------------------------------------------------------
# Notes on phrasing common to both:
# - We do NOT include the original CRITICAL_INSTRUCTION block — its rule
#   "Only coloring of the objects in the original image is allowed. Do NOT
#   change, add, or move the objects" directly contradicts "render a digit",
#   and edit models defer to the strong constraint.
# - We keep the HIDDEN OBJECT clause from DEFINITIONS so the count includes
#   hidden support objects to match the ground truth.

# Two-image-input prompt: framing mirrors tasks 10/11 ("modify IMAGE 2,
# return IMAGE 2"), which Hunyuan handles natively.
MAIN_TASK_PROMPT_TWOIMAGE = """You are given TWO images:
- IMAGE 1 is a 3D scene of stacked objects (boxes, cubes, cylinders, cans, or mugs).
- IMAGE 2 is a completely blank, solid white canvas of the SAME size as IMAGE 1.

DEFINITIONS:
- A VISIBLE OBJECT is an object with at least one face fully or partially visible in IMAGE 1.
- A HIDDEN OBJECT is an object with no visible faces, but which must exist to support a visible object directly above it.
- The TOTAL count includes BOTH visible and hidden objects.

YOUR TASK:
1. Count the TOTAL number of objects in the 3D scene shown in IMAGE 1 (including hidden support objects).
2. Take IMAGE 2 (the blank white canvas) and render the integer count as a SINGLE large bold black digit (e.g., "7" or "12"), centered on the white canvas. The canvas must contain ONLY that number — no scene, no extra text, no decoration. The rest of the canvas stays solid white.

OUTPUT: Return the modified IMAGE 2 — the white canvas with only the centered black integer on it. Do NOT return IMAGE 1 or any version of the 3D scene.
"""

# Composite-input prompt: matches inference_image_edit_qwen_main.py verbatim
# so the two model runs are directly comparable.
MAIN_TASK_PROMPT_COMPOSITE = """The image is a SIDE-BY-SIDE composite split exactly down the middle:
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


def _validate_image_path(path: str) -> None:
    if "_gt" in os.path.basename(path):
        raise ValueError(f"Refusing to use ground-truth image as model input: {path}")


def _make_blank_canvas(scene_path: str, dest_path: str) -> Tuple[int, int]:
    """Write a same-size solid-white PNG to dest_path; return (w, h)."""
    with Image.open(scene_path) as scene:
        w, h = scene.size
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.save(dest_path)
    return w, h


def _make_composite(scene_path: str, dest_path: str) -> int:
    """Write [scene | blank white canvas] (width = 2*scene_w) to dest_path.
    Returns the scene width so the right half can be cropped from the model output.
    """
    with Image.open(scene_path) as scene:
        scene_rgb = scene.convert("RGB")
        w, h = scene_rgb.size
        composite = Image.new("RGB", (w * 2, h), (255, 255, 255))
        composite.paste(scene_rgb, (0, 0))
    composite.save(dest_path)
    return w


def _crop_right_half(img: Image.Image, scene_w: int) -> Image.Image:
    """Crop the right half (the answer canvas) out of the composite output.
    If the model rescaled the output, fall back to a midpoint split.
    """
    w, h = img.size
    if abs(w - 2 * scene_w) <= 4:
        return img.crop((scene_w, 0, w, h))
    return img.crop((w // 2, 0, w, h))


def load_hunyuan_model(
    model_path: str,
    *,
    attn_impl: str = "sdpa",
    moe_impl: str = "eager",
) -> Any:
    """Load HunyuanImage 3.0 Instruct-Distil sharded across all visible GPUs."""
    from transformers import AutoModelForCausalLM

    logger.info("Loading HunyuanImage from: %s", model_path)
    t0 = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
        moe_impl=moe_impl,
        moe_drop_tokens=True,
    )
    model.load_tokenizer(model_path)

    elapsed = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs", elapsed)
    return model


def _generate_image(
    model: Any,
    image_paths: List[str],
    prompt: str,
    *,
    seed: int,
    diff_steps: int,
    bot_task: str,
) -> Tuple[Optional[str], Any]:
    """Call HunyuanImage's generate_image() with the given input images and prompt."""
    for p in image_paths:
        _validate_image_path(p)

    cot_text, samples = model.generate_image(
        prompt=prompt,
        image=image_paths,
        seed=seed,
        image_size="auto",
        use_system_prompt="en_unified",
        bot_task=bot_task,
        infer_align_image_size=True,
        diff_infer_steps=diff_steps,
        verbose=0,
    )

    cot_str = None
    if cot_text:
        cot_str = "\n".join(cot_text) if isinstance(cot_text, list) else str(cot_text)

    out_image = samples[0] if samples else None
    return cot_str, out_image


def run_main_task(
    samples: List[Tuple[str, List[Tuple[int, str]]]],
    model: Any,
    *,
    override: bool,
    seed: int,
    diff_steps: int,
    bot_task: str,
    pred_stem_suffix: str,
    save_cot: bool,
    save_composite: bool,
    composite_mode: bool,
    canvas_tmp_dir: str,
) -> None:
    total_views = sum(len(views) for _, views in samples)
    mode_label = "composite" if composite_mode else "two-image"
    print(f"SpatialIQ MAIN task (HunyuanImage {mode_label}): {total_views} views")
    print(f"  Diff steps: {diff_steps}, bot_task: {bot_task}, suffix: {pred_stem_suffix}\n")

    prompt = MAIN_TASK_PROMPT_COMPOSITE if composite_mode else MAIN_TASK_PROMPT_TWOIMAGE
    tmp_prefix = "hunyuan_composite_" if composite_mode else "hunyuan_canvas_"

    done = 0
    for label, views in samples:
        for v_idx, scene_path in views:
            done += 1
            out_dir = _get_output_dir(os.path.dirname(scene_path))
            out_path = os.path.join(out_dir, f"task_main_pred{pred_stem_suffix}.png")

            print(f"[{done}/{total_views}] {label} view_{v_idx} -> {os.path.basename(out_path)}")

            if not override and os.path.isfile(out_path):
                print("    (exists, skipped)")
                continue

            with tempfile.NamedTemporaryFile(
                suffix=".png", prefix=f"{tmp_prefix}v{v_idx}_",
                dir=canvas_tmp_dir, delete=False,
            ) as tmp:
                tmp_path = tmp.name
            try:
                if composite_mode:
                    scene_w = _make_composite(scene_path, tmp_path)
                    image_paths = [tmp_path]
                else:
                    _make_blank_canvas(scene_path, tmp_path)
                    image_paths = [scene_path, tmp_path]

                t0 = time.monotonic()
                cot, out_img = _generate_image(
                    model, image_paths, prompt,
                    seed=seed, diff_steps=diff_steps, bot_task=bot_task,
                )
                elapsed = time.monotonic() - t0

                if out_img is not None:
                    if composite_mode:
                        if save_composite:
                            base, ext = os.path.splitext(out_path)
                            out_img.save(f"{base}_composite{ext}")
                        answer_img = _crop_right_half(out_img, scene_w)
                        answer_img.save(out_path)
                    else:
                        out_img.save(out_path)
                    print(f"    Image saved. ({elapsed:.1f}s)")
                else:
                    print(f"    No image generated. ({elapsed:.1f}s)")

                if save_cot and cot:
                    cot_path = os.path.splitext(out_path)[0] + "_cot.txt"
                    with open(cot_path, "w") as f:
                        f.write(cot)
                    print(f"    CoT saved: {os.path.basename(cot_path)}")
            except Exception as e:
                elapsed = time.monotonic() - t0 if 't0' in locals() else 0.0
                print(f"    Error: {e} ({elapsed:.1f}s)")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    print("\nSpatialIQ MAIN task done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HunyuanImage 3.0 on the SpatialIQ MAIN task. Default "
                    "uses two-image input (scene + blank canvas) and saves "
                    "task_main_pred_image_hunyuan_canvas.png. Pass --composite "
                    "to instead concatenate the scene and a blank canvas into a "
                    "single side-by-side image (matching inference_image_edit_qwen_main.py); "
                    "in that mode the right half is cropped from the model output "
                    "and saved as task_main_pred_image_hunyuan_composite.png."
    )
    parser.add_argument(
        "folder",
        help="Sample folder, top-level folder, or a .txt file with one folder path per line.",
    )
    _default_model_path = os.environ.get(
        "HUNYUAN_MODEL_PATH",
        os.path.join(
            os.environ.get("SPATIALIQ_INFERENCE_DIR",
                           "spatialiq-inference"),
            "weights", "HunyuanImage-3-Instruct-Distil",
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=_default_model_path,
        metavar="DIR",
        help="Path to HunyuanImage-3-Instruct-Distil weights "
             "(default: spatialiq-inference/weights/... or HUNYUAN_MODEL_PATH).",
    )
    parser.add_argument("--override", action="store_true", help="Overwrite existing PNGs.")
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Use side-by-side composite input ([scene | blank canvas]) instead of "
             "the default two-image input. Matches inference_image_edit_qwen_main.py. "
             "Changes the default output filename to task_main_pred_image_hunyuan_composite.png.",
    )
    parser.add_argument(
        "--pred-stem-suffix",
        type=str,
        default=None,
        metavar="SUFFIX",
        help=f"Stem suffix after task_main_pred (default: _image_{MODEL_SLUG_TWOIMAGE} "
             f"in two-image mode, _image_{MODEL_SLUG_COMPOSITE} in --composite mode).",
    )
    parser.add_argument(
        "--diff-steps",
        type=int,
        default=int(os.environ.get("HUNYUAN_STEPS", "8")),
        metavar="N",
        help="Diffusion inference steps (default: 8 for Distil; use 50 for full variant).",
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        default="think_recaption",
        choices=["think_recaption", "recaption", "image"],
        help="HunyuanImage bot_task mode (default: think_recaption = CoT + generate).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2"],
        help="Attention implementation (default: sdpa).",
    )
    parser.add_argument(
        "--moe-impl",
        type=str,
        default="eager",
        choices=["eager", "flashinfer"],
        help="MoE implementation (default: eager; flashinfer is ~3x faster if installed).",
    )
    parser.add_argument(
        "--save-cot",
        action="store_true",
        help="Also save the CoT text alongside the image (suffix _cot.txt). "
             "Useful as a fallback signal when the digit rendering fails.",
    )
    parser.add_argument(
        "--save-composite",
        action="store_true",
        help="Composite mode only: also save the full uncropped composite output "
             "(suffix _composite). Lets you check whether the model leaked edits "
             "into the scene half.",
    )
    parser.add_argument(
        "--canvas-tmp-dir",
        type=str,
        default=os.environ.get("TMPDIR", "/tmp"),
        metavar="DIR",
        help="Where to write the per-view canvas/composite temp file (default: $TMPDIR or /tmp).",
    )
    args = parser.parse_args()

    if args.save_composite and not args.composite:
        raise SystemExit("--save-composite is only meaningful with --composite mode.")

    if args.pred_stem_suffix is None:
        slug = MODEL_SLUG_COMPOSITE if args.composite else MODEL_SLUG_TWOIMAGE
        args.pred_stem_suffix = f"_image_{slug}"

    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        raise SystemExit(f"Model weights not found: {model_path}")

    folders = _resolve_folders(args.folder)
    samples = _scan_folders_cached(args.folder, folders)
    if not samples:
        raise SystemExit("No view.png files found in any of the specified folder(s).")

    os.makedirs(args.canvas_tmp_dir, exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Diff steps: {args.diff_steps}, bot_task: {args.bot_task}")
    print(f"Attn: {args.attn_impl}, MoE: {args.moe_impl}")
    print(f"Seed: {args.seed}\n")

    model = load_hunyuan_model(
        model_path,
        attn_impl=args.attn_impl,
        moe_impl=args.moe_impl,
    )

    run_main_task(
        samples,
        model,
        override=args.override,
        seed=args.seed,
        diff_steps=args.diff_steps,
        bot_task=args.bot_task,
        pred_stem_suffix=args.pred_stem_suffix,
        save_cot=args.save_cot,
        save_composite=args.save_composite,
        composite_mode=args.composite,
        canvas_tmp_dir=args.canvas_tmp_dir,
    )


if __name__ == "__main__":
    main()
