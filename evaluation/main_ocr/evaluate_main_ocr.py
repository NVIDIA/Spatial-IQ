#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluate the SpatialIQ "main task" (count of total blocks) for image-output
models by running OCR on the predicted black-number-on-white-background image.

Three OCR systems vote per image; 2/3 consensus wins. Disagreements are
flagged for manual review.

Systems:
    1. Tesseract (digit-only whitelist via pytesseract)
    2. PaddleOCR
    3. Qwen2.5-VL via local vLLM OpenAI-compatible server

Models supported (--model):
    gemini, hunyuan, qwen-image-edit

Subset (optional, --subset PATH):
    A .txt file with one view-dir path per line (lines starting with '#' or
    blank are ignored). Each line should contain '/eval_set/<...>/view_<NN>';
    the part after '/eval_set/' becomes a view-dir rel_key, expanded with the
    four offset/fov/dist subdirs (3000 leaves from 750 view-dirs in the case
    of inference_results/subset_eval.txt). Without --subset, the entire
    pred-root is walked.

Prediction layout (per leaf):
    <pred-root>/data_*/level_1/sample_*/view_*/offset_*/task_main_pred_image_<slug>.png

For hunyuan, we union the main pred-root with --hunyuan-extra-root.

Ground truth layout (mirrors the relative path under dataset/eval_set/):
    <gt-root>/data_*/level_1/sample_*/view_*/offset_*/gt.json
The integer comes from gt.json["task_main"].

Outputs (under --output-dir):
    eval_main_ocr_<model>.json   — full per-image results + aggregate
    eval_main_ocr_<model>.csv    — flat per-image table
    eval_main_ocr_<model>_review.csv  — only the flagged-for-review rows
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# Quiet PaddleOCR's noisy logger before it loads.
os.environ.setdefault("FLAGS_minloglevel", "2")
logging.getLogger("ppocr").setLevel(logging.ERROR)

DEFAULT_PRED_ROOT = os.environ.get("PRED_ROOT", "inference_results/dataset/eval_set")
DEFAULT_HUNYUAN_EXTRA_ROOT = os.environ.get("HUNYUAN_EXTRA_ROOT", "inference_results/hunyuan_inference_results")
DEFAULT_GT_ROOT = os.environ.get("GT_ROOT", "dataset/eval_set")
DEFAULT_VLLM_BASE_URLS = (
    os.environ.get("VLLM_BASE_URLS")
    or os.environ.get("VLLM_BASE_URL")
    or "http://localhost:8000/v1"
)
DEFAULT_VLLM_MODEL = os.environ.get("QWEN_MODEL", "qwen3.5-27b")

MODEL_TO_FILENAME_SLUG = {
    "gemini": "gemini",
    "hunyuan": "hunyuan",
    "hunyuan-composite": "hunyuan_composite",
    "qwen-image-edit": "qwen_image_edit",
}

QWEN_OCR_PROMPT = (
    "The image contains exactly one integer in black on a white background. "
    "Output only that integer. No other characters, no explanation."
)

# View-dir → leaf-dir expansion (matches evaluation_human_image.py).
CAMERA_DIRS = [
    "offset12.5_fov1_dist1",
    "offset12.5_fov3_dist0.3",
    "offset3_fov1_dist1",
    "offset3_fov3_dist0.3",
]

EVAL_SET_MARKER = "/eval_set/"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _read_subset_file(path: str) -> List[str]:
    """Read a subset_eval.txt-style file of view-dir paths and return a list of
    leaf rel_keys (i.e. data_X/level_1/sample_Y/view_Z/offset_F_D).

    Each non-blank, non-# line should contain '/eval_set/<...>/view_<NN>'. The
    portion after '/eval_set/' is taken as the view-dir rel_key, then expanded
    with the four CAMERA_DIRS to produce leaf rel_keys.
    """
    rel_keys: List[str] = []
    skipped: List[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            idx = line.find(EVAL_SET_MARKER)
            if idx < 0:
                skipped.append(line)
                continue
            view_rel = line[idx + len(EVAL_SET_MARKER):].rstrip("/")
            for cam in CAMERA_DIRS:
                rel_keys.append(os.path.join(view_rel, cam))
    if skipped:
        print(f"  WARN: subset file: {len(skipped)} line(s) had no '{EVAL_SET_MARKER}' marker; skipped.",
              file=sys.stderr)
        for s in skipped[:3]:
            print(f"    - {s}", file=sys.stderr)
    return rel_keys


def discover_pred_files(
    model: str,
    pred_root: str,
    hunyuan_extra_root: Optional[str],
    subset_rel_keys: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Find every task_main_pred_image_<slug>.png for the model.

    Returns list of (rel_key, abs_path) where rel_key is the path under the
    eval_set directory (i.e. "data_X/level_1/sample_Y/view_Z/offset_F_D"). For
    hunyuan, files in pred_root take precedence over hunyuan_extra_root.

    If subset_rel_keys is given, restricts to those leaf rel_keys (no os.walk).
    """
    slug = MODEL_TO_FILENAME_SLUG[model]
    target_filename = f"task_main_pred_image_{slug}.png"

    roots: List[str] = [pred_root]
    if model == "hunyuan" and hunyuan_extra_root:
        roots.append(hunyuan_extra_root)

    found: Dict[str, str] = {}

    if subset_rel_keys is not None:
        # Direct probes — much faster than walking when the subset is small.
        missing = 0
        for rel_key in subset_rel_keys:
            for root in roots:
                p = os.path.join(root, rel_key, target_filename)
                if os.path.isfile(p):
                    if rel_key not in found:
                        found[rel_key] = p
                    break
            else:
                missing += 1
        if missing:
            print(f"  WARN: subset: {missing}/{len(subset_rel_keys)} leaf(s) had no "
                  f"{target_filename} under any pred root.", file=sys.stderr)
        return sorted(found.items())

    # Default: walk each root.
    def _scan(root: str) -> None:
        if not os.path.isdir(root):
            print(f"  WARN: root not found, skipping: {root}", file=sys.stderr)
            return
        root_abs = os.path.abspath(root)
        for dirpath, _, filenames in os.walk(root_abs):
            if target_filename in filenames:
                rel = os.path.relpath(dirpath, root_abs)
                if rel not in found:
                    found[rel] = os.path.join(dirpath, target_filename)

    for root in roots:
        _scan(root)

    return sorted(found.items())


def load_gt(gt_root: str, rel_key: str) -> Optional[int]:
    gt_path = os.path.join(gt_root, rel_key, "gt.json")
    if not os.path.isfile(gt_path):
        return None
    try:
        with open(gt_path) as f:
            data = json.load(f)
        v = data.get("task_main")
        if v is None:
            v = data.get("total_blocks")
        return int(v) if v is not None else None
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# OCR backends (lazy-init; each returns Optional[int])
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r"-?\d+")


def _extract_int(text: str) -> Optional[int]:
    """Pull the first integer out of an OCR/VLM response string."""
    if not text:
        return None
    m = _DIGIT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


class TesseractOCR:
    def __init__(self) -> None:
        import pytesseract
        self._pt = pytesseract

    def __call__(self, image_path: str) -> Optional[int]:
        from PIL import Image
        try:
            img = Image.open(image_path).convert("L")
            cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = self._pt.image_to_string(img, config=cfg)
            return _extract_int(text)
        except Exception:
            return None


class PaddleOCRBackend:
    def __init__(self) -> None:
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        # PaddleOCR's C++ predictor is not thread-safe; concurrent calls from
        # ThreadPoolExecutor SIGSEGV deep in conv2d. Serialize at the python
        # boundary. Tesseract and Qwen still run in parallel.
        self._lock = threading.Lock()

    def __call__(self, image_path: str) -> Optional[int]:
        try:
            with self._lock:
                result = self._ocr.ocr(image_path, cls=False)
        except Exception:
            return None
        # PaddleOCR 2.x returns [[[box, (text, score)], ...]] (per-image list).
        # Concatenate all detected text and pull the first integer.
        if not result:
            return None
        page = result[0] if isinstance(result, list) and result else result
        if page is None:
            return None
        texts: List[Tuple[str, float]] = []
        try:
            for det in page:
                # det is [box, (text, score)]
                if len(det) >= 2 and isinstance(det[1], (list, tuple)) and len(det[1]) >= 2:
                    texts.append((str(det[1][0]), float(det[1][1])))
        except (TypeError, IndexError):
            return None
        if not texts:
            return None
        # Prefer the highest-confidence detection that contains a digit.
        texts.sort(key=lambda t: t[1], reverse=True)
        for t, _score in texts:
            v = _extract_int(t)
            if v is not None:
                return v
        return None


class QwenVLMBackend:
    """Round-robins across one or more vLLM endpoints (each ideally TP=1 on its
    own GPU) for high throughput on short-context OCR queries.
    """

    def __init__(self, base_urls: List[str], model: str) -> None:
        from openai import OpenAI
        if not base_urls:
            raise ValueError("QwenVLMBackend needs at least one base URL")
        self._clients = [OpenAI(base_url=u, api_key="EMPTY") for u in base_urls]
        self._model = model
        self._idx = 0
        self._idx_lock = threading.Lock()

    def _next_client(self):
        with self._idx_lock:
            c = self._clients[self._idx % len(self._clients)]
            self._idx += 1
        return c

    def __call__(self, image_path: str) -> Optional[int]:
        client = self._next_client()
        try:
            with open(image_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode("ascii")
            data_url = f"data:image/png;base64,{b64}"
            resp = client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": QWEN_OCR_PROMPT},
                    ],
                }],
                temperature=0,
                max_tokens=16,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = (resp.choices[0].message.content or "").strip()
            return _extract_int(text)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

SYSTEM_PRIORITY = ("qwen_vlm", "paddle", "tesseract")


def consensus(votes: Dict[str, Optional[int]]) -> Tuple[Optional[int], bool, str]:
    """Return (consensus_int, flagged_for_review, reason).

    Threshold scales with the number of valid (non-None) readings:
        n=0 → flag, "all_failed"
        n=1 → accept the single reading           (1/1)
        n=2 → accept the more popular reading;    (1/2)
              if the two disagree, tie-break by SYSTEM_PRIORITY
        n=3 → accept if any value has >=2 votes;  (2/3)
              else flag, "no_majority"
    """
    valid_pairs = [(name, v) for name, v in votes.items() if v is not None]
    n = len(valid_pairs)
    if n == 0:
        return None, True, "all_failed"

    if n == 1:
        return valid_pairs[0][1], False, "single_valid"

    counts = Counter(v for _, v in valid_pairs)
    top_val, top_cnt = counts.most_common(1)[0]

    if n == 2:
        if top_cnt == 2:
            return top_val, False, "consensus_2_of_2"
        # Two readings, two distinct values — tie-break by priority.
        for sys_name in SYSTEM_PRIORITY:
            for s, v in valid_pairs:
                if s == sys_name:
                    return v, False, f"tiebreak_to_{sys_name}"
        # Fallback (shouldn't hit if SYSTEM_PRIORITY covers all systems).
        return valid_pairs[0][1], False, "tiebreak_first"

    # n == 3
    if top_cnt >= 2:
        return top_val, False, "consensus_2_of_3" if top_cnt == 2 else "consensus_3_of_3"
    return None, True, "no_majority"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def evaluate_one(
    rel_key: str,
    pred_path: str,
    gt: Optional[int],
    tesseract: Optional[TesseractOCR],
    paddle: Optional[PaddleOCRBackend],
    qwen: Optional[QwenVLMBackend],
) -> Dict[str, Any]:
    votes: Dict[str, Optional[int]] = {}
    if tesseract is not None:
        votes["tesseract"] = tesseract(pred_path)
    if paddle is not None:
        votes["paddle"] = paddle(pred_path)
    if qwen is not None:
        votes["qwen_vlm"] = qwen(pred_path)

    cons_val, flagged, reason = consensus(votes)

    # Auto-resolve 3-way OCR disagreement when GT is known. If any reading
    # matches GT, that's almost certainly the correct read — accept it as
    # consensus. If none match, the model's answer (whatever it actually is)
    # cannot equal GT, so mark wrong without manual review.
    if reason == "no_majority" and gt is not None:
        valid_values = {v for v in votes.values() if v is not None}
        if gt in valid_values:
            cons_val = gt
            reason = "no_majority_one_matches_gt"
        else:
            reason = "no_majority_all_wrong"
        flagged = False

    correct: Optional[bool]
    if reason == "no_majority_all_wrong":
        correct = False
    elif cons_val is None or gt is None:
        correct = None
    else:
        correct = bool(cons_val == gt)

    return {
        "rel_key": rel_key,
        "pred_path": pred_path,
        "gt": gt,
        "votes": votes,
        "consensus": cons_val,
        "flagged_for_review": flagged,
        "reason": reason,
        "correct": correct,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    n_consensus = sum(1 for r in results if r["consensus"] is not None)
    n_flagged = sum(1 for r in results if r["flagged_for_review"])
    n_missing_gt = sum(1 for r in results if r["gt"] is None)
    n_correct = sum(1 for r in results if r["correct"] is True)
    n_scored = sum(1 for r in results if r["correct"] is not None)
    pct = (n_correct / n_scored * 100.0) if n_scored else None

    # Per-system raw success-rate (returned an int) and agreement-with-GT.
    per_system: Dict[str, Dict[str, Any]] = {}
    for sys_name in ("tesseract", "paddle", "qwen_vlm"):
        if not any(sys_name in r["votes"] for r in results):
            continue
        n_returned = sum(1 for r in results if r["votes"].get(sys_name) is not None)
        n_match_gt = sum(
            1 for r in results
            if r["gt"] is not None and r["votes"].get(sys_name) == r["gt"]
        )
        n_gt_present = sum(
            1 for r in results
            if r["gt"] is not None and sys_name in r["votes"]
        )
        per_system[sys_name] = {
            "n_returned_int": n_returned,
            "n_match_gt": n_match_gt,
            "n_gt_present": n_gt_present,
            "pct_match_gt": (n_match_gt / n_gt_present * 100.0) if n_gt_present else None,
        }

    by_reason = Counter(r["reason"] for r in results)

    return {
        "n_total": n,
        "n_consensus": n_consensus,
        "n_flagged_for_review": n_flagged,
        "n_missing_gt": n_missing_gt,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "pct_correct": pct,
        "per_system": per_system,
        "by_reason": dict(by_reason),
    }


def format_summary(model: str, agg: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Main-task OCR Evaluation  (model={model})")
    lines.append(f"{'='*60}")
    lines.append(f"  total images          : {agg['n_total']}")
    lines.append(f"  consensus reached     : {agg['n_consensus']}")
    lines.append(f"  flagged for review    : {agg['n_flagged_for_review']}")
    lines.append(f"  missing ground truth  : {agg['n_missing_gt']}")
    lines.append(f"  scored (consensus+GT) : {agg['n_scored']}")
    lines.append(f"  correct               : {agg['n_correct']}")
    pct = agg.get("pct_correct")
    pct_s = f"{pct:.2f}%" if pct is not None else "N/A"
    lines.append(f"  accuracy              : {pct_s}")
    lines.append(f"{'-'*60}")
    lines.append(f"  Per-system match-with-GT (raw, ignoring consensus):")
    for sys_name, d in agg.get("per_system", {}).items():
        p = d.get("pct_match_gt")
        ps = f"{p:.2f}%" if p is not None else "N/A"
        lines.append(
            f"    {sys_name:<12}  returned={d['n_returned_int']:>4}  "
            f"match_gt={d['n_match_gt']:>4}/{d['n_gt_present']:<4}  ({ps})"
        )
    by_reason = agg.get("by_reason", {})
    if by_reason:
        lines.append(f"{'-'*60}")
        lines.append(f"  Decisions by reason:")
        for reason in sorted(by_reason):
            lines.append(f"    {reason:<22} {by_reason[reason]:>5}")
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csvs(results: List[Dict[str, Any]], full_csv: str, review_csv: str) -> None:
    cols = [
        "rel_key", "gt",
        "tesseract", "paddle", "qwen_vlm",
        "consensus", "correct",
        "flagged_for_review", "reason",
        "pred_path",
    ]

    def _row(r: Dict[str, Any]) -> List[Any]:
        return [
            r["rel_key"],
            "" if r["gt"] is None else r["gt"],
            r["votes"].get("tesseract", ""),
            r["votes"].get("paddle", ""),
            r["votes"].get("qwen_vlm", ""),
            "" if r["consensus"] is None else r["consensus"],
            "" if r["correct"] is None else int(r["correct"]),
            int(r["flagged_for_review"]),
            r["reason"],
            r["pred_path"],
        ]

    with open(full_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow(_row(r))

    flagged = [r for r in results if r["flagged_for_review"]]
    with open(review_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in flagged:
            w.writerow(_row(r))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODEL_TO_FILENAME_SLUG.keys()))
    parser.add_argument("--subset", default=None, metavar="TXT",
                        help="Path to a .txt file with one view-dir path per line "
                             "(e.g. inference_results/subset_eval.txt). Each view dir "
                             "is expanded to its 4 offset/fov/dist leaves; only those "
                             "are evaluated. Default: walk the whole pred root.")
    parser.add_argument("--pred-root", default=DEFAULT_PRED_ROOT)
    parser.add_argument("--hunyuan-extra-root", default=DEFAULT_HUNYUAN_EXTRA_ROOT)
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: <repo>/evaluation_results/main_ocr_eval/)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images (for smoke testing).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel images. Tesseract+Paddle run inside a per-image lock.")
    parser.add_argument(
        "--vllm-base-urls",
        default=DEFAULT_VLLM_BASE_URLS,
        metavar="URL[,URL,...]",
        help="One or more comma-separated vLLM endpoints. With multiple "
             "endpoints, requests are round-robined across them — useful when "
             "you launch N×TP=1 vLLM instances (one per GPU) instead of a "
             "single TP=N instance, since short-context OCR queries are "
             "throughput-bound and TP overhead dominates. "
             "Env: VLLM_BASE_URLS (preferred) or VLLM_BASE_URL.",
    )
    parser.add_argument("--vllm-model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--no-tesseract", action="store_true")
    parser.add_argument("--no-paddle", action="store_true")
    parser.add_argument("--no-qwen", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = args.output_dir or os.path.join(repo_root, "evaluation_results", "main_ocr_eval")
    os.makedirs(out_dir, exist_ok=True)

    subset_rel_keys: Optional[List[str]] = None
    if args.subset:
        subset_rel_keys = _read_subset_file(args.subset)
        print(f"Subset file: {args.subset}")
        print(f"  Expanded to {len(subset_rel_keys)} leaf rel_keys "
              f"({len(subset_rel_keys) // len(CAMERA_DIRS)} view-dirs x {len(CAMERA_DIRS)} cameras).")

    print(f"Discovering predictions for model={args.model}...")
    files = discover_pred_files(args.model, args.pred_root, args.hunyuan_extra_root,
                                subset_rel_keys=subset_rel_keys)
    if args.limit is not None:
        files = files[: args.limit]
    print(f"  Found {len(files)} prediction images.")
    if not files:
        sys.exit("No prediction images found. Check --pred-root / --model.")

    # Initialize backends once.
    print("Initializing OCR backends...")
    tesseract = None if args.no_tesseract else TesseractOCR()
    paddle = None if args.no_paddle else PaddleOCRBackend()
    qwen = None
    qwen_urls: List[str] = []
    if not args.no_qwen:
        qwen_urls = [u.strip() for u in args.vllm_base_urls.split(",") if u.strip()]
        if not qwen_urls:
            sys.exit("--vllm-base-urls produced an empty list.")
        qwen = QwenVLMBackend(qwen_urls, args.vllm_model)
        print(f"  Qwen VLM: {len(qwen_urls)} endpoint(s) (round-robin), model={args.vllm_model}")
        for u in qwen_urls:
            print(f"    - {u}")

    # tqdm if available
    try:
        from tqdm import tqdm
        progress = lambda it, **kw: tqdm(it, **kw)  # noqa: E731
    except ImportError:
        progress = lambda it, **kw: it  # noqa: E731

    # Prefetch GT (cheap, sequential).
    print("Loading ground truths...")
    gt_by_key: Dict[str, Optional[int]] = {}
    for rel_key, _ in progress(files, desc="  gt.json"):
        gt_by_key[rel_key] = load_gt(args.gt_root, rel_key)

    # Run OCR. Use threads — most time is local OCR + remote VLM I/O.
    # Note: PaddleOCR is not thread-safe in older versions. Serialize by using
    # workers=1 if you hit issues; default is 4 which works for most builds.
    print(f"Running OCR ({args.workers} worker(s))...")
    results: List[Dict[str, Any]] = [None] * len(files)  # type: ignore

    def _job(idx: int, rel_key: str, pred_path: str) -> Tuple[int, Dict[str, Any]]:
        return idx, evaluate_one(
            rel_key, pred_path, gt_by_key.get(rel_key),
            tesseract, paddle, qwen,
        )

    if args.workers <= 1:
        for i, (rel_key, pred_path) in progress(list(enumerate(files)),
                                                 total=len(files), desc="  ocr"):
            _, r = _job(i, rel_key, pred_path)
            results[i] = r
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_job, i, k, p) for i, (k, p) in enumerate(files)]
            for fut in progress(as_completed(futs), total=len(futs), desc="  ocr"):
                idx, r = fut.result()
                results[idx] = r

    # Aggregate + dump
    agg = aggregate(results)
    print(format_summary(args.model, agg))

    # Tag output files with the subset stem so subset and full runs don't clobber.
    suffix = ""
    if args.subset:
        subset_stem = os.path.splitext(os.path.basename(args.subset))[0]
        suffix = f"_{subset_stem}"

    json_path = os.path.join(out_dir, f"eval_main_ocr_{args.model}{suffix}.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model,
            "config": {
                "subset": args.subset,
                "pred_root": args.pred_root,
                "hunyuan_extra_root": args.hunyuan_extra_root if args.model == "hunyuan" else None,
                "gt_root": args.gt_root,
                "vllm_base_urls": qwen_urls if not args.no_qwen else None,
                "vllm_model": args.vllm_model if not args.no_qwen else None,
                "systems_used": [
                    name for name, on in (
                        ("tesseract", not args.no_tesseract),
                        ("paddle", not args.no_paddle),
                        ("qwen_vlm", not args.no_qwen),
                    ) if on
                ],
            },
            "aggregate": agg,
            "per_image": results,
        }, f, indent=2)
    print(f"  Full results: {json_path}")

    full_csv = os.path.join(out_dir, f"eval_main_ocr_{args.model}{suffix}.csv")
    review_csv = os.path.join(out_dir, f"eval_main_ocr_{args.model}{suffix}_review.csv")
    write_csvs(results, full_csv, review_csv)
    print(f"  Full CSV    : {full_csv}")
    print(f"  Review CSV  : {review_csv}  ({sum(1 for r in results if r['flagged_for_review'])} rows)")


if __name__ == "__main__":
    main()
