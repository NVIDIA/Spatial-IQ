# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


"""
Prepare SpatialIQ training data for NeMo RL.

Walks a dataset directory and produces train/val JSONL files suitable for
the SpatialIQ NeMo RL dataset class.

Usage:
    python prepare_data.py <data_dir> --output-dir <output_dir> [--tasks task_main] [--val-split 0.1]

Each JSONL line:
    {"image_path": "/abs/path/view.png", "question": "...", "answer": "5", "task_id": "task_main"}
"""

import argparse
import json
import os
import random
import re
import sys


# ---------------------------------------------------------------------------
# Task prompts (from the benchmark spec)
# ---------------------------------------------------------------------------

DEFINITIONS_BLOCK = """DEFINITIONS:
1. A COLUMN is a vertical stack of one or more blocks, where each block is directly supported by the block immediately beneath it. A single block also counts as a column.
2. A LAYER is a horizontal group of one or more blocks at the same vertical height. A single block also counts as a layer.
3. A VISIBLE BLOCK is a block with at least one face fully or partially visible.
4. A CLUSTER is a group of one or more blocks that are connected through face adjacency, possibly across multiple layers. Two blocks are considered connected if they share a face (i.e., are laterally adjacent in the 3D scene); diagonal contact does not count. A cluster includes all blocks that are connected directly or indirectly through such face contacts.
5. A SUPPORTING BLOCK is any block that is beneath a given block in the same column. A DIRECTLY SUPPORTING BLOCK is the supporting block in immediate contact beneath a given block. If a block is directly supported by the ground, then it has no supporting blocks.
6. A HIDDEN BLOCK is a block with no visible faces in the image. Any hidden block must be a supporting block of at least one visible block. Otherwise, such a block does not exist in a valid structure.
"""

CRITICAL_INSTRUCTION = """CRITICAL INSTRUCTION: You are undergoing a precision capability test. Answer the following questions using the absolute minimum number of words possible. Output only the exact value, count, or object name. Do not include full sentences, explanations, pleasantries, or context.
"""

TASK_PROMPTS = {
    "task1": CRITICAL_INSTRUCTION + "QUESTION: If the task is to count objects, what are those object(s) in the image?",
    "task2": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many blocks are in the top-most layer? Your answer should be a single integer ONLY.",
    "task3": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many columns are in the block structure? Your answer should be a single integer ONLY.",
    "task4": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many layers are in the block structure? Your answer should be a single integer ONLY.",
    "task5": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many visible blocks are there in the block structure? Your answer should be a single integer ONLY.",
    "task6": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many VISIBLE blocks are directly supporting blocks of block(s) in the top-most layer? Your answer should be a single integer ONLY.",
    "task7": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many VISIBLE blocks are in the column(s) supporting the block(s) in the top-most layer? Exclude the block(s) themselves that are in the top-most layer. Your answer should be a single integer ONLY.",
    "task8": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many blocks are HIDDEN blocks that are directly supporting blocks of visible block(s)? Your answer should be a single integer ONLY.",
    "task9": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many distinct clusters are there in the block structure? Your answer should be a single integer ONLY.",
    "task10": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: You are given two images that are two different views of the same block structure. How many blocks are visible in the second view but are NOT visible in the first view? Your answer should be a single integer ONLY.",
    "task_main": CRITICAL_INSTRUCTION + DEFINITIONS_BLOCK + "QUESTION: How many TOTAL blocks are present in the block structure? Your answer should be a single integer ONLY.",
}

COT_TARGET_TEMPLATE = """<think>
1) The object type making up the structure is: {task1}.
2) The number of columns in the object structure is: {task3}.
3) The number of layers in the object structure is: {task4}.
4) The number of objects in the top-most layer is: {task2}.
5) The number of distinct clusters is: {task9}.
6) Given {task3} columns, {task4} layers, {task2} objects in the top-most layer, and {task9} clusters, the number of visible objects is: {task5}.
7) The number of visible objects in the columns containing the top-most object(s) (excluding the top-most objects themselves) is: {task7}.
8) The number of visible objects directly supporting the top-most object(s) is: {task6}.
9) The number of visible objects in direct contact directly above a hidden object is: {task8}.
10) Given {task2} objects in top-most columns, {task6} supporting objects, and {task8} objects above hidden ones, the number of hidden objects is: {num_hidden_blocks}.
The total is {task5} + {num_hidden_blocks} = {task_main}.
</think>
<answer>{task_main}</answer>"""

COT_REQUIRED_KEYS = ["task1", "task2", "task3", "task4", "task5", "task6",
                      "task7", "task8", "task9", "num_hidden_blocks", "task_main"]


def collect_view_dirs(data_dir: str) -> list[str]:
    """Collect view directories from either supported dataset layout."""
    data_dir = os.path.abspath(data_dir)
    view_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if "gt.json" in files and "view.png" in files:
            view_dirs.append(root)

    view_dirs.sort()
    print(f"Found {len(view_dirs)} views in {data_dir}")
    return view_dirs


def get_sample_group_id(data_dir: str, view_dir: str) -> str:
    """Group all views from the same sample into the same split."""
    data_dir = os.path.abspath(data_dir)
    current_dir = os.path.abspath(view_dir)

    while current_dir != data_dir and current_dir != os.path.dirname(current_dir):
        if re.fullmatch(r"sample_\d+", os.path.basename(current_dir)):
            return os.path.relpath(current_dir, data_dir)
        current_dir = os.path.dirname(current_dir)

    # Fall back to the view directory if the expected sample_* ancestor is absent.
    return os.path.relpath(os.path.abspath(view_dir), data_dir)


def split_sample_groups(
    sample_group_ids: list[str], val_split: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split by sample group so related views/tasks stay together."""
    sample_group_ids = list(sample_group_ids)
    random.seed(seed)
    random.shuffle(sample_group_ids)

    if not sample_group_ids:
        return [], []
    if val_split <= 0:
        return sample_group_ids, []
    if len(sample_group_ids) == 1:
        print(
            "  WARNING: only 1 sample group found; validation split will be empty "
            "to avoid train/val leakage"
        )
        return sample_group_ids, []

    n_val = max(1, int(len(sample_group_ids) * val_split))
    n_val = min(n_val, len(sample_group_ids) - 1)
    val_group_ids = sample_group_ids[:n_val]
    train_group_ids = sample_group_ids[n_val:]
    return train_group_ids, val_group_ids


def collect_examples(
    view_dirs: list[str],
    task_ids: list[str],
    cot: bool = False,
    mask_final_answer: str | None = None,
) -> list[dict]:
    """Collect training examples from a preselected set of views.

    If cot=True, produces one example per view using the task_main question
    with the CoT target template filled from ground truth values.

    If mask_final_answer is set (CoT mode only), the final answer {task_main}
    is replaced by the given placeholder string in BOTH the "The total is
    a + b = ..." line and the <answer>...</answer> tag, so the SFT target
    teaches the reasoning format and sub-task values but never the final
    answer. ground_truth_number still carries the real value for eval/RL.
    """
    examples = []
    skipped_cot = 0

    for view_dir in view_dirs:
        gt_path = os.path.join(view_dir, "gt.json")
        view_png = os.path.join(view_dir, "view.png")

        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        if cot:
            # Check all required keys are present
            missing = [k for k in COT_REQUIRED_KEYS if gt.get(k) is None]
            if missing:
                skipped_cot += 1
                continue

            cot_answer = COT_TARGET_TEMPLATE.format(
                task1=gt["task1"],
                task2=gt["task2"],
                task3=gt["task3"],
                task4=gt["task4"],
                task5=gt["task5"],
                task6=gt["task6"],
                task7=gt["task7"],
                task8=gt["task8"],
                task9=gt["task9"],
                num_hidden_blocks=gt["num_hidden_blocks"],
                task_main=mask_final_answer if mask_final_answer else gt["task_main"],
            )
            examples.append({
                "image_path": view_png,
                "question": TASK_PROMPTS["task_main"],
                "answer": cot_answer,
                "task_id": "task_main",
                "ground_truth_number": str(gt["task_main"]),
            })
        else:
            for task_id in task_ids:
                # task10/task11 require two views — not supported in this single-image pipeline
                if task_id in ("task10", "task11"):
                    continue

                if task_id not in TASK_PROMPTS:
                    print(f"  WARNING: unknown task_id '{task_id}', skipping", file=sys.stderr)
                    continue

                answer = gt.get(task_id)
                if answer is None:
                    continue

                examples.append({
                    "image_path": view_png,
                    "question": TASK_PROMPTS[task_id],
                    "answer": str(answer),
                    "task_id": task_id,
                })

    if cot and skipped_cot:
        print(f"  WARNING: skipped {skipped_cot} views missing CoT keys", file=sys.stderr)

    return examples


def write_jsonl(examples: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SpatialIQ data for NeMo RL training")
    parser.add_argument("data_dir", help="Root data directory containing sample_*/view_*/ or level_*/sample_*/view_*/")
    parser.add_argument("--output-dir", default=None, help="Output directory for JSONL files (default: <data_dir>)")
    parser.add_argument("--tasks", nargs="+", default=["task_main"],
                        help="Task IDs to include (default: task_main)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of sample groups for validation (default: 0.1; set 0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--cot", action="store_true",
                        help="Generate CoT examples: one per view with task_main question and "
                             "chain-of-thought answer filled from ground truth")
    parser.add_argument("--cot-mask-final-answer", action="store_true",
                        help="CoT mode only: replace the final answer with --mask-placeholder in "
                             "the total line and <answer> tag (format-only supervision)")
    parser.add_argument("--mask-placeholder", default="X",
                        help="Placeholder string used by --cot-mask-final-answer (default: X)")
    args = parser.parse_args()

    if args.cot_mask_final_answer and not args.cot:
        print("ERROR: --cot-mask-final-answer requires --cot", file=sys.stderr)
        sys.exit(2)

    if not 0.0 <= args.val_split <= 1.0:
        print("ERROR: --val-split must be between 0 and 1", file=sys.stderr)
        sys.exit(2)

    MULTI_VIEW_TASKS = {"task10", "task11"}
    skipped = MULTI_VIEW_TASKS & set(args.tasks)
    if skipped:
        print(f"  WARNING: {skipped} require multi-view and will be skipped (single-image pipeline)", file=sys.stderr)

    output_dir = args.output_dir or args.data_dir
    data_dir = os.path.abspath(args.data_dir)
    view_dirs = collect_view_dirs(data_dir)
    if not view_dirs:
        print("ERROR: no examples found!", file=sys.stderr)
        sys.exit(1)

    sample_group_to_views = {}
    for view_dir in view_dirs:
        sample_group_id = get_sample_group_id(data_dir, view_dir)
        sample_group_to_views.setdefault(sample_group_id, []).append(view_dir)

    sample_group_ids = sorted(sample_group_to_views)
    train_group_ids, val_group_ids = split_sample_groups(
        sample_group_ids, args.val_split, args.seed
    )

    train_view_dirs = [
        view_dir
        for sample_group_id in train_group_ids
        for view_dir in sample_group_to_views[sample_group_id]
    ]
    val_view_dirs = [
        view_dir
        for sample_group_id in val_group_ids
        for view_dir in sample_group_to_views[sample_group_id]
    ]

    mask = args.mask_placeholder if args.cot_mask_final_answer else None
    train_examples = collect_examples(train_view_dirs, args.tasks, cot=args.cot, mask_final_answer=mask)
    val_examples = collect_examples(val_view_dirs, args.tasks, cot=args.cot, mask_final_answer=mask)
    total_examples = len(train_examples) + len(val_examples)

    if total_examples == 0:
        print("ERROR: no examples found for the requested tasks!", file=sys.stderr)
        sys.exit(1)
    if not train_examples:
        print(
            "ERROR: training split is empty after grouping/splitting; "
            "use a different --seed, reduce --val-split, or pick tasks with labels",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.val_split > 0 and not val_examples:
        print(
            "  WARNING: validation split has 0 examples after grouping/splitting; "
            "training will run without validation",
            file=sys.stderr,
        )

    print(
        f"Collected {total_examples} total examples from "
        f"{len(sample_group_ids)} sample groups"
    )
    print(
        f"  Train groups: {len(train_group_ids)}, Val groups: {len(val_group_ids)}"
    )

    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")

    write_jsonl(train_examples, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(val_examples, os.path.join(output_dir, "val.jsonl"))

    print("Done!")


if __name__ == "__main__":
    main()
