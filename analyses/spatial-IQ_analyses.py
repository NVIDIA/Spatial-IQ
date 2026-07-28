#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Public Spatial-IQ analysis and figure-generation script.

This script is organized around the paper outputs in
``Spatial-IQ: Deconstructing Spatial Intelligence via Hierarchical Capability
Tests``:

1. Load raw ``eval_raw_*.csv`` files from ``inference_results/`` by default.
2. Export analysis-ready CSV tables with paper figure/table numbers.
3. Render figures only from those exported CSV tables.

The raw CSV interface is kept flexible so new model CSVs can be dropped into
``inference_results/`` and picked up by discovery without touching the plotting
code.

Expected input filenames follow a small set of conventions:
``eval_raw_*_text.csv`` for text responses, ``eval_raw_*_mcq.csv`` for MCQ
responses, ``eval_raw_image_*.csv`` for image-output responses, and
``eval_raw_human_frq.csv`` for human free-response text data. If a requested
group has no matching files, the script emits a warning with the expected
pattern rather than silently producing empty outputs.

The implementation keeps the public-release path deliberately explicit:
analysis functions produce reviewer-facing CSVs with counts, uncertainty, and
test statistics; plotting functions then consume only those CSVs. This mirrors
the paper workflow and makes it easier to audit a figure without re-reading the
raw model outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Matplotlib may try to create a user-level cache on first import. A temporary
# cache keeps this script portable on clusters, CI, and read-only home folders.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "spatialiq-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy import stats

# Appendix F hierarchy-dependency statistics live in a sibling module so they can
# also be run standalone with a --validate printout. ``sys.path[0]`` is this
# file's directory, so the plain import resolves from any working directory.
import hierarchy_dependency as hdep

warnings.filterwarnings("ignore", category=FutureWarning)

# Common analysis input type used throughout this file.
#
# Keys are public model identifiers inferred from CSV filenames, such as
# ``"human"``, ``"gpt"``, ``"qwen4CQ"``, or ``"dapo-32b-tight"``. Values are
# normalized raw evaluation DataFrames: one row per evaluated scene/request, plus
# binary task-correctness columns, optional numeric prediction columns, and
# derived camera/difficulty metadata.
ModelResults = dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# Paper output registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperOutput:
    """Manifest entry linking one paper output to its reproducible artifact.

    Static figures are included so the manifest matches the PDF numbering even
    when the figure is drawn externally rather than generated from CSV results.
    """

    paper_id: str
    kind: str
    title: str
    status: str
    analysis_csv: str | None = None
    figure_file: str | None = None
    notes: str = ""


# The registry is the single source of truth for public paper numbering. It
# keeps figure/table names stable even if helper functions or source CSVs change.
PAPER_OUTPUTS: list[PaperOutput] = [
    PaperOutput("Figure 1", "static", "Motivating hierarchy example", "static"),
    PaperOutput("Figure 2", "static", "Task suite and developmental tags", "static"),
    PaperOutput("Figure 3", "static", "Pre-specified hierarchy relations", "static"),
    PaperOutput("Figure 4", "figure", "Text per-task accuracy", "generated", "tables/table_06_text_task_accuracy.csv", "figures/figure_04_text_task_accuracy.png"),
    PaperOutput("Figure 5", "figure", "5CQ wrong-answer preferences", "generated", "tables/figure_05_mcq5_wrong_answer_preferences.csv", "figures/figure_05_mcq5_wrong_answer_preferences.png"),
    PaperOutput("Figure 6", "figure", "Text difficulty controls", "generated", "tables/table_09_text_difficulty_bins.csv", "figures/figure_06_text_difficulty_controls.png"),
    PaperOutput("Figure 7", "figure", "Text summation mechanism", "generated", "tables/figure_07_text_summation_mechanism.csv", "figures/figure_07_text_summation_mechanism.png"),
    PaperOutput("Figure 8", "static", "MCQ option-grid examples", "static"),
    PaperOutput("Figure 9", "static", "Text human-study interface", "static"),
    PaperOutput("Figure 10", "static", "MCQ human-study interface", "static"),
    PaperOutput("Figure 11", "static", "Image-editing human-scoring interface", "static"),
    PaperOutput("Figure 12", "static", "Developmental-framework summary", "static"),
    PaperOutput("Figure 13", "figure", "Mental-rotation dependency correlations", "generated", "tables/figure_13_text_mental_rotation_correlations.csv", "figures/figure_13_text_mental_rotation_correlations.png"),
    PaperOutput("Figure 14", "figure", "Mental-rotation bundle versus counting mechanism", "generated", "tables/figure_14_text_bundle_correlations.csv", "figures/figure_14_text_bundle_correlations.png"),
    PaperOutput("Figure 15", "figure", "Mixed-CQ chance-adjusted task accuracy", "generated", "tables/figure_15_mcq345_task_accuracy.csv", "figures/figure_15_mcq345_task_accuracy.png"),
    PaperOutput("Figure 16", "figure", "Mixed-CQ wrong-answer preferences", "generated", "tables/figure_16_mcq345_wrong_answer_preferences.csv", "figures/figure_16_mcq345_wrong_answer_preferences.png"),
    PaperOutput("Figure 17", "figure", "Image-output per-task accuracy", "generated", "tables/table_08_image_task_accuracy.csv", "figures/figure_17_image_task_accuracy.png"),
    PaperOutput("Figure 18", "figure", "Text camera-parameter effects", "generated", "tables/figure_18_text_camera_effects.csv", "figures/figure_18_text_camera_effects.png"),
    PaperOutput("Figure 19", "figure", "Image-output difficulty controls", "generated", "tables/figure_19_image_difficulty_bins.csv", "figures/figure_19_image_difficulty_controls.png"),
    PaperOutput("Figure 20", "figure", "Text object-conditioned accuracy", "generated", "tables/figure_20_text_object_accuracy.csv", "figures/figure_20_text_object_accuracy.png"),
    PaperOutput("Figure 21", "figure", "Text object and difficulty slices", "generated", "tables/figure_21_text_object_difficulty_slices.csv", "figures/figure_21_text_object_difficulty_slices.png"),
    PaperOutput("Figure 22", "figure", "Text composition diagnostics", "generated", "tables/figure_22_text_composition_diagnostics.csv", "figures/figure_22_text_composition_diagnostics.png"),
    PaperOutput("Figure 23", "figure", "MCQ summation mechanism", "generated", "tables/figure_23_mcq345_summation_mechanism.csv", "figures/figure_23_mcq345_summation_mechanism.png"),
    PaperOutput("Figure 24", "figure", "Image-output summation mechanism", "generated", "tables/figure_24_image_summation_mechanism.csv", "figures/figure_24_image_summation_mechanism.png"),
    PaperOutput("Figure 25", "figure", "Trained-text per-task accuracy", "generated", "tables/figure_25_trained_text_task_accuracy.csv", "figures/figure_25_trained_text_task_accuracy.png"),
    PaperOutput("Figure 26", "figure", "Trained-text object-conditioned accuracy", "generated", "tables/figure_26_trained_text_object_accuracy.csv", "figures/figure_26_trained_text_object_accuracy.png"),
    PaperOutput("Figure 27", "figure", "Trained-text summation mechanism", "generated", "tables/figure_27_trained_text_summation_mechanism.csv", "figures/figure_27_trained_text_summation_mechanism.png"),
    PaperOutput("Figure 28", "figure", "Trained-text composition diagnostics", "generated", "tables/figure_28_trained_text_composition_diagnostics.csv", "figures/figure_28_trained_text_composition_diagnostics.png"),
    PaperOutput("Table 1", "table", "Text modality evaluative summary", "generated", "tables/table_01_text_summary.csv"),
    PaperOutput("Table 2", "table", "MCQ modality evaluative summary", "generated", "tables/table_02_mcq_summary.csv"),
    PaperOutput("Table 3", "table", "Image-editing modality evaluative summary", "generated", "tables/table_03_image_summary.csv"),
    PaperOutput("Table 4", "table", "Trained-checkpoint evaluative summary", "generated", "tables/table_04_trained_text_summary.csv"),
    PaperOutput("Table 5", "table", "Task-specific query text", "generated", "tables/table_05_task_queries.csv"),
    PaperOutput("Table 6", "table", "Text per-task accuracy", "generated", "tables/table_06_text_task_accuracy.csv"),
    PaperOutput("Table 7", "table", "5CQ per-task accuracy", "generated", "tables/table_07_mcq5_task_accuracy.csv"),
    PaperOutput("Table 8", "table", "Image-output per-task accuracy", "generated", "tables/table_08_image_task_accuracy.csv"),
    PaperOutput("Table 9", "table", "Text difficulty-bin accuracy", "generated", "tables/table_09_text_difficulty_bins.csv"),
    # Appendix F ("Full Hierarchy Dependency Analysis") is reported in prose in the
    # paper; these tables carry the underlying numbers. Computed by
    # hierarchy_dependency.py, which the analysis stage calls directly.
    PaperOutput("Appendix F", "table", "Per-edge and joint within-scene lifts", "generated", "tables/tableF_within_scene_lift_text.csv"),
    PaperOutput("Appendix F", "table", "Joint relation integrity per responder", "generated", "tables/tableF_joint_integrity_text.csv",
                notes="variant column: prespecified_s8 (Fig. 3) and submitted_s7 (as computed for the submitted appendix)"),
    PaperOutput("Appendix F", "table", "Bundle-to-Object-Counting within-scene coupling", "generated", "tables/tableF_bundle_coupling_text.csv"),
    PaperOutput("Appendix F", "table", "Responder-controlled lift (two-way scene + responder FE)", "generated", "tables/tableF_responder_controlled_lift_text.csv",
                notes="subset column: all responders and models-only (human baseline dropped)"),
    PaperOutput("Appendix F", "table", "Per-responder marginal lift with cell sizes and exclusions", "generated", "tables/tableF_per_responder_lift_text.csv"),
    PaperOutput("Appendix F", "table", "Bundle coupling, scene versus scene+responder demeaned", "generated", "tables/tableF_bundle_coupling_twoway_text.csv"),
    PaperOutput("Table 6", "table", "All-pairs McNemar significance on Object Counting", "generated", "tables/table6_mcnemar_object_counting_text.csv"),
    PaperOutput("Figure 18", "table", "Within-structure paired camera effects per task", "generated", "tables/figF18_camera_within_structure_text.csv",
                notes="paired within-structure test with BH-FDR; supersedes the unpaired camera_effect_table variant"),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Internal CSV task names come from the original development code. The order
# below follows the paper narrative: primitive perception, visible counting,
# support/occlusion reasoning, object counting, and mental rotation.
TASK_ORDER = [
    "task1",
    "task9",
    "task3",
    "task4",
    "task5",
    "task2",
    "task6",
    "task7",
    "task8",
    "main",
    "task10",
    "task11",
]

TASK_DISPLAY = {
    "task1": "Object\ncategory",
    "task9": "Clusters",
    "task3": "Columns",
    "task4": "Layers",
    "task5": "Visible\ncount",
    "task2": "Top\nlayer",
    "task6": "Direct\nsupport",
    "task7": "Support\ncolumn",
    "task8": "Hidden\ncount",
    "main": "Object\ncount",
    "task10": "Mental\nrotation 90",
    "task11": "Mental\nrotation 180",
}

TASK_LONG_NAME = {
    "task1": "Object Categorization",
    "task9": "Cluster Count",
    "task3": "Column Count",
    "task4": "Layer Count",
    "task5": "Visible Object Count",
    "task2": "Top Layer",
    "task6": "Direct Support",
    "task7": "Support Column",
    "task8": "Hidden Object Count",
    "main": "Object Counting",
    "task10": "Mental Rotation 90",
    "task11": "Mental Rotation 180",
}

TASK_QUERY_TEXT = {
    "task1": "What object category composes the stacked structure?",
    "task9": "How many non-laterally-adjacent clusters of objects are present?",
    "task3": "How many columns are in the stacked structure?",
    "task4": "How many horizontal layers are in the stacked structure?",
    "task5": "How many visible objects are present?",
    "task2": "How many objects are in the top-most layer?",
    "task6": "How many visible objects directly support the top-most object(s)?",
    "task7": "How many visible objects are in columns supporting the top-most object(s)?",
    "task8": "How many hidden objects must exist to support the visible structure?",
    "main": "How many objects are in the stacked structure in total?",
    "task10": "Which objects are visible in the adjacent view but not the reference view?",
    "task11": "Which objects are visible in the opposite view but not the reference view?",
}

# Public task IDs in the paper are not the same as the raw CSV column numbers.
# ``source_task`` preserves the original column name, while ``task`` uses this
# mapping so exported tables match the manuscript and appendix.
PAPER_SUBTASK_NUMBER = {
    "task1": 1,
    "task9": 2,
    "task3": 3,
    "task4": 4,
    "task5": 5,
    "task2": 6,
    "task6": 7,
    "task7": 8,
    "task8": 9,
}

DIFFICULTY_FACTORS = ["total_blocks", "num_hidden_blocks", "num_layers", "num_columns", "fill_ratio"]

# ``perspective`` is derived from the field-of-view code embedded in ``view``.
# The paper labels this as a perspective effect rather than a FoV effect.
CAMERA_FACTORS = ["perspective", "offset"]
MODEL_FAMILY_ORDER = ["human", "claude", "qwen", "gemini", "gpt", "glm", "qwen3b", "vla-0", "kimi"]

# Color families are shared across figures so readers can track the same model
# across modalities. MCQ choice-count variants and 7B checkpoints are lightened
# from their base family color in ``model_color``.
MODEL_COLOR_MAP = {
    "human": "#212121",
    "claude": "#1d6fbb",
    "qwen": "#e67e22",
    "gemini": "#27ae60",
    "gpt": "#c0392b",
    "glm": "#8e44ad",
    "qwen3b": "#16a085",
    "vla-0": "#d81b60",
    "kimi": "#795548",
    "hunyuan": "#2563eb",
    "gemini-image": "#27ae60",
    "qwen-image-edit": "#e67e22",
    "dapo": "#2ca58d",
    "sft-cot": "#cc503e",
    "sft-plain": "#8d6e63",
    "qwen2.5-vl": "#f28e2b",
}

PAPER_MODEL_COLOR_MAP = {
    "human": "#333333",
    "claude": "#2F7AC0",
    "qwen": "#E88834",
    "gemini": "#38B46C",
    "gpt": "#C5493C",
    "glm": "#9753B3",
    "qwen3b": "#29A78E",
    "vla-0": "#DB2D6C",
    "kimi": "#836256",
}

ROTATION_HIGHLIGHT_COLOR = "#B8860B"
ROTATION_HIGHLIGHT_ALPHA = 0.10
WRONG_ANSWER_CMAP = "Reds"
WRONG_ANSWER_VMAX = 0.40

E1A_CMAP = LinearSegmentedColormap.from_list(
    "spatialiq_yellow_red",
    ["#fff7bc", "#fec44f", "#fb8c00", "#ef4444", "#991b1b"],
)
PAPER_E1A_CMAP = plt.get_cmap("YlGn")

# Figure 14 compares cross-view mental-rotation signals against the single-view
# counting chain used elsewhere in the paper.
CROSSVIEW_BUNDLE_VARIANTS = {
    "Mental Rotation 90": ["task10"],
    "Mental Rotation 180": ["task11"],
    "Mental Rotation bundle": ["task10", "task11"],
    "Referential chain": ["task2", "task6", "task7"],
    "Visible + hidden mechanism": ["task5", "task8"],
}

OBJECT_DISPLAY = {
    "cube": "Cube",
    "factory_box": "Factory box",
    "soup_can": "Soup can",
    "spam": "Ham can",
}

OBJECT_COLORS = {
    "cube": "#2563eb",
    "factory_box": "#dc2626",
    "soup_can": "#059669",
    "spam": "#9333ea",
}


# ---------------------------------------------------------------------------
# Small statistical helpers
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson binomial confidence interval for accuracy estimates."""

    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def binom_se(p: float, n: int) -> float:
    """Return the standard error of a binomial proportion."""

    if n <= 0 or pd.isna(p):
        return np.nan
    return float(np.sqrt(p * (1 - p) / n))


def safe_corr(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    """Pearson correlation with guards for missing or constant vectors."""

    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3 or len(np.unique(x_arr[mask])) < 2 or len(np.unique(y_arr[mask])) < 2:
        return np.nan, np.nan
    r, p = stats.pearsonr(x_arr[mask], y_arr[mask])
    return float(r), float(p)


def fixed_effect_corr(df: pd.DataFrame, x_col: str, y_col: str, fe_col: str = "view") -> tuple[float, float, int]:
    """Estimate a within-scene correlation by demeaning within ``fe_col``.

    This supports Figure 14, where we want correlations that are less dominated
    by scene identity and more sensitive to variation across model responses.
    """

    if fe_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        if x_col in df.columns and y_col in df.columns:
            r, p = safe_corr(df[x_col], df[y_col])
            return r, p, int(df[[x_col, y_col]].dropna().shape[0])
        return np.nan, np.nan, 0
    sub = df[[fe_col, x_col, y_col]].dropna().copy()
    if len(sub) < 3:
        return np.nan, np.nan, int(len(sub))
    sub["_x_dm"] = sub[x_col].astype(float) - sub.groupby(fe_col)[x_col].transform("mean")
    sub["_y_dm"] = sub[y_col].astype(float) - sub.groupby(fe_col)[y_col].transform("mean")
    r, p = safe_corr(sub["_x_dm"], sub["_y_dm"])
    return r, p, int(len(sub))


def choice_count(model: str) -> int:
    """Infer the number of MCQ choices from a model name such as ``qwen3CQ``."""

    match = re.search(r"([345])cq\b", model.lower())
    if match:
        return int(match.group(1))
    return 5


def chance_level(group: str, model: str) -> float | None:
    """Return the random-choice baseline for MCQ groups, otherwise ``None``."""

    if group.startswith("mcq"):
        return 1.0 / choice_count(model)
    return None


def chance_adjust(value: float, group: str, model: str) -> float:
    """Chance-adjust MCQ accuracy as (observed - chance) / (1 - chance)."""

    chance = chance_level(group, model)
    if chance is None or pd.isna(value):
        return value
    return (float(value) - chance) / (1.0 - chance)


def chance_adjust_ci(lo: float, hi: float, group: str, model: str) -> tuple[float, float]:
    return chance_adjust(lo, group, model), chance_adjust(hi, group, model)


def p_above_chance(k: int, n: int, group: str, model: str) -> float:
    """One-sided binomial test that accuracy exceeds the MCQ chance level."""

    chance = chance_level(group, model)
    if chance is None or n <= 0:
        return np.nan
    return float(stats.binomtest(k, n, chance, alternative="greater").pvalue)


def sig_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def mean_ignore_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def relation_integrity(df: pd.DataFrame, parent: str, child: str, group: str, model: str) -> float:
    """Compute appendix-style relation integrity for one hierarchy edge.

    Appendix F defines relation integrity as ``I(P -> C) = P(P=1 | C=1)``.
    MCQ summary tables place every column on a chance-adjusted scale, so for
    MCQ groups we apply the same chance adjustment to the conditional score.
    """

    if parent not in df.columns or child not in df.columns:
        return np.nan
    sub = df[[parent, child]].dropna()
    sub = sub[sub[child] == 1]
    if sub.empty:
        return np.nan
    score = float(sub[parent].mean())
    return chance_adjust(score, group, model) if group.startswith("mcq") else score


# Two of the manuscript's summary tables take the relation conditional in the
# other direction. Table 1 (text) and Table 3 (image) score a relation as
# P(prerequisite correct | dependent correct), the definition given in Section 5
# and Appendix F. Table 2 (multiple choice) and the trained-checkpoint rows of
# Table 4 instead score P(dependent correct | prerequisite correct). Each table
# is reproduced here as published; the revision unifies them on the Section 5
# definition. Table 4's human row is the text-modality reference row, carried
# over from Table 1, and so follows the text rule.
def relation_direction(group: str, model: str) -> str:
    """Which way round the relation conditional is taken for a summary row."""

    if group == "mcq345":
        return "dependent_given_prerequisite"
    if group == "trained_text" and model != "human":
        return "dependent_given_prerequisite"
    return "prerequisite_given_dependent"


def _edge_score(df: pd.DataFrame, parent: str, child: str, group: str, model: str, direction: str) -> float:
    given, scored = (parent, child) if direction == "dependent_given_prerequisite" else (child, parent)
    return relation_integrity(df, scored, given, group, model)


def chain_integrity(df: pd.DataFrame, edges: list[tuple[str, str]], group: str, model: str) -> float:
    """Referential-chain integrity across the chain's edges.

    Under the Section 5 definition the chain's edges have different numbers of
    testable rows, so the score is pooled across them -- one minus the share of
    violations among all rows where an edge's dependent task is correct -- rather
    than averaged per edge. The dependent-conditioned tables average instead.
    """

    direction = relation_direction(group, model)
    if direction == "dependent_given_prerequisite":
        scores = [_edge_score(df, parent, child, group, model, direction) for parent, child in edges]
        return mean_ignore_nan(scores)
    testable = violations = 0
    for parent, child in edges:
        if parent not in df.columns or child not in df.columns:
            continue
        sub = df[[parent, child]].dropna()
        sub = sub[sub[child] == 1]
        testable += len(sub)
        violations += int((sub[parent] == 0).sum())
    if testable == 0:
        return np.nan
    score = 1.0 - violations / testable
    return chance_adjust(score, group, model) if group.startswith("mcq") else score


def support_integrity(df: pd.DataFrame, edges: list[tuple[str, str]], group: str, model: str) -> float:
    """Support-hierarchy integrity: the mean of the relation's per-edge scores.

    Under the Section 5 definition both edges of a support relation condition on
    the same dependent task, so they are estimable together or not at all, and a
    relation with an undefined edge is reported as undefined rather than as a
    score derived from the remaining edge. The dependent-conditioned tables
    condition on the prerequisites, which are estimable separately, so there an
    undefined edge is dropped and the remaining edge carries the relation.
    """

    direction = relation_direction(group, model)
    scores = [_edge_score(df, parent, child, group, model, direction) for parent, child in edges]
    if direction == "dependent_given_prerequisite":
        return mean_ignore_nan(scores)
    if not scores or any(pd.isna(s) for s in scores):
        return np.nan
    return float(np.mean(scores))


def summation_mechanism(df: pd.DataFrame, group: str, model: str) -> float:
    """Summation Mechanism column.

    Where numeric predictions exist (text-like modalities) this is the rate at
    which a responder's total-count error decomposes into its visible- and
    hidden-count errors, i.e. ``(M-M*) - [(V-V*) + (H-H*)] == 0``. Working on
    errors rather than on the raw identity ``M == V + H`` matters because the
    hidden sub-task counts visible objects resting on an occluded one, so the
    raw identity does not hold for every scene and a perfect responder would
    otherwise be capped below 1. Multiple-choice and image-editing answers are
    not counts that can be composed this way, so for those modalities the column
    reports ``P(Object Counting correct | Visible and Hidden correct)`` instead.
    """

    pred_cols = ["task5_pred", "task5_gt", "task8_pred", "task8_gt", "main_pred", "main_gt"]
    if not group.startswith("mcq") and set(pred_cols).issubset(df.columns):
        s = df[pred_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if s.empty:
            return np.nan
        composed = (s["task5_pred"] - s["task5_gt"]) + (s["task8_pred"] - s["task8_gt"])
        residual = (s["main_pred"] - s["main_gt"]) - composed
        return float((residual == 0).mean())
    if not {"task5", "task8", "main"}.issubset(df.columns):
        return np.nan
    sub = df[["task5", "task8", "main"]].dropna()
    sub = sub[(sub["task5"] == 1) & (sub["task8"] == 1)]
    if sub.empty:
        return np.nan
    score = float(sub["main"].mean())
    return chance_adjust(score, group, model) if group.startswith("mcq") else score


def object_display_name(obj: str) -> str:
    return OBJECT_DISPLAY.get(str(obj), str(obj).replace("_", " ").title())


def family_name(model: str) -> str:
    m = model.lower()
    if m == "human":
        return "human"
    if m.startswith("qwen2.5-vl"):
        return "qwen2.5-vl"
    if m.startswith("sft-") and "-cot-" in m:
        return "sft-cot"
    if m.startswith("sft-") and "-plain-" in m:
        return "sft-plain"
    if m.startswith("dapo"):
        return "dapo"
    for family in MODEL_FAMILY_ORDER:
        if m.startswith(family):
            return family
    if "image" in m and "qwen" in m:
        return "qwen-image-edit"
    if "image" in m and "gemini" in m:
        return "gemini-image"
    return m


def model_color(model: str) -> str:
    """Return the stable plotting color for a model or model variant."""

    family = family_name(model)
    base = MODEL_COLOR_MAP.get(family, "#6b7280")
    if re.search(r"3cq\b", model.lower()):
        return blend_color(base, "#ffffff", 0.60)
    if re.search(r"4cq\b", model.lower()):
        return blend_color(base, "#ffffff", 0.38)
    if "7b" in model.lower():
        return blend_color(base, "#ffffff", 0.50)
    return base


def paper_model_color(model: str) -> str:
    """Return the notebook-style model color for paper-specific replacement figures."""

    family = family_name(model)
    base = PAPER_MODEL_COLOR_MAP.get(family, model_color(model))
    if re.search(r"3cq\b", model.lower()):
        return blend_color(base, "#ffffff", 0.60)
    if re.search(r"4cq\b", model.lower()):
        return blend_color(base, "#ffffff", 0.38)
    if "7b" in model.lower():
        return blend_color(base, "#ffffff", 0.50)
    return base


def blend_color(color: str, target: str, amount: float) -> str:
    c = np.array(matplotlib.colors.to_rgb(color))
    t = np.array(matplotlib.colors.to_rgb(target))
    mixed = (1 - amount) * c + amount * t
    return matplotlib.colors.to_hex(mixed)


def display_model(model: str) -> str:
    """Convert raw file/model identifiers into compact figure labels."""

    m = model.lower()
    if m == "human":
        return "Human"
    if m.startswith("gpt"):
        return "GPT" + model[3:].upper().replace("CQ", "CQ")
    if m.startswith("glm"):
        return "GLM" + model[3:].upper().replace("CQ", "CQ")
    if m.startswith("qwen3b"):
        return "Qwen3B"
    if m.startswith("qwen") and "cq" in m:
        return "Qwen " + re.search(r"([345])cq", m).group(1) + "CQ"
    if m.startswith("claude") and "cq" in m:
        return "Claude " + re.search(r"([345])cq", m).group(1) + "CQ"
    if m.startswith("gemini") and "cq" in m:
        return "Gemini " + re.search(r"([345])cq", m).group(1) + "CQ"
    if m in {"claude", "qwen", "gemini", "kimi"}:
        return m.capitalize()
    if m == "vla-0":
        return "VLA-0"
    if m == "qwen-image-edit":
        return "Qwen Image Edit"
    if m == "gemini-image":
        return "Gemini Image"
    return model


def model_sort_key(model: str, score: float | None = None) -> tuple:
    """Sort models by performance first, then by family for visual stability."""

    if score is not None and not pd.isna(score):
        primary = -float(score)
    else:
        primary = 0.0
    family = family_name(model)
    family_rank = MODEL_FAMILY_ORDER.index(family) if family in MODEL_FAMILY_ORDER else len(MODEL_FAMILY_ORDER)
    cq = choice_count(model) if "cq" in model.lower() else 99
    return (primary, 0 if model.lower() == "human" else 1, family_rank, cq, model.lower())


def label_from_choice_type(raw: str) -> str:
    """Clean raw MCQ error labels into human-readable wrong-answer reasons."""

    label = str(raw).replace("_fallback_unique_0", "").replace("_fallback_unique_1", "")
    label = re.sub(r"_fallback_unique_[0-9]+", "", label)
    return label.replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# CSV discovery and normalization
# ---------------------------------------------------------------------------


def infer_model_name(path: Path) -> str:
    """Infer the public model identifier from the raw CSV filename."""

    stem = path.stem
    stem = re.sub(r"^eval_raw_", "", stem)
    stem = stem.replace("_text", "").replace("_mcq", "")
    stem = stem.replace("image_", "")
    if stem == "human_frq":
        return "human"
    if stem == "human":
        return "human"
    return stem


def warn_no_group_files(group: str, input_dir: Path, patterns: list[str], note: str = "") -> None:
    """Warn when a requested group has no files matching its naming convention."""

    pattern_text = ", ".join(patterns)
    message = (
        f"No input CSV files found for group '{group}' in {input_dir}. "
        f"Expected filename pattern(s): {pattern_text}."
    )
    if note:
        message += f" {note}"
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def discover_group_files(input_dir: Path, group: str) -> list[tuple[str, Path]]:
    """Discover input CSVs for one modality/group without hardcoded models.

    The public script is intended to survive new model additions. Naming
    conventions decide membership: text, MCQ, image-output, or trained
    checkpoint. If a requested group has no matching files, a warning reports
    the expected filename pattern so incorrectly named inputs are easier to
    diagnose. The returned model names are later used for labels and colors.
    """

    if group == "text":
        patterns = ["eval_raw_*_text.csv", "eval_raw_human_frq.csv"]
        candidates = list(input_dir.glob("eval_raw_*_text.csv"))
        candidates += list(input_dir.glob("eval_raw_human_frq.csv"))
        files = [
            p for p in candidates
            if not is_trained_checkpoint_name(infer_model_name(p))
            and "eval_raw_trained_ckpts" not in str(p)
        ]
        note = "Trained checkpoint text files are intentionally excluded from this base text group."
    elif group == "trained_text":
        patterns = [
            "eval_raw_trained_ckpts/eval_raw_*_text.csv",
            "eval_raw_*_text.csv with model names beginning dapo*, sft-*, or Qwen2.5-VL*",
            "eval_raw_human_frq.csv",
        ]
        files = list((input_dir / "eval_raw_trained_ckpts").glob("eval_raw_*_text.csv"))
        files += [p for p in input_dir.glob("eval_raw_*_text.csv") if is_trained_checkpoint_name(infer_model_name(p))]
        human = input_dir / "eval_raw_human_frq.csv"
        if human.exists():
            files.append(human)
        note = "The human free-response CSV is optional but included when present."
    elif group == "mcq5":
        patterns = ["eval_raw_*_mcq.csv without 3CQ or 4CQ in the filename"]
        files = [p for p in input_dir.glob("eval_raw_*_mcq.csv") if not re.search(r"[34]CQ", p.name, re.I)]
        note = "For 5-choice MCQ, filenames containing 3CQ or 4CQ are reserved for mixed-CQ analyses."
    elif group == "mcq345":
        patterns = ["eval_raw_*_mcq.csv for responders evaluated at 3CQ, 4CQ and 5CQ"]
        all_mcq = list(input_dir.glob("eval_raw_*_mcq.csv"))
        # The mixed-CQ analysis compares a responder against itself as the answer
        # set shrinks, so it covers only responders evaluated under all three
        # sizes. Responders run at five choices alone belong to the mcq5 group.
        # The human baseline is the exception that anchors the comparison; it was
        # collected at five choices only.
        complete = {
            base for base in {re.sub(r"[34]CQ$", "", infer_model_name(p), flags=re.I) for p in all_mcq}
            if all((input_dir / f"eval_raw_{base}{cq}_mcq.csv").exists() for cq in ("3CQ", "4CQ"))
        }
        files = [
            p for p in all_mcq
            if re.sub(r"[34]CQ$", "", infer_model_name(p), flags=re.I) in complete
            or infer_model_name(p) == "human"
        ]
        note = "Responders evaluated only at five choices are excluded here and covered by the mcq5 group."
    elif group == "image":
        patterns = ["eval_raw_image_*.csv"]
        files = list(input_dir.glob("eval_raw_image_*.csv"))
        note = ""
    else:
        raise ValueError(f"Unknown group: {group}")

    if not files:
        warn_no_group_files(group, input_dir, patterns, note)

    pairs = [(infer_model_name(p), p) for p in sorted(set(files))]
    return pairs


def is_trained_checkpoint_name(model: str) -> bool:
    """Identify trained checkpoint outputs that should not mix with base text."""

    m = model.lower()
    return m.startswith("dapo") or m.startswith("sft-") or m.startswith("qwen2.5-vl")


def normalize_raw_df(path: Path) -> pd.DataFrame:
    """Load one raw evaluation CSV and add common derived analysis columns.

    This preserves the original task columns while normalizing small historical
    differences from the development script, such as ``task_main`` versus
    ``main``. Camera and density variables are parsed here so every downstream
    table uses the same definitions.
    """

    df = pd.read_csv(path)
    if "task_main" in df.columns and "main" not in df.columns:
        df = df.rename(columns={"task_main": "main"})
    if "view" in df.columns:
        offset = df["view"].astype(str).str.extract(r"offset([\d.]+)_")[0]
        fov = df["view"].astype(str).str.extract(r"fov(\d+)_")[0]
        if offset.notna().any():
            df["offset"] = pd.to_numeric(offset, errors="coerce")
        if fov.notna().any():
            df["fov"] = pd.to_numeric(fov, errors="coerce")
            df["perspective"] = (df["fov"] >= 3).astype(int)
    if {"total_blocks", "num_columns", "num_layers"}.issubset(df.columns):
        denom = df["num_columns"] * df["num_layers"]
        df["fill_ratio"] = df["total_blocks"] / denom.replace(0, np.nan)
    for col in TASK_ORDER:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_group(input_dir: Path, group: str) -> ModelResults:
    """Load all CSVs for a modality/group into a ``model -> DataFrame`` map.

    The returned dictionary keys are model identifiers inferred from filenames;
    values are normalized raw evaluation DataFrames with one row per evaluated
    scene/request and task-correctness columns such as ``task5`` and ``main``.
    """

    out: ModelResults = {}
    for model, path in discover_group_files(input_dir, group):
        df = normalize_raw_df(path)
        task_cols = [c for c in TASK_ORDER if c in df.columns]
        if not task_cols:
            continue
        out[model] = df
    return out


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Create and return the standard public output directories."""

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


# ---------------------------------------------------------------------------
# Analysis CSV exporters
# ---------------------------------------------------------------------------


def export_manifest(output_dir: Path) -> None:
    """Write a machine-readable index of every paper figure and table."""

    rows = [asdict(item) for item in PAPER_OUTPUTS]
    pd.DataFrame(rows).to_csv(output_dir / "paper_output_manifest.csv", index=False)
    (output_dir / "paper_output_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def export_task_query_table(table_dir: Path) -> None:
    """Export the public task IDs, names, and query text used in the paper."""

    rows = []
    for task in TASK_ORDER:
        rows.append({
            "task": paper_task_id(task),
            "source_task": task,
            "task_name": TASK_LONG_NAME[task],
            "query_text": TASK_QUERY_TEXT[task],
        })
    pd.DataFrame(rows).to_csv(table_dir / "table_05_task_queries.csv", index=False)


def task_to_subtask_number(task: str) -> int | str:
    """Return the manuscript sub-task number, if the task has one."""

    return PAPER_SUBTASK_NUMBER.get(task, "")


def paper_task_id(task: str) -> str:
    """Convert an internal raw task column into the paper-facing task ID."""

    if task in PAPER_SUBTASK_NUMBER:
        return f"Task{PAPER_SUBTASK_NUMBER[task]}"
    if task == "main":
        return "Target1"
    if task == "task10":
        return "Target2a"
    if task == "task11":
        return "Target2b"
    return task


def task_accuracy_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Build per-model, per-task accuracy rows for Tables 6-8 and Figures 4/15/17/25.

    ``data`` maps each discovered model name to its normalized raw evaluation
    table. Each DataFrame contains one row per evaluated scene/request and
    binary correctness columns for the Spatial-IQ tasks.

    Each row includes raw counts, Wilson intervals, binomial standard error,
    chance-adjusted accuracy where applicable, and a one-sided chance test for
    MCQ outputs.
    """

    rows = []
    for model, df in data.items():
        for task in TASK_ORDER:
            if task not in df.columns:
                continue
            vals = df[task].dropna().astype(float)
            n = int(vals.size)
            if n == 0:
                continue
            k = int(vals.sum())
            acc = k / n
            lo, hi = wilson_ci(k, n)
            rows.append({
                "group": group,
                "model": model,
                "display_name": display_model(model),
                "task": paper_task_id(task),
                "source_task": task,
                "task_name": TASK_LONG_NAME[task],
                "accuracy": acc,
                "n_correct": k,
                "n_incorrect": n - k,
                "n": n,
                "binom_se": binom_se(acc, n),
                "ci_low_wilson": lo,
                "ci_high_wilson": hi,
                "chance_level": chance_level(group, model),
                "chance_adjusted_accuracy": chance_adjust(acc, group, model),
                "chance_adjusted_ci_low": chance_adjust(lo, group, model),
                "chance_adjusted_ci_high": chance_adjust(hi, group, model),
                "above_chance_p": p_above_chance(k, n, group, model),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        main_scores = out[out["source_task"] == "main"].set_index("model")["chance_adjusted_accuracy"].to_dict()
        out["_sort"] = out["model"].map(lambda m: model_sort_key(m, main_scores.get(m))[0])
        out["_task_order"] = out["source_task"].map(lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
        out = out.sort_values(["_sort", "model", "_task_order"], ascending=[True, True, True]).drop(columns=["_sort", "_task_order"])
    return out


def summary_table(data: ModelResults, task_acc: pd.DataFrame, group: str) -> pd.DataFrame:
    """Collapse raw and task-level results into modality-level summary tables.

    Raw task accuracies come from ``task_acc``. The hierarchy columns follow the
    appendix dependency definitions and are recomputed directly from the raw
    responder tables in ``data`` as mean relation-integrity scores over the
    pre-specified edges for each structure.
    """

    if task_acc.empty:
        return pd.DataFrame()
    task_col = "source_task" if "source_task" in task_acc.columns else "task"
    pivot = task_acc.pivot_table(index=["model", "display_name"], columns=task_col, values="chance_adjusted_accuracy" if group.startswith("mcq") else "accuracy")
    rows = []
    for (model, display_name), row in pivot.iterrows():
        rows.append({
            "model": model,
            "display_name": display_name,
            "object_count": row.get("main", np.nan),
            "visible_count": row.get("task5", np.nan),
            "hidden_count": row.get("task8", np.nan),
            "referential_chain": chain_integrity(
                data[model], [("task2", "task6"), ("task6", "task7")], group, model,
            ) if model in data else np.nan,
            "visible_support": support_integrity(
                data[model], [("task3", "task5"), ("task4", "task5")], group, model,
            ) if model in data else np.nan,
            "hidden_support": support_integrity(
                data[model], [("task3", "task8"), ("task7", "task8")], group, model,
            ) if model in data else np.nan,
            "summation_mechanism": summation_mechanism(data[model], group, model) if model in data else np.nan,
            "mental_rotation_mean": mean_ignore_nan([row.get("task10", np.nan), row.get("task11", np.nan)]),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("object_count", ascending=False)


def difficulty_bin_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Aggregate Object Counting accuracy over pre-defined difficulty bins.

    ``data`` is a model-result map: keys are discovered model names, and values
    are normalized raw result tables with scene-level difficulty metadata.
    """

    rows = []
    for model, df in data.items():
        if "main" not in df.columns:
            continue
        for factor in DIFFICULTY_FACTORS:
            if factor not in df.columns:
                continue
            bins, order = difficulty_bins(df, factor)
            work = df.assign(_bin=bins)
            for bin_label in order:
                sub = work[work["_bin"] == bin_label]
                vals = sub["main"].dropna().astype(float)
                n = int(vals.size)
                if n == 0:
                    continue
                k = int(vals.sum())
                acc = k / n
                lo, hi = wilson_ci(k, n)
                rows.append({
                    "group": group,
                    "model": model,
                    "display_name": display_model(model),
                    "factor": factor,
                    "factor_name": factor_display(factor),
                    "bin": bin_label,
                    "accuracy": acc,
                    "n_correct": k,
                    "n_incorrect": n - k,
                    "n": n,
                    "binom_se": binom_se(acc, n),
                    "ci_low_wilson": lo,
                    "ci_high_wilson": hi,
                    "chance_adjusted_accuracy": chance_adjust(acc, group, model),
                    "chance_adjusted_ci_low": chance_adjust(lo, group, model),
                    "chance_adjusted_ci_high": chance_adjust(hi, group, model),
                })
    return pd.DataFrame(rows)


def difficulty_bins(df: pd.DataFrame, factor: str) -> tuple[pd.Series, list[str]]:
    """Return paper-style bin labels and their display order for one factor."""

    if factor == "total_blocks":
        order = ["4-8", "9-12", "13-18", "19-25", "26+"]
        return pd.cut(df[factor], [0, 8, 12, 18, 25, 100], labels=order).astype(str), order
    if factor == "num_hidden_blocks":
        order = ["0", "1", "2-3", "4+"]
        return pd.cut(df[factor], [-1, 0, 1, 3, 100], labels=order).astype(str), order
    if factor == "num_layers":
        order = [str(v) for v in sorted(pd.Series(df[factor].dropna().astype(int).unique()).tolist())]
        return df[factor].astype("Int64").astype(str), order
    if factor == "num_columns":
        order = ["3-5", "6-8", "9-11", "12-14"]
        return pd.cut(df[factor], [2.5, 5.5, 8.5, 11.5, 14.5], labels=order).astype(str), order
    if factor == "fill_ratio":
        order = ["Dense", "Medium", "Sparse"]
        labels = ["Sparse", "Medium", "Dense"]
        return pd.cut(df[factor], [0.0, 0.55, 0.67, 1.01], labels=labels).astype(str), order
    order = sorted(df[factor].dropna().astype(str).unique().tolist())
    return df[factor].astype(str), order


def factor_display(factor: str) -> str:
    return {
        "total_blocks": "Number of objects",
        "num_hidden_blocks": "Number of hidden objects",
        "num_layers": "Number of layers",
        "num_columns": "Number of columns",
        "fill_ratio": "Fill ratio",
    }.get(factor, factor)


def e1a_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Export the 2x2 visible/hidden-to-main mechanism table.

    ``data`` maps model identifiers to normalized raw result tables containing
    ``task5`` visible-count correctness, ``task8`` hidden-count correctness, and
    ``main`` Object Counting correctness.

    This backs Figures 7, 23, 24, and 27. The denominator for each cell is the
    number of scenes falling into that visible-correct/hidden-correct state, so
    cell percentages are conditional accuracies rather than parts of a 100% sum.
    """

    rows = []
    for model, df in data.items():
        needed = {"task5", "task8", "main"}
        if not needed.issubset(df.columns):
            continue
        valid = df[list(needed)].dropna()
        total_n = len(valid)
        main_correct = int(valid["main"].sum()) if total_n else 0
        for v_ok in [1, 0]:
            for h_ok in [1, 0]:
                sub = valid[(valid["task5"] == v_ok) & (valid["task8"] == h_ok)]
                n = int(len(sub))
                k = int(sub["main"].sum()) if n else 0
                acc = k / n if n else np.nan
                rows.append({
                    "group": group,
                    "model": model,
                    "display_name": display_model(model),
                    "visible_correct": v_ok,
                    "hidden_correct": h_ok,
                    "cell": f"V{'+' if v_ok else '-'} H{'+' if h_ok else '-'}",
                    "main_accuracy": acc,
                    "main_correct": k,
                    "main_incorrect": n - k,
                    "n": n,
                    "share_of_scenes": n / total_n if total_n else np.nan,
                    "overall_main_accuracy": main_correct / total_n if total_n else np.nan,
                    "overall_main_correct": main_correct,
                    "overall_n": total_n,
                })
    return pd.DataFrame(rows)


def composition_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Measure whether numerical component predictions compose into Main.

    ``data`` maps model identifiers to normalized raw result tables. This
    exporter only uses models whose DataFrames include numeric prediction
    columns such as ``task5_pred``, ``task8_pred``, and ``main_pred``.

    For text-like modalities with numeric predictions, this table reports the
    correlation between ``task5_pred + task8_pred`` and ``main_pred`` together
    with the composition residual -- the total-count error minus the summed
    component errors -- and its exact-agreement rate, magnitude, and signed
    bias. See ``summation_mechanism`` for why composition is scored on errors
    rather than on the raw prediction identity.
    """

    rows = []
    for model, df in data.items():
        required = {"task5_pred", "task5_gt", "task8_pred", "task8_gt", "main_pred", "main_gt"}
        if not required.issubset(df.columns):
            continue
        work = df.dropna(subset=list(required)).copy()
        if work.empty:
            continue
        comp = pd.to_numeric(work["task5_pred"], errors="coerce") + pd.to_numeric(work["task8_pred"], errors="coerce")
        main_pred = pd.to_numeric(work["main_pred"], errors="coerce")
        r, p = safe_corr(comp, main_pred)
        # Composition is scored on errors rather than on the raw prediction
        # identity: the hidden sub-task counts visible objects resting on an
        # occluded one, so ``main = visible + hidden`` does not hold for every
        # scene and comparing predictions directly would penalise a responder
        # for the scenes where it does not. The residual below is zero exactly
        # when the total-count error equals the sum of the component errors.
        composed_err = ((pd.to_numeric(work["task5_pred"], errors="coerce") - pd.to_numeric(work["task5_gt"], errors="coerce"))
                        + (pd.to_numeric(work["task8_pred"], errors="coerce") - pd.to_numeric(work["task8_gt"], errors="coerce")))
        residual = (main_pred - pd.to_numeric(work["main_gt"], errors="coerce")) - composed_err
        exact = residual == 0
        both_wrong = pd.Series(False, index=work.index)
        if {"task5", "task8"}.issubset(work.columns):
            both_wrong = (pd.to_numeric(work["task5"], errors="coerce") == 0) & (pd.to_numeric(work["task8"], errors="coerce") == 0)
        both_wrong_n = int(both_wrong.sum())
        both_wrong_exact = float(exact[both_wrong].mean()) if both_wrong_n else np.nan
        rows.append({
            "group": group,
            "model": model,
            "display_name": display_model(model),
            "prediction_correlation_r": r,
            "prediction_correlation_p": p,
            "exact_match_rate": float(exact.mean()),
            "both_wrong_exact_match_rate": both_wrong_exact,
            "mean_absolute_residual": mean_ignore_nan(np.abs(residual)),
            "residual_bias": mean_ignore_nan(residual),
            "both_wrong_n": both_wrong_n,
            "n": int(len(work)),
        })
    return pd.DataFrame(rows)


def object_accuracy_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Aggregate Object Counting accuracy separately by object appearance.

    ``data`` maps model identifiers to normalized raw result tables that include
    ``object_type`` labels when object-conditioned analyses are available.
    """

    rows = []
    for model, df in data.items():
        if "object_type" not in df.columns or "main" not in df.columns:
            continue
        for obj, sub in df.groupby("object_type", dropna=True):
            vals = sub["main"].dropna().astype(float)
            n = int(vals.size)
            if n == 0:
                continue
            k = int(vals.sum())
            acc = k / n
            lo, hi = wilson_ci(k, n)
            rows.append({
                "group": group,
                "model": model,
                "display_name": display_model(model),
                "object_type": str(obj),
                "accuracy": acc,
                "n_correct": k,
                "n_incorrect": n - k,
                "n": n,
                "ci_low_wilson": lo,
                "ci_high_wilson": hi,
                "chance_adjusted_accuracy": chance_adjust(acc, group, model),
            })
    return pd.DataFrame(rows)


def object_difficulty_slice_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Create pooled object/difficulty slices for Figure 21.

    ``data`` maps model identifiers to normalized raw result tables. The tables
    are pooled across models after adding an internal model column so this figure
    summarizes object and difficulty effects at the modality level.

    The rows are intentionally pooled across models to summarize whether object
    appearance and scene difficulty change the overall task profile.
    """

    rows = []
    pooled = []
    for model, df in data.items():
        if "main" not in df.columns:
            continue
        pooled.append(df.assign(_model=model))
    if not pooled:
        return pd.DataFrame()
    full = pd.concat(pooled, ignore_index=True)
    selected_tasks = ["task2", "task5", "task6", "task7", "task8", "main"]
    if "object_type" in full.columns:
        objects = [o for o in ["cube", "factory_box", "soup_can", "spam"] if o in set(full["object_type"].dropna())]
        objects += [o for o in sorted(full["object_type"].dropna().astype(str).unique()) if o not in objects]
        for task in selected_tasks:
            if task not in full.columns:
                continue
            for obj in objects:
                sub = full[full["object_type"].astype(str) == obj]
                rows.append(slice_row(group, "object", str(obj), task, sub[task]))
    if "total_blocks" in full.columns:
        bins, order = difficulty_bins(full, "total_blocks")
        full = full.assign(_bin=bins)
        for task in selected_tasks:
            if task not in full.columns:
                continue
            for b in order:
                sub = full[full["_bin"] == b]
                rows.append(slice_row(group, "difficulty", b, task, sub[task]))
    if {"object_type", "_bin"}.issubset(full.columns):
        objects = [o for o in ["cube", "factory_box", "soup_can", "spam"] if o in set(full["object_type"].dropna())]
        objects += [o for o in sorted(full["object_type"].dropna().astype(str).unique()) if o not in objects]
        for task in ["main", "task5", "task8"]:
            if task not in full.columns:
                continue
            for obj in objects:
                obj_sub = full[full["object_type"].astype(str) == obj]
                for b in order:
                    sub = obj_sub[obj_sub["_bin"] == b]
                    rows.append(slice_row(group, "object_difficulty", f"{obj}|{b}", task, sub[task]))
    return pd.DataFrame(rows)


def slice_row(group: str, slice_type: str, slice_value: str, task: str, values: pd.Series) -> dict:
    """Small helper for object/difficulty slice tables."""

    vals = values.dropna().astype(float)
    n = int(vals.size)
    k = int(vals.sum()) if n else 0
    acc = k / n if n else np.nan
    return {
        "group": group,
        "slice_type": slice_type,
        "slice_value": slice_value,
        "task": paper_task_id(task),
        "source_task": task,
        "task_name": TASK_LONG_NAME[task],
        "accuracy": acc,
        "n_correct": k,
        "n": n,
    }


def wrong_answer_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Export MCQ wrong-answer preferences and significance annotations.

    ``data`` maps MCQ model identifiers to normalized raw result tables with
    ``*_choice_type`` columns that record the selected wrong-answer category.

    Figures 5 and 16 display only wrong-answer categories with raw p < 0.05
    against the relevant MCQ chance level. The full table keeps every category
    so appendix readers can inspect non-displayed errors as well.
    """

    rows = []
    for model, df in data.items():
        for task in TASK_ORDER:
            choice_col = f"{task}_choice_type"
            if choice_col not in df.columns:
                continue
            valid = df[choice_col].dropna().astype(str)
            n = int(len(valid))
            if n == 0:
                continue
            for raw_label, k in valid.value_counts().items():
                if raw_label == "correct":
                    continue
                p_raw = k / n
                baseline = chance_level(group, model)
                p_val = float(stats.binomtest(int(k), n, baseline, alternative="greater").pvalue) if baseline else np.nan
                rows.append({
                    "group": group,
                    "model": model,
                    "display_name": display_model(model),
                    "task": paper_task_id(task),
                    "source_task": task,
                    "task_name": TASK_LONG_NAME.get(task, task),
                    "raw_label": raw_label,
                    "reason": label_from_choice_type(raw_label),
                    "n_error": int(k),
                    "n_total": n,
                    "raw_proportion": p_raw,
                    "chance_level": baseline,
                    "chance_adjusted_preference": chance_adjust(p_raw, group, model),
                    "above_chance_p": p_val,
                    "displayed_in_figure": bool(p_val < 0.05) if not pd.isna(p_val) else False,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_task_order"] = out["source_task"].map(lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
    return out.sort_values(["_task_order", "reason", "above_chance_p", "model"]).drop(columns="_task_order")


def mental_rotation_corr_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Compute Figure 13 correlations for mental-rotation targets.

    ``data`` maps text-model identifiers to normalized raw result tables. Each
    DataFrame contributes task correctness columns and the ``view`` identifier
    used to define within-scene subsets.

    ``naive`` correlations use all rows. ``within_scene`` correlations restrict
    to scenes where at least one compared task varies across model outputs,
    reducing correlations that arise only because some scenes are globally easy
    or hard.
    """

    rows = []
    targets = [t for t in TASK_ORDER if t in {"task10", "task11"}]
    compare = [t for t in TASK_ORDER if t not in targets]
    compare += targets
    combined = pd.concat([df.assign(_model=model) for model, df in data.items()], ignore_index=True) if data else pd.DataFrame()
    varying_views: dict[str, set] = {}
    if "view" in combined.columns:
        for task in compare:
            if task in combined.columns:
                rng = combined.groupby("view")[task].agg(lambda x: x.max() - x.min())
                varying_views[task] = set(rng[rng > 0].index)
    for model, df in data.items():
        for focus in targets:
            if focus not in df.columns:
                continue
            for task in compare:
                if task not in df.columns:
                    continue
                for corr_type in ["naive", "within_scene"]:
                    if corr_type == "within_scene" and "view" in df.columns:
                        views = varying_views.get(focus, set()) | varying_views.get(task, set())
                        work = df[df["view"].isin(views)] if views else df.iloc[0:0]
                    else:
                        work = df
                    r, p = safe_corr(work[focus], work[task]) if not work.empty else (np.nan, np.nan)
                    rows.append({
                        "group": group,
                        "model": model,
                        "display_name": display_model(model),
                        "correlation_type": corr_type,
                        "focus_task": paper_task_id(focus),
                        "source_focus_task": focus,
                        "focus_name": TASK_LONG_NAME[focus],
                        "task": paper_task_id(task),
                        "source_task": task,
                        "task_name": TASK_LONG_NAME[task],
                        "pearson_r": r if task != focus else np.nan,
                        "p_value": p if task != focus else np.nan,
                        "n": int(work[[focus, task]].dropna().shape[0]) if not work.empty else 0,
                    })
    return pd.DataFrame(rows)


def bundle_corr_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Compute Figure 14 bundle-to-Object-Counting correlations.

    ``data`` maps model identifiers to normalized raw result tables. The tables
    are concatenated for an aggregate fixed-effect correlation and also analyzed
    separately for per-model within-varying-scene correlations.

    Bundles contrast the cross-view mental-rotation branch with the single-view
    referential and visible+hidden mechanisms used for object counting.
    """

    rows = []
    if not data:
        return pd.DataFrame()
    combined_parts = []
    for model, df in data.items():
        if "main" in df.columns:
            combined_parts.append(df.assign(_model=model))
    if not combined_parts:
        return pd.DataFrame()
    combined = pd.concat(combined_parts, ignore_index=True)
    for bundle, tasks in CROSSVIEW_BUNDLE_VARIANTS.items():
        present = [t for t in tasks if t in combined.columns]
        if not present:
            continue
        score_col = f"_bundle_{re.sub(r'[^a-zA-Z0-9]+', '_', bundle).strip('_')}"
        combined[score_col] = combined[present].mean(axis=1)
        r_agg, p_agg, n_agg = fixed_effect_corr(combined, score_col, "main")
        if "view" in combined.columns:
            rng = combined.groupby("view")[score_col].agg(lambda x: x.max() - x.min())
            varying_views = set(rng[rng > 0].index)
        else:
            varying_views = set()
        rows.append({
            "group": group,
            "level": "aggregate",
            "model": "aggregate",
            "display_name": "Aggregate",
            "bundle": bundle,
            "pearson_r_with_object_count": r_agg,
            "p_value": p_agg,
            "n": n_agg,
            "n_varying_scenes": len(varying_views),
            "n_tasks": len(present),
        })
        for model, df in data.items():
            if "main" not in df.columns:
                continue
            present_model = [t for t in tasks if t in df.columns]
            if not present_model:
                continue
            work = df.copy()
            work[score_col] = work[present_model].mean(axis=1)
            if varying_views and "view" in work.columns:
                work = work[work["view"].isin(varying_views)]
            r, p = safe_corr(work[score_col], work["main"])
            rows.append({
                "group": group,
                "level": "per_model",
                "model": model,
                "display_name": display_model(model),
                "bundle": bundle,
                "pearson_r_with_object_count": r,
                "p_value": p,
                "n": int(work[[score_col, "main"]].dropna().shape[0]),
                "n_varying_scenes": len(varying_views),
                "n_tasks": len(present_model),
            })
    return pd.DataFrame(rows)


def camera_effect_table(data: ModelResults, group: str) -> pd.DataFrame:
    """Estimate perspective and offset effects for Figure 18.

    ``data`` maps model identifiers to normalized raw result tables with camera
    metadata derived from the ``view`` filename field.

    For each binary camera factor, the reported delta is high-level accuracy
    minus low-level accuracy, with a Welch t-test and normal-approximation CI.
    """

    rows = []
    for model, df in data.items():
        for task in TASK_ORDER:
            if task not in df.columns:
                continue
            for factor in CAMERA_FACTORS:
                if factor not in df.columns:
                    continue
                levels = sorted(df[factor].dropna().unique().tolist())
                if len(levels) != 2:
                    continue
                lo_level, hi_level = levels[0], levels[-1]
                lo_vals = df.loc[df[factor] == lo_level, task].dropna().astype(float)
                hi_vals = df.loc[df[factor] == hi_level, task].dropna().astype(float)
                if len(lo_vals) == 0 or len(hi_vals) == 0:
                    continue
                delta = hi_vals.mean() - lo_vals.mean()
                se = np.sqrt(lo_vals.var(ddof=1) / len(lo_vals) + hi_vals.var(ddof=1) / len(hi_vals))
                try:
                    p_val = float(stats.ttest_ind(hi_vals, lo_vals, equal_var=False, nan_policy="omit").pvalue)
                except Exception:
                    p_val = np.nan
                rows.append({
                    "group": group,
                    "model": model,
                    "display_name": display_model(model),
                    "task": paper_task_id(task),
                    "source_task": task,
                    "task_name": TASK_LONG_NAME[task],
                    "factor": factor,
                    "low_level": lo_level,
                    "high_level": hi_level,
                    "delta_high_minus_low": delta,
                    "se": se,
                    "p_value": p_val,
                    "ci_low": delta - 1.96 * se if not pd.isna(se) else np.nan,
                    "ci_high": delta + 1.96 * se if not pd.isna(se) else np.nan,
                    "n_low": len(lo_vals),
                    "n_high": len(hi_vals),
                })
    return pd.DataFrame(rows)


def export_hierarchy_dependency_tables(data: ModelResults, table_dir: Path, group: str) -> None:
    """Export the Appendix F hierarchy-dependency tables via ``hierarchy_dependency``.

    Appendix F reports these statistics in prose, so they have no figure/table
    number; the CSVs written here are the numbers behind that text, including the
    responder-controlled robustness checks. Running this module standalone with
    ``--validate`` produces the same files plus a stdout summary. The two-way
    fixed-effect bootstrap dominates the runtime (roughly a minute).
    """

    exporters = {
        f"tableF_within_scene_lift_{group}.csv": hdep.within_scene_lift_table,
        f"tableF_joint_integrity_{group}.csv": hdep.joint_integrity_table,
        f"tableF_bundle_coupling_{group}.csv": hdep.bundle_coupling_table,
        f"tableF_responder_controlled_lift_{group}.csv": hdep.responder_controlled_lift_table,
        f"tableF_per_responder_lift_{group}.csv": hdep.per_responder_lift_table,
        f"tableF_bundle_coupling_twoway_{group}.csv": hdep.bundle_coupling_twoway_table,
        f"table6_mcnemar_object_counting_{group}.csv": hdep.mcnemar_main_table,
        f"figF18_camera_within_structure_{group}.csv": hdep.camera_within_structure_table,
    }
    for filename, exporter in exporters.items():
        exporter(data, group).to_csv(table_dir / filename, index=False)


def run_analysis(input_dir: Path, output_dir: Path, groups: list[str]) -> None:
    """Run all requested analysis exporters before any figure rendering."""

    table_dir, _ = ensure_dirs(output_dir)
    export_manifest(output_dir)
    export_task_query_table(table_dir)

    loaded = {group: load_group(input_dir, group) for group in groups}

    if "text" in loaded:
        text_acc = task_accuracy_table(loaded["text"], "text")
        text_acc.to_csv(table_dir / "table_06_text_task_accuracy.csv", index=False)
        summary_table(loaded["text"], text_acc, "text").to_csv(table_dir / "table_01_text_summary.csv", index=False)
        difficulty_bin_table(loaded["text"], "text").to_csv(table_dir / "table_09_text_difficulty_bins.csv", index=False)
        e1a_table(loaded["text"], "text").to_csv(table_dir / "figure_07_text_summation_mechanism.csv", index=False)
        composition_table(loaded["text"], "text").to_csv(table_dir / "figure_22_text_composition_diagnostics.csv", index=False)
        object_accuracy_table(loaded["text"], "text").to_csv(table_dir / "figure_20_text_object_accuracy.csv", index=False)
        object_difficulty_slice_table(loaded["text"], "text").to_csv(table_dir / "figure_21_text_object_difficulty_slices.csv", index=False)
        mental_rotation_corr_table(loaded["text"], "text").to_csv(table_dir / "figure_13_text_mental_rotation_correlations.csv", index=False)
        bundle_corr_table(loaded["text"], "text").to_csv(table_dir / "figure_14_text_bundle_correlations.csv", index=False)
        camera_effect_table(loaded["text"], "text").to_csv(table_dir / "figure_18_text_camera_effects.csv", index=False)
        export_hierarchy_dependency_tables(loaded["text"], table_dir, "text")

    if "mcq5" in loaded:
        mcq5_acc = task_accuracy_table(loaded["mcq5"], "mcq5")
        mcq5_acc.to_csv(table_dir / "table_07_mcq5_task_accuracy.csv", index=False)
        wrong_answer_table(loaded["mcq5"], "mcq5").to_csv(table_dir / "figure_05_mcq5_wrong_answer_preferences.csv", index=False)

    if "mcq345" in loaded:
        mcq345_acc = task_accuracy_table(loaded["mcq345"], "mcq345")
        mcq345_acc.to_csv(table_dir / "figure_15_mcq345_task_accuracy.csv", index=False)
        summary_table(loaded["mcq345"], mcq345_acc, "mcq345").to_csv(table_dir / "table_02_mcq_summary.csv", index=False)
        wrong_answer_table(loaded["mcq345"], "mcq345").to_csv(table_dir / "figure_16_mcq345_wrong_answer_preferences.csv", index=False)
        e1a_table(loaded["mcq345"], "mcq345").to_csv(table_dir / "figure_23_mcq345_summation_mechanism.csv", index=False)

    if "image" in loaded:
        image_acc = task_accuracy_table(loaded["image"], "image")
        image_acc.to_csv(table_dir / "table_08_image_task_accuracy.csv", index=False)
        summary_table(loaded["image"], image_acc, "image").to_csv(table_dir / "table_03_image_summary.csv", index=False)
        difficulty_bin_table(loaded["image"], "image").to_csv(table_dir / "figure_19_image_difficulty_bins.csv", index=False)
        e1a_table(loaded["image"], "image").to_csv(table_dir / "figure_24_image_summation_mechanism.csv", index=False)

    if "trained_text" in loaded:
        trained_acc = task_accuracy_table(loaded["trained_text"], "trained_text")
        trained_acc.to_csv(table_dir / "figure_25_trained_text_task_accuracy.csv", index=False)
        summary_table(loaded["trained_text"], trained_acc, "trained_text").to_csv(table_dir / "table_04_trained_text_summary.csv", index=False)
        object_accuracy_table(loaded["trained_text"], "trained_text").to_csv(table_dir / "figure_26_trained_text_object_accuracy.csv", index=False)
        e1a_table(loaded["trained_text"], "trained_text").to_csv(table_dir / "figure_27_trained_text_summation_mechanism.csv", index=False)
        composition_table(loaded["trained_text"], "trained_text").to_csv(table_dir / "figure_28_trained_text_composition_diagnostics.csv", index=False)


# ---------------------------------------------------------------------------
# Figure renderers. These read only the exported CSVs above.
# ---------------------------------------------------------------------------


def put_grid_behind(ax, axis: str = "y", **kwargs) -> None:
    """Apply light grid lines behind plotted marks."""

    ax.set_axisbelow(True)
    params = {"color": "#e5e7eb", "linewidth": 0.8, "zorder": 0}
    params.update(kwargs)
    ax.grid(axis=axis, **params)


def shade_task_regions(ax, tasks: list[str]) -> None:
    """Shade target and mental-rotation task regions consistently across figures."""

    mental_inds = [i for i, task in enumerate(tasks) if task in {"task10", "task11"}]
    if mental_inds:
        ax.axvspan(min(mental_inds) - 0.5, max(mental_inds) + 0.5, color="#f4efe7", alpha=0.9, zorder=0)
    if "main" in tasks:
        idx = tasks.index("main")
        ax.axvspan(idx - 0.5, idx + 0.5, color="#f8fafc", alpha=0.95, zorder=0)
        ax.axvline(idx - 0.5, color="#9ca3af", lw=1.0, ls=":", zorder=1)
        ax.axvline(idx + 0.5, color="#9ca3af", lw=1.0, ls=":", zorder=1)


def set_model_tick_colors(ticklabels: list, models: list[str]) -> None:
    """Color model tick labels with the same family colors as plotted marks."""

    for tick, model in zip(ticklabels, models):
        tick.set_color(model_color(model))


def rotation_bg_rgba(alpha: float = ROTATION_HIGHLIGHT_ALPHA) -> tuple[float, float, float, float]:
    """Return the shared mental-rotation background color from the paper figure code."""

    return matplotlib.colors.to_rgba(ROTATION_HIGHLIGHT_COLOR, alpha=alpha)


def paper_task_axis_label(task: str) -> str:
    """Return compact labels used by the paper-style task figures."""

    return {
        "task1": "Object\nCategory",
        "task9": "Clusters",
        "task3": "Columns",
        "task4": "Layers",
        "task5": "Visible\nCount",
        "task2": "Top\nLayer",
        "task6": "Direct\nSupport",
        "task7": "Support\nColumn",
        "task8": "Hidden\nCount",
        "main": "Object\nCount",
        "task10": "90° Mental\nRotation",
        "task11": "180° Mental\nRotation",
    }.get(task, TASK_DISPLAY.get(task, task))


def paper_factor_label(factor_name: str) -> str:
    """Normalize difficulty-control factor labels to the paper figure style."""

    return {
        "Number of objects": "Number of Objects",
        "Number of hidden objects": "Number of Hidden Objects",
        "Number of layers": "Number of Layers",
        "Number of columns": "Number of Columns",
        "Fill ratio": "Fill Ratio",
    }.get(factor_name, factor_name)


def split_label_near_midpoint(text: str) -> str:
    """Split a label at a midpoint space to keep heatmap columns readable."""

    label = str(text).title()
    for old, new in [
        ("Block", "Obj"),
        ("Object", "Obj"),
        ("Objects", "Objs"),
        ("One", "1"),
        ("Two", "2"),
        ("Phantom", "New"),
    ]:
        label = label.replace(old, new)
    if " " not in label:
        return label
    midpoint = len(label) / 2
    spaces = [idx for idx, char in enumerate(label) if char == " "]
    best_space = min(spaces, key=lambda idx: abs(idx - midpoint))
    return label[:best_space] + "\n" + label[best_space + 1:]


def render_all_figures(output_dir: Path) -> None:
    """Render every generated paper figure from the exported table CSVs."""

    table_dir, figure_dir = ensure_dirs(output_dir)
    render_paper_task_accuracy(table_dir / "table_06_text_task_accuracy.csv", figure_dir / "figure_04_text_task_accuracy.png", value_col="accuracy")
    render_wrong_answer_heatmap(table_dir / "figure_05_mcq5_wrong_answer_preferences.csv", figure_dir / "figure_05_mcq5_wrong_answer_preferences.png")
    render_paper_difficulty(table_dir / "table_09_text_difficulty_bins.csv", figure_dir / "figure_06_text_difficulty_controls.png", value_col="accuracy")
    render_paper_e1a(table_dir / "figure_07_text_summation_mechanism.csv", figure_dir / "figure_07_text_summation_mechanism.png")
    render_correlation_heatmap(table_dir / "figure_13_text_mental_rotation_correlations.csv", figure_dir / "figure_13_text_mental_rotation_correlations.png")
    render_bundle_bars(table_dir / "figure_14_text_bundle_correlations.csv", figure_dir / "figure_14_text_bundle_correlations.png")
    render_task_accuracy(table_dir / "figure_15_mcq345_task_accuracy.csv", figure_dir / "figure_15_mcq345_task_accuracy.png", value_col="chance_adjusted_accuracy", ylabel="Chance-adjusted accuracy")
    render_wrong_answer_heatmap(table_dir / "figure_16_mcq345_wrong_answer_preferences.csv", figure_dir / "figure_16_mcq345_wrong_answer_preferences.png", value_col="chance_adjusted_preference")
    render_task_accuracy(table_dir / "table_08_image_task_accuracy.csv", figure_dir / "figure_17_image_task_accuracy.png", value_col="accuracy", ylabel="Accuracy")
    render_camera_effects(table_dir / "figure_18_text_camera_effects.csv", figure_dir / "figure_18_text_camera_effects.png")
    render_difficulty(table_dir / "figure_19_image_difficulty_bins.csv", figure_dir / "figure_19_image_difficulty_controls.png", value_col="accuracy", ylabel="Object Counting accuracy")
    render_object_accuracy(table_dir / "figure_20_text_object_accuracy.csv", figure_dir / "figure_20_text_object_accuracy.png")
    render_object_difficulty_slices(table_dir / "figure_21_text_object_difficulty_slices.csv", figure_dir / "figure_21_text_object_difficulty_slices.png")
    render_composition(table_dir / "figure_22_text_composition_diagnostics.csv", figure_dir / "figure_22_text_composition_diagnostics.png")
    render_e1a(table_dir / "figure_23_mcq345_summation_mechanism.csv", figure_dir / "figure_23_mcq345_summation_mechanism.png")
    render_e1a(table_dir / "figure_24_image_summation_mechanism.csv", figure_dir / "figure_24_image_summation_mechanism.png")
    render_task_accuracy(table_dir / "figure_25_trained_text_task_accuracy.csv", figure_dir / "figure_25_trained_text_task_accuracy.png", value_col="accuracy", ylabel="Accuracy")
    render_object_accuracy(table_dir / "figure_26_trained_text_object_accuracy.csv", figure_dir / "figure_26_trained_text_object_accuracy.png")
    render_e1a(table_dir / "figure_27_trained_text_summation_mechanism.csv", figure_dir / "figure_27_trained_text_summation_mechanism.png")
    render_composition(table_dir / "figure_28_trained_text_composition_diagnostics.csv", figure_dir / "figure_28_trained_text_composition_diagnostics.png")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a generated table if present; return an empty frame otherwise."""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def source_task_col(df: pd.DataFrame) -> str:
    """Prefer the raw task column when a public table contains both task IDs."""

    return "source_task" if "source_task" in df.columns else "task"


def ordered_models_from_table(df: pd.DataFrame, value_col: str) -> list[str]:
    """Order models by Main score when available, then by stable family order."""

    if df.empty:
        return []
    task_col = source_task_col(df) if "task" in df.columns or "source_task" in df.columns else None
    if task_col is not None and "main" in set(df[task_col]):
        scores = df[df[task_col] == "main"].set_index("model")[value_col].to_dict()
    else:
        scores = df.groupby("model")[value_col].mean().to_dict()
    return sorted(df["model"].drop_duplicates(), key=lambda m: model_sort_key(m, scores.get(m)))


def render_paper_task_accuracy(csv_path: Path, out_path: Path, value_col: str = "accuracy") -> None:
    """Render Figure 4 using the notebook-provided paper task-accuracy style."""

    df = read_csv_if_exists(csv_path)
    if df.empty or value_col not in df.columns:
        return
    task_col = source_task_col(df)
    tasks = [task for task in TASK_ORDER if task in set(df[task_col])]
    models = ordered_models_from_table(df, value_col)
    if not tasks or not models:
        return

    label_font_size = 12
    tick_font_size = 10
    bar_width = 0.08
    bar_gap = 0.01
    x = np.arange(len(tasks))
    total_group_width = (len(models) * bar_width) + ((len(models) - 1) * bar_gap)

    fig, ax = plt.subplots(figsize=(18, 3.8))
    for j, task in enumerate(tasks):
        if task in {"task10", "task11"}:
            ax.add_patch(
                Rectangle(
                    (j - 0.5, 0),
                    1,
                    1.1,
                    lw=0,
                    fc=rotation_bg_rgba(),
                    zorder=0,
                    transform=ax.get_xaxis_transform(),
                )
            )

    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index(task_col)
        vals = [sub.loc[task, value_col] if task in sub.index else np.nan for task in tasks]
        offset = (-total_group_width / 2) + (i * (bar_width + bar_gap)) + (bar_width / 2)
        ax.bar(x + offset, vals, bar_width, label=display_model(model), color=paper_model_color(model), zorder=2)

    ax.set_ylabel("Accuracy", fontsize=label_font_size, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([paper_task_axis_label(task) for task in tasks], rotation=0, ha="center", fontsize=tick_font_size)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=tick_font_size)
    ax.set_ylim(0, 1)
    padding = (total_group_width / 2) + (10 * bar_gap)
    ax.set_xlim(-padding, (len(tasks) - 1) + padding)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=max(1, len(models)),
        frameon=False,
        fontsize=tick_font_size,
    )
    for label in ax.get_xticklabels():
        if "Rotation" in label.get_text():
            label.set_color(ROTATION_HIGHLIGHT_COLOR)
            label.set_weight("bold")
        if "Object\nCount" in label.get_text():
            label.set_weight("bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_paper_difficulty(csv_path: Path, out_path: Path, value_col: str = "accuracy") -> None:
    """Render Figure 6 using the notebook-provided difficulty-control style."""

    df = read_csv_if_exists(csv_path)
    if df.empty or value_col not in df.columns:
        return
    factors = [factor for factor in DIFFICULTY_FACTORS if factor in set(df["factor"])]
    models = ordered_models_from_table(df, value_col)
    if not factors or not models:
        return

    label_font_size = 11
    tick_font_size = 9
    legend_font_size = 9
    fig, axes = plt.subplots(1, len(factors), figsize=(3.8 * len(factors), 3.6), sharey=True, squeeze=False)
    for ax, factor in zip(axes.flatten(), factors):
        fact_df = df[df["factor"] == factor]
        bins = fact_df["bin"].drop_duplicates().tolist()
        x_indices = np.arange(len(bins))
        for model in models:
            sub = fact_df[fact_df["model"] == model].set_index("bin")
            vals = [sub.loc[bin_label, value_col] if bin_label in sub.index else np.nan for bin_label in bins]
            ax.plot(
                x_indices,
                vals,
                color=paper_model_color(model),
                label=display_model(model),
                marker="o",
                linestyle="-",
                linewidth=2,
                zorder=2,
            )
        factor_label = paper_factor_label(fact_df["factor_name"].dropna().iloc[0] if fact_df["factor_name"].notna().any() else factor_display(factor))
        ax.set_xlabel(factor_label, weight="bold", fontsize=label_font_size)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(bins, fontsize=tick_font_size)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
        ax.set_axisbelow(True)

    first_ax = axes.flatten()[0]
    first_ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    first_ax.tick_params(axis="y", labelsize=tick_font_size)
    first_ax.set_ylabel("Accuracy", fontsize=label_font_size, weight="bold")
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=legend_font_size,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=max(1, len(models)),
        frameon=False,
    )
    fig.subplots_adjust(top=0.78, bottom=0.24, wspace=0.24)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_paper_e1a(csv_path: Path, out_path: Path) -> None:
    """Render Figure 7 using the notebook-provided 2x2 mechanism-matrix style."""

    df = read_csv_if_exists(csv_path)
    if df.empty:
        return
    models = ordered_models_from_table(df.rename(columns={"main_accuracy": "accuracy"}), "overall_main_accuracy")
    if not models:
        models = df["model"].drop_duplicates().tolist()

    title_font_size = 8
    acc_font_size = 7
    label_font_size = 11
    cell_font_size = 7
    n_font_size = 6
    axis_linewidth = 2
    plot_spacing = 0.30

    fig, axes = plt.subplots(1, len(models), figsize=(len(models) * 1.35, 2.35), sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    cmap = PAPER_E1A_CMAP

    for i, model in enumerate(models):
        ax = axes_flat[i]
        sub = df[df["model"] == model]
        mat = np.full((2, 2), np.nan)
        nmat = np.zeros((2, 2), dtype=int)
        for _, row in sub.iterrows():
            r = 0 if int(row["visible_correct"]) == 1 else 1
            c = 0 if int(row["hidden_correct"]) == 1 else 1
            mat[r, c] = row["main_accuracy"]
            nmat[r, c] = int(row["n"])
        overall_acc = sub["overall_main_accuracy"].dropna().iloc[0] if sub["overall_main_accuracy"].notna().any() else np.nan

        for r_idx in range(2):
            for c_idx in range(2):
                val = mat[r_idx, c_idx]
                count = nmat[r_idx, c_idx]
                val_for_color = 0.0 if not np.isfinite(val) else val
                color = cmap(np.clip(val_for_color, 0, 1))
                text_color = "white" if val_for_color > 0.6 else "black"
                ax.add_patch(Rectangle((c_idx, 1 - r_idx), 1, 1, facecolor=color, edgecolor="black", lw=1))
                if np.isfinite(val):
                    ax.text(
                        c_idx + 0.5,
                        1 - r_idx + 0.60,
                        f"{val * 100:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=cell_font_size,
                        weight="bold",
                        color=text_color,
                    )
                    ax.text(
                        c_idx + 0.5,
                        1 - r_idx + 0.35,
                        f"n={count}",
                        ha="center",
                        va="center",
                        fontsize=n_font_size,
                        color=text_color,
                    )

        ax.text(0.5, 1.12, display_model(model), transform=ax.transAxes, ha="center", va="bottom", color=paper_model_color(model), weight="bold", fontsize=title_font_size)
        if np.isfinite(overall_acc):
            ax.text(0.5, 1.02, f"({overall_acc * 100:.1f}%)", transform=ax.transAxes, ha="center", va="bottom", color=paper_model_color(model), fontsize=acc_font_size)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect("equal")
        if i == 0:
            ax.text(-0.4, 1.5, r"V$\checkmark$", ha="center", va="center", fontsize=label_font_size, weight="bold")
            ax.text(-0.4, 0.5, r"V$\times$", ha="center", va="center", fontsize=label_font_size, weight="bold")
        ax.text(0.5, -0.3, r"H$\checkmark$", ha="center", va="center", fontsize=label_font_size, weight="bold")
        ax.text(1.5, -0.3, r"H$\times$", ha="center", va="center", fontsize=label_font_size, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine_name, spine in ax.spines.items():
            if spine_name in {"left", "bottom"}:
                spine.set_visible(True)
                spine.set_color(paper_model_color(model))
                spine.set_linewidth(axis_linewidth)
            else:
                spine.set_visible(False)

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.20, top=0.78, wspace=plot_spacing)
    pos = axes_flat[-1].get_position()
    cbar_ax = fig.add_axes([0.895, pos.y0, 0.010, pos.height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("$p_{correct}$", weight="bold", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_task_accuracy(csv_path: Path, out_path: Path, value_col: str, ylabel: str) -> None:
    """Render grouped task-accuracy bars for base, MCQ, image, or checkpoint figures."""

    df = read_csv_if_exists(csv_path)
    if df.empty or value_col not in df.columns:
        return
    task_col = source_task_col(df)
    tasks = [t for t in TASK_ORDER if t in set(df[task_col])]
    models = ordered_models_from_table(df, value_col)
    fig_w = max(12.8, 1.0 * len(tasks) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, 3.9))
    x = np.arange(len(tasks))
    bar_w = 0.82 / max(1, len(models))
    shade_task_regions(ax, tasks)
    for i, model in enumerate(models):
        vals = []
        sub = df[df["model"] == model].set_index(task_col)
        for task in tasks:
            vals.append(sub.loc[task, value_col] if task in sub.index else np.nan)
        ax.bar(
            x + (i - (len(models) - 1) / 2) * bar_w,
            vals,
            width=bar_w * 0.94,
            color=model_color(model),
            edgecolor="white",
            linewidth=0.7,
            label=display_model(model),
            zorder=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks], fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(min(0, np.nanmin(df[value_col]) - 0.05) if value_col.startswith("chance") else 0, 1.0)
    put_grid_behind(ax, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_difficulty(csv_path: Path, out_path: Path, value_col: str, ylabel: str) -> None:
    """Render difficulty-control panels from binned accuracy CSVs."""

    df = read_csv_if_exists(csv_path)
    if df.empty or value_col not in df.columns:
        return
    factors = [f for f in DIFFICULTY_FACTORS if f in set(df["factor"])]
    models = ordered_models_from_table(df, value_col)
    fig, axes = plt.subplots(1, len(factors), figsize=(4.1 * len(factors), 3.8), sharey=True, squeeze=False)
    for ax, factor in zip(axes.flatten(), factors):
        fdf = df[df["factor"] == factor]
        bins = fdf["bin"].drop_duplicates().tolist()
        x = np.arange(len(bins))
        bar_w = 0.82 / max(1, len(models))
        for i, model in enumerate(models):
            sub = fdf[fdf["model"] == model].set_index("bin")
            vals = [sub.loc[b, value_col] if b in sub.index else np.nan for b in bins]
            ax.bar(
                x + (i - (len(models) - 1) / 2) * bar_w,
                vals,
                width=bar_w * 0.94,
                color=model_color(model),
                edgecolor="white",
                linewidth=0.6,
                label=display_model(model),
                zorder=2,
            )
        ax.set_title(factor_display(factor), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, fontsize=8.5)
        put_grid_behind(ax, axis="y")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes.flatten()[0].set_ylabel(ylabel)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.995, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_e1a(csv_path: Path, out_path: Path) -> None:
    """Render the visible/hidden 2x2 conditional mechanism panels.

    All modalities use the same layout and color scale so the mechanism figures
    can be compared directly across text, MCQ, image-output, and trained text.
    """

    df = read_csv_if_exists(csv_path)
    if df.empty:
        return
    models = ordered_models_from_table(df.rename(columns={"main_accuracy": "accuracy"}), "overall_main_accuracy")
    if not models:
        models = df["model"].drop_duplicates().tolist()
    fig_w = max(5.5, 1.08 * len(models) + 0.8)
    fig = plt.figure(figsize=(fig_w, 2.7))
    gs = fig.add_gridspec(1, len(models), wspace=0.16)
    heat = None
    first_ax = None
    last_ax = None
    for i, model in enumerate(models):
        ax = fig.add_subplot(gs[0, i], sharey=first_ax)
        if first_ax is None:
            first_ax = ax
        last_ax = ax
        sub = df[df["model"] == model]
        mat = np.full((2, 2), np.nan)
        nmat = np.zeros((2, 2), dtype=int)
        for _, row in sub.iterrows():
            r = 0 if int(row["visible_correct"]) == 1 else 1
            c = 0 if int(row["hidden_correct"]) == 1 else 1
            mat[r, c] = row["main_accuracy"]
            nmat[r, c] = int(row["n"])
        heat = ax.imshow(mat, vmin=0, vmax=1, cmap=E1A_CMAP, aspect="equal")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["H+", "H-"], fontsize=7)
        ax.set_yticks([0, 1])
        if i == 0:
            ax.set_yticklabels(["V+", "V-"], fontsize=7)
        else:
            ax.set_yticklabels([])
        main_acc = sub["overall_main_accuracy"].dropna().iloc[0] if sub["overall_main_accuracy"].notna().any() else np.nan
        ax.set_title(f"{display_model(model)}\nMain {main_acc:.1%}", fontsize=6.4, color=model_color(model), pad=7)
        for r in range(2):
            for c in range(2):
                if np.isfinite(mat[r, c]):
                    ax.text(c, r, f"{mat[r,c]:.0%}\nn={nmat[r,c]}", ha="center", va="center", fontsize=5.0)
        for spine in ax.spines.values():
            spine.set_edgecolor(model_color(model))
            spine.set_linewidth(1.2)
    if heat is not None and last_ax is not None:
        cax = fig.add_axes([0.925, 0.23, 0.014, 0.55])
        cbar = fig.colorbar(heat, cax=cax)
        cbar.set_label("P(Object count correct)", fontsize=7)
        cbar.ax.tick_params(labelsize=6.2)
    fig.subplots_adjust(left=0.055, right=0.90, top=0.84, bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_wrong_answer_heatmap(csv_path: Path, out_path: Path, value_col: str = "raw_proportion") -> None:
    """Render Figures 5 and 16 using the notebook-provided wrong-answer heatmap style."""

    df = read_csv_if_exists(csv_path)
    if df.empty or value_col not in df.columns:
        return
    displayed = df["displayed_in_figure"].astype(bool) if "displayed_in_figure" in df.columns else pd.Series(False, index=df.index)
    shown = df[displayed].copy()
    if shown.empty:
        shown = df.sort_values("above_chance_p").head(80).copy()
    if shown.empty:
        return

    task_col = source_task_col(shown)
    rows = sorted(shown["model"].drop_duplicates(), key=lambda model: model_sort_key(model, None))
    grouped_cols: list[tuple[str, str]] = []
    spans: list[tuple[str, int, int]] = []
    x_ptr = 0
    for task in [task for task in TASK_ORDER if task in set(shown[task_col])]:
        task_df = shown[shown[task_col] == task]
        if task_df.empty:
            continue
        reason_scores = task_df.groupby("reason")[value_col].sum().sort_values(ascending=False)
        start = x_ptr
        for reason in reason_scores.index.tolist():
            grouped_cols.append((task, reason))
            x_ptr += 1
        spans.append((task, start, x_ptr - 1))
    if not grouped_cols:
        return

    mat = np.full((len(rows), len(grouped_cols)), np.nan)
    for _, row in shown.iterrows():
        key = (row[task_col], row["reason"])
        if row["model"] in rows and key in grouped_cols:
            mat[rows.index(row["model"]), grouped_cols.index(key)] = row[value_col]

    cell_aspect = 1.6
    cell_font_size = 7 if len(grouped_cols) <= 22 else 5.8
    title_font_size = 8
    y_label_font_size = 8
    x_label_font_size = 6.3
    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(grouped_cols) + 3.0), max(3.0, 0.34 * len(rows) + 1.3)))
    cmap = plt.get_cmap(WRONG_ANSWER_CMAP)
    rotation_bg = rotation_bg_rgba()

    for i, model in enumerate(rows):
        for j, (task, _) in enumerate(grouped_cols):
            bg = rotation_bg if task in {"task10", "task11"} else "white"
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0, fc=bg, zorder=0))
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            color_val = np.clip(val / WRONG_ANSWER_VMAX, 0, 1)
            face_color = cmap(color_val)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0.5, ec="black", fc=face_color, zorder=1))
            ax.text(
                j,
                i,
                f"{val * 100:.0f}%",
                ha="center",
                va="center",
                color="white" if val > WRONG_ANSWER_VMAX * 0.50 else "black",
                fontsize=cell_font_size,
                weight="bold",
                zorder=2,
            )

    for idx, (task, start, end) in enumerate(spans):
        is_rotation = task in {"task10", "task11"}
        display = paper_task_axis_label(task).replace("\n", " ")
        if is_rotation:
            display = display.replace(" Rotation", "\nRotation")
        else:
            display = display.replace(" ", "\n")
        ax.text(
            (start + end) / 2,
            -0.62,
            display,
            ha="center",
            va="bottom",
            fontsize=title_font_size,
            weight="bold",
            color=ROTATION_HIGHLIGHT_COLOR if is_rotation else "black",
            clip_on=False,
        )
        if idx < len(spans) - 1:
            ax.axvline(end + 0.5, color="black", lw=1.5, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([display_model(model) for model in rows], weight="bold", fontsize=y_label_font_size)
    for tick, model in zip(ax.get_yticklabels(), rows):
        tick.set_color(paper_model_color(model))
    ax.set_xticks(range(len(grouped_cols)))
    ax.set_xticklabels([split_label_near_midpoint(reason) for _, reason in grouped_cols], rotation=0, ha="center", fontsize=x_label_font_size)
    ax.set(xlim=(-0.5, len(grouped_cols) - 0.5), ylim=(len(rows) - 0.5, -0.5), aspect=1 / cell_aspect)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, WRONG_ANSWER_VMAX)),
        cax=cax,
        ticks=[0, 0.1, 0.2, 0.3, 0.4],
    )
    cbar.set_label("P(wrong type)", weight="bold", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_correlation_heatmap(csv_path: Path, out_path: Path) -> None:
    """Render Figure 13 mental-rotation correlation matrices."""

    df = read_csv_if_exists(csv_path)
    if df.empty:
        return
    rows = ordered_models_from_table(df.rename(columns={"pearson_r": "accuracy"}), "accuracy")
    focus_col = "source_focus_task" if "source_focus_task" in df.columns else "focus_task"
    task_col = source_task_col(df)
    focus = [t for t in ["task10", "task11"] if t in set(df[focus_col])]
    tasks = [t for t in TASK_ORDER if t in set(df[task_col])]
    modes = [("naive", "Naive phi"), ("within_scene", "Within-scene phi")]
    fig, axes = plt.subplots(2, 2, figsize=(max(12.4, len(tasks) * 0.98 + 1.0), max(7.2, len(rows) * 0.54 + 4.8)), squeeze=False)
    im = None
    for ri, ftask in enumerate(focus):
        for ci, (mode, mode_label) in enumerate(modes):
            ax = axes[ri, ci]
            sub = df[(df[focus_col] == ftask) & (df["correlation_type"] == mode)] if "correlation_type" in df.columns else df[df[focus_col] == ftask]
            mat = np.full((len(rows), len(tasks)), np.nan)
            pmat = np.full((len(rows), len(tasks)), np.nan)
            for _, row in sub.iterrows():
                row_task = row[task_col]
                if row["model"] in rows and row_task in tasks:
                    mat[rows.index(row["model"]), tasks.index(row_task)] = row["pearson_r"]
                    pmat[rows.index(row["model"]), tasks.index(row_task)] = row.get("p_value", np.nan)
            im = ax.imshow(np.ma.masked_invalid(mat), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            ax.set_title(f"{TASK_LONG_NAME[ftask]}\n{mode_label}", fontsize=8.8, pad=10)
            ax.set_yticks(np.arange(len(rows)))
            ax.set_yticklabels([display_model(m) for m in rows], fontsize=7.5)
            set_model_tick_colors(ax.get_yticklabels(), rows)
            ax.set_xticks(np.arange(len(tasks)))
            if ri == len(focus) - 1:
                ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks], rotation=90, fontsize=7)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            if "main" in tasks:
                main_idx = tasks.index("main")
                ax.axvline(main_idx - 0.5, color="black", lw=1.1, ls=":")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = mat[r, c]
                    if np.isfinite(val):
                        ax.text(c, r, f"{val:+.2f}{sig_stars(pmat[r,c])}", ha="center", va="center", fontsize=5.8, color="white" if abs(val) > 0.45 else "black")
    cax = fig.add_axes([0.925, 0.22, 0.016, 0.52])
    fig.colorbar(im, cax=cax, label="r")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.26, top=0.90, wspace=0.20, hspace=0.62)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_bundle_bars(csv_path: Path, out_path: Path) -> None:
    """Render Figure 14 aggregate and per-model bundle correlations."""

    df = read_csv_if_exists(csv_path)
    if df.empty:
        return
    bundles = [b for b in CROSSVIEW_BUNDLE_VARIANTS if b in set(df["bundle"])]
    aggregate = df[df.get("level", "per_model") == "aggregate"].set_index("bundle")
    per_df = df[df.get("level", "per_model") == "per_model"].copy()
    models = ordered_models_from_table(per_df.rename(columns={"pearson_r_with_object_count": "accuracy"}), "accuracy")
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(max(12, len(bundles) * 2.0), max(5.6, len(models) * 0.45 + 2.7)),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    colors = ["#2563eb", "#3b82f6", "#0f766e", "#7c3aed", "#dc2626"][:len(bundles)]
    x = np.arange(len(bundles))
    vals = [aggregate.loc[b, "pearson_r_with_object_count"] if b in aggregate.index else np.nan for b in bundles]
    pvals = [aggregate.loc[b, "p_value"] if b in aggregate.index else np.nan for b in bundles]
    ax_a.bar(x, vals, color=colors, alpha=0.86, width=0.72, zorder=2)
    ax_a.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    for xi, (val, p_val, color) in enumerate(zip(vals, pvals, colors)):
        if np.isfinite(val):
            ax_a.text(xi, val + (0.012 if val >= 0 else -0.03), sig_stars(p_val), ha="center", va="bottom", fontsize=10, color=color)
    ax_a.set_ylabel("Aggregate within-scene r")
    ax_a.set_title("Aggregate bundle -> Object Counting", fontsize=9.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(bundles, rotation=20, ha="right", fontsize=8)
    put_grid_behind(ax_a, axis="y", alpha=0.7)

    mat = np.full((len(models), len(bundles)), np.nan)
    pmat = np.full_like(mat, np.nan)
    for _, row in per_df.iterrows():
        if row["model"] in models and row["bundle"] in bundles:
            mat[models.index(row["model"]), bundles.index(row["bundle"])] = row["pearson_r_with_object_count"]
            pmat[models.index(row["model"]), bundles.index(row["bundle"])] = row.get("p_value", np.nan)
    im = ax_b.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_b.set_title("Per-model within-varying-scene r", fontsize=9.5)
    ax_b.set_xticks(np.arange(len(bundles)))
    ax_b.set_xticklabels(bundles, rotation=20, ha="right", fontsize=8)
    ax_b.set_yticks(np.arange(len(models)))
    ax_b.set_yticklabels([display_model(m) for m in models], fontsize=8)
    set_model_tick_colors(ax_b.get_yticklabels(), models)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            if np.isfinite(val):
                ax_b.text(c, r, f"{val:+.2f}{sig_stars(pmat[r,c])}", ha="center", va="center", fontsize=6.3, color="white" if abs(val) > 0.45 else "black")
    fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.025, label="r")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_camera_effects(csv_path: Path, out_path: Path) -> None:
    """Render Figure 18 perspective and offset effect bars."""

    df = read_csv_if_exists(csv_path)
    if df.empty:
        return
    task_col = source_task_col(df)
    tasks = [t for t in TASK_ORDER if t in set(df[task_col])]
    models = sorted(df["model"].drop_duplicates(), key=lambda m: model_sort_key(m, None))
    fig, axes = plt.subplots(2, 1, figsize=(max(15.5, len(tasks) * 1.18 + 2.2), 6.2), sharex=True)
    x = np.arange(len(tasks))
    bar_w = 0.84 / max(1, len(models))
    max_abs = max(0.08, float(np.nanmax(np.abs(df["delta_high_minus_low"]))) * 1.18) if df["delta_high_minus_low"].notna().any() else 0.08
    for ax, factor in zip(axes, CAMERA_FACTORS):
        shade_task_regions(ax, tasks)
        fdf = df[df["factor"] == factor]
        for i, model in enumerate(models):
            sub = fdf[fdf["model"] == model].set_index(task_col)
            vals = [sub.loc[t, "delta_high_minus_low"] if t in sub.index else np.nan for t in tasks]
            pvals = [sub.loc[t, "p_value"] if t in sub.index and "p_value" in sub.columns else np.nan for t in tasks]
            offset = (i - (len(models) - 1) / 2) * bar_w
            ax.bar(x + offset, vals, width=bar_w * 0.94, color=model_color(model), edgecolor="white", linewidth=0.7, label=display_model(model), zorder=2)
            for xi, val, p_val in zip(x + offset, vals, pvals):
                if np.isfinite(val) and sig_stars(p_val):
                    ax.text(xi, val + (0.004 if val >= 0 else -0.010), sig_stars(p_val), ha="center", va="bottom" if val >= 0 else "top", fontsize=6.8, color=model_color(model))
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_ylim(-max_abs, max_abs)
        ax.set_ylabel("Accuracy")
        ax.set_title("Perspective effect" if factor == "perspective" else "Offset effect", fontsize=11, pad=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        put_grid_behind(ax, axis="y")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([TASK_DISPLAY[t] for t in tasks], fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=max(1, len(models)), frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.14, top=0.90, hspace=0.32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_object_accuracy(csv_path: Path, out_path: Path) -> None:
    """Render Object Counting accuracy by object appearance."""

    df = read_csv_if_exists(csv_path)
    if df.empty:
        return
    value_col = "chance_adjusted_accuracy" if df["group"].astype(str).str.startswith("mcq").any() else "accuracy"
    models = ordered_models_from_table(df.rename(columns={value_col: "accuracy"}), "accuracy")
    objects = df["object_type"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(objects) * 1.45 + 4), 3.8))
    x = np.arange(len(objects))
    bar_w = 0.82 / max(1, len(models))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("object_type")
        vals = [sub.loc[o, value_col] if o in sub.index else np.nan for o in objects]
        ax.bar(x + (i - (len(models) - 1) / 2) * bar_w, vals, width=bar_w * 0.94, color=model_color(model), label=display_model(model), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(objects, fontsize=9)
    ax.set_ylabel("Object Counting accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    put_grid_behind(ax, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_object_difficulty_slices(csv_path: Path, out_path: Path) -> None:
    """Render Figure 21 object and difficulty slice summaries."""

    df = read_csv_if_exists(csv_path)
    if df.empty:
        return
    task_col = source_task_col(df)
    selected = [t for t in ["task2", "task5", "task6", "task7", "task8", "main"] if t in set(df[task_col])]
    object_rows = df[df["slice_type"] == "object"]
    difficulty_rows = df[df["slice_type"] == "difficulty"]
    object_difficulty = df[df["slice_type"] == "object_difficulty"]
    objects = [o for o in ["cube", "factory_box", "soup_can", "spam"] if o in set(object_rows["slice_value"])]
    objects += [o for o in object_rows["slice_value"].drop_duplicates().tolist() if o not in objects]
    bins = difficulty_rows["slice_value"].drop_duplicates().tolist()
    heat_labels = [TASK_DISPLAY[t].replace("\n", " ") for t in selected]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_obj, ax_bin, ax_main, ax_key = axes.ravel()

    obj_mat = np.full((len(objects), len(selected)), np.nan)
    for _, row in object_rows.iterrows():
        task = row[task_col]
        if row["slice_value"] in objects and task in selected:
            obj_mat[objects.index(row["slice_value"]), selected.index(task)] = row["accuracy"]
    im1 = ax_obj.imshow(np.ma.masked_invalid(obj_mat), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax_obj.set_xticks(np.arange(len(selected)))
    ax_obj.set_xticklabels(heat_labels, rotation=20, ha="right", fontsize=8)
    ax_obj.set_yticks(np.arange(len(objects)))
    ax_obj.set_yticklabels([object_display_name(o) for o in objects], fontsize=8.5)
    ax_obj.set_title("Mean performance by object type")
    for i in range(obj_mat.shape[0]):
        for j in range(obj_mat.shape[1]):
            val = obj_mat[i, j]
            if np.isfinite(val):
                ax_obj.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.2, color="white" if val > 0.65 else "black")
    fig.colorbar(im1, ax=ax_obj, fraction=0.046, pad=0.04, label="Accuracy")

    bin_mat = np.full((len(bins), len(selected)), np.nan)
    for _, row in difficulty_rows.iterrows():
        task = row[task_col]
        if row["slice_value"] in bins and task in selected:
            bin_mat[bins.index(row["slice_value"]), selected.index(task)] = row["accuracy"]
    im2 = ax_bin.imshow(np.ma.masked_invalid(bin_mat), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax_bin.set_xticks(np.arange(len(selected)))
    ax_bin.set_xticklabels(heat_labels, rotation=20, ha="right", fontsize=8)
    ax_bin.set_yticks(np.arange(len(bins)))
    ax_bin.set_yticklabels(bins, fontsize=8.5)
    ax_bin.set_title("Mean performance by Number of objects")
    for i in range(bin_mat.shape[0]):
        for j in range(bin_mat.shape[1]):
            val = bin_mat[i, j]
            if np.isfinite(val):
                ax_bin.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.2, color="white" if val > 0.65 else "black")
    fig.colorbar(im2, ax=ax_bin, fraction=0.046, pad=0.04, label="Accuracy")

    x = np.arange(len(bins))
    for obj in objects:
        vals = []
        for b in bins:
            key = f"{obj}|{b}"
            sub = object_difficulty[(object_difficulty["slice_value"] == key) & (object_difficulty[task_col] == "main")]
            vals.append(sub["accuracy"].iloc[0] if not sub.empty else np.nan)
        vals_arr = np.asarray(vals, dtype=float)
        ok = np.isfinite(vals_arr)
        if ok.any():
            ax_main.plot(x[ok], vals_arr[ok], marker="o", ms=5, lw=1.8, color=OBJECT_COLORS.get(obj, "gray"), label=object_display_name(obj), alpha=0.9)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(bins, rotation=20)
    ax_main.set_ylim(0, 1)
    ax_main.set_ylabel("Object Counting accuracy")
    ax_main.set_title("Object Counting by object x difficulty")
    ax_main.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    put_grid_behind(ax_main, axis="y")
    ax_main.legend(fontsize=8)

    key_tasks = [("main", "Object count", "#111827"), ("task5", "Visible count", "#2563eb"), ("task8", "Hidden count", "#dc2626")]
    for task, label, color in key_tasks:
        vals = []
        for b in bins:
            sub = difficulty_rows[(difficulty_rows["slice_value"] == b) & (difficulty_rows[task_col] == task)]
            vals.append(sub["accuracy"].iloc[0] if not sub.empty else np.nan)
        vals_arr = np.asarray(vals, dtype=float)
        ok = np.isfinite(vals_arr)
        if ok.any():
            ax_key.plot(x[ok], vals_arr[ok], marker="o", ms=5, lw=2.0, color=color, label=label)
    ax_key.set_xticks(x)
    ax_key.set_xticklabels(bins, rotation=20)
    ax_key.set_ylim(0, 1)
    ax_key.set_ylabel("Accuracy")
    ax_key.set_title("Key tasks by difficulty")
    ax_key.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    put_grid_behind(ax_key, axis="y")
    ax_key.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_composition(csv_path: Path, out_path: Path) -> None:
    """Render numerical composition diagnostics for Figures 22 and 28."""

    df = read_csv_if_exists(csv_path)
    if df.empty:
        return
    metrics = [
        ("prediction_correlation_r", "Pred. corr.", "corr"),
        ("exact_match_rate", "Exact match", "rate"),
        ("both_wrong_exact_match_rate", "Both wrong exact", "rate"),
        ("mean_absolute_residual", "MAD", "positive"),
        ("residual_bias", "Bias", "signed"),
    ]
    sort_col = "prediction_correlation_r" if "prediction_correlation_r" in df.columns else metrics[0][0]
    model_scores = df.set_index("model")[sort_col].to_dict()
    models = sorted(df["model"].drop_duplicates(), key=lambda m: (-float(model_scores.get(m, -np.inf)) if np.isfinite(model_scores.get(m, np.nan)) else np.inf, model_sort_key(m, model_scores.get(m))))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.05 * len(metrics), max(3.2, 0.34 * len(models) + 1.0)), sharey=True)
    y = np.arange(len(models))
    sub = df.set_index("model")
    for ax, (col, label, kind) in zip(axes, metrics):
        sub = df.set_index("model")
        vals = [sub.loc[m, col] if m in sub.index else np.nan for m in models]
        vals_arr = np.asarray(vals, dtype=float)
        valid = np.isfinite(vals_arr)
        ax.barh(y[valid], vals_arr[valid], color=np.asarray([model_color(m) for m in models], dtype=object)[valid], alpha=0.88, zorder=2)
        ax.set_title(label, fontsize=10)
        if kind == "signed":
            limit = max(0.5, float(np.nanmax(np.abs(vals_arr))) * 1.2) if valid.any() else 0.5
            ax.set_xlim(-limit, limit)
            ax.axvline(0, color="gray", linewidth=0.9, linestyle="--", zorder=1)
        elif kind == "positive":
            xmax = max(1.0, float(np.nanmax(vals_arr)) * 1.15) if valid.any() else 1.0
            ax.set_xlim(0, xmax)
            ax.axvline(0, color="black", linewidth=0.8, zorder=1)
        else:
            ax.set_xlim(0, 1.0)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.axvline(0, color="black", linewidth=0.8, zorder=1)
        for yi, val in zip(y, vals_arr):
            if not np.isfinite(val):
                continue
            if kind == "rate":
                txt = f"{val:.0%}"
                xpos = val + 0.015
                ha = "left"
            elif kind == "signed":
                txt = f"{val:+.2f}"
                lo, hi = ax.get_xlim()
                xpos = hi - 0.04 * (hi - lo)
                ha = "right"
            else:
                txt = f"{val:.2f}"
                xpos = val + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                ha = "left"
            ax.text(xpos, yi, txt, va="center", ha=ha, fontsize=7.3)
        put_grid_behind(ax, axis="x", alpha=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([display_model(m) for m in models], fontsize=8)
    set_model_tick_colors(axes[0].get_yticklabels(), models)
    axes[0].invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse public-release CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("inference_results"), help="Directory containing eval_raw_*.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("analyses_results"), help="Directory for generated analysis tables and figures")
    parser.add_argument("--stage", choices=["all", "analysis", "figures", "manifest"], default="all")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["text", "mcq5", "mcq345", "image", "trained_text"],
        help="Data groups to process: text mcq5 mcq345 image trained_text",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "analysis", "manifest"}:
        export_manifest(args.output_dir)
    if args.stage in {"all", "analysis"}:
        run_analysis(args.input_dir, args.output_dir, args.groups)
    if args.stage in {"all", "figures"}:
        render_all_figures(args.output_dir)
    print(f"Public Spatial-IQ outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
