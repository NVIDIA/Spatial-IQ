#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Appendix F hierarchy-dependency statistics for Spatial-IQ.

Reproduces the numbers behind the "Full Hierarchy Dependency Analysis"
(Appendix F): per-edge within-scene lifts, joint relation integrity,
bundle-to-Object-Counting couplings, and their significance. These quantities are
reported in the paper prose/figures but are not produced by the
figure-generation script ``spatial-IQ_analyses.py``; this module fills that gap
so every quantitative claim in Appendix F is reproducible from the released code
and data.

Method (matches the reference analysis):
  - within-scene lift  Delta = P(child=1 | parent met) - P(child=1 | parent unmet),
    restricted to scenes (``view``) where the parent varies across responders, so
    scene difficulty is held fixed. Significance via a 2x2 chi-square on the
    within-scene contingency table and a view-blocked permutation null that
    shuffles the parent labels within each responder (preserving each
    responder's marginal accuracy), with Benjamini-Hochberg FDR across relations.
  - joint relation integrity I(P1 & P2 -> C) = P(P1=1 & P2=1 | C=1), the stricter
    conjunction variant of the per-edge integrity averaged in Table 1.
  - bundle-to-Main coupling r_ws = within-``view`` demeaned (fixed-effect)
    Pearson correlation between the bundle score (fraction of bundle tasks
    correct) and Object Counting.
  - responder-controlled robustness: a two-way (scene + responder) fixed-effect
    slope, the marginal lift inside each responder, and both statistics with the
    human baseline dropped.

Hidden-Count Hierarchy input: the Support Column (task7 = S8), consistent with
Figure 3 and the Table 1 conditional-accuracy columns. The submitted Appendix F
prose instead instantiated this relation with Direct Support (task6 = S7), and
its Hidden-relation statistics were computed accordingly; the S7 variant is
retained in ``LEGACY_JOINT_EDGES`` so both the submitted and the corrected values
are reproducible from this module. Depends only on pandas, numpy, and scipy.

Internal task columns map to paper sub-tasks as:
  task2=S6 Top Layer, task3=S3 Column Count, task4=S4 Layer Count,
  task5=S5 Visible Object Count, task6=S7 Direct Support,
  task7=S8 Support Column, task8=S9 Hidden Object Count, main=T1 Object Counting.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Pre-specified relations (Hidden standardized on Support Column, task7 = S8)
# ---------------------------------------------------------------------------

# Unary edges: (parent, child, group_label). Parent is the lower-level task.
UNARY_EDGES = [
    ("task2", "task6", "Internal Referential Chain"),   # Top Layer -> Direct Support
    ("task6", "task7", "Internal Referential Chain"),   # Direct Support -> Support Column
    ("task3", "task5", "Visible Count Hierarchy"),       # Column -> Visible
    ("task4", "task5", "Visible Count Hierarchy"),       # Layer -> Visible
    ("task3", "task8", "Hidden Count Hierarchy"),         # Column -> Hidden
    ("task7", "task8", "Hidden Count Hierarchy"),         # Support Column (S8) -> Hidden
    ("task5", "main", "Summation Mechanism"),            # Visible -> Total
    ("task8", "main", "Summation Mechanism"),            # Hidden -> Total
]

# Joint (conjunction) edges: (parent_1, parent_2, child, group_label).
JOINT_EDGES = [
    ("task3", "task4", "task5", "Visible Count Hierarchy"),   # Column & Layer -> Visible
    ("task3", "task7", "task8", "Hidden Count Hierarchy"),    # Column & Support Column (S8) -> Hidden
]

# Variant kept only to reproduce the submitted Appendix F numbers, which
# instantiated the Hidden-Count Hierarchy with Direct Support (task6 = S7)
# instead of the pre-specified Support Column (task7 = S8) shown in Figure 3.
LEGACY_JOINT_EDGES = [
    ("task3", "task6", "task8", "Hidden Count Hierarchy (submitted S7 variant)"),
]

# Bundles for bundle-to-Main coupling (Hidden = Support Column, task7 = S8).
HIERARCHY_BUNDLES = [
    ("Internal Referential Chain", ["task2", "task6", "task7"]),
    ("Visible Count Hierarchy", ["task3", "task4", "task5"]),
    ("Hidden Count Hierarchy", ["task3", "task7", "task8"]),
    ("Summation Mechanism", ["task5", "task8"]),
]

TASK_COLS = [f"task{i}" for i in range(1, 12)] + ["main"]

TASK_DISPLAY = {
    "task2": "Top Layer", "task3": "Column Count", "task4": "Layer Count",
    "task5": "Visible Object Count", "task6": "Direct Support",
    "task7": "Support Column", "task8": "Hidden Object Count", "main": "Object Counting",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _is_trained(model: str) -> bool:
    m = model.lower()
    return m.startswith("dapo") or m.startswith("sft-") or m.startswith("qwen2.5-vl")


def load_text_group(input_dir: Path, group: str = "text") -> dict[str, pd.DataFrame]:
    """Load ``eval_raw_*_text.csv`` (+ human) into ``model -> DataFrame``.

    ``group='text'`` excludes trained checkpoints; ``group='trained_text'`` keeps
    only trained checkpoints (+ human).
    """
    out: dict[str, pd.DataFrame] = {}
    files = sorted(input_dir.glob("eval_raw_*_text.csv"))
    human = input_dir / "eval_raw_human_frq.csv"
    if human.exists():
        files.append(human)
    for path in files:
        stem = path.stem.replace("eval_raw_", "").replace("_text", "")
        model = "human" if stem in ("human_frq", "human") else stem
        if group == "text" and _is_trained(model):
            continue
        if group == "trained_text" and not (_is_trained(model) or model == "human"):
            continue
        df = pd.read_csv(path)
        if "task_main" in df.columns and "main" not in df.columns:
            df = df.rename(columns={"task_main": "main"})
        for c in TASK_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out[model] = df
    return out


def _pool(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in data.items():
        d = df.copy()
        d["_model"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (numpy-only, no new deps)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if not ok.any():
        return out
    q = p[ok]
    n = q.size
    order = np.argsort(q)
    ranked = q[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def _chi2_p(met: pd.Series, unmet: pd.Series) -> float:
    if len(met) == 0 or len(unmet) == 0:
        return np.nan
    table = [[int(met.sum()), len(met) - int(met.sum())],
             [int(unmet.sum()), len(unmet) - int(unmet.sum())]]
    try:
        _, p, _, _ = stats.chi2_contingency(table, correction=False)
        return float(p)
    except ValueError:
        return np.nan


def _fe_corr(frame: pd.DataFrame, xcol: str, ycol: str, view: str = "view") -> tuple[float, float]:
    """Within-``view`` demeaned (fixed-effect) Pearson correlation."""
    sub = frame[[view, xcol, ycol]].dropna().copy()
    if len(sub) < 3:
        return np.nan, np.nan
    sub["_x"] = sub[xcol] - sub.groupby(view)[xcol].transform("mean")
    sub["_y"] = sub[ycol] - sub.groupby(view)[ycol].transform("mean")
    valid = sub["_x"].notna() & sub["_y"].notna()
    if valid.sum() < 3 or sub.loc[valid, "_x"].std() == 0 or sub.loc[valid, "_y"].std() == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(sub.loc[valid, "_x"], sub.loc[valid, "_y"])
    return float(r), float(p)


def _bootstrap_ci(ws_full: pd.DataFrame, pcol: str, ccol: str, varying: list,
                  n_boot: int, seed: int) -> tuple[float, float]:
    """Scene-level cluster bootstrap 95% CI for the within-scene lift.

    Resamples scenes (``view``) with replacement and pools their rows, matching a
    cluster bootstrap. Vectorized via per-view sufficient statistics so cost is
    O(n_boot * n_views) rather than concatenating frames each iteration."""
    varying = list(varying)
    if len(varying) < 4:
        return np.nan, np.nan
    d = ws_full[ws_full["view"].isin(varying)].dropna(subset=[pcol, ccol])
    met, unmet = d[d[pcol] == 1], d[d[pcol] == 0]
    mc = met.groupby("view")[ccol].sum().reindex(varying, fill_value=0).to_numpy()
    mn = met.groupby("view")[ccol].size().reindex(varying, fill_value=0).to_numpy()
    uc = unmet.groupby("view")[ccol].sum().reindex(varying, fill_value=0).to_numpy()
    un = unmet.groupby("view")[ccol].size().reindex(varying, fill_value=0).to_numpy()
    rng = np.random.default_rng(seed)
    v = len(varying)
    lifts = []
    for _ in range(n_boot):
        idx = rng.integers(0, v, v)
        mn_s, un_s = mn[idx].sum(), un[idx].sum()
        if mn_s > 0 and un_s > 0:
            lifts.append(mc[idx].sum() / mn_s - uc[idx].sum() / un_s)
    if len(lifts) < 10:
        return np.nan, np.nan
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


def _perm_null(ws_perm: pd.DataFrame, pcol: str, ccol: str, obs_lift: float,
               n_perm: int, seed: int) -> tuple[float, float]:
    """View-blocked permutation null: shuffle parent labels within each responder
    (preserving each responder's marginal accuracy), recompute pooled lift."""
    if n_perm <= 0 or np.isnan(obs_lift):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    groups = [g for _, g in ws_perm.groupby("_model", sort=False)]
    child_by_g = [g[ccol].values for g in groups]
    parent_by_g = [g[pcol].values.copy() for g in groups]
    null_lifts = []
    for _ in range(n_perm):
        met_c = met_n = unmet_c = unmet_n = 0.0
        for parent, child in zip(parent_by_g, child_by_g):
            perm = rng.permutation(parent)
            met_mask = perm == 1
            met_c += child[met_mask].sum()
            met_n += met_mask.sum()
            unmet_c += child[~met_mask].sum()
            unmet_n += (~met_mask).sum()
        if met_n > 0 and unmet_n > 0:
            null_lifts.append(met_c / met_n - unmet_c / unmet_n)
    if not null_lifts:
        return np.nan, np.nan
    null_lifts = np.asarray(null_lifts)
    perm_p = (1 + int((null_lifts >= obs_lift).sum())) / (1 + len(null_lifts))
    return float(perm_p), float(null_lifts.mean())


# ---------------------------------------------------------------------------
# Within-scene lift
# ---------------------------------------------------------------------------


def _lift_stats(sub, pcol, ccol, varying, n_perm, n_boot, seed):
    """Within-scene lift record. ``sub`` has [view, _model, pcol, ccol];
    ``varying`` is the set/list of scene ids where the relation's parent(s) vary
    across responders (computed by the caller so unary and joint relations can
    use their respective varying-scene definitions)."""
    if len(sub) < 10:
        return None
    naive_lift = float(sub[sub[pcol] == 1][ccol].mean() - sub[sub[pcol] == 0][ccol].mean())
    varying = list(varying)
    ws = sub[sub["view"].isin(varying)]
    met = ws[ws[pcol] == 1][ccol].dropna()
    unmet = ws[ws[pcol] == 0][ccol].dropna()
    if len(met) == 0 or len(unmet) == 0:
        return None
    ws_met, ws_unmet = float(met.mean()), float(unmet.mean())
    ws_lift = ws_met - ws_unmet
    ci_lo, ci_hi = _bootstrap_ci(sub[["view", pcol, ccol]], pcol, ccol, varying, n_boot, seed)
    r_within, p_within = _fe_corr(sub, pcol, ccol)
    perm_p, null_mean = _perm_null(ws[["_model", pcol, ccol]], pcol, ccol, ws_lift, n_perm, seed)
    return {
        "naive_lift": naive_lift, "ws_met": ws_met, "ws_unmet": ws_unmet, "ws_lift": ws_lift,
        "ws_ci_low": ci_lo, "ws_ci_high": ci_hi, "p_ws_lift": _chi2_p(met, unmet),
        "r_within": r_within, "p_within": p_within, "null_mean": null_mean,
        "perm_p": perm_p, "n_varying": len(varying),
    }


def within_scene_lift_table(data: dict[str, pd.DataFrame], modality: str = "text",
                            n_perm: int = 500, n_boot: int = 1000,
                            seed: int = 42) -> pd.DataFrame:
    combined = _pool(data)
    if combined.empty or "view" not in combined.columns:
        return pd.DataFrame()
    rows = []
    for pc, nc, grp in UNARY_EDGES:
        if pc not in combined.columns or nc not in combined.columns:
            continue
        sub = combined[["view", "_model", pc, nc]].dropna()
        rng = sub.groupby("view")[pc].agg(lambda x: x.max() - x.min())
        varying = set(rng[rng > 0].index)   # scenes where the parent varies across responders
        rec = _lift_stats(sub, pc, nc, varying, n_perm, n_boot, seed)
        if rec:
            rec.update({"group": grp, "relation": f"{pc}->{nc}", "kind": "unary",
                        "parent_1": pc, "parent_2": "", "target": nc})
            rows.append(rec)
    for p1, p2, nc, grp in JOINT_EDGES:
        if any(c not in combined.columns for c in (p1, p2, nc)):
            continue
        jc = f"_joint_{p1}_{p2}"
        combined[jc] = ((combined[p1] == 1) & (combined[p2] == 1)).astype(float)
        combined.loc[combined[p1].isna() | combined[p2].isna(), jc] = np.nan
        sub = combined[["view", "_model", p1, p2, jc, nc]].dropna()
        r1 = sub.groupby("view")[p1].agg(lambda x: x.max() - x.min())
        r2 = sub.groupby("view")[p2].agg(lambda x: x.max() - x.min())
        varying = set(r1[(r1 > 0) | (r2 > 0)].index)   # scenes where EITHER parent varies
        rec = _lift_stats(sub, jc, nc, varying, n_perm, n_boot, seed)
        if rec:
            rec.update({"group": grp, "relation": f"{p1}&{p2}->{nc}", "kind": "joint",
                        "parent_1": p1, "parent_2": p2, "target": nc})
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_ws_lift_fdr"] = bh_fdr(out["p_ws_lift"].values)
        out["perm_p_fdr"] = bh_fdr(out["perm_p"].values)
        out.insert(0, "modality", modality)
    return out


# ---------------------------------------------------------------------------
# Bundle-to-Main coupling
# ---------------------------------------------------------------------------


def bundle_coupling_table(data: dict[str, pd.DataFrame], modality: str = "text",
                          main_col: str = "main") -> pd.DataFrame:
    combined = _pool(data)
    if combined.empty or "view" not in combined.columns or main_col not in combined.columns:
        return pd.DataFrame()
    agg_rows, per_rows = [], []
    per_p_keys, per_p_vals = [], []
    for name, tasks in HIERARCHY_BUNDLES:
        cols = [t for t in tasks if t in combined.columns]
        if not cols:
            continue
        bcol = f"_bundle_{name[:6]}"
        combined[bcol] = combined[cols].mean(axis=1)
        r_ws, p_ws = _fe_corr(combined, bcol, main_col)
        sub = combined[["view", bcol]].dropna()
        rng = sub.groupby("view")[bcol].agg(lambda x: x.max() - x.min())
        n_varying = int((rng > 0).sum())
        agg_rows.append({"modality": modality, "bundle": name, "model": "aggregate",
                         "r_ws": r_ws, "p_ws": p_ws, "n_varying": n_varying})
        varying = set(rng[rng > 0].index)
        for mname, df in data.items():
            cm = [t for t in tasks if t in df.columns]
            if not cm or main_col not in df.columns or "view" not in df.columns:
                continue
            d = df.copy()
            d[bcol] = d[cm].mean(axis=1)
            wm = d[d["view"].isin(varying)][[bcol, main_col]].dropna()
            if len(wm) > 2 and wm[bcol].std() > 0 and wm[main_col].std() > 0:
                r_m, p_m = stats.pearsonr(wm[bcol].values, wm[main_col].values)
            else:
                r_m, p_m = np.nan, np.nan
            per_rows.append({"modality": modality, "bundle": name, "model": mname,
                             "r_ws": float(r_m), "p_ws": float(p_m), "n_varying": n_varying})
            per_p_keys.append(len(per_rows) - 1)
            per_p_vals.append(p_m)
    per_df = pd.DataFrame(per_rows)
    if not per_df.empty:
        per_df["p_ws_fdr"] = np.nan
        per_df.loc[per_p_keys, "p_ws_fdr"] = bh_fdr(per_p_vals)
    agg_df = pd.DataFrame(agg_rows)
    if not agg_df.empty:
        agg_df["p_ws_fdr"] = bh_fdr(agg_df["p_ws"].values)
    return pd.concat([agg_df, per_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Joint relation integrity
# ---------------------------------------------------------------------------


def joint_integrity_table(data: dict[str, pd.DataFrame], modality: str = "text",
                          include_legacy: bool = True) -> pd.DataFrame:
    """Joint relation integrity I(P1 & P2 -> C) = P(P1=1 & P2=1 | C=1) per responder.

    This is the stricter conjunction variant reported in the Appendix F prose. It
    is *not* the Table 1 quantity: Table 1 averages the two per-edge integrities
    I(P -> C), so the two differ for the same relation and responder. Rows tagged
    ``variant='submitted_s7'`` reproduce the Hidden values as computed for the
    submitted appendix; ``variant='prespecified_s8'`` is the corrected one.
    """
    edges = [(*e, "prespecified_s8") for e in JOINT_EDGES]
    if include_legacy:
        edges += [(*e, "submitted_s7") for e in LEGACY_JOINT_EDGES]
    rows = []
    for p1, p2, nc, grp, variant in edges:
        for name, df in data.items():
            if any(c not in df.columns for c in (p1, p2, nc)):
                continue
            sub = df[[p1, p2, nc]].dropna()
            sub = sub[sub[nc] == 1]
            if sub.empty:
                continue
            rows.append({
                "modality": modality, "variant": variant, "group": grp,
                "relation": f"{p1}&{p2}->{nc}", "model": name,
                "joint_integrity": float(((sub[p1] == 1) & (sub[p2] == 1)).mean()),
                "n_child_correct": int(len(sub)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Responder-controlled robustness (two-way scene + responder fixed effects)
# ---------------------------------------------------------------------------
#
# The within-scene lift above holds the scene fixed but varies the parent across
# responders, which only *partially* controls for difficulty (an abler responder
# tends to do better on everything). These functions add a responder fixed
# effect via a two-way (scene + responder) within transform, so the slope is
# identified from within-scene, between-responder variation net of each
# responder's own mean accuracy. Effect sizes shrink but sign/significance are
# preserved, showing the dependence is not merely a responder-ability artifact.


def _demean_twoway(frame, cols, g1="view", g2="_model", iters=50, tol=1e-10):
    """Two-way within transform by alternating projections on g1 and g2."""
    out = frame[cols].astype(float).copy()
    k1, k2 = frame[g1].values, frame[g2].values
    for _ in range(iters):
        prev = out.values.copy()
        out = out - out.groupby(k1).transform("mean")
        out = out - out.groupby(k2).transform("mean")
        if np.nanmax(np.abs(out.values - prev)) < tol:
            break
    return out


def _twoway_slope(sub, xcol, ycol, n_boot, seed):
    """Two-way FE slope of ycol on xcol with a scene-cluster bootstrap CI."""
    dm = _demean_twoway(sub, [xcol, ycol])
    x, y = dm[xcol].values, dm[ycol].values
    denom = float(np.dot(x, x))
    slope = float(np.dot(x, y) / denom) if denom > 0 else np.nan
    rng = np.random.default_rng(seed)
    views = sub["view"].unique()
    idx_by_view = {v: np.flatnonzero(sub["view"].values == v) for v in views}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(views, size=len(views), replace=True)
        rows = np.concatenate([idx_by_view[v] for v in pick])
        bs = sub.iloc[rows].copy()
        bs["view"] = np.repeat(np.arange(len(pick)), [len(idx_by_view[v]) for v in pick])
        bdm = _demean_twoway(bs, [xcol, ycol])
        bx, by = bdm[xcol].values, bdm[ycol].values
        d = float(np.dot(bx, bx))
        if d > 0:
            boots.append(np.dot(bx, by) / d)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return slope, float(lo), float(hi), float(np.mean(np.asarray(boots) <= 0))
    return slope, np.nan, np.nan, np.nan


def per_responder_lift_table(data: dict[str, pd.DataFrame], modality: str = "text",
                             min_cell: int = 20) -> pd.DataFrame:
    """Marginal lift for each joint hierarchy *inside* each responder.

    This is responder-controlled but **not** scene-controlled: it pools a single
    responder's scenes, so it complements rather than replaces the two-way FE
    slope. Responders with fewer than ``min_cell`` observations in either parent
    condition are reported with ``included=False`` and a NaN lift, so the
    denominator behind the "positive in k of n responders" claim is explicit.
    """
    rows = []
    for p1, p2, nc, grp in JOINT_EDGES:
        for name, df in data.items():
            if any(c not in df.columns for c in (p1, p2, nc)):
                continue
            sub = df[[p1, p2, nc]].dropna()
            met = (sub[p1] == 1) & (sub[p2] == 1)
            n_met, n_unmet = int(met.sum()), int((~met).sum())
            included = n_met >= min_cell and n_unmet >= min_cell
            rows.append({
                "modality": modality, "relation": f"{p1}&{p2}->{nc}", "group": grp,
                "model": name, "n_met": n_met, "n_unmet": n_unmet, "included": included,
                "marginal_lift": (float(sub.loc[met, nc].mean() - sub.loc[~met, nc].mean())
                                  if included else np.nan),
            })
    return pd.DataFrame(rows)


def responder_controlled_lift_table(data: dict[str, pd.DataFrame], modality: str = "text",
                                    n_boot: int = 2000, seed: int = 1,
                                    min_cell: int = 20,
                                    include_models_only: bool = True) -> pd.DataFrame:
    """Scene-only lift vs two-way (scene + responder) FE slope for each joint
    hierarchy, plus the count of responders whose own marginal lift is positive.

    Emits one row per (relation, responder subset). ``subset='all'`` pools the
    model systems and the human baseline; ``subset='models_only'`` drops the
    human baseline, since the human row is far more accurate than any model and
    could otherwise carry the pooled contrast on its own.
    """
    subsets = [("all", data)]
    if include_models_only and "human" in data:
        subsets.append(("models_only", {k: v for k, v in data.items() if k != "human"}))
    rows = []
    for subset_name, subset in subsets:
        combined = _pool(subset)
        if combined.empty or "view" not in combined.columns:
            continue
        per_resp = per_responder_lift_table(subset, modality, min_cell)
        for p1, p2, nc, grp in JOINT_EDGES:
            if any(c not in combined.columns for c in (p1, p2, nc)):
                continue
            sub = combined[["view", "_model", p1, p2, nc]].dropna().copy()
            sub["_x"] = (sub[p1].to_numpy() * sub[p2].to_numpy()).astype(float)
            r1 = sub.groupby("view")[p1].agg(lambda x: x.max() - x.min())
            r2 = sub.groupby("view")[p2].agg(lambda x: x.max() - x.min())
            varying = set(r1[(r1 > 0) | (r2 > 0)].index)
            ws = sub[sub["view"].isin(varying)]
            met, unmet = ws.loc[ws["_x"] == 1, nc], ws.loc[ws["_x"] == 0, nc]
            scene_only = float(met.mean() - unmet.mean()) if len(met) and len(unmet) else np.nan
            slope, lo, hi, p_le0 = _twoway_slope(sub, "_x", nc, n_boot, seed)
            marg = per_resp[(per_resp["relation"] == f"{p1}&{p2}->{nc}") & per_resp["included"]]
            rows.append({
                "modality": modality, "subset": subset_name,
                "relation": f"{p1}&{p2}->{nc}", "group": grp,
                "scene_only_lift": scene_only, "twoway_fe_slope": slope,
                "ci_low": lo, "ci_high": hi, "p_boot_le0": p_le0,
                "responders_positive": int((marg["marginal_lift"] > 0).sum()),
                "responders_total": int(len(marg)),
                "responders_excluded": int(len(data) - len(marg)) if subset_name == "all" else np.nan,
                "n": len(sub),
            })
    return pd.DataFrame(rows)


def bundle_coupling_twoway_table(data: dict[str, pd.DataFrame], modality: str = "text",
                                 main_col: str = "main") -> pd.DataFrame:
    """Bundle-to-Object-Counting coupling: one-way (scene) vs two-way (scene +
    responder) demeaned correlation, for each hierarchy bundle."""
    combined = _pool(data)
    if combined.empty or "view" not in combined.columns or main_col not in combined.columns:
        return pd.DataFrame()
    rows = []
    for name, tasks in HIERARCHY_BUNDLES:
        cols = [t for t in tasks if t in combined.columns]
        if not cols:
            continue
        sub = pd.DataFrame({
            "view": combined["view"], "_model": combined["_model"],
            "cc": combined[cols].mean(axis=1), "mc": combined[main_col],
        }).dropna()
        if len(sub) < 3:
            continue
        one = sub[["cc", "mc"]] - sub.groupby("view")[["cc", "mc"]].transform("mean")
        two = _demean_twoway(sub, ["cc", "mc"])
        r_one = float(stats.pearsonr(one["cc"], one["mc"])[0]) if one["cc"].std() else np.nan
        r_two = float(stats.pearsonr(two["cc"], two["mc"])[0]) if two["cc"].std() else np.nan
        rows.append({"modality": modality, "bundle": name,
                     "r_scene": r_one, "r_scene_responder": r_two, "n": len(sub)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise McNemar on Object Counting (Table 6 between-model significance)
# ---------------------------------------------------------------------------


def mcnemar_main_table(data: dict[str, pd.DataFrame], modality: str = "text",
                       main_col: str = "main", min_common: int = 10) -> pd.DataFrame:
    """All-pairs exact McNemar on Object Counting with BH-FDR (Table 6).

    Models are ranked by Object Counting accuracy; each pair is aligned on shared
    scene ids (``view``) and tested with an exact two-sided McNemar (binomial on
    the discordant pairs). ``is_adjacent`` flags consecutive pairs in the ranking.
    """
    accs = {m: float(df[main_col].mean()) for m, df in data.items()
            if main_col in df.columns and "view" in df.columns}
    names = sorted(accs, key=lambda m: -accs[m])
    idx = {m: data[m].set_index("view") for m in names}
    rows, pvals = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = idx[a].index.intersection(idx[b].index)
            if len(common) < min_common:
                p, n_b, n_c = np.nan, 0, 0
            else:
                y1 = idx[a].loc[common, main_col].fillna(0).to_numpy().astype(int)
                y2 = idx[b].loc[common, main_col].fillna(0).to_numpy().astype(int)
                n_b = int(((y1 == 0) & (y2 == 1)).sum())
                n_c = int(((y1 == 1) & (y2 == 0)).sum())
                p = 1.0 if (n_b + n_c) == 0 else float(
                    stats.binomtest(n_b, n_b + n_c, 0.5, alternative="two-sided").pvalue)
            rows.append({
                "modality": modality, "model_a": a, "model_b": b,
                "acc_a": accs[a], "acc_b": accs[b], "n_common": int(len(common)),
                "a_wrong_b_right": n_b, "a_right_b_wrong": n_c,
                "p_mcnemar": p, "is_adjacent": (j == i + 1),
            })
            pvals.append(p)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_mcnemar_fdr"] = bh_fdr(out["p_mcnemar"].values)
    return out


# ---------------------------------------------------------------------------
# Within-structure camera-parameter effects (Fig. 18)
# ---------------------------------------------------------------------------


def _add_struct_id(df: pd.DataFrame) -> pd.DataFrame:
    """Derive structure id and camera factors from the ``view`` path."""
    d = df.copy()
    d["_struct_id"] = d["view"].str.replace(r"/offset[\d.]+_fov\d+_dist[\d.]+$", "", regex=True)
    fov = d["view"].str.extract(r"fov(\d+)_")[0].astype(float)
    d["_perspective"] = (fov >= 3).astype(float)
    d["_offset"] = d["view"].str.extract(r"offset([\d.]+)_")[0].astype(float)
    return d


def camera_within_structure_table(data: dict[str, pd.DataFrame], modality: str = "text",
                                  min_struct: int = 5) -> pd.DataFrame:
    """Within-structure paired perspective/offset effects per task, BH-FDR per model.

    For each structure (same scene, varied camera), the perspective delta is
    acc(strong fov) - acc(weak fov) averaged over offsets (and vice-versa for the
    offset delta). A one-sample t-test over structures tests each delta against 0;
    BH-FDR is applied across tasks within each model. Matches the paper's Fig. 18
    (unlike an unpaired pooled test, this holds structure fixed)."""
    def _paired_delta(df, tcol, block, contrast, hi, lo):
        """Per-structure delta = acc(contrast=hi) - acc(contrast=lo), averaged over
        the ``block`` factor. Vectorized via groupby/pivot."""
        g = (df.dropna(subset=[tcol])
               .groupby(["_struct_id", block, contrast])[tcol].mean().rename("m").reset_index())
        if g.empty:
            return np.array([])
        piv = g.pivot_table(index=["_struct_id", block], columns=contrast, values="m")
        if hi not in piv.columns or lo not in piv.columns:
            return np.array([])
        delta = piv[hi] - piv[lo]
        per_struct = delta.groupby(level="_struct_id").mean()  # avg over block, skipping NaN cells
        return per_struct.dropna().to_numpy()

    task_cols = [t for t in TASK_COLS if any(t in df.columns for df in data.values())]
    rows = []
    for name, df0 in data.items():
        df = _add_struct_id(df0)
        persp = sorted(df["_perspective"].dropna().unique())
        offs = sorted(df["_offset"].dropna().unique())
        for tcol in task_cols:
            if tcol not in df.columns or df[tcol].isna().all():
                continue
            dp = _paired_delta(df, tcol, "_offset", "_perspective", persp[-1], persp[0]) if len(persp) >= 2 else np.array([])
            do = _paired_delta(df, tcol, "_perspective", "_offset", offs[-1], offs[0]) if len(offs) >= 2 else np.array([])
            tp = pp = to = po = np.nan
            if len(dp) >= min_struct:
                tp, pp = stats.ttest_1samp(np.asarray(dp), 0.0)
            if len(do) >= min_struct:
                to, po = stats.ttest_1samp(np.asarray(do), 0.0)
            rows.append({
                "modality": modality, "model": name, "task": tcol,
                "task_display": TASK_DISPLAY.get(tcol, tcol),
                "perspective_delta": float(np.mean(dp)) if len(dp) else np.nan,
                "t_perspective": float(tp) if not np.isnan(tp) else np.nan,
                "p_perspective": float(pp) if not np.isnan(pp) else np.nan,
                "n_struct_perspective": len(dp),
                "offset_delta": float(np.mean(do)) if len(do) else np.nan,
                "t_offset": float(to) if not np.isnan(to) else np.nan,
                "p_offset": float(po) if not np.isnan(po) else np.nan,
                "n_struct_offset": len(do),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_perspective_fdr"] = np.nan
    out["p_offset_fdr"] = np.nan
    for _, g in out.groupby("model"):
        out.loc[g.index, "p_perspective_fdr"] = bh_fdr(g["p_perspective"].values)
        out.loc[g.index, "p_offset_fdr"] = bh_fdr(g["p_offset"].values)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("inference_results"))
    ap.add_argument("--output-dir", type=Path, default=Path("analyses_results/tables"))
    ap.add_argument("--group", choices=["text", "trained_text"], default="text")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-fe", type=int, default=2000,
                    help="scene-cluster bootstrap draws for the two-way FE slope CIs")
    ap.add_argument("--validate", action="store_true", help="print key numbers to stdout")
    args = ap.parse_args()

    data = load_text_group(args.input_dir, args.group)
    if not data:
        print(f"No text CSVs found in {args.input_dir}")
        return
    print(f"Loaded {len(data)} responders: {sorted(data)}")

    lift = within_scene_lift_table(data, args.group, n_perm=args.n_perm, n_boot=args.n_boot)
    bundle = bundle_coupling_table(data, args.group)
    integrity = joint_integrity_table(data, args.group)
    resp = responder_controlled_lift_table(data, args.group, n_boot=args.n_boot_fe)
    per_resp = per_responder_lift_table(data, args.group)
    coup2 = bundle_coupling_twoway_table(data, args.group)
    mcnemar = mcnemar_main_table(data, args.group)
    camera = camera_within_structure_table(data, args.group)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lift.to_csv(args.output_dir / f"tableF_within_scene_lift_{args.group}.csv", index=False)
    bundle.to_csv(args.output_dir / f"tableF_bundle_coupling_{args.group}.csv", index=False)
    integrity.to_csv(args.output_dir / f"tableF_joint_integrity_{args.group}.csv", index=False)
    resp.to_csv(args.output_dir / f"tableF_responder_controlled_lift_{args.group}.csv", index=False)
    per_resp.to_csv(args.output_dir / f"tableF_per_responder_lift_{args.group}.csv", index=False)
    coup2.to_csv(args.output_dir / f"tableF_bundle_coupling_twoway_{args.group}.csv", index=False)
    mcnemar.to_csv(args.output_dir / f"table6_mcnemar_object_counting_{args.group}.csv", index=False)
    camera.to_csv(args.output_dir / f"figF18_camera_within_structure_{args.group}.csv", index=False)
    print(f"Wrote tables to {args.output_dir}")

    if args.validate:
        print("\n== within-scene JOINT lift (Hidden = Support Column, S8) ==")
        for _, r in lift[lift["kind"] == "joint"].iterrows():
            print(f"  {r['relation']:24s} lift={r['ws_lift']:+.4f}  r_within={r['r_within']:.3f}"
                  f"  perm_p={r['perm_p']:.4f}  n_var={r['n_varying']}")
        print("\n== bundle-to-Main coupling (aggregate, Hidden = S8) ==")
        for _, r in bundle[bundle["model"] == "aggregate"].iterrows():
            print(f"  {r['bundle']:28s} r_ws={r['r_ws']:+.4f}  p_fdr={r['p_ws_fdr']:.2e}  n_var={r['n_varying']}")
        print("\n== joint relation integrity, human row (S8 pre-specified vs submitted S7) ==")
        for _, r in integrity[integrity["model"] == "human"].iterrows():
            print(f"  {r['relation']:24s} [{r['variant']:16s}] {100*r['joint_integrity']:.1f}%"
                  f"  n={int(r['n_child_correct'])}")
        print("\n== responder-controlled robustness (scene-only vs two-way FE) ==")
        for _, r in resp.iterrows():
            print(f"  [{r['subset']:11s}] {r['relation']:24s} scene-only={100*r['scene_only_lift']:+.1f}pp  "
                  f"two-way FE={100*r['twoway_fe_slope']:+.1f}pp "
                  f"[{100*r['ci_low']:+.1f},{100*r['ci_high']:+.1f}]  "
                  f"resp+={int(r['responders_positive'])}/{int(r['responders_total'])}")
        print("  per-responder marginal lift (responder-controlled, scene-uncontrolled):")
        for rel, g in per_resp.groupby("relation"):
            cells = "  ".join(
                f"{r['model']}:" + (f"{100*r['marginal_lift']:+.1f}" if r["included"]
                                    else f"excluded(n_met={int(r['n_met'])})")
                for _, r in g.iterrows())
            print(f"    {rel:24s} {cells}")
        for _, r in coup2.iterrows():
            print(f"  coupling {r['bundle']:22s} scene={r['r_scene']:+.3f}  scene+responder={r['r_scene_responder']:+.3f}")
        print("\n== McNemar on Object Counting: adjacent pairs (Table 6) ==")
        for _, r in mcnemar[mcnemar["is_adjacent"]].iterrows():
            print(f"  {r['model_a']:>10s}({r['acc_a']:.1%}) vs {r['model_b']:<10s}({r['acc_b']:.1%})"
                  f"  p_fdr={r['p_mcnemar_fdr']:.2e}")
        print("\n== camera perspective effect (Fig.18): Columns / Visible / Hidden per model ==")
        cam = camera[camera["task"].isin(["task3", "task5", "task8"])]
        for model in sorted(cam["model"].unique()):
            g = cam[cam["model"] == model].set_index("task")
            def _d(t):
                return f"{g.loc[t,'perspective_delta']*100:+.1f}pp" if t in g.index else "  n/a"
            print(f"  {model:>8s}:  Col {_d('task3')}   Vis {_d('task5')}   Hid {_d('task8')}")


if __name__ == "__main__":
    main()
