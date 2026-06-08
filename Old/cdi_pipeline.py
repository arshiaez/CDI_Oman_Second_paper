"""
CDI Analytics Pipeline — Two-Stage GP Framework with Lk (Lambda-kappa) Proxy
Implements T-01..T-07 from CDI_RA_Report_Arshia.pdf

Validation design
-----------------
- LOCO-CV: leave-one-condition-out (10 folds).
- Operating conditions are the independent validation units.
- All preprocessing, scaling, and GP hyperparameter optimisation are fitted
  only on the training fold (no leakage).
- Both cycle-level and condition-weighted metrics are reported; equal weight
  per condition so conditions with more cycles do not dominate.

Uncertainty calibration (T-04)
-------------------------------
- OOF residuals from LOCO-CV supply the calibration signal.
- Multiplier is selected on K-1 conditions; coverage is assessed on the
  remaining condition (nested LOO). The reported coverage is this honest
  cross-validated estimate -- NOT the in-sample nominal coverage.

Flow in Stage 2 (T-03)
-----------------------
- ARD lengthscales are inspected on the full dataset (structural decision,
  not a predictive fit).
- Flow is included if ARD ls_Q < ARD_FLOW_THRESHOLD in at least one target
  and >= MIN_UNIQUE_FLOW unique flow values exist; otherwise it is fixed to
  the observed median and the decision is documented.

Improvement percentages and Pareto results are PROVISIONAL until the
pipeline is run on real experimental data.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE      = 42
COVERAGE_TARGET   = 0.90
FIG_DPI           = 300
ARD_FLOW_THRESHOLD = 3.0   # in standardised space; ls > this → flow uninformative
MIN_UNIQUE_FLOW   = 3      # need at least 3 distinct flow values to include as variable
N_GP_RESTARTS     = 5


# ---------------------------------------------------------------------------
# GP kernel and fitting
# ---------------------------------------------------------------------------

def _make_kernel(n_features: int):
    ls = np.ones(n_features)
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(ls, [(1e-2, 1e2)] * n_features)
        + WhiteKernel(1e-3, (1e-6, 1e1))
    )


def _fit_gp(X_tr: np.ndarray, y_tr: np.ndarray) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(
        kernel=_make_kernel(X_tr.shape[1]),
        n_restarts_optimizer=N_GP_RESTARTS,
        normalize_y=True,
        random_state=RANDOM_STATE,
    )
    gp.fit(X_tr, y_tr)
    return gp


def _ard_lengthscales(gp: GaussianProcessRegressor) -> np.ndarray:
    """Extract per-feature ARD lengthscales from a fitted (C * RBF + White) kernel."""
    try:
        return gp.kernel_.k1.k2.length_scale
    except AttributeError:
        return np.atleast_1d(gp.kernel_.k1.length_scale)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _cycle_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae":     float(mae),
        "rmse":    float(rmse),
        "rel_mae": float(mae / (np.mean(np.abs(y_true)) + 1e-12)),
        "r2":      float(r2_score(y_true, y_pred)),
        "n":       int(len(y_true)),
    }


def _condition_weighted_metrics(df: pd.DataFrame,
                                true_col: str, pred_col: str) -> dict:
    """Equal weight per condition; ignores cycle-count imbalance."""
    per = df.groupby("condition_id").apply(
        lambda g: mean_absolute_error(g[true_col].values, g[pred_col].values)
    )
    rms = df.groupby("condition_id").apply(
        lambda g: np.sqrt(mean_squared_error(g[true_col].values, g[pred_col].values))
    )
    return {
        "cond_mae_mean":  float(per.mean()),
        "cond_mae_std":   float(per.std()),
        "cond_rmse_mean": float(rms.mean()),
        "per_condition":  {int(k): float(v) for k, v in per.items()},
    }


def report_metrics(df: pd.DataFrame,
                   true_col: str, pred_col: str, label: str = "") -> dict:
    valid = df[pred_col].notna()
    cyc = _cycle_metrics(df.loc[valid, true_col].values,
                         df.loc[valid, pred_col].values)
    cnd = _condition_weighted_metrics(df[valid], true_col, pred_col)
    tag = f" [{label}]" if label else ""
    print(f"\n  {true_col}{tag}")
    print(f"    Cycle-level     MAE={cyc['mae']:.4f}  RMSE={cyc['rmse']:.4f}  "
          f"RelMAE={cyc['rel_mae']:.3f}  R²={cyc['r2']:.4f}  N={cyc['n']}")
    print(f"    Cond-weighted   MAE={cnd['cond_mae_mean']:.4f}"
          f"±{cnd['cond_mae_std']:.4f}  RMSE={cnd['cond_rmse_mean']:.4f}")
    return {**{f"cycle_{k}": v for k, v in cyc.items()},
            **{f"cond_{k}": v for k, v in cnd.items()}}


# ---------------------------------------------------------------------------
# Stage 1 LOCO-CV  (T-01 fix: iterates ALL cycles per held-out condition)
# ---------------------------------------------------------------------------

def run_loco_cv_stage1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict Lk from [C_in, Q, V] via LOCO-CV.
    Scaler and GP hyperparameters fitted on training fold only.
    Adds columns: lambda_kappa_oof, lambda_kappa_std_oof.
    """
    conditions = sorted(df["condition_id"].unique())
    feat_cols  = ["C_in", "Q", "V"]
    oof_pred   = np.full(len(df), np.nan)
    oof_std    = np.full(len(df), np.nan)

    for held_out in conditions:
        mask_te = (df["condition_id"] == held_out).values
        mask_tr = ~mask_te
        n_te = mask_te.sum()

        X_tr = df.loc[mask_tr, feat_cols].values
        X_te = df.loc[mask_te, feat_cols].values
        y_tr = df.loc[mask_tr, "lambda_kappa"].values

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)  # fitted on train only
        X_te_s   = scaler.transform(X_te)

        gp = _fit_gp(X_tr_s, y_tr)
        pred, std = gp.predict(X_te_s, return_std=True)
        oof_pred[mask_te] = pred
        oof_std[mask_te]  = std

        mae = mean_absolute_error(df.loc[mask_te, "lambda_kappa"].values, pred)
        print(f"    cond={held_out:2d}  n_test={n_te:3d}  n_train={mask_tr.sum():3d}  MAE={mae:.4f}")

    df = df.copy()
    df["lambda_kappa_oof"]     = oof_pred
    df["lambda_kappa_std_oof"] = oof_std
    return df


# ---------------------------------------------------------------------------
# T-03: Flow-in-Stage-2 decision
# ---------------------------------------------------------------------------

def decide_flow_in_stage2(df: pd.DataFrame) -> tuple[bool, dict, str]:
    """
    Fit Stage-2 GPs with full ARD on all data (structural decision only, not
    used for prediction). Inspect Q's lengthscale in standardised space.
    Returns (include_flow, ard_info, justification_text).
    """
    feat_cols = ["C_in", "Q", "V", "lambda_kappa"]
    q_idx     = feat_cols.index("Q")
    q_unique  = int(df["Q"].nunique())

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(df[feat_cols].values)

    ls_by_target: dict[str, np.ndarray] = {}
    for tgt in ["SEC_Wh_m3", "R_kappa_pos"]:
        gp = _fit_gp(X_s, df[tgt].values)
        ls_by_target[tgt] = _ard_lengthscales(gp)

    ls_q_sec = float(ls_by_target["SEC_Wh_m3"][q_idx])
    ls_q_rk  = float(ls_by_target["R_kappa_pos"][q_idx])
    q_median = float(df["Q"].median())

    # Include flow only if at least one target is sensitive to it
    # AND there are enough distinct flow values
    informative = (min(ls_q_sec, ls_q_rk) < ARD_FLOW_THRESHOLD)
    include_flow = informative and (q_unique >= MIN_UNIQUE_FLOW)

    ard_info = {
        "ls_q_sec": ls_q_sec,
        "ls_q_rk":  ls_q_rk,
        "q_unique": q_unique,
        "q_median": q_median,
    }

    if include_flow:
        just = (
            f"Flow included in Stage 2. ARD lengthscale for Q in standardised space: "
            f"SEC={ls_q_sec:.2f}, R_kappa={ls_q_rk:.2f} (threshold {ARD_FLOW_THRESHOLD}); "
            f"{q_unique} unique flow values."
        )
    else:
        just = (
            f"Flow fixed to median Q={q_median:.2f} in Stage 2 optimisation grid. "
            f"ARD lengthscale for Q in standardised space: SEC={ls_q_sec:.2f}, "
            f"R_kappa={ls_q_rk:.2f} — both exceed threshold {ARD_FLOW_THRESHOLD} "
            f"or fewer than {MIN_UNIQUE_FLOW} unique flow values ({q_unique}). "
            f"Insufficient independent variation to justify flow as a Stage-2 "
            f"decision variable. The claim is limited to fixed-flow optimisation."
        )

    print(f"\n  T-03 Flow decision: {'INCLUDE' if include_flow else 'FIX to median'}")
    print(f"    {just}")
    return include_flow, ard_info, just


# ---------------------------------------------------------------------------
# Stage 2 LOCO-CV  (T-01 fix: all cycles iterated)
# ---------------------------------------------------------------------------

def run_loco_cv_stage2(df: pd.DataFrame, include_flow: bool) -> pd.DataFrame:
    """
    Predict SEC and R_kappa_pos via LOCO-CV.
    Training fold: TRUE lambda_kappa (measured for those conditions).
    Test fold: OOF lambda_kappa from Stage 1 (propagates Stage-1 uncertainty).
    Scaler fitted on training fold only.
    Adds columns: {target}_pred, {target}_std for each target.
    """
    conditions   = sorted(df["condition_id"].unique())
    # Test fold features use OOF Lk; training fold features use true Lk
    feat_oof = ["C_in", "Q", "V", "lambda_kappa_oof"] if include_flow else ["C_in", "V", "lambda_kappa_oof"]
    feat_tr  = [c.replace("lambda_kappa_oof", "lambda_kappa") for c in feat_oof]
    targets  = ["SEC_Wh_m3", "R_kappa_pos"]

    oof: dict[str, dict] = {
        t: {"pred": np.full(len(df), np.nan), "std": np.full(len(df), np.nan)}
        for t in targets
    }

    for held_out in conditions:
        mask_te = (df["condition_id"] == held_out).values
        mask_tr = ~mask_te

        X_tr = df.loc[mask_tr, feat_tr].values
        X_te = df.loc[mask_te, feat_oof].values

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        for t in targets:
            y_tr = df.loc[mask_tr, t].values
            gp   = _fit_gp(X_tr_s, y_tr)
            pred, std = gp.predict(X_te_s, return_std=True)
            oof[t]["pred"][mask_te] = pred
            oof[t]["std"][mask_te]  = std

        print(f"    cond={held_out:2d}  n_test={mask_te.sum():3d}")

    df = df.copy()
    for t in targets:
        df[f"{t}_pred"] = oof[t]["pred"]
        df[f"{t}_std"]  = oof[t]["std"]
    return df


# ---------------------------------------------------------------------------
# T-04: Uncertainty calibration — nested LOO over conditions
# ---------------------------------------------------------------------------

def calibrate_uncertainty_nested(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    std_col: str,
    target_coverage: float = COVERAGE_TARGET,
) -> dict:
    """
    For each condition i:
      1. Select multiplier alpha on OOF residuals from conditions j != i.
      2. Assess coverage on condition i with that alpha.
    Reported coverage is the cross-validated (honest) estimate.
    The final multiplier (for deployment) is selected on all conditions, but
    its in-sample coverage is NOT reported as the generalisation estimate.
    """
    conditions = sorted(df["condition_id"].unique())
    cv_coverages: list[float] = []

    resid_all = (df[true_col] - df[pred_col]).abs().values
    std_all   = df[std_col].values

    for held_out in conditions:
        mask_cal = (df["condition_id"] != held_out).values
        mask_ass = (df["condition_id"] == held_out).values

        scores_cal = np.where(
            std_all[mask_cal] > 0,
            resid_all[mask_cal] / std_all[mask_cal],
            np.inf,
        )
        alpha = float(np.quantile(scores_cal, target_coverage))

        resid_ass = resid_all[mask_ass]
        std_ass   = std_all[mask_ass]
        coverage  = float((resid_ass <= alpha * std_ass).mean())
        cv_coverages.append(coverage)

    # Final multiplier on all residuals (used only for deployment bands)
    scores_all  = np.where(std_all > 0, resid_all / std_all, np.inf)
    alpha_final = float(np.quantile(scores_all, target_coverage))
    nominal_cov = float((resid_all <= alpha_final * std_all).mean())

    return {
        "multiplier":              alpha_final,
        "nominal_coverage":        nominal_cov,    # in-sample — do NOT cite as generalisation
        "cv_coverage_mean":        float(np.mean(cv_coverages)),
        "cv_coverage_std":         float(np.std(cv_coverages)),
        "per_condition_coverage":  {int(c): float(v) for c, v in zip(conditions, cv_coverages)},
    }


# ---------------------------------------------------------------------------
# Ablation study (T-07)
# ---------------------------------------------------------------------------

def _loco_ridge(df: pd.DataFrame, feat_cols: list[str], target: str) -> np.ndarray:
    """Ridge with alpha tuned by nested LOO inside each training fold (no leakage)."""
    conditions = sorted(df["condition_id"].unique())
    oof = np.full(len(df), np.nan)
    ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

    for held_out in conditions:
        mask_te = (df["condition_id"] == held_out).values
        mask_tr = ~mask_te
        df_tr   = df[mask_tr].reset_index(drop=True)
        tr_conds = sorted(df_tr["condition_id"].unique())

        # Nested LOO over training conditions to select Ridge alpha
        best_alpha, best_mae = ALPHAS[0], np.inf
        for alpha in ALPHAS:
            inner_preds = np.full(len(df_tr), np.nan)
            for c in tr_conds:
                mi_te = (df_tr["condition_id"] == c).values
                mi_tr = ~mi_te
                sc2   = StandardScaler()
                Xtr2  = sc2.fit_transform(df_tr.loc[mi_tr, feat_cols].values)
                Xte2  = sc2.transform(df_tr.loc[mi_te, feat_cols].values)
                ridge = Ridge(alpha=alpha)
                ridge.fit(Xtr2, df_tr.loc[mi_tr, target].values)
                inner_preds[mi_te] = ridge.predict(Xte2)
            valid = ~np.isnan(inner_preds)
            mae = mean_absolute_error(df_tr[target].values[valid], inner_preds[valid])
            if mae < best_mae:
                best_mae, best_alpha = mae, alpha

        # Fit final Ridge on full training fold
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(df.loc[mask_tr, feat_cols].values)
        X_te_s = scaler.transform(df.loc[mask_te, feat_cols].values)
        ridge  = Ridge(alpha=best_alpha)
        ridge.fit(X_tr_s, df.loc[mask_tr, target].values)
        oof[mask_te] = ridge.predict(X_te_s)

    return oof


def _loco_gp(df: pd.DataFrame, feat_tr_cols: list[str],
             feat_te_cols: list[str], target: str) -> np.ndarray:
    """Generic LOCO-CV GP with separate train/test feature columns."""
    conditions = sorted(df["condition_id"].unique())
    oof = np.full(len(df), np.nan)
    for held_out in conditions:
        mask_te = (df["condition_id"] == held_out).values
        mask_tr = ~mask_te
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(df.loc[mask_tr, feat_tr_cols].values)
        X_te_s  = scaler.transform(df.loc[mask_te, feat_te_cols].values)
        gp      = _fit_gp(X_tr_s, df.loc[mask_tr, target].values)
        oof[mask_te], _ = gp.predict(X_te_s, return_std=True)
    return oof


def run_ablation(df: pd.DataFrame, include_flow_s2: bool) -> pd.DataFrame:
    """
    Three approaches for Stage 2 (both targets), all via LOCO-CV.
    Ridge alpha is tuned inside each training fold (nested LOO).
    GP hyperparameters are optimised on training fold only.
    """
    targets = ["SEC_Wh_m3", "R_kappa_pos"]
    feat_base = ["C_in", "Q", "V"]
    feat_full_te = (["C_in", "Q", "V", "lambda_kappa_oof"] if include_flow_s2
                    else ["C_in", "V", "lambda_kappa_oof"])
    feat_full_tr = [c.replace("lambda_kappa_oof", "lambda_kappa") for c in feat_full_te]

    rows = []
    for target in targets:
        print(f"    ablation target={target}")
        ridge_oof   = _loco_ridge(df, feat_base, target)
        gp_nolk_oof = _loco_gp(df, feat_base, feat_base, target)
        gp_lk_oof   = _loco_gp(df, feat_full_tr, feat_full_te, target)

        for approach, oof_pred in [
            ("Ridge",       ridge_oof),
            ("GP_no_Lk",   gp_nolk_oof),
            ("GP_with_Lk", gp_lk_oof),
        ]:
            valid   = ~np.isnan(oof_pred)
            df_v    = df[valid].copy()
            df_v["_pred"] = oof_pred[valid]
            cyc = _cycle_metrics(df_v[target].values, df_v["_pred"].values)
            cnd = _condition_weighted_metrics(df_v, target, "_pred")
            rows.append({
                "target":   target,
                "approach": approach,
                **{f"cycle_{k}": v for k, v in cyc.items()},
                **{f"cond_{k}": v for k, v in cnd.items()},
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pareto front and knee  (T-05)
# ---------------------------------------------------------------------------

def _pareto_front_2d(cost: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """O(n log n) 2D Pareto: minimise cost, maximise benefit."""
    idx       = np.lexsort((-benefit, cost))  # cost asc, benefit desc
    is_pareto = np.zeros(len(cost), dtype=bool)
    max_ben   = -np.inf
    for i in idx:
        if benefit[i] > max_ben:
            is_pareto[i] = True
            max_ben = benefit[i]
    return is_pareto


def pareto_search(
    df: pd.DataFrame,
    include_flow: bool,
    flow_fixed: float,
    calib: dict[str, dict],
    n_grid: int = 30,
) -> dict:
    """
    Fits final Stage-1 and Stage-2 GPs on ALL data, then grid-searches for
    Pareto-optimal candidates.
    The knee is labelled 'model-suggested candidate for experimental validation'.
    All results are PROVISIONAL until rerun on corrected real data.
    """
    # Final Stage-1 GP
    sc_s1  = StandardScaler()
    X_s1   = sc_s1.fit_transform(df[["C_in", "Q", "V"]].values)
    gp_s1  = _fit_gp(X_s1, df["lambda_kappa"].values)

    feat_tr = (["C_in", "Q", "V", "lambda_kappa"] if include_flow
               else ["C_in", "V", "lambda_kappa"])
    sc_s2   = StandardScaler()
    X_s2    = sc_s2.fit_transform(df[feat_tr].values)
    gp_sec  = _fit_gp(X_s2, df["SEC_Wh_m3"].values)
    gp_rkp  = _fit_gp(X_s2, df["R_kappa_pos"].values)

    # Grid within observed range
    c_lo, c_hi = df["C_in"].min(), df["C_in"].max()
    v_lo, v_hi = df["V"].min(),   df["V"].max()
    c_g = np.linspace(c_lo, c_hi, n_grid)
    v_g = np.linspace(v_lo, v_hi, n_grid)

    if include_flow:
        q_lo, q_hi = df["Q"].min(), df["Q"].max()
        q_g = np.linspace(q_lo, q_hi, n_grid)
        C, Q_, V_ = np.meshgrid(c_g, q_g, v_g, indexing="ij")
        grid_cqv = np.column_stack([C.ravel(), Q_.ravel(), V_.ravel()])
    else:
        C, V_ = np.meshgrid(c_g, v_g, indexing="ij")
        grid_cqv = np.column_stack([
            C.ravel(), np.full(C.size, flow_fixed), V_.ravel()
        ])

    lk_grid, _ = gp_s1.predict(sc_s1.transform(grid_cqv), return_std=True)

    if include_flow:
        grid_s2 = np.column_stack([grid_cqv[:, 0], grid_cqv[:, 1], grid_cqv[:, 2], lk_grid])
    else:
        grid_s2 = np.column_stack([grid_cqv[:, 0], grid_cqv[:, 2], lk_grid])

    X_grid_s2 = sc_s2.transform(grid_s2)
    sec_pred, sec_std = gp_sec.predict(X_grid_s2, return_std=True)
    rkp_pred, rkp_std = gp_rkp.predict(X_grid_s2, return_std=True)

    # Apply calibrated multipliers to uncertainty bands
    alpha_sec = calib.get("SEC_Wh_m3", {}).get("multiplier", 1.0)
    alpha_rkp = calib.get("R_kappa_pos", {}).get("multiplier", 1.0)

    pareto_mask = _pareto_front_2d(sec_pred, rkp_pred)

    # Knee: closest to ideal corner in normalised Pareto space
    sec_p = sec_pred[pareto_mask]
    rkp_p = rkp_pred[pareto_mask]
    sec_n = (sec_p - sec_p.min()) / (np.ptp(sec_p) + 1e-12)
    rkp_n = (rkp_p - rkp_p.min()) / (np.ptp(rkp_p) + 1e-12)
    knee_local  = int(np.argmin(np.hypot(sec_n, 1 - rkp_n)))
    knee_global = int(np.where(pareto_mask)[0][knee_local])

    # Near-observed flag (within 1 std of nearest training condition)
    obs_cqv  = df[["C_in", "Q", "V"]].drop_duplicates().values
    sc_obs   = StandardScaler().fit(obs_cqv)
    dists    = cdist(sc_obs.transform(grid_cqv),
                     sc_obs.transform(obs_cqv)).min(axis=1)
    near_mask = dists < 1.0

    k = knee_global
    knee = {
        "C_in":         float(grid_cqv[k, 0]),
        "Q":            float(grid_cqv[k, 1]),
        "V":            float(grid_cqv[k, 2]),
        "SEC_pred":     float(sec_pred[k]),
        "SEC_lo":       float(sec_pred[k] - alpha_sec * sec_std[k]),
        "SEC_hi":       float(sec_pred[k] + alpha_sec * sec_std[k]),
        "R_kappa_pred": float(rkp_pred[k]),
        "R_kappa_lo":   float(rkp_pred[k] - alpha_rkp * rkp_std[k]),
        "R_kappa_hi":   float(rkp_pred[k] + alpha_rkp * rkp_std[k]),
        "near_observed": bool(near_mask[k]),
        "label": "model-suggested candidate for experimental validation "
                 "(PROVISIONAL — subject to corrected-pipeline rerun)",
    }

    return {
        "grid_cqv":    grid_cqv,
        "sec_pred":    sec_pred,
        "rkp_pred":    rkp_pred,
        "sec_lo":      sec_pred - alpha_sec * sec_std,
        "sec_hi":      sec_pred + alpha_sec * sec_std,
        "rkp_lo":      rkp_pred - alpha_rkp * rkp_std,
        "rkp_hi":      rkp_pred + alpha_rkp * rkp_std,
        "pareto_mask": pareto_mask,
        "near_mask":   near_mask,
        "knee":        knee,
    }


# ---------------------------------------------------------------------------
# Figures (T-06) — all saved at >= 300 DPI
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str, out_dir: Path) -> None:
    p = out_dir / name
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {p.name}")


def _fig1_pipeline(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    steps = [
        "Raw data",
        "Cycle\nsegmentation",
        "Cycle metrics\n(Lk, SEC, Rk)",
        "Stage 1 GP\npredict Lk",
        "Stage 2 GP\nSEC + Rk",
        "Pareto\nfront",
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="#d0e8f7", ec="#2c7bb6"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.055, 0.5), xytext=(x + 0.055, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#2c7bb6", lw=1.5))
    ax.set_title("Fig. 1 — CDI Analytics Pipeline Schematic", fontweight="bold", fontsize=11)
    _save_fig(fig, "fig1_pipeline.pdf", out_dir)


def _fig2_cycle_example(out_dir: Path) -> None:
    """Simulated cycle segmentation example (replace with real data if available)."""
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 1, 300)
    # Adsorption (0–0.6) then desorption (0.6–1.0)
    cond = np.where(t < 0.6,
                    1.0 - 0.25 * (1 - np.exp(-8 * t)),
                    0.75 + 0.25 * (1 - np.exp(-8 * (t - 0.6))))
    cond += rng.normal(0, 0.005, len(t))
    curr = np.where(t < 0.6,
                    0.4 * np.exp(-5 * t),
                    -0.3 * np.exp(-5 * (t - 0.6)))
    curr += rng.normal(0, 0.005, len(t))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t, curr, color="#2c7bb6")
    ax1.axvline(0.6, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("Current (A)")
    ax1.text(0.28, ax1.get_ylim()[1] * 0.85, "Adsorption", ha="center", fontsize=8, color="gray")
    ax1.text(0.78, ax1.get_ylim()[1] * 0.85, "Desorption", ha="center", fontsize=8, color="gray")

    ax2.fill_between(t[t < 0.6], 1.0, cond[t < 0.6], alpha=0.3, color="steelblue",
                     label="Rk area (integrated)")
    ax2.plot(t, cond, color="steelblue")
    ax2.axvline(0.6, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("Normalised cycle time")
    ax2.set_ylabel("Conductivity (normalised)")
    ax2.legend(fontsize=8)

    fig.suptitle("Fig. 2 — Cycle Segmentation Example\n"
                 "(simulated; replace with experimental trace)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, "fig2_cycle_segmentation.pdf", out_dir)


def _fig3_lk_profile(df: pd.DataFrame, out_dir: Path) -> None:
    cond_m = df.groupby("condition_id").agg(
        C_in=("C_in", "mean"), V=("V", "mean"),
        lk=("lambda_kappa", "mean"), lk_s=("lambda_kappa", "std")
    ).reset_index()
    cond_m["lk_s"] = cond_m["lk_s"].fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.errorbar(cond_m["C_in"], cond_m["lk"], yerr=cond_m["lk_s"],
                 fmt="o", capsize=4, color="#2c7bb6")
    ax1.set_xlabel("Inlet concentration (ppm)")
    ax1.set_ylabel("Lk (conductivity-native proxy)")
    ax1.set_title("Lk vs C_in")

    ax2.errorbar(cond_m["V"], cond_m["lk"], yerr=cond_m["lk_s"],
                 fmt="s", capsize=4, color="coral")
    ax2.set_xlabel("Applied voltage (V)")
    ax2.set_title("Lk vs Voltage")

    fig.suptitle("Fig. 3 — Lk Profile by Operating Condition\n"
                 "(error bars = std over repeated cycles)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, "fig3_lk_profile.pdf", out_dir)


def _fig4_ard_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    feat_cols = ["C_in", "Q", "V"]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(df[feat_cols].values)
    gp  = _fit_gp(X_s, df["lambda_kappa"].values)
    ls  = _ard_lengthscales(gp)
    # Use only the first len(feat_cols) entries in case kernel shape differs
    ls = ls[:len(feat_cols)]
    sens = 1.0 / (ls + 1e-12)
    sens /= sens.sum()

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(feat_cols, sens, color=["#2c7bb6", "coral", "seagreen"])
    for bar, v in zip(bars, sens):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Relative ARD sensitivity (1/ls, normalised)")
    ax.set_title("Fig. 4 — Lk ARD Sensitivity Surface\n"
                 "(larger = more informative to Lk)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, "fig4_ard_sensitivity.pdf", out_dir)


def _fig5_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    pairs = [
        ("lambda_kappa",  "lambda_kappa_oof",  "Lk"),
        ("SEC_Wh_m3",    "SEC_Wh_m3_pred",    "SEC (Wh/m³)"),
        ("R_kappa_pos",  "R_kappa_pos_pred",  "R_κ_pos"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (true_col, pred_col, label) in zip(axes, pairs):
        if pred_col not in df.columns:
            ax.set_visible(False)
            continue
        valid   = df[pred_col].notna()
        y_true  = df.loc[valid, true_col].values
        y_pred  = df.loc[valid, pred_col].values
        mae     = mean_absolute_error(y_true, y_pred)
        r2      = r2_score(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.5, s=14, c="#2c7bb6")
        lo, hi  = y_true.min(), y_true.max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1")
        ax.set_xlabel(f"Observed {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}\nMAE={mae:.4f}  R²={r2:.3f}", fontsize=9)
    fig.suptitle("Fig. 5 — Predicted vs Observed (LOCO-CV, all cycles)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, "fig5_scatter.pdf", out_dir)


def _fig6_pareto(pareto: dict, out_dir: Path) -> None:
    sec   = pareto["sec_pred"]
    rkp   = pareto["rkp_pred"]
    pm    = pareto["pareto_mask"]
    near  = pareto["near_mask"]
    knee  = pareto["knee"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sec[~pm &  near], rkp[~pm &  near], c="lightblue",   s=6, alpha=0.3, label="Near-obs, dominated")
    ax.scatter(sec[~pm & ~near], rkp[~pm & ~near], c="lightyellow", s=6, alpha=0.3, label="Far, dominated")
    ax.scatter(sec[ pm],         rkp[ pm],         c="#2c7bb6",     s=16, alpha=0.7, label="Pareto front")

    xerr = [[knee["SEC_pred"] - knee["SEC_lo"]], [knee["SEC_hi"] - knee["SEC_pred"]]]
    yerr = [[knee["R_kappa_pred"] - knee["R_kappa_lo"]], [knee["R_kappa_hi"] - knee["R_kappa_pred"]]]
    ax.errorbar(knee["SEC_pred"], knee["R_kappa_pred"],
                xerr=xerr, yerr=yerr,
                fmt="*", color="red", ms=14, capsize=5,
                label="Pareto knee (candidate)")
    ax.set_xlabel("SEC (Wh/m³) — minimise")
    ax.set_ylabel("R_κ_pos — maximise")
    ax.set_title("Fig. 6 — Pareto Front\n"
                 "Knee = model-suggested candidate for experimental validation (PROVISIONAL)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "fig6_pareto.pdf", out_dir)


def _fig7_region_validation(df: pd.DataFrame, out_dir: Path) -> None:
    if "SEC_Wh_m3_pred" not in df.columns:
        return
    obs_cqv = df[["C_in", "Q", "V"]].values
    sc_obs  = StandardScaler().fit(obs_cqv)
    rows    = []
    conditions = sorted(df["condition_id"].unique())

    for held_out in conditions:
        mask  = (df["condition_id"] == held_out).values
        x_te  = sc_obs.transform(df.loc[mask, ["C_in", "Q", "V"]].values)
        x_tr  = sc_obs.transform(df.loc[~mask, ["C_in", "Q", "V"]].values)
        dist  = cdist(x_te, x_tr).min(axis=1).mean()
        region = "Near-observed" if dist < 1.0 else "Far/extrapolation"
        for t, pc in [("SEC_Wh_m3", "SEC_Wh_m3_pred"),
                      ("R_kappa_pos", "R_kappa_pos_pred")]:
            valid = df.loc[mask, pc].notna()
            if valid.sum() == 0:
                continue
            mae = mean_absolute_error(df.loc[mask & (df["condition_id"] == held_out), t].values,
                                      df.loc[mask, pc][valid.values].values)
            rows.append({"condition": str(held_out), "region": region,
                         "target": t, "mae": mae})

    if not rows:
        return
    reg = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"Near-observed": "#2c7bb6", "Far/extrapolation": "coral"}
    for ax, tgt in zip(axes, ["SEC_Wh_m3", "R_kappa_pos"]):
        sub = reg[reg["target"] == tgt]
        width, x0 = 0.35, np.arange(len(conditions))
        for k, (region, color) in enumerate(colors.items()):
            s = sub[sub["region"] == region].set_index("condition")
            vals = [s.loc[str(c), "mae"] if str(c) in s.index else 0.0 for c in conditions]
            ax.bar(x0 + k * width, vals, width, label=region, color=color, alpha=0.75)
        ax.set_xticks(x0 + width / 2)
        ax.set_xticklabels([str(c) for c in conditions], fontsize=8)
        ax.set_xlabel("Condition ID")
        ax.set_ylabel("MAE")
        ax.set_title(tgt)
        ax.legend(fontsize=8)
    fig.suptitle("Fig. 7 — Region-Based Validation (LOCO-CV MAE per condition)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, "fig7_region_validation.pdf", out_dir)


def make_figures(df: pd.DataFrame, pareto: dict, out_dir: Path) -> None:
    print("\n--- Generating figures ---")
    _fig1_pipeline(out_dir)
    _fig2_cycle_example(out_dir)
    _fig3_lk_profile(df, out_dir)
    _fig4_ard_sensitivity(df, out_dir)
    _fig5_scatter(df, out_dir)
    _fig6_pareto(pareto, out_dir)
    _fig7_region_validation(df, out_dir)


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def _print_ablation_table(abl: pd.DataFrame) -> None:
    print(f"\n{'Approach':<14} {'Target':<14} {'CyclMAE':>9} {'CyclRMSE':>10} "
          f"{'RelMAE':>8} {'CondMAE':>9}")
    print("-" * 68)
    for _, r in abl.iterrows():
        print(f"{r['approach']:<14} {r['target']:<14} "
              f"{r['cycle_mae']:>9.4f} {r['cycle_rmse']:>10.4f} "
              f"{r['cycle_rel_mae']:>8.3f} {r['cond_cond_mae_mean']:>9.4f}")

    print()
    for tgt in abl["target"].unique():
        sub = abl[abl["target"] == tgt].set_index("approach")
        if "GP_with_Lk" in sub.index and "Ridge" in sub.index:
            imp_ridge = ((sub.loc["Ridge", "cycle_mae"] -
                          sub.loc["GP_with_Lk", "cycle_mae"]) /
                         sub.loc["Ridge", "cycle_mae"] * 100)
            print(f"  {tgt}: GP+Lk vs Ridge cycle-MAE improvement: {imp_ridge:.1f}%  "
                  f"[PROVISIONAL — rerun after T-01 fix on real data]")
        if "GP_with_Lk" in sub.index and "GP_no_Lk" in sub.index:
            imp_gp = ((sub.loc["GP_no_Lk", "cycle_mae"] -
                       sub.loc["GP_with_Lk", "cycle_mae"]) /
                      sub.loc["GP_no_Lk", "cycle_mae"] * 100)
            print(f"  {tgt}: GP+Lk vs GP-no-Lk cycle-MAE improvement: {imp_gp:.1f}%  "
                  f"[PROVISIONAL]")


def _print_calib_table(calib: dict) -> None:
    print(f"\n{'Model':<22} {'Multiplier':>10} {'NomCov':>8} {'CV Cov':>8} {'CV Std':>8}")
    print("-" * 60)
    for model, c in calib.items():
        print(f"{model:<22} {c['multiplier']:>10.3f} "
              f"{c['nominal_coverage']:>8.3f} "
              f"{c['cv_coverage_mean']:>8.3f} "
              f"{c['cv_coverage_std']:>8.3f}")
    print("  NomCov = in-sample (do NOT cite as generalisation). "
          "CV Cov = honest cross-validated coverage.")


def _print_pareto_knee(knee: dict) -> None:
    print(f"\n  Pareto knee ({knee['label']}):")
    print(f"    C_in = {knee['C_in']:.0f} ppm | Q = {knee['Q']:.2f} mL/min | V = {knee['V']:.3f} V")
    print(f"    SEC  = {knee['SEC_pred']:.3f} [{knee['SEC_lo']:.3f}, {knee['SEC_hi']:.3f}] Wh/m³")
    print(f"    Rk   = {knee['R_kappa_pred']:.4f} [{knee['R_kappa_lo']:.4f}, {knee['R_kappa_hi']:.4f}]")
    print(f"    Near-observed region: {knee['near_observed']}")


# ---------------------------------------------------------------------------
# Synthetic data (testing only — replace with real experimental CSV)
# ---------------------------------------------------------------------------

def generate_synthetic_data(seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    10 operating conditions, ~10-13 cycles each (~107 total).
    Physics-inspired relationships so Stage-1→Stage-2 cascade is meaningful.
    NOT a substitute for real CDI experimental data.
    """
    rng = np.random.default_rng(seed)
    # Concentration sweep + voltage sweep
    conds = [
        (500,  5.0, 0.80), (750,  5.0, 0.85), (1000, 5.0, 0.90),
        (1000, 7.5, 0.90), (1000,10.0, 0.90), (1500, 5.0, 0.95),
        (1500, 7.5, 0.95), (2000, 5.0, 1.00), (2000, 7.5, 1.00),
        (2000,10.0, 1.00),
    ]
    rows = []
    for cid, (c, q, v) in enumerate(conds):
        n_cyc = int(rng.integers(9, 14))
        for _ in range(n_cyc):
            lk  = 0.40 + 0.30 * v - 0.04 * q / 10 - 0.08 * (c - 1000) / 1000
            lk += rng.normal(0, 0.018)
            lk  = float(np.clip(lk, 0.05, 1.5))
            sec = 12.0 + 9.0 * v + 0.004 * c - 0.6 * q
            sec += rng.normal(0, 1.2)
            sec = float(np.clip(sec, 4.0, 80.0))
            rkp = 0.20 + 0.18 * lk + 0.12 * v - 0.008 * q
            rkp += rng.normal(0, 0.015)
            rkp = float(np.clip(rkp, 0.03, 0.92))
            rows.append({"condition_id": cid, "C_in": float(c), "Q": float(q),
                         "V": float(v), "lambda_kappa": lk,
                         "SEC_Wh_m3": sec, "R_kappa_pos": rkp})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"condition_id", "C_in", "Q", "V",
                 "lambda_kappa", "SEC_Wh_m3", "R_kappa_pos"}


def run_pipeline(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    n_cyc   = len(df)
    n_conds = df["condition_id"].nunique()
    print(f"\n=== CDI Pipeline  ({n_cyc} cycles across {n_conds} conditions) ===")
    print(df.groupby("condition_id").agg(n=("lambda_kappa", "count"),
                                         C=("C_in", "first"), Q=("Q", "first"),
                                         V=("V", "first")).to_string())

    # ------------------------------------------------------------------
    # T-01 / T-02 — Stage 1 LOCO-CV (all cycles in each held-out condition)
    # ------------------------------------------------------------------
    print("\n--- Stage 1 LOCO-CV (Lk) ---")
    df = run_loco_cv_stage1(df)
    results["stage1"] = report_metrics(df, "lambda_kappa", "lambda_kappa_oof", "Stage1")

    # ------------------------------------------------------------------
    # T-03 — Flow decision
    # ------------------------------------------------------------------
    include_flow, ard_info, flow_just = decide_flow_in_stage2(df)
    flow_fixed = ard_info["q_median"]
    results["flow_decision"] = {
        "include_flow": include_flow, **ard_info, "justification": flow_just
    }

    # ------------------------------------------------------------------
    # T-01 / T-02 — Stage 2 LOCO-CV
    # ------------------------------------------------------------------
    print("\n--- Stage 2 LOCO-CV (SEC + R_kappa) ---")
    df = run_loco_cv_stage2(df, include_flow)
    results["stage2_SEC"]    = report_metrics(df, "SEC_Wh_m3",   "SEC_Wh_m3_pred",   "Stage2-SEC")
    results["stage2_Rkappa"] = report_metrics(df, "R_kappa_pos", "R_kappa_pos_pred", "Stage2-Rk")

    # ------------------------------------------------------------------
    # T-04 — Uncertainty calibration (nested LOO, no circular assessment)
    # ------------------------------------------------------------------
    print("\n--- Uncertainty calibration (nested LOO) ---")
    calib: dict[str, dict] = {}
    for true_col, pred_col, std_col in [
        ("lambda_kappa",  "lambda_kappa_oof",  "lambda_kappa_std_oof"),
        ("SEC_Wh_m3",    "SEC_Wh_m3_pred",    "SEC_Wh_m3_std"),
        ("R_kappa_pos",  "R_kappa_pos_pred",  "R_kappa_pos_std"),
    ]:
        if pred_col not in df.columns or std_col not in df.columns:
            continue
        calib[true_col] = calibrate_uncertainty_nested(df, true_col, pred_col, std_col)
    results["calibration"] = calib
    _print_calib_table(calib)

    # ------------------------------------------------------------------
    # T-07 — Ablation
    # ------------------------------------------------------------------
    print("\n--- Ablation study ---")
    abl = run_ablation(df, include_flow)
    results["ablation"] = abl.to_dict("records")
    _print_ablation_table(abl)
    abl.to_csv(out_dir / "ablation_table.csv", index=False)

    # ------------------------------------------------------------------
    # T-05 — Pareto
    # ------------------------------------------------------------------
    print("\n--- Pareto search ---")
    pareto = pareto_search(df, include_flow, flow_fixed, calib)
    results["pareto_knee"] = pareto["knee"]
    _print_pareto_knee(pareto["knee"])

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    df.to_csv(out_dir / "oof_predictions.csv", index=False)
    with open(out_dir / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)
    with open(out_dir / "pareto_knee.json", "w") as f:
        json.dump(pareto["knee"], f, indent=2)
    with open(out_dir / "flow_decision.json", "w") as f:
        json.dump(results["flow_decision"], f, indent=2)

    # ------------------------------------------------------------------
    # T-06 — Figures
    # ------------------------------------------------------------------
    make_figures(df, pareto, out_dir)

    print(f"\nAll outputs in: {out_dir.resolve()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CDI Analytics Pipeline")
    parser.add_argument(
        "--data", type=str, default=None,
        help=(
            "Path to cycle-level CSV with columns: "
            "condition_id, C_in, Q, V, lambda_kappa, SEC_Wh_m3, R_kappa_pos. "
            "Omit to run on synthetic demo data."
        ),
    )
    parser.add_argument("--out", type=str, default="cdi_output",
                        help="Output directory (default: cdi_output)")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        print(f"Loaded {len(df)} cycles from {args.data}")
    else:
        print("No --data file specified. Running on synthetic demo data.")
        df = generate_synthetic_data()

    run_pipeline(df, Path(args.out))


if __name__ == "__main__":
    main()
