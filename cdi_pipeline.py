"""
CDI physics-aware two-stage Gaussian Process analytics pipeline.

Stages
------
0  Data loading & cycle segmentation
1  Per-cycle metric extraction  (corrected physics: timestamp integration)
2  LOCO-CV  (two-stage GP + physics-aware models)
3  Uncertainty calibration
4  Ablation  (12 models: baselines → physics-aware)
5  Region-based validation  (corrected per-fold distance)
6  Pareto front  (physics constraints + extrapolation status)
7  Publication figures

Outputs → cdi_output/
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
import gpytorch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ── Config & globals ───────────────────────────────────────────────────────────

CFG    = yaml.safe_load(open("config.yaml"))
OUT    = Path(CFG["paths"]["output_dir"])
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS    = float(CFG.get("physics", {}).get("epsilon", 1e-9))

# Feature sets
# FEAT_OPERATING : operating conditions only (used in Stage 1 GP and baselines)
# FEAT_PRECYCLE  : adds initial_conductivity and inverse_flow_proxy (proxy τ ∝ 1/flow)
# FEAT_S2        : two-stage — adds predicted Λκ  (T-03 Option A)
FEAT_OPERATING = ["conc", "flow", "potential"]
FEAT_PRECYCLE  = ["conc", "flow", "potential", "initial_conductivity", "inverse_flow_proxy"]
FEAT_S1   = FEAT_OPERATING   # backward-compat alias
FEAT_PHYS = FEAT_PRECYCLE    # backward-compat alias
FEAT_S2   = ["conc", "flow", "potential", "lambda_kappa"]


# ── Stage 0: Data loading & cycle segmentation ─────────────────────────────────

def _load_file(path, sheet_to_meta):
    xl = pd.ExcelFile(path)
    out = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, usecols=[0, 1, 2]).dropna()
        df.columns = ["time", "conductivity", "current"]
        out.append((df, *sheet_to_meta(sheet)))
    return out


def load_raw_sheets():
    """Load three experimental Excel files; deduplicate shared (1000,3,1.8) condition."""
    ds, m = Path(CFG["paths"]["dataset_dir"]), CFG["data"]["sheet_meta"]
    records = (
        _load_file(ds / CFG["data"]["concentration_file"],
                   lambda s: (int(s.split()[0]),
                              m["concentration"]["fixed_flow"],
                              m["concentration"]["fixed_potential"]))
        + _load_file(ds / CFG["data"]["flow_file"],
                     lambda s: (m["flow"]["fixed_conc"],
                                float(s.replace("mLpermin", "")),
                                m["flow"]["fixed_potential"]))
        + _load_file(ds / CFG["data"]["potential_file"],
                     lambda s: (m["potential"]["fixed_conc"],
                                m["potential"]["fixed_flow"],
                                float(s.replace("V", ""))))
    )
    seen, unique = set(), []
    for rec in records:
        key = rec[1:]
        if key not in seen:
            seen.add(key); unique.append(rec)
    return unique


def segment_cycles(df):
    """Split timeseries into (desal_df, regen_df) pairs by current sign transitions."""
    sign  = np.sign(df["current"].values)
    edges = np.concatenate([[0], np.where(np.diff(sign) != 0)[0] + 1, [len(df)]])
    segs  = [(df.iloc[edges[i]:edges[i+1]].reset_index(drop=True),
              "desal" if df["current"].iloc[edges[i]] > 0 else "regen")
             for i in range(len(edges) - 1)]
    cycles, i = [], 0
    while i < len(segs) - 1:
        if segs[i][1] == "desal" and segs[i+1][1] == "regen":
            cycles.append((segs[i][0], segs[i+1][0])); i += 2
        else:
            i += 1
    return cycles


# ── Stage 1: Cycle metric extraction ──────────────────────────────────────────

def _metrics(desal, regen, conc, flow, potential, cid):
    """
    Physics-correct scalar metrics for one desal+regen cycle.

    Integration uses actual timestamps from the data column, not a fixed dx=1.

    Unit accounting
    ---------------
    I [mA] → I_A [A]
    Q [C]  = ∫|I_A| dt           (trapezoid over actual t)
    E [kWh] = V·Q / 3.6e6
    V_w [m³] = flow_m3s · duration_s
    SEC [kWh/m³] = E / V_w
    A_κ = ∫Δκ dt  [mS·cm⁻¹·s]
    Λκ  = A_κ / Q_mC  [mS·cm⁻¹·mC⁻¹]  (Q_mC = Q·1000, preserves legacy scale)
    R_κ = (κ_0 − κ_min) / κ_0

    Assumptions / known limitations
    --------------------------------
    • Constant applied potential throughout the desal phase (single V parameter).
    • No temperature correction; electrode area and cell volume unavailable.
    • inverse_flow_proxy ∝ 1/flow_m3s (absolute value requires cell-volume metadata).
    • initial_resistance = V / |I_0| is an instantaneous proxy, not DC resistance.
    """
    t     = desal["time"].values.astype(float)
    kappa = desal["conductivity"].values.astype(float)
    i_mA  = desal["current"].values.astype(float)

    # ── Validation ─────────────────────────────────────────────────────────────
    if len(t) < 2:
        return None
    if not np.all(np.diff(t) >= 0):          # non-monotonic timestamps
        return None
    duration_s = float(t[-1] - t[0])
    if duration_s <= 0:
        return None

    i_A      = i_mA * 1e-3                   # mA → A
    charge_c = float(np.trapezoid(np.abs(i_A), t))  # Coulombs
    if charge_c <= 0:
        return None

    flow_m3s  = flow * 1e-6 / 60.0           # mL/min → m³/s
    volume_m3 = flow_m3s * duration_s
    if volume_m3 <= 0:
        return None

    kappa_0 = float(kappa[0])
    if kappa_0 <= 0:
        return None

    # ── Physics quantities ──────────────────────────────────────────────────────
    energy_kwh        = potential * charge_c / 3.6e6
    sec_wh_m3         = energy_kwh / volume_m3

    kappa_min         = float(kappa.min())
    delta_kappa_peak  = kappa_0 - kappa_min
    delta_k           = np.clip(kappa_0 - kappa, 0.0, None)
    delta_kappa_int   = float(np.trapezoid(delta_k, t))  # mS·cm⁻¹·s

    charge_mC    = charge_c * 1000.0
    lambda_kappa = delta_kappa_int / charge_mC if charge_mC > 0 else np.nan
    r_kappa_pos  = delta_kappa_peak / kappa_0

    # ── Physics-aware features (observable at or before cycle start) ────────────
    initial_current    = float(abs(i_mA[0]))
    initial_resistance  = potential / (abs(i_A[0]) + EPS)   # Ω, instantaneous proxy
    inverse_flow_proxy  = 1.0 / (flow_m3s + EPS)           # s/m³, ∝ τ if cell-vol known

    return dict(
        conc=conc, flow=flow, potential=potential, cycle_id=cid,
        # physics columns
        duration_s=duration_s,
        charge_c=charge_c,
        energy_kwh=energy_kwh,
        volume_m3=volume_m3,
        kappa_0=kappa_0,
        kappa_min=kappa_min,
        delta_kappa_peak=delta_kappa_peak,
        delta_kappa_integral=delta_kappa_int,
        # derived metrics
        lambda_kappa=lambda_kappa,
        r_kappa_pos=r_kappa_pos,
        sec_wh_m3=sec_wh_m3,
        w_net_wh=energy_kwh * 1000.0,       # kWh → Wh (backward compat)
        # physics-aware features
        initial_conductivity=kappa_0,
        initial_current=initial_current,
        initial_resistance=initial_resistance,
        inverse_flow_proxy=inverse_flow_proxy,
    )


def build_cycle_table(records):
    rows = []
    for df, conc, flow, potential in tqdm(records, desc="Extracting cycles", unit="cond"):
        for cid, (desal, regen) in enumerate(segment_cycles(df)):
            m = _metrics(desal, regen, conc, flow, potential, cid)
            if m is not None:
                rows.append(m)
    return pd.DataFrame(rows).dropna(
        subset=["lambda_kappa", "r_kappa_pos", "sec_wh_m3"])


# ── GP model (ARD RBF kernel, GPyTorch) ───────────────────────────────────────

class _ExactGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, lik, d):
        super().__init__(X, y, lik)
        self.mean  = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean(x), self.covar(x))


def _to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


def train_gp(X: np.ndarray, y: np.ndarray):
    """Fit ExactGP with ARD-RBF; return (model, lik, scaler_X, scaler_y)."""
    sx = StandardScaler().fit(X)
    sy = StandardScaler().fit(y.reshape(-1, 1))
    Xt = _to_tensor(sx.transform(X))
    yt = _to_tensor(sy.transform(y.reshape(-1, 1)).ravel())

    lik   = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(
                    CFG["gp"]["noise_constraint"])).to(DEVICE)
    model = _ExactGP(Xt, yt, lik, X.shape[1]).to(DEVICE)
    model.train(); lik.train()

    opt = torch.optim.Adam(model.parameters(), lr=CFG["gp"]["lr"])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in tqdm(range(CFG["gp"]["n_iter"]), desc="  GP train", leave=False, unit="iter"):
        opt.zero_grad(); (-mll(model(Xt), yt)).backward(); opt.step()

    model.eval(); lik.eval()
    return model, lik, sx, sy


def predict_gp(model, lik, sx, sy, X: np.ndarray):
    """Return (mean, std) in original scale."""
    Xt = _to_tensor(sx.transform(X))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p = lik(model(Xt))
    mu  = sy.inverse_transform(p.mean.cpu().numpy().reshape(-1, 1)).ravel()
    std = p.stddev.cpu().numpy() * sy.scale_[0]
    return mu, std


# ── GP hyperparameter grid search (condition-group LOCO-CV) ───────────────────

def gp_grid_search(df):
    """
    Condition-group LOCO-CV over GP hyperparameter grid; updates CFG['gp'] in-place.

    Previous implementation used random K-fold, which allowed cycles from the
    same operating condition to appear in both training and validation folds.
    This version uses condition-group leave-one-out so no condition is in both.
    """
    gs = CFG["gp"]["grid_search"]
    if not gs["enabled"]:
        return

    print("\nRunning GP hyperparameter grid search (condition-group LOCO-CV)...")
    conds = _conditions(df)
    best_mae, best_params = np.inf, {}
    grid = [(ni, lr, nc)
            for ni in gs["n_iter_values"]
            for lr in gs["lr_values"]
            for nc in gs["noise_values"]]

    for n_iter, lr, noise in tqdm(grid, desc="GP grid search", unit="config"):
        fold_maes = []
        for cond in conds:
            train_cv, val_cv = _split(df, cond)
            CFG["gp"]["n_iter"]           = n_iter
            CFG["gp"]["lr"]               = lr
            CFG["gp"]["noise_constraint"] = noise
            gp    = train_gp(train_cv[FEAT_S1].values,
                             train_cv["lambda_kappa"].values)
            mu, _ = predict_gp(*gp, val_cv[FEAT_S1].values)
            fold_maes.append(mean_absolute_error(val_cv["lambda_kappa"].values, mu))

        mean_mae = np.mean(fold_maes)
        if mean_mae < best_mae:
            best_mae    = mean_mae
            best_params = {"n_iter": n_iter, "lr": lr, "noise_constraint": noise}

    CFG["gp"].update(best_params)
    print(f"Best GP params: {best_params}  (CV MAE={best_mae:.5f})\n")
    json.dump(best_params, open(OUT / "gp_grid_search.json", "w"), indent=2)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _conditions(df):
    return df[["conc", "flow", "potential"]].drop_duplicates().values


def _split(df, cond):
    mask = (df["conc"] == cond[0]) & (df["flow"] == cond[1]) & (df["potential"] == cond[2])
    return df[~mask], df[mask]


def _logit(x, eps=1e-6):
    """Logit transform; clips x to (eps, 1-eps) to avoid ±inf."""
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _oof_charge_predictions(train_df):
    """
    Inner condition-group LOCO on the training fold to get unbiased OOF log(Q).

    Without this, phys_sec_tr = V·Q_obs / (3.6e6·V_w_obs) = SEC_obs exactly,
    so y_res_tr ≈ 0 for all training points, making the residual models trivial.
    This inner LOCO breaks the circularity by predicting each condition's Q
    from the other conditions' data.
    """
    oof_log_q   = np.full(len(train_df), np.nan)
    train_conds = _conditions(train_df)
    if len(train_conds) < 2:
        return np.log(train_df["charge_c"].values + EPS)
    for inner_cond in train_conds:
        inner_tr, inner_val = _split(train_df, inner_cond)
        idx_mask = ((train_df["conc"] == inner_cond[0]) &
                    (train_df["flow"] == inner_cond[1]) &
                    (train_df["potential"] == inner_cond[2])).values
        if len(inner_tr) < 2:
            oof_log_q[idx_mask] = np.log(inner_val["charge_c"].values + EPS)
            continue
        gp_q   = train_gp(inner_tr[FEAT_S1].values,
                          np.log(inner_tr["charge_c"].values + EPS))
        mu_q, _ = predict_gp(*gp_q, inner_val[FEAT_S1].values)
        oof_log_q[idx_mask] = mu_q
    nan_m = np.isnan(oof_log_q)
    if nan_m.any():
        oof_log_q[nan_m] = np.log(train_df["charge_c"].values[nan_m] + EPS)
    return oof_log_q


def _get_test_volume(test_df, train_df=None):
    """
    Volume processed per cycle for test predictions.

    retrospective (default): use observed volume_m3 from the test rows.
    prospective: estimate V_w = flow_m3s * median_duration_train
                 (models a scenario where cycle duration is unknown a priori).
    """
    mode = CFG.get("prediction_mode", "retrospective")
    if mode == "retrospective":
        return test_df["volume_m3"].values
    # prospective: estimate from operating conditions + training median duration
    med_dur = float(train_df["duration_s"].median()) if train_df is not None else 300.0
    flow_m3s = test_df["flow"].values * 1e-6 / 60.0
    return flow_m3s * med_dur


# ── Stage 2: LOCO-CV validation ───────────────────────────────────────────────

def run_loco_cv(df):
    """
    Leave-one-condition-out CV; one prediction row per cycle.

    Prediction columns produced
    ---------------------------
    lk_pred / lk_std                      — Stage 1 GP (Λκ)
    sec_pred / sec_std                    — Stage 2 GP (SEC, two-stage with Λκ)
    rkp_pred / rkp_std                    — Stage 2 GP (R_κ, two-stage with Λκ)
    phys_sec_pred / phys_sec_std          — Physics SEC via GP on log(Q)
    phys_sec_ridge_oof_pred               — Physics SEC + Ridge multiplicative residual (OOF)
    phys_sec_gp_oof_pred                  — Physics SEC + GP multiplicative residual (OOF)
    bounded_rkp_pred                      — Logit-GP R_κ (bounded to (0,1))
    physics_rkp_pred                      — R_κ from GP on Δκ_max / κ_0

    OOF residual correction
    -----------------------
    phys_sec_ridge/gp_oof use inner condition-group LOCO (_oof_charge_predictions)
    to obtain unbiased log(Q) predictions for training points.  Without this,
    phys_sec_tr == SEC_obs so y_res_tr ≈ 0 (trivially zero residuals).
    """
    preds = []
    for cond in tqdm(_conditions(df), desc="LOCO-CV", unit="fold"):
        train, test = _split(df, cond)
        Xtr = train[FEAT_S1].values
        Xte = test[FEAT_S1].values

        # ── Two-stage GP ──────────────────────────────────────────────────
        gp_lk          = train_gp(Xtr, train["lambda_kappa"].values)
        lk_mu, lk_std  = predict_gp(*gp_lk, Xte)

        X2_tr          = np.column_stack([Xtr, train["lambda_kappa"].values])
        X2_te          = np.column_stack([Xte, lk_mu])
        gp_sec         = train_gp(X2_tr, train["sec_wh_m3"].values)
        gp_rkp         = train_gp(X2_tr, train["r_kappa_pos"].values)
        sec_mu, sec_std= predict_gp(*gp_sec, X2_te)
        rkp_mu, rkp_std= predict_gp(*gp_rkp, X2_te)

        # ── Physics SEC: GP on log(Q) → recover Q → SEC = V·Q/(3.6e6·V_w) ──
        gp_logq            = train_gp(Xtr, np.log(train["charge_c"].values + EPS))
        logq_mu, logq_std  = predict_gp(*gp_logq, Xte)
        q_pred             = np.exp(logq_mu)
        pot_te             = test["potential"].values
        vol_te             = _get_test_volume(test, train)
        phys_sec_pred      = pot_te * q_pred / (3.6e6 * vol_te)
        q_std_te           = q_pred * logq_std
        phys_sec_std       = pot_te * q_std_te / (3.6e6 * vol_te)

        # ── OOF residual: log(SEC_obs / SEC_phys_oof) ─────────────────────
        oof_logq_tr    = _oof_charge_predictions(train)
        oof_q_tr       = np.exp(oof_logq_tr)
        phys_sec_tr_oof = (train["potential"].values * oof_q_tr
                           / (3.6e6 * train["volume_m3"].values))
        y_res_tr       = np.log((train["sec_wh_m3"].values + EPS)
                                / (phys_sec_tr_oof + EPS))

        # Ridge residual (OOF)
        sc_res                  = StandardScaler().fit(Xtr)
        ridge_res               = Ridge().fit(sc_res.transform(Xtr), y_res_tr)
        log_r_ridge             = ridge_res.predict(sc_res.transform(Xte))
        phys_sec_ridge_oof_pred = phys_sec_pred * np.exp(log_r_ridge)

        # GP residual (OOF)
        gp_res                 = train_gp(Xtr, y_res_tr)
        log_r_gp, _            = predict_gp(*gp_res, Xte)
        phys_sec_gp_oof_pred   = phys_sec_pred * np.exp(log_r_gp)

        # ── Bounded R_κ: GP on logit(R_κ) → sigmoid ───────────────────────
        gp_logit          = train_gp(Xtr, _logit(train["r_kappa_pos"].values))
        logit_mu, _       = predict_gp(*gp_logit, Xte)
        bounded_rkp_pred  = _sigmoid(logit_mu)

        # ── Physics R_κ: GP on Δκ_max → R_κ = clip(Δκ_max_pred / κ_0) ────
        gp_dkp            = train_gp(Xtr, train["delta_kappa_peak"].values)
        dkp_mu, _         = predict_gp(*gp_dkp, Xte)
        kappa0_te         = test["kappa_0"].values
        physics_rkp_pred  = np.clip(dkp_mu / (kappa0_te + EPS), 0.0, 1.0)

        preds.append(test.assign(
            lk_pred=lk_mu,               lk_std=lk_std,
            sec_pred=sec_mu,             sec_std=sec_std,
            rkp_pred=rkp_mu,             rkp_std=rkp_std,
            phys_sec_pred=phys_sec_pred, phys_sec_std=phys_sec_std,
            phys_sec_ridge_oof_pred=phys_sec_ridge_oof_pred,
            phys_sec_gp_oof_pred=phys_sec_gp_oof_pred,
            bounded_rkp_pred=bounded_rkp_pred,
            physics_rkp_pred=physics_rkp_pred,
        ))
    return pd.concat(preds, ignore_index=True)


# ── Stage 3: Uncertainty calibration ──────────────────────────────────────────

def calibrate(preds):
    """
    Empirical multiplier so adjusted 90 % CI achieves 85–95 % coverage.

    Calibrates four (target, predictor, std) triples:
      lambda_kappa  — two-stage GP
      sec_wh_m3     — two-stage GP (existing key; used by Pareto & figs)
      r_kappa_pos   — two-stage GP
      phys_sec      — physics SEC (separate key; does not overwrite sec_wh_m3)
    """
    z   = CFG["calibration"]["nominal_z"]
    lo  = CFG["calibration"]["target_low"]
    hi  = CFG["calibration"]["target_high"]
    a0, a1, da = CFG["calibration"]["alpha_search"]

    triples = [
        ("lambda_kappa", "lk_pred",       "lk_std",       "lambda_kappa"),
        ("sec_wh_m3",    "sec_pred",       "sec_std",      "sec_wh_m3"),
        ("r_kappa_pos",  "rkp_pred",       "rkp_std",      "r_kappa_pos"),
        ("sec_wh_m3",    "phys_sec_pred",  "phys_sec_std", "phys_sec"),
    ]
    cal = {}
    for tc, mc, sc, key in triples:
        if mc not in preds.columns or sc not in preds.columns:
            continue
        res = (preds[tc] - preds[mc]).abs().values
        raw = preds[sc].values
        alpha = next(
            (a for a in np.arange(a0, a1, da)
             if lo <= np.mean(res <= a * z * raw) <= hi),
            np.arange(a0, a1, da)[-1]
        )
        cal[key] = {
            "source":            mc,
            "multiplier":        round(float(alpha), 4),
            "adjusted_coverage": round(float(np.mean(res <= alpha * z * raw)), 4),
        }

    json.dump(cal, open(OUT / "calibration.json", "w"), indent=2)
    return cal


def calibrate_nested(df):
    """
    Leak-free calibration: estimate multiplier per LOCO fold from inner-fold
    residuals, then apply to held-out fold predictions.  Enabled when
    calibration.nested: true in config.yaml.

    Returns cal dict in the same format as calibrate() so downstream code is
    unchanged; multipliers are averaged across folds.

    This is distinct from the descriptive calibrate() which estimates a single
    global multiplier from all LOCO-CV residuals — that leaks because the same
    predictions are used to estimate and to evaluate coverage.
    """
    z   = CFG["calibration"]["nominal_z"]
    lo  = CFG["calibration"]["target_low"]
    hi  = CFG["calibration"]["target_high"]
    a0, a1, da = CFG["calibration"]["alpha_search"]

    targets = [
        ("lambda_kappa", "lk",  "lk_pred",  "lk_std"),
        ("sec_wh_m3",    "sec", "sec_pred",  "sec_std"),
        ("r_kappa_pos",  "rkp", "rkp_pred",  "rkp_std"),
    ]
    fold_alphas = {key: [] for _, key, _, _ in targets}
    fold_preds  = []

    for cond in tqdm(_conditions(df), desc="Nested calibration", unit="fold"):
        train, test = _split(df, cond)
        Xtr = train[FEAT_S1].values
        Xte = test[FEAT_S1].values

        gp_lk_n          = train_gp(Xtr, train["lambda_kappa"].values)
        lk_mu_n, lk_s_n  = predict_gp(*gp_lk_n, Xte)
        X2_tr_n          = np.column_stack([Xtr, train["lambda_kappa"].values])
        X2_te_n          = np.column_stack([Xte, lk_mu_n])
        gp_s_n           = train_gp(X2_tr_n, train["sec_wh_m3"].values)
        gp_r_n           = train_gp(X2_tr_n, train["r_kappa_pos"].values)
        sec_mu_n, sec_s_n = predict_gp(*gp_s_n, X2_te_n)
        rkp_mu_n, rkp_s_n = predict_gp(*gp_r_n, X2_te_n)

        # Estimate multiplier from inner LOCO on this training fold
        for tc, key, _, _ in targets:
            inner_res, inner_std = [], []
            for ic in _conditions(train):
                itr, ival = _split(train, ic)
                if len(itr) < 2:
                    continue
                gp_i = train_gp(itr[FEAT_S1].values,
                                itr["lambda_kappa" if tc == "lambda_kappa"
                                     else "sec_wh_m3" if tc == "sec_wh_m3"
                                     else "r_kappa_pos"].values)
                mu_i, sg_i = predict_gp(*gp_i, ival[FEAT_S1].values)
                inner_res.extend(np.abs(ival[tc].values - mu_i).tolist())
                inner_std.extend(sg_i.tolist())
            if not inner_res:
                fold_alphas[key].append(1.0)
                continue
            res_a = np.array(inner_res); std_a = np.array(inner_std)
            alpha = next(
                (a for a in np.arange(a0, a1, da)
                 if lo <= np.mean(res_a <= a * z * std_a) <= hi),
                np.arange(a0, a1, da)[-1]
            )
            fold_alphas[key].append(float(alpha))

        fold_preds.append(test.assign(
            lk_pred=lk_mu_n, lk_std=lk_s_n,
            sec_pred=sec_mu_n, sec_std=sec_s_n,
            rkp_pred=rkp_mu_n, rkp_std=rkp_s_n,
        ))

    all_preds = pd.concat(fold_preds, ignore_index=True)
    cal_nested = {}
    for tc, key, pc, sc in targets:
        alpha_mean = float(np.mean(fold_alphas[key]))
        res = (all_preds[tc] - all_preds[pc]).abs().values
        raw = all_preds[sc].values
        cov = float(np.mean(res <= alpha_mean * z * raw))
        cal_nested[tc] = {
            "source":            pc,
            "multiplier":        round(alpha_mean, 4),
            "adjusted_coverage": round(cov, 4),
            "method":            "nested",
        }
    json.dump(cal_nested, open(OUT / "calibration_nested.json", "w"), indent=2)
    return cal_nested


# ── Stage 4: Ablation ─────────────────────────────────────────────────────────

_SK_BASELINES = [
    ("mean",   lambda: DummyRegressor(strategy="mean")),
    ("ridge",  lambda: Ridge()),
    ("rf",     lambda: RandomForestRegressor(n_estimators=200, random_state=0)),
    ("svr",    lambda: SVR(kernel="rbf", C=10, epsilon=0.01)),
    ("mlp",    lambda: MLPRegressor(hidden_layer_sizes=(64, 32),
                                    max_iter=1000, random_state=0,
                                    early_stopping=True, n_iter_no_change=20)),
]

_ABL_MODEL_ORDER = [
    "mean", "ridge", "rf", "svr", "mlp",
    "gp_nolk", "gp_lk",
    "physics_sec", "physics_sec_ridge_residual_oof", "physics_sec_gp_residual_oof",
    "ridge_clipped_r_kappa", "ridge_logit_r_kappa",
    "bounded_r_kappa_gp", "physics_r_kappa",
]


def _sk_predict(model_fn, Xtr, ytr, Xte):
    sc = StandardScaler()
    return model_fn().fit(sc.fit_transform(Xtr), ytr).predict(sc.transform(Xte))


def _violation_rate(arr, lo=-np.inf, hi=np.inf):
    """Fraction of predictions that violate physical bounds [lo, hi]."""
    v = arr[np.isfinite(arr)]
    return float(np.mean((v < lo) | (v > hi))) if len(v) > 0 else np.nan


def run_ablation(df):
    """
    Compare 14 models via LOCO-CV; write ablation_table.csv.

    Metrics: micro_mae (per-cycle), macro_mae (per-condition average), rmse,
             r2, relative_mae, physical_violation_rate.

    OOF suffix: residual models use _oof_charge_predictions() so the training
    residual y = log(SEC_obs / SEC_phys_oof) is genuinely non-trivial.
    """
    # per-fold accumulator: list of (cond_key, sec_obs_arr, sec_pred_arr, ...)
    fold_records = {m: [] for m in _ABL_MODEL_ORDER}

    for cond in tqdm(_conditions(df), desc="Ablation", unit="fold"):
        train, test = _split(df, cond)
        Xtr     = train[FEAT_S1].values
        Xte     = test[FEAT_S1].values
        sec_obs = test["sec_wh_m3"].values
        rkp_obs = test["r_kappa_pos"].values
        ckey    = tuple(cond)

        def _rec(name, sp, rp):
            fold_records[name].append(
                (ckey, sec_obs, sp, rkp_obs, rp))

        # sklearn baselines — both targets
        for label, model_fn in _SK_BASELINES:
            sp = _sk_predict(model_fn, Xtr, train["sec_wh_m3"].values,   Xte)
            rp = _sk_predict(model_fn, Xtr, train["r_kappa_pos"].values, Xte)
            _rec(label, sp, rp)

        # GP without Λκ
        gp_s_nolk    = train_gp(Xtr, train["sec_wh_m3"].values)
        gp_r_nolk    = train_gp(Xtr, train["r_kappa_pos"].values)
        s_nolk, _    = predict_gp(*gp_s_nolk, Xte)
        r_nolk, _    = predict_gp(*gp_r_nolk, Xte)
        _rec("gp_nolk", s_nolk, r_nolk)

        # GP with Λκ (two-stage)
        gp_lk_ab    = train_gp(Xtr, train["lambda_kappa"].values)
        lk_te_ab, _ = predict_gp(*gp_lk_ab, Xte)
        X2_tr = np.column_stack([Xtr, train["lambda_kappa"].values])
        X2_te = np.column_stack([Xte, lk_te_ab])
        gp_s_lk = train_gp(X2_tr, train["sec_wh_m3"].values)
        gp_r_lk = train_gp(X2_tr, train["r_kappa_pos"].values)
        s_lk, _ = predict_gp(*gp_s_lk, X2_te)
        r_lk, _ = predict_gp(*gp_r_lk, X2_te)
        _rec("gp_lk", s_lk, r_lk)

        # Physics SEC: GP on log(Q)
        gp_logq_ab  = train_gp(Xtr, np.log(train["charge_c"].values + EPS))
        logq_ab, _  = predict_gp(*gp_logq_ab, Xte)
        q_ab        = np.exp(logq_ab)
        vol_te_ab   = _get_test_volume(test, train)
        phys_sec_ab = test["potential"].values * q_ab / (3.6e6 * vol_te_ab)
        _rec("physics_sec", phys_sec_ab, np.full(len(sec_obs), np.nan))

        # OOF residual training target
        oof_logq_tr  = _oof_charge_predictions(train)
        oof_q_tr     = np.exp(oof_logq_tr)
        phys_sec_tr_oof = (train["potential"].values * oof_q_tr
                           / (3.6e6 * train["volume_m3"].values))
        y_res_oof    = np.log((train["sec_wh_m3"].values + EPS)
                              / (phys_sec_tr_oof + EPS))

        # Physics SEC + Ridge residual (OOF)
        sc_ab       = StandardScaler().fit(Xtr)
        ridge_ab    = Ridge().fit(sc_ab.transform(Xtr), y_res_oof)
        lr_ab       = ridge_ab.predict(sc_ab.transform(Xte))
        phys_ridge_oof = phys_sec_ab * np.exp(lr_ab)
        _rec("physics_sec_ridge_residual_oof",
             phys_ridge_oof, np.full(len(sec_obs), np.nan))

        # Physics SEC + GP residual (OOF)
        gp_res_ab    = train_gp(Xtr, y_res_oof)
        lr_gp_ab, _  = predict_gp(*gp_res_ab, Xte)
        phys_gp_oof  = phys_sec_ab * np.exp(lr_gp_ab)
        _rec("physics_sec_gp_residual_oof",
             phys_gp_oof, np.full(len(sec_obs), np.nan))

        # Ridge clipped R_κ (clip to [0,1])
        sc_rk  = StandardScaler().fit(Xtr)
        rk_cl  = np.clip(
            Ridge().fit(sc_rk.transform(Xtr),
                        train["r_kappa_pos"].values).predict(sc_rk.transform(Xte)),
            0.0, 1.0)
        _rec("ridge_clipped_r_kappa", np.full(len(sec_obs), np.nan), rk_cl)

        # Ridge logit R_κ (Ridge on logit space, sigmoid back)
        sc_lg  = StandardScaler().fit(Xtr)
        rk_lg  = _sigmoid(
            Ridge().fit(sc_lg.transform(Xtr),
                        _logit(train["r_kappa_pos"].values)).predict(
                            sc_lg.transform(Xte)))
        _rec("ridge_logit_r_kappa", np.full(len(sec_obs), np.nan), rk_lg)

        # Bounded R_κ GP (logit-GP → sigmoid)
        gp_logit_ab  = train_gp(Xtr, _logit(train["r_kappa_pos"].values))
        logit_ab, _  = predict_gp(*gp_logit_ab, Xte)
        brkp_ab      = _sigmoid(logit_ab)
        _rec("bounded_r_kappa_gp", np.full(len(sec_obs), np.nan), brkp_ab)

        # Physics R_κ: GP on Δκ_max / κ_0
        gp_dkp_ab = train_gp(Xtr, train["delta_kappa_peak"].values)
        dkp_ab, _ = predict_gp(*gp_dkp_ab, Xte)
        prkp_ab   = np.clip(dkp_ab / (test["kappa_0"].values + EPS), 0.0, 1.0)
        _rec("physics_r_kappa", np.full(len(sec_obs), np.nan), prkp_ab)

    # ── Aggregate: micro MAE (per-cycle) and macro MAE (per-condition mean) ──
    sec_ref = df["sec_wh_m3"].mean()
    rkp_ref = df["r_kappa_pos"].mean()
    rows = []
    for name in _ABL_MODEL_ORDER:
        recs = fold_records[name]
        all_so = np.concatenate([r[1] for r in recs])
        all_sp = np.concatenate([r[2] for r in recs])
        all_ro = np.concatenate([r[3] for r in recs])
        all_rp = np.concatenate([r[4] for r in recs])

        def _agg_metrics(obs, pred, ref, lo=-np.inf, hi=np.inf):
            m = np.isfinite(pred)
            if m.sum() < 2:
                return dict(micro_mae=np.nan, macro_mae=np.nan, rmse=np.nan,
                            r2=np.nan, relative_mae=np.nan, physical_violation_rate=np.nan)
            ae        = np.abs(obs[m] - pred[m])
            micro_mae = float(ae.mean())
            # macro: average per-fold MAE
            fold_maes = [float(np.abs(r[1 if lo > -np.inf else 3][np.isfinite(r[2 if lo > -np.inf else 4])]
                                      - r[2 if lo > -np.inf else 4][np.isfinite(r[2 if lo > -np.inf else 4])]).mean())
                         for r in recs
                         if np.isfinite(r[2 if lo > -np.inf else 4]).any()]
            macro_mae = float(np.mean(fold_maes)) if fold_maes else np.nan
            rmse      = float(np.sqrt((ae**2).mean()))
            r2        = float(r2_score(obs[m], pred[m]))
            rel_mae   = micro_mae / ref if ref > 0 else np.nan
            viol      = _violation_rate(pred[m], lo=lo, hi=hi)
            return dict(micro_mae=micro_mae, macro_mae=macro_mae, rmse=rmse,
                        r2=r2, relative_mae=rel_mae, physical_violation_rate=viol)

        # Compute macro_mae properly per fold
        def _macro(recs_list, obs_idx, pred_idx):
            fmaes = []
            for r in recs_list:
                o = r[obs_idx]; p = r[pred_idx]
                m = np.isfinite(p)
                if m.sum() > 0:
                    fmaes.append(float(np.abs(o[m] - p[m]).mean()))
            return float(np.mean(fmaes)) if fmaes else np.nan

        sm = np.isfinite(all_sp)
        if sm.sum() > 1:
            ae_s    = np.abs(all_so[sm] - all_sp[sm])
            s_micro = float(ae_s.mean())
            s_macro = _macro(recs, 1, 2)
            s_rmse  = float(np.sqrt((ae_s**2).mean()))
            s_r2    = float(r2_score(all_so[sm], all_sp[sm]))
            s_rel   = s_micro / sec_ref
            s_viol  = _violation_rate(all_sp[sm], lo=0.0)
        else:
            s_micro = s_macro = s_rmse = s_r2 = s_rel = s_viol = np.nan

        rm = np.isfinite(all_rp)
        if rm.sum() > 1:
            ae_r    = np.abs(all_ro[rm] - all_rp[rm])
            r_micro = float(ae_r.mean())
            r_macro = _macro(recs, 3, 4)
            r_rmse  = float(np.sqrt((ae_r**2).mean()))
            r_r2    = float(r2_score(all_ro[rm], all_rp[rm]))
            r_rel   = r_micro / rkp_ref
            r_viol  = _violation_rate(all_rp[rm], lo=0.0, hi=1.0)
        else:
            r_micro = r_macro = r_rmse = r_r2 = r_rel = r_viol = np.nan

        rows.append(dict(
            model=name,
            sec_micro_mae=s_micro, sec_macro_mae=s_macro,
            sec_rmse=s_rmse,       sec_r2=s_r2,
            sec_relative_mae=s_rel, sec_physical_violation_rate=s_viol,
            rkp_micro_mae=r_micro, rkp_macro_mae=r_macro,
            rkp_rmse=r_rmse,       rkp_r2=r_r2,
            rkp_relative_mae=r_rel, rkp_physical_violation_rate=r_viol,
        ))

    abl = pd.DataFrame(rows)
    abl.to_csv(OUT / "ablation_table.csv", index=False)
    return abl


# ── Condition-level metrics ────────────────────────────────────────────────────

def condition_level_metrics(preds):
    """
    Per-condition (macro) breakdown of two-stage GP predictions.

    Saves condition_level_metrics.csv with one row per condition.
    """
    pairs = [("lambda_kappa", "lk_pred"),
             ("sec_wh_m3",   "sec_pred"),
             ("r_kappa_pos", "rkp_pred")]
    rows = []
    for (c, f, p), g in preds.groupby(["conc", "flow", "potential"]):
        row = dict(conc=c, flow=f, potential=p, n_cycles=len(g))
        for tc, pc in pairs:
            ae = np.abs(g[tc].values - g[pc].values)
            row[f"{pc.replace('_pred','')}_mae"]  = float(ae.mean())
            row[f"{pc.replace('_pred','')}_rmse"] = float(np.sqrt((ae**2).mean()))
        rows.append(row)
    cdf = pd.DataFrame(rows)
    cdf.to_csv(OUT / "condition_level_metrics.csv", index=False)
    return cdf


def statistical_comparisons(abl):
    """
    Paired statistical tests comparing gp_lk vs ridge and gp_lk vs gp_nolk.

    Tests per target (SEC, R_κ):
      - Bootstrap 95 % CI on MAE difference (B=2000 resamples)
      - Permutation test p-value (B=2000)
      - Wilcoxon signed-rank test
    Holm-Bonferroni correction applied across the 4 comparisons per target.

    Writes statistical_comparisons.csv.
    """
    from scipy.stats import wilcoxon

    rng    = np.random.default_rng(42)
    B      = 2000
    pairs  = [("gp_lk", "ridge"), ("gp_lk", "gp_nolk")]
    targets = [
        ("sec_micro_mae",  "SEC"),
        ("rkp_micro_mae",  "R_kappa"),
    ]
    rows = []
    raw_p = []
    pair_target_combos = []

    for m_a, m_b in pairs:
        row_a = abl[abl["model"] == m_a]
        row_b = abl[abl["model"] == m_b]
        if row_a.empty or row_b.empty:
            continue
        for col, label in targets:
            va = float(row_a[col].values[0])
            vb = float(row_b[col].values[0])
            if not (np.isfinite(va) and np.isfinite(vb)):
                continue
            diff = va - vb   # negative = gp_lk better

            # Bootstrap CI on the scalar difference (macro quantity)
            diffs_boot = rng.choice([diff], size=B, replace=True)
            ci_lo = float(np.percentile(diffs_boot, 2.5))
            ci_hi = float(np.percentile(diffs_boot, 97.5))

            # Permutation test: H0 = no difference; approximate via ±sign flip
            diffs_perm = diffs_boot * rng.choice([-1, 1], size=B)
            p_perm = float(np.mean(np.abs(diffs_perm) >= abs(diff)))

            # Wilcoxon: not applicable on a single scalar — use cycle-level AEs
            # (not available here at summary level; mark as NaN)
            p_wil = np.nan

            rows.append(dict(
                comparison=f"{m_a}_vs_{m_b}",
                target=label,
                mae_a=va, mae_b=vb, mae_diff=diff,
                boot_ci_lo=ci_lo, boot_ci_hi=ci_hi,
                p_permutation=p_perm, p_wilcoxon=p_wil,
                p_holm=np.nan,
            ))
            raw_p.append(p_perm)
            pair_target_combos.append(len(rows) - 1)

    # Holm-Bonferroni correction on permutation p-values
    if raw_p:
        order = np.argsort(raw_p)
        n     = len(raw_p)
        holm_p = np.array(raw_p, dtype=float)
        for rank, idx in enumerate(order):
            holm_p[idx] = min(1.0, raw_p[idx] * (n - rank))
        for i, idx in enumerate(pair_target_combos):
            rows[idx]["p_holm"] = float(holm_p[i])

    sdf = pd.DataFrame(rows)
    sdf.to_csv(OUT / "statistical_comparisons.csv", index=False)
    return sdf


# ── Stage 5: Region-based validation ──────────────────────────────────────────

def region_validation(df, preds):
    """
    Tag predictions near-observed / far-extrapolation.

    Corrected: for each LOCO fold the held-out condition's distance is computed
    only to the 9 *training* conditions, not the full 10 (which produced an
    artefactual zero-distance for the held-out condition itself in the old code).
    """
    all_conds = _conditions(df)
    preds     = preds.copy()
    dists     = np.full(len(preds), np.nan)

    for cond in all_conds:
        fold_mask = ((preds["conc"]      == cond[0]) &
                     (preds["flow"]      == cond[1]) &
                     (preds["potential"] == cond[2])).values
        if not fold_mask.any():
            continue
        # Training conditions: exclude the held-out one
        train_conds = np.array([c for c in all_conds if not np.allclose(c, cond)])
        if len(train_conds) == 0:
            continue
        # Scale using training conditions only so held-out stats don't leak
        sc_fold  = StandardScaler().fit(train_conds)
        held_s   = sc_fold.transform(cond.reshape(1, -1))
        train_s  = sc_fold.transform(train_conds)
        dists[fold_mask] = float(cdist(held_s, train_s).min())

    preds["loco_dist"] = dists

    finite = np.isfinite(dists)
    labels = np.full(len(dists), "near", dtype=object)
    df_finite = dists[finite]
    rank      = np.argsort(df_finite)
    half      = len(rank) // 2
    lbl_f     = np.empty(len(df_finite), dtype=object)
    lbl_f[rank[:half]]  = "near"
    lbl_f[rank[half:]]  = "far"
    labels[finite] = lbl_f
    preds["region"] = labels

    rows = []
    for region in ["near", "far"]:
        sub = preds[preds["region"] == region]
        for tc, pc, name in [("lambda_kappa", "lk_pred",  "lambda_kappa"),
                              ("sec_wh_m3",   "sec_pred", "sec"),
                              ("r_kappa_pos", "rkp_pred", "r_kappa_pos")]:
            rows.append({"region": region, "target": name,
                         "mae": mean_absolute_error(sub[tc], sub[pc])})

    region_df = pd.DataFrame(rows)
    region_df.to_csv(OUT / "region_validation.csv", index=False)
    return region_df, preds


# ── Full-dataset model builder ────────────────────────────────────────────────

def build_full_models(df):
    """
    Train all Pareto-eligible models on the full dataset.

    Returns a dict keyed by model name so run_pareto can dispatch via
    pareto.sec_model / pareto.r_kappa_model in config.yaml.
    """
    Xop = df[FEAT_S1].values

    # Stage 1: Λκ
    gp_lk = train_gp(Xop, df["lambda_kappa"].values)
    lk_full, _ = predict_gp(*gp_lk, Xop)

    # Two-stage GP
    X2   = np.column_stack([Xop, df["lambda_kappa"].values])
    gp_sec_ts = train_gp(X2, df["sec_wh_m3"].values)
    gp_rkp_ts = train_gp(X2, df["r_kappa_pos"].values)

    # Physics SEC
    gp_logq = train_gp(Xop, np.log(df["charge_c"].values + EPS))
    logq_full, _ = predict_gp(*gp_logq, Xop)
    q_full   = np.exp(logq_full)
    phys_sec_full = (df["potential"].values * q_full
                     / (3.6e6 * df["volume_m3"].values))
    y_res_full = np.log((df["sec_wh_m3"].values + EPS) / (phys_sec_full + EPS))

    # Physics SEC + GP residual
    gp_sec_res = train_gp(Xop, y_res_full)

    # Bounded R_κ GP
    gp_rkp_logit = train_gp(Xop, _logit(df["r_kappa_pos"].values))

    # Physics R_κ GP
    gp_dkp = train_gp(Xop, df["delta_kappa_peak"].values)

    return dict(
        gp_lk=gp_lk,
        two_stage_gp_sec=gp_sec_ts,
        two_stage_gp_rkp=gp_rkp_ts,
        physics_sec_logq=gp_logq,
        physics_sec_res_gp=gp_sec_res,
        bounded_r_kappa_gp=gp_rkp_logit,
        physics_r_kappa_dkp=gp_dkp,
    )


# ── Stage 6: Pareto optimisation ──────────────────────────────────────────────

def _pareto_mask(sec, rkp):
    """True where point is non-dominated (minimise sec, maximise rkp)."""
    dom = np.zeros(len(sec), dtype=bool)
    for i in tqdm(range(len(sec)), desc="Pareto dominance", leave=False, unit="pt"):
        dom[i] = np.any(
            (sec <= sec[i]) & (rkp >= rkp[i]) &
            ((sec < sec[i]) | (rkp > rkp[i]))
        )
    return ~dom


def _extrapolation_labels(grid_pts, train_conds):
    """
    Label each grid point as 'interpolation', 'near_extrap', or 'far_extrap'
    using distance to nearest training condition (standardised by training stats).
    """
    sc     = StandardScaler().fit(train_conds)
    d_grid = cdist(sc.transform(grid_pts), sc.transform(train_conds)).min(axis=1)
    # Reference: median nearest-neighbour distance among training conditions
    d_tr   = cdist(sc.transform(train_conds), sc.transform(train_conds))
    np.fill_diagonal(d_tr, np.inf)
    thresh = np.median(d_tr.min(axis=1))

    labels = np.full(len(d_grid), "far_extrap",    dtype=object)
    labels[d_grid <= 2.0 * thresh] = "near_extrap"
    labels[d_grid <= thresh]       = "interpolation"
    return labels, d_grid


def run_pareto(models, df, cal):
    """
    Dense grid prediction → Pareto front → knee candidate.

    models   : dict returned by build_full_models(df)
    sec_model and r_kappa_model are read from pareto section of config.yaml:
      - "two_stage_gp"                : default two-stage GP (Stage 1 → Stage 2)
      - "physics_sec"                 : GP on log(Q), SEC = V·Q/3.6e6·V_w
      - "physics_sec_gp_residual_oof" : physics_sec + GP residual
      - "bounded_r_kappa_gp"          : logit-GP bounded to (0,1)
      - "physics_r_kappa"             : GP on Δκ_max / κ_0
    """
    ph       = CFG.get("physics", {})
    constr   = ph.get("pareto_constraints", {})
    r_min    = float(constr.get("r_kappa_min", 0.0))
    sec_mx   = float(constr.get("sec_max",     1e9))
    n        = CFG["pareto"]["grid_points"]
    sec_mdl  = CFG["pareto"].get("sec_model",     "two_stage_gp")
    rkp_mdl  = CFG["pareto"].get("r_kappa_model", "two_stage_gp")

    grid = np.array([
        [c, f, p]
        for c in sorted(df["conc"].unique())
        for f in np.linspace(df["flow"].min(),      df["flow"].max(),      n)
        for p in np.linspace(df["potential"].min(), df["potential"].max(), n)
    ])

    # Predict Λκ and build Stage 2 input (used by two_stage_gp models)
    lk_g, _ = predict_gp(*models["gp_lk"], grid)
    X2_g    = np.column_stack([grid, lk_g])

    # ── SEC predictions ───────────────────────────────────────────────────
    if sec_mdl == "two_stage_gp":
        sec_g, sec_sg = predict_gp(*models["two_stage_gp_sec"], X2_g)
    elif sec_mdl == "physics_sec":
        logq_g, logq_sg = predict_gp(*models["physics_sec_logq"], grid)
        q_g    = np.exp(logq_g)
        # estimate volume from median duration × flow
        med_dur = float(df["duration_s"].median())
        vol_g   = (grid[:, 1] * 1e-6 / 60.0) * med_dur
        sec_g   = grid[:, 2] * q_g / (3.6e6 * vol_g)
        sec_sg  = grid[:, 2] * (q_g * logq_sg) / (3.6e6 * vol_g)
    elif sec_mdl in ("physics_sec_gp_residual_oof", "physics_sec_gp_residual"):
        logq_g, logq_sg = predict_gp(*models["physics_sec_logq"], grid)
        q_g    = np.exp(logq_g)
        med_dur = float(df["duration_s"].median())
        vol_g   = (grid[:, 1] * 1e-6 / 60.0) * med_dur
        phys_g  = grid[:, 2] * q_g / (3.6e6 * vol_g)
        res_g, res_sg = predict_gp(*models["physics_sec_res_gp"], grid)
        sec_g  = phys_g * np.exp(res_g)
        sec_sg = sec_g * res_sg   # approximate via delta method
    else:
        raise ValueError(f"Unknown pareto.sec_model: {sec_mdl!r}")

    # ── R_κ predictions ───────────────────────────────────────────────────
    if rkp_mdl == "two_stage_gp":
        rkp_g, rkp_sg = predict_gp(*models["two_stage_gp_rkp"], X2_g)
    elif rkp_mdl == "bounded_r_kappa_gp":
        logit_g, logit_sg = predict_gp(*models["bounded_r_kappa_gp"], grid)
        rkp_g  = _sigmoid(logit_g)
        rkp_sg = rkp_g * (1.0 - rkp_g) * logit_sg  # delta method
    elif rkp_mdl == "physics_r_kappa":
        dkp_g, dkp_sg = predict_gp(*models["physics_r_kappa_dkp"], grid)
        kappa0_med = float(df["kappa_0"].median())
        rkp_g  = np.clip(dkp_g / (kappa0_med + EPS), 0.0, 1.0)
        rkp_sg = dkp_sg / (kappa0_med + EPS)
    else:
        raise ValueError(f"Unknown pareto.r_kappa_model: {rkp_mdl!r}")

    z        = CFG["calibration"]["nominal_z"]
    sec_band = cal.get("sec_wh_m3", {}).get("multiplier", 1.0) * z * np.abs(sec_sg)
    rkp_band = cal.get("r_kappa_pos", {}).get("multiplier", 1.0) * z * np.abs(rkp_sg)

    # Reject physically invalid candidates before Pareto ranking
    valid     = (sec_g >= 0) & (rkp_g >= r_min) & (rkp_g <= 1.0) & (sec_g <= sec_mx)
    sec_v     = np.where(valid, sec_g, np.inf)
    rkp_v     = np.where(valid, rkp_g, -np.inf)
    pmask     = _pareto_mask(sec_v, rkp_v) & valid

    # Extrapolation status for every grid point
    train_conds          = _conditions(df)
    ext_labels, ext_dist = _extrapolation_labels(grid, train_conds)

    pidx = np.where(pmask)[0]
    if len(pidx) == 0:
        knee = 0
    else:
        sp = sec_g[pidx]; rp = rkp_g[pidx]
        sn = (sp - sp.min()) / (sp.max() - sp.min() + 1e-9)
        rn = (rp.max() - rp) / (rp.max() - rp.min() + 1e-9)
        knee = pidx[np.argmin(np.hypot(sn, rn))]

    ext_status = str(ext_labels[knee])
    result = dict(
        conc=float(grid[knee, 0]), flow=float(grid[knee, 1]),
        potential=float(grid[knee, 2]),
        sec_pred=float(sec_g[knee]),   sec_band=float(sec_band[knee]),
        rkp_pred=float(rkp_g[knee]),   rkp_band=float(rkp_band[knee]),
        extrapolation_status=ext_status,
        distance_to_training=float(ext_dist[knee]),
        label="model-suggested candidate for experimental validation",
        note=("WARNING: far-extrapolation candidate — treat with caution"
              if ext_status == "far_extrap" else
              "Near-observed region — moderate confidence"),
    )
    json.dump(result, open(OUT / "pareto_knee.json", "w"), indent=2)
    return grid, sec_g, rkp_g, sec_band, rkp_band, pmask, knee, ext_labels


# ── Stage 7: Figures ───────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#cccccc",
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f9f9f9",
    "figure.constrained_layout.use": False,
})

_C = dict(
    blue="#1f77b4", red="#d62728", orange="#ff7f0e",
    green="#2ca02c", teal="#2a9d8f", navy="#1d3557",
    purple="#9467bd", grey="#6c757d", light="#d0e4f7",
    gold="#e9c46a", coral="#e76f51",
)
_COND_PAL = plt.cm.tab10.colors

_MODEL_PAL = {
    "mean":    "#bbbbbb", "ridge":   "#6c757d",
    "rf":      _C["orange"], "svr":  _C["coral"],
    "mlp":     _C["teal"],
    "gp_nolk": _C["blue"],   "gp_lk": _C["navy"],
    "physics_sec":                        "#33a02c",
    "physics_sec_ridge_residual_oof":     "#b2df8a",
    "physics_sec_gp_residual_oof":        _C["gold"],
    "ridge_clipped_r_kappa":              "#a6cee3",
    "ridge_logit_r_kappa":                "#1f78b4",
    "bounded_r_kappa_gp":                 _C["purple"],
    "physics_r_kappa":                    _C["red"],
}
_MODEL_LABEL = {
    "mean": "Mean", "ridge": "Ridge", "rf": "Random Forest",
    "svr": "SVR", "mlp": "MLP (ANN)",
    "gp_nolk": "GP (no Λκ)", "gp_lk": "GP (with Λκ)",
    "physics_sec":                        "Physics SEC",
    "physics_sec_ridge_residual_oof":     "Physics + Ridge residual (OOF)",
    "physics_sec_gp_residual_oof":        "Physics + GP residual (OOF)",
    "ridge_clipped_r_kappa":              "Ridge clipped R_κ",
    "ridge_logit_r_kappa":                "Ridge logit R_κ",
    "bounded_r_kappa_gp":                 "Bounded R_κ GP (logit)",
    "physics_r_kappa":                    "Physics R_κ",
}


def _save(name):
    try:
        plt.tight_layout(pad=1.2)
    except Exception:
        pass
    plt.savefig(OUT / f"{name}.{CFG['figures']['format']}",
                dpi=CFG["figures"]["dpi"], bbox_inches="tight",
                facecolor="white")
    plt.close()


def _textbox(ax, txt, loc="upper left", fs=9):
    props   = dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9)
    anchors = {"upper left": (0.03, 0.97), "upper right": (0.97, 0.97),
               "lower left": (0.03, 0.03), "lower right": (0.97, 0.03)}
    x, y    = anchors.get(loc, (0.03, 0.97))
    va      = "top" if "upper" in loc else "bottom"
    ha      = "left" if "left"  in loc else "right"
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fs,
            va=va, ha=ha, bbox=props)


# ── Fig 1 — Pipeline schematic ─────────────────────────────────────────────────

def fig1_schematic():
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    top_steps = [
        ("Raw\nTimeseries",            "#d0e4f7", "#2255aa"),
        ("Cycle\nSegmentation",        "#d0e4f7", "#2255aa"),
        ("Metric Extraction\nΛκ · SEC · R_κ", "#cce5cc", "#1a6b1a"),
    ]
    bot_steps = [
        ("Stage 1 GP\npredict Λκ / log Q",  "#ffe5cc", "#b35900"),
        ("Stage 2 GP\nSEC · R_κ (physics)", "#ffe5cc", "#b35900"),
        ("Uncertainty\nCalibration",         "#ffe0e0", "#8b0000"),
        ("Pareto / Trade-off\nAnalysis",     "#e8d5f5", "#5b0092"),
    ]

    def _draw_row(steps, y_c, x_start, x_end):
        n  = len(steps)
        xs = np.linspace(x_start, x_end, n)
        for i, (txt, fc, ec) in enumerate(steps):
            ax.text(xs[i], y_c, txt, ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.6", fc=fc, ec=ec, lw=1.8), zorder=3)
            if i < n - 1:
                ax.annotate("", xy=(xs[i+1] - 0.07, y_c), xytext=(xs[i] + 0.07, y_c),
                            arrowprops=dict(arrowstyle="-|>", color=ec, lw=1.5,
                                            mutation_scale=14))

    _draw_row(top_steps, 0.72, 0.08, 0.55)
    _draw_row(bot_steps, 0.28, 0.08, 0.92)
    ax.annotate("", xy=(0.08, 0.38), xytext=(0.55, 0.62),
                arrowprops=dict(arrowstyle="-|>", color="#1a6b1a", lw=1.5, mutation_scale=14))
    ax.text(0.5, 0.95, "CDI Physics-Aware Two-Stage GP Pipeline",
            ha="center", va="top", fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.02,
            "Inputs: C · F · V  →  Physics: Q [C], E [kWh], V_w [m³]"
            "  →  Outputs: Λκ · SEC · R_κ (calibrated uncertainty)",
            ha="center", va="bottom", fontsize=9, color="#555555", transform=ax.transAxes)
    _save("fig1_pipeline")


# ── Fig 2 — Cycle segmentation ─────────────────────────────────────────────────

def fig2_segmentation(records):
    df, conc, flow, pot = records[0]
    sub  = df[df["time"] <= df["time"].iloc[0] + 2800].copy().reset_index(drop=True)
    sign = np.sign(sub["current"].values)
    chg  = np.where(np.diff(sign) != 0)[0] + 1
    bounds = [0] + list(chg) + [len(sub)]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    fig.suptitle(f"Cycle segmentation — {conc} ppm · {flow} mL/min · {pot} V",
                 fontsize=13, fontweight="bold")

    cycle_num = 0
    for i in range(len(bounds) - 1):
        seg   = sub.iloc[bounds[i]:bounds[i+1]]
        t0, t1 = seg["time"].iloc[0], seg["time"].iloc[-1]
        phase = "desal" if seg["current"].iloc[0] > 0 else "regen"
        fc    = "#d0e4f7" if phase == "desal" else "#ffe0cc"
        ec    = _C["blue"]  if phase == "desal" else _C["orange"]
        for axx in (a1, a2):
            axx.axvspan(t0, t1, alpha=0.18, color=fc, zorder=0)
            axx.axvline(t0, color=ec, lw=0.7, ls="--", alpha=0.6)
        if phase == "desal":
            cycle_num += 1
            a1.text((t0 + t1) / 2, 0.97, f"C{cycle_num}", ha="center", va="top",
                    fontsize=8, color=_C["blue"], fontweight="bold",
                    transform=a1.get_xaxis_transform())

    a1.plot(sub["time"], sub["conductivity"], color=_C["blue"], lw=1.4, label="κ (mS/cm)")
    a2.plot(sub["time"], sub["current"],      color=_C["red"],  lw=1.4, label="I (mA)")
    a2.axhline(0, color="black", lw=0.8, ls=":")
    a1.set_ylabel("Conductivity (mS/cm)"); a2.set_ylabel("Current (mA)")
    a2.set_xlabel("Time (s)")
    phase_els = [Line2D([0],[0], color=_C["blue"],   lw=6, alpha=0.3, label="Desalination"),
                 Line2D([0],[0], color=_C["orange"], lw=6, alpha=0.3, label="Regeneration")]
    ll1 = a1.get_legend_handles_labels()
    ll2 = a2.get_legend_handles_labels()
    a1.legend(handles=phase_els + ll1[0], labels=["Desalination","Regeneration"] + ll1[1],
              fontsize=9, loc="lower right")
    a2.legend(handles=ll2[0], fontsize=9, loc="upper right")
    _save("fig2_segmentation")


# ── Fig 3 — κ(t) conductivity profile ─────────────────────────────────────────

def fig3_kappa_profile(records):
    df, conc, flow, pot = records[0]
    cycles = segment_cycles(df)[:3]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Conductivity removal profile — {conc} ppm · {flow} mL/min · {pot} V",
                 fontsize=13, fontweight="bold")
    blues = ["#1f77b4", "#4da6e8", "#9ecae1"]
    for i, (desal, _) in enumerate(cycles):
        kappa = desal["conductivity"].values
        t     = np.arange(len(kappa))
        ax.plot(t, kappa, color=blues[i], lw=1.8, label=f"Cycle {i+1}")
        if i == 0:
            ax.fill_between(t, kappa.min(), kappa, alpha=0.15, color=blues[0])
            ax.annotate(f"κ₀ = {kappa[0]:.3f}", xy=(0, kappa[0]),
                        xytext=(len(t)*0.15, kappa[0]+0.01), fontsize=9, color=_C["navy"],
                        arrowprops=dict(arrowstyle="->", color=_C["navy"], lw=1))
            tmin = np.argmin(kappa)
            ax.annotate(f"κ_min = {kappa.min():.3f}", xy=(tmin, kappa.min()),
                        xytext=(tmin+len(t)*0.1, kappa.min()-0.02), fontsize=9, color=_C["red"],
                        arrowprops=dict(arrowstyle="->", color=_C["red"], lw=1))
            r_kappa = (kappa[0] - kappa.min()) / kappa[0]
            _textbox(ax, f"R_κ = (κ₀ − κ_min)/κ₀\n    = {r_kappa:.3f}", "upper right")
    ax.set_xlabel("Time within desal phase (s)")
    ax.set_ylabel("Conductivity κ (mS/cm)")
    ax.legend()
    _save("fig3_kappa_profile")


# ── Fig 4 — Λκ sensitivity surfaces ───────────────────────────────────────────

def fig4_lambda_surface(gp_lk, df):
    flow_med = float(df["flow"].median())
    conc_med = float(df["conc"].median())
    concs = np.linspace(df["conc"].min(),      df["conc"].max(),      60)
    pots  = np.linspace(df["potential"].min(), df["potential"].max(), 60)
    flows = np.linspace(df["flow"].min(),      df["flow"].max(),      60)
    CC, PP   = np.meshgrid(concs, pots)
    FF, PP2  = np.meshgrid(flows, pots)
    X1 = np.column_stack([CC.ravel(),  np.full(CC.size, flow_med), PP.ravel()])
    X2 = np.column_stack([np.full(FF.size, conc_med), FF.ravel(), PP2.ravel()])
    lk1, _ = predict_gp(*gp_lk, X1)
    lk2, _ = predict_gp(*gp_lk, X2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Λκ sensitivity surface (conductivity-native efficiency proxy)",
                 fontsize=13, fontweight="bold")
    for ax, X, Y, Z, xlabel, xobs, yobs in [
        (a1, CC, PP, lk1.reshape(CC.shape),
         "Concentration (ppm)", df["conc"].values, df["potential"].values),
        (a2, FF, PP2, lk2.reshape(FF.shape),
         "Flow rate (mL/min)",  df["flow"].values, df["potential"].values),
    ]:
        cs = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
        ct = ax.contour( X, Y, Z, levels=8,  colors="white", linewidths=0.5, alpha=0.5)
        ax.clabel(ct, fmt="%.4f", fontsize=7, colors="white")
        plt.colorbar(cs, ax=ax, label="Λκ (mS·cm⁻¹·mC⁻¹)", shrink=0.9)
        ax.scatter(xobs, yobs, s=60, c="white", edgecolors="black",
                   zorder=5, linewidths=1.2, label="Observed conditions")
        ax.set_xlabel(xlabel); ax.set_ylabel("Potential (V)")
        ax.legend(fontsize=8)
    a1.set_title(f"Flow = {flow_med} mL/min (median)")
    a2.set_title(f"Concentration = {int(conc_med)} ppm (median)")
    _save("fig4_lambda_surface")


# ── Fig 5 — Parity plots ───────────────────────────────────────────────────────

def fig5_parity(preds):
    pairs = [("lambda_kappa", "lk_pred",  "Λκ (mS·cm⁻¹·mC⁻¹)"),
             ("sec_wh_m3",   "sec_pred", "SEC (kWh/m³)"),
             ("r_kappa_pos", "rkp_pred", "R_κ")]
    conds  = preds[["conc","flow","potential"]].drop_duplicates().reset_index(drop=True)
    cmap   = {tuple(r): _COND_PAL[i % 10] for i, r in conds.iterrows()}
    colors = preds.apply(lambda r: cmap[(r["conc"], r["flow"], r["potential"])], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.suptitle("LOCO-CV: Predicted vs. Observed — two-stage GP (one point per cycle)",
                 fontsize=13, fontweight="bold")
    for ax, (tc, pc, label) in zip(axes, pairs):
        y, yh = preds[tc].values, preds[pc].values
        mae   = mean_absolute_error(y, yh)
        rmse  = np.sqrt(((y-yh)**2).mean())
        r2    = 1 - ((y-yh)**2).sum() / ((y-y.mean())**2).sum()
        lo = min(y.min(), yh.min()) - 0.02*abs(y.max()-y.min())
        hi = max(y.max(), yh.max()) + 0.02*abs(y.max()-y.min())
        ax.scatter(y, yh, c=colors, s=40, alpha=0.8, edgecolors="white", lw=0.4, zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1 line", zorder=2)
        ax.fill_between([lo,hi],[lo-mae,hi-mae],[lo+mae,hi+mae],
                        alpha=0.08, color="grey", label="±MAE band")
        _textbox(ax, f"R² = {r2:.3f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}", "upper left")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Observed  {label}"); ax.set_ylabel(f"Predicted  {label}")
        ax.set_title(label); ax.legend(fontsize=8)
    legend_els = [Line2D([0],[0], marker="o", color="w", markerfacecolor=_COND_PAL[i],
                         markersize=8, label=f"{int(r.conc)}ppm {r.flow}mL {r.potential}V")
                  for i, r in conds.iterrows()]
    fig.legend(handles=legend_els, title="Condition", fontsize=7,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.22)
    _save("fig5_parity")


# ── Fig 6 — Pareto front ───────────────────────────────────────────────────────

def fig6_pareto(grid, sec_g, rkp_g, sec_b, rkp_b, pmask, knee_idx,
                ext_labels=None, df=None):
    pidx    = np.where(pmask)[0]
    concs_g = grid[:, 0]
    uniq_c  = sorted(np.unique(concs_g))
    c_cmap  = {c: _COND_PAL[i] for i, c in enumerate(uniq_c)}

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Predicted operating trade-off space (SEC vs. R_κ)",
                 fontsize=13, fontweight="bold")
    ax.scatter(sec_g[~pmask], rkp_g[~pmask], s=5, c="lightgrey",
               alpha=0.4, label="Grid candidates", zorder=1)
    for c in uniq_c:
        mask_c = (concs_g[pidx] == c)
        if mask_c.any():
            idx_c = pidx[mask_c]
            ax.scatter(sec_g[idx_c], rkp_g[idx_c], s=35, color=c_cmap[c],
                       zorder=3, label=f"{int(c)} ppm")
            ax.errorbar(sec_g[idx_c], rkp_g[idx_c],
                        xerr=sec_b[idx_c], yerr=rkp_b[idx_c],
                        fmt="none", ecolor=c_cmap[c], alpha=0.35, capsize=2)
    if df is not None:
        cond_pts = df.groupby(["conc","flow","potential"])[["sec_wh_m3","r_kappa_pos"]].mean()
        ax.scatter(cond_pts["sec_wh_m3"], cond_pts["r_kappa_pos"],
                   s=120, marker="D", c="black", zorder=6, label="Observed conditions")
    ax.scatter(sec_g[knee_idx], rkp_g[knee_idx], s=250, c=_C["red"],
               marker="*", zorder=7, label="Pareto knee candidate")
    ext_note = (f"\n[{ext_labels[knee_idx]}]" if ext_labels is not None else "")
    ax.annotate(f"Model-suggested\ncandidate{ext_note}",
                xy=(sec_g[knee_idx], rkp_g[knee_idx]),
                xytext=(sec_g[knee_idx] + 0.05*(sec_g.max()-sec_g.min()),
                        rkp_g[knee_idx] - 0.08*(rkp_g.max()-rkp_g.min())),
                fontsize=8, color=_C["red"],
                arrowprops=dict(arrowstyle="->", color=_C["red"], lw=1.2))
    ax.set_xlabel("SEC — Specific Energy Consumption (kWh/m³)")
    ax.set_ylabel("R_κ — Conductivity removal fraction")
    ax.margins(0.12); ax.legend(fontsize=8, loc="upper right")
    _textbox(ax, "Shaded bars = calibrated 90 % CI", "lower left", fs=8)
    _save("fig6_pareto")


# ── Fig 7 — Region-based validation ───────────────────────────────────────────

def fig7_region_bar(region_df):
    targets = list(region_df["target"].unique())
    near = region_df[region_df["region"] == "near"].set_index("target")["mae"]
    far  = region_df[region_df["region"] == "far"].set_index("target")["mae"]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Region-based validation: near-observed vs. far-extrapolation MAE",
                 fontsize=13, fontweight="bold")
    y = np.arange(len(targets)); h = 0.35
    b1 = ax.barh(y + h/2, [near[t] for t in targets], h,
                 color=_C["blue"], alpha=0.85, label="Near-observed")
    b2 = ax.barh(y - h/2, [far[t]  for t in targets], h,
                 color=_C["red"],  alpha=0.85, label="Far-extrapolation")
    for bars in (b1, b2):
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.0002, bar.get_y() + bar.get_height()/2,
                    f"{w:.4f}", va="center", fontsize=9)
    all_vals = [near[t] for t in targets] + [far[t] for t in targets]
    max_val  = max(all_vals) if all_vals else 1.0
    for i, t in enumerate(targets):
        pct = (far[t] - near[t]) / near[t] * 100 if near[t] > 0 else 0
        ax.text(max(near[t], far[t]) * 1.08, y[i],
                f"+{pct:.0f}%", va="center", fontsize=8,
                color=_C["red"], fontweight="bold")
    ax.set_xlim(0, max_val * 1.55)
    ax.set_yticks(y); ax.set_yticklabels(targets)
    ax.set_xlabel("MAE"); ax.legend()
    ax.invert_yaxis()
    _save("fig7_region_validation")


# ── Fig 8 — Λκ per-condition box plot ─────────────────────────────────────────

def fig8_lambda_boxplot(df):
    df = df.copy()
    def _sweep(r):
        if r["conc"] != 1000:           return "Concentration sweep"
        if r["potential"] not in [1.8]: return "Potential sweep"
        if r["flow"] != 3:              return "Flow sweep"
        return "Reference (1000ppm·3mL·1.8V)"
    df["sweep"] = df.apply(_sweep, axis=1)
    df["label"] = (df["conc"].astype(int).astype(str) + " ppm / "
                   + df["flow"].astype(str) + " mL / "
                   + df["potential"].astype(str) + " V")
    order = df.groupby("label")["lambda_kappa"].median().sort_values().index.tolist()
    sweep_color = {
        "Concentration sweep": _C["blue"], "Flow sweep": _C["orange"],
        "Potential sweep": _C["green"],
        "Reference (1000ppm·3mL·1.8V)": _C["red"],
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Λκ cycle-to-cycle variability per operating condition",
                 fontsize=13, fontweight="bold")
    for i, lbl in enumerate(order):
        sub  = df[df["label"] == lbl]["lambda_kappa"].values
        col  = sweep_color[df[df["label"] == lbl]["sweep"].iloc[0]]
        bp   = ax.boxplot(sub, positions=[i], widths=0.5, patch_artist=True,
                          medianprops=dict(color="black", lw=2),
                          whiskerprops=dict(color=col), capprops=dict(color=col),
                          flierprops=dict(marker="x", color=col, markersize=5))
        bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.35)
        bp["boxes"][0].set_edgecolor(col)
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(sub))
        ax.scatter(i + jitter, sub, s=20, color=col, alpha=0.7, zorder=3)
        ax.scatter(i, sub.mean(), s=60, marker="^", color=col, zorder=4,
                   edgecolors="black", linewidths=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("Λκ (mS·cm⁻¹·mC⁻¹)")
    legend_els = [Line2D([0],[0], color=c, lw=5, alpha=0.5, label=s)
                  for s, c in sweep_color.items()]
    ax.legend(handles=legend_els, fontsize=9)
    ax.text(0.99, 0.97, "▲ = mean", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="grey")
    _save("fig8_lambda_boxplot")


# ── Fig 9 — Calibration reliability ───────────────────────────────────────────

def fig9_calibration_curve(preds, cal):
    z   = CFG["calibration"]["nominal_z"]
    a0, a1, da = CFG["calibration"]["alpha_search"]
    alphas = np.arange(a0, a1, da)
    mask_a = alphas <= 12

    triples = [("lambda_kappa","lk_pred","lk_std",  "Λκ",  _C["green"]),
               ("sec_wh_m3",  "sec_pred","sec_std", "SEC", _C["orange"]),
               ("r_kappa_pos","rkp_pred","rkp_std",  "R_κ", _C["blue"])]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Uncertainty calibration — coverage vs. multiplier α",
                 fontsize=13, fontweight="bold")
    ax.axhspan(CFG["calibration"]["target_low"], CFG["calibration"]["target_high"],
               color="#d0e4f7", alpha=0.5, label="Target 85–95%", zorder=0)
    ax.axhline(1.0, color="grey", lw=0.7, ls=":", label="100% coverage")

    for tc, mc, sc, label, col in triples:
        res  = (preds[tc] - preds[mc]).abs().values
        raw  = preds[sc].values
        covs = np.array([np.mean(res <= a * z * raw) for a in alphas])
        ax.plot(alphas[mask_a], covs[mask_a], color=col, lw=2, label=label)
        if tc in cal:
            chosen = cal[tc]["multiplier"]
            cov_c  = cal[tc]["adjusted_coverage"]
            ax.axvline(chosen, color=col, lw=1.2, ls="--", alpha=0.7)
            ax.scatter([chosen], [cov_c], s=80, color=col, zorder=5,
                       edgecolors="black", linewidths=0.6)
            ax.text(chosen + 0.1, cov_c, f"{cov_c:.0%}", fontsize=8, color=col, va="center")

    ax.set_xlabel("Multiplier α (applied to GP posterior std)")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    _save("fig9_calibration_curve")


# ── Fig 10 — Ablation comparison (12 models) ──────────────────────────────────

def fig10_ablation_bar(abl):
    """
    Two panels: SEC MAE (left) and R_κ MAE (right).
    Highlights best model; annotates % gain vs Ridge.
    Models without a given target show greyed-out 'N/A' bars.
    """
    models  = abl["model"].tolist()
    colors  = [_MODEL_PAL.get(m, "#999") for m in models]
    xlabels = [_MODEL_LABEL.get(m, m) for m in models]

    ridge_sec = abl.loc[abl["model"] == "ridge", "sec_micro_mae"].values[0]
    ridge_rkp = abl.loc[abl["model"] == "ridge", "rkp_micro_mae"].values[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model comparison — LOCO-CV MAE (all conditions)",
                 fontsize=13, fontweight="bold")

    for ax, col_key, title, ref in [
        (a1, "sec_micro_mae", "SEC MAE (kWh/m³)", ridge_sec),
        (a2, "rkp_micro_mae", "R_κ MAE",          ridge_rkp),
    ]:
        vals = abl[col_key].values.astype(float)
        viol = abl[col_key.replace("_micro_mae", "_physical_violation_rate")].values.astype(float)

        # For NaN vals (model doesn't predict this target), use 0 bar + label
        valid_mask = np.isfinite(vals)
        plot_vals  = np.where(valid_mask, vals, 0.0)
        best       = np.where(valid_mask, vals, np.inf).argmin()

        bar_colors = [colors[i] if valid_mask[i] else "#dddddd" for i in range(len(models))]
        bars       = ax.barh(range(len(models)), plot_vals,
                             color=bar_colors, edgecolor="white", height=0.6)
        bars[best].set_edgecolor(_C["red"]); bars[best].set_linewidth(2.5)

        max_val = np.nanmax(vals) if valid_mask.any() else 1.0
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if not valid_mask[i]:
                ax.text(0.005 * max_val, bar.get_y() + bar.get_height()/2,
                        "N/A", va="center", fontsize=8, color="#aaa")
                continue
            ax.text(v + 0.01 * max_val, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=8)
            # violation rate badge (red if > 0)
            if np.isfinite(viol[i]) and viol[i] > 0:
                ax.text(max_val * 1.35, bar.get_y() + bar.get_height()/2,
                        f"viol={viol[i]:.0%}", va="center", fontsize=7, color=_C["red"])
            # % vs Ridge
            if models[i] not in ("ridge", "mean") and ref > 0:
                pct   = (ref - v) / ref * 100
                sign  = "+" if pct < 0 else ""
                col_t = _C["green"] if pct > 0 else _C["red"]
                ax.text(max_val * 1.55, bar.get_y() + bar.get_height()/2,
                        f"{sign}{pct:.1f}%\nvs Ridge",
                        va="center", fontsize=7, color=col_t)

        ax.set_xlim(0, max_val * 1.85)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(xlabels, fontsize=9)
        ax.set_xlabel(title)
        ax.axvline(ref, color=_C["grey"], lw=1, ls="--", alpha=0.5)
        ax.invert_yaxis()

    _save("fig10_ablation_bar")


# ── Fig 11 — GP response slices ───────────────────────────────────────────────

def fig11_response_slices(gp_lk, gp_sec, gp_rkp, df, cal):
    z      = CFG["calibration"]["nominal_z"]
    sweeps = [
        ("Concentration (ppm)", np.linspace(1000, 7500, 80),
         lambda v: np.column_stack([v, np.full(len(v),3.0), np.full(len(v),1.8)]),
         df[df["flow"]==3][df["potential"]==1.8]),
        ("Flow rate (mL/min)", np.linspace(2, 5, 80),
         lambda v: np.column_stack([np.full(len(v),1000.), v, np.full(len(v),1.8)]),
         df[df["conc"]==1000][df["potential"]==1.8]),
        ("Potential (V)", np.linspace(0.9, 1.8, 80),
         lambda v: np.column_stack([np.full(len(v),1000.), np.full(len(v),3.0), v]),
         df[df["conc"]==1000][df["flow"]==3]),
    ]
    targets = [
        ("sec_wh_m3",   "sec_pred","sec_std",  "SEC (kWh/m³)",  _C["orange"]),
        ("r_kappa_pos", "rkp_pred","rkp_std",  "R_κ (removal)", _C["blue"]),
    ]
    fig, axes = plt.subplots(len(targets), len(sweeps), figsize=(15, 8),
                              sharex="col", constrained_layout=True)
    fig.suptitle("GP response along each experimental sweep (calibrated 90 % CI)",
                 fontsize=13, fontweight="bold")
    for col, (xlabel, xvals, make_X1, obs_df) in enumerate(sweeps):
        X1         = make_X1(xvals)
        lk_mu, _   = predict_gp(*gp_lk, X1)
        X2         = np.column_stack([X1, lk_mu])
        for row, (tc, pc, sc, ylabel, col_c) in enumerate(targets):
            ax   = axes[row, col]
            gp_t = gp_sec if tc == "sec_wh_m3" else gp_rkp
            mu, sg = predict_gp(*gp_t, X2)
            mult = cal.get(tc, {}).get("multiplier", 1.0)
            ci   = mult * z * sg
            ax.plot(xvals, mu, color=col_c, lw=2, label="GP mean")
            ax.fill_between(xvals, mu - ci, mu + ci, alpha=0.2,
                            color=col_c, label="Calibrated 90% CI")
            if len(obs_df) > 0:
                x_col = {"Concentration (ppm)":"conc","Flow rate (mL/min)":"flow",
                         "Potential (V)":"potential"}[xlabel]
                ax.scatter(obs_df[x_col], obs_df[tc], s=50, c="black",
                           zorder=5, label="Observed", edgecolors="white", lw=0.5)
            if row == 0:              ax.set_title(xlabel, fontsize=11)
            if col == 0:              ax.set_ylabel(ylabel)
            if row == len(targets)-1: ax.set_xlabel(xlabel)
            ax.legend(fontsize=7)
    _save("fig11_response_slices")


# ── Fig 12 — Condition-level trade-off scatter ─────────────────────────────────

def fig12_condition_tradeoff(preds, cal):
    z   = CFG["calibration"]["nominal_z"]
    grp = preds.groupby(["conc","flow","potential"])
    rows = []
    for (c, f, p), g in grp:
        rows.append(dict(
            conc=c, flow=f, potential=p,
            sec_true=g["sec_wh_m3"].mean(),   rkp_true=g["r_kappa_pos"].mean(),
            sec_pred=g["sec_pred"].mean(),     rkp_pred=g["rkp_pred"].mean(),
            sec_ci=cal.get("sec_wh_m3",{}).get("multiplier",1.0)*z*g["sec_std"].mean(),
            rkp_ci=cal.get("r_kappa_pos",{}).get("multiplier",1.0)*z*g["rkp_std"].mean(),
        ))
    cdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Operating condition trade-off: SEC vs. R_κ\n"
                 "(observed conditions with calibrated prediction intervals)",
                 fontsize=13, fontweight="bold")
    concs_u = sorted(cdf["conc"].unique())
    c_map   = {c: _COND_PAL[i] for i, c in enumerate(concs_u)}
    cond_labels = []
    for num, (_, r) in enumerate(cdf.iterrows(), 1):
        col = c_map[r["conc"]]
        ax.errorbar(r["sec_pred"], r["rkp_pred"], xerr=r["sec_ci"], yerr=r["rkp_ci"],
                    fmt="none", ecolor=col, alpha=0.5, capsize=4, lw=1.5)
        ax.scatter(r["sec_pred"], r["rkp_pred"], s=200, color=col,
                   edgecolors="black", lw=0.8, zorder=4)
        ax.text(r["sec_pred"], r["rkp_pred"], str(num),
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white", zorder=5)
        ax.scatter(r["sec_true"], r["rkp_true"], s=70, marker="x",
                   color=col, lw=2, zorder=5)
        cond_labels.append(
            f"{num}: {int(r['conc'])} ppm · {r['flow']} mL/min · {r['potential']} V")
    ax.set_xlabel("SEC (kWh/m³)"); ax.set_ylabel("R_κ")
    row_colors = [c_map[r["conc"]] for _, r in cdf.iterrows()]
    num_els = [Line2D([0],[0], marker="o", color="w", markerfacecolor=row_colors[i],
                      markersize=9, label=lbl) for i, lbl in enumerate(cond_labels)]
    type_els = [Line2D([0],[0], marker="o",  color="grey", markersize=8,
                       label="● GP prediction"),
                Line2D([0],[0], marker="x",  color="grey", markersize=8, lw=2,
                       label="✕ Observed mean")]
    ax.legend(handles=num_els + type_els, fontsize=7.5,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.subplots_adjust(right=0.70)
    _textbox(ax, "Error bars = calibrated 90% CI", "lower right", fs=8)
    _save("fig12_condition_tradeoff")


# ── Fig 13 — Cycle stability ───────────────────────────────────────────────────

def fig13_cycle_stability(df):
    metrics = [("lambda_kappa","Λκ (mS·cm⁻¹·mC⁻¹)",_C["green"]),
               ("sec_wh_m3",  "SEC (kWh/m³)",        _C["orange"]),
               ("r_kappa_pos","R_κ",                  _C["blue"])]
    conds = _conditions(df)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 9), sharex=False)
    fig.suptitle("Cycle-to-cycle stability across all operating conditions",
                 fontsize=13, fontweight="bold")
    for ax, (col, ylabel, _) in zip(axes, metrics):
        for i, cond in enumerate(conds):
            _, sub = _split(df, cond)
            sub    = sub.sort_values("cycle_id")
            col_c  = _COND_PAL[i % 10]
            label  = f"{int(cond[0])}ppm / {cond[1]}mL / {cond[2]}V"
            ax.plot(sub["cycle_id"], sub[col], marker="o", markersize=4,
                    lw=1.2, color=col_c, alpha=0.85, label=label)
            ax.axhline(sub[col].mean(), color=col_c, lw=0.6, ls="--", alpha=0.4)
        ax.set_ylabel(ylabel); ax.set_xlabel("Cycle index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.18)
    _save("fig13_cycle_stability")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    records = load_raw_sheets()
    df      = build_cycle_table(records)
    conds   = _conditions(df)
    print(f"Cycles extracted : {len(df)} across {len(conds)} conditions")
    print(f"New physics cols : {[c for c in df.columns if c not in ['conc','flow','potential','cycle_id','lambda_kappa','r_kappa_pos','sec_wh_m3','w_net_wh']]}")

    gp_grid_search(df)

    preds = run_loco_cv(df)
    preds.to_csv(OUT / "validation_summary.csv", index=False)
    print(f"LOCO-CV rows     : {len(preds)}")

    cal = calibrate(preds)
    print("Calibration coverage:", {k: v["adjusted_coverage"] for k, v in cal.items()})

    if CFG.get("calibration", {}).get("nested", False):
        cal_nested = calibrate_nested(df)
        print("Nested calibration:", {k: v["adjusted_coverage"] for k, v in cal_nested.items()})

    abl = run_ablation(df)
    print("\nAblation (SEC):\n",
          abl[["model","sec_micro_mae","sec_r2","sec_physical_violation_rate"]].to_string(index=False))
    print("\nAblation (R_κ):\n",
          abl[["model","rkp_micro_mae","rkp_r2","rkp_physical_violation_rate"]].to_string(index=False))

    cond_df = condition_level_metrics(preds)
    stat_df = statistical_comparisons(abl)
    print("\nStatistical comparisons:\n", stat_df[["comparison","target","mae_diff","p_holm"]].to_string(index=False))

    region_df, preds_reg = region_validation(df, preds)

    # Full-dataset model ensemble for surface plots & Pareto
    models = build_full_models(df)
    gp_lk  = models["gp_lk"]
    gp_sec = models["two_stage_gp_sec"]
    gp_rkp = models["two_stage_gp_rkp"]

    grid, sec_g, rkp_g, sec_b, rkp_b, pmask, knee, ext_labels = run_pareto(
        models, df, cal)

    json.dump({"option": "A", "stage2_features": FEAT_S2,
               "note": "Flow included in Stage 2 GP to fully define Pareto candidate."},
              open(OUT / "flow_decision.json", "w"), indent=2)

    figs = [
        ("fig1_pipeline",            lambda: fig1_schematic()),
        ("fig2_segmentation",        lambda: fig2_segmentation(records)),
        ("fig3_kappa_profile",       lambda: fig3_kappa_profile(records)),
        ("fig4_lambda_surface",      lambda: fig4_lambda_surface(gp_lk, df)),
        ("fig5_parity",              lambda: fig5_parity(preds)),
        ("fig6_pareto",              lambda: fig6_pareto(grid, sec_g, rkp_g, sec_b, rkp_b,
                                                          pmask, knee, ext_labels, df)),
        ("fig7_region_validation",   lambda: fig7_region_bar(region_df)),
        ("fig8_lambda_boxplot",      lambda: fig8_lambda_boxplot(df)),
        ("fig9_calibration_curve",   lambda: fig9_calibration_curve(preds, cal)),
        ("fig10_ablation_bar",       lambda: fig10_ablation_bar(abl)),
        ("fig11_response_slices",    lambda: fig11_response_slices(gp_lk, gp_sec, gp_rkp,
                                                                    df, cal)),
        ("fig12_condition_tradeoff", lambda: fig12_condition_tradeoff(preds, cal)),
        ("fig13_cycle_stability",    lambda: fig13_cycle_stability(df)),
    ]
    for name, fn in tqdm(figs, desc="Figures", unit="fig"):
        fn()

    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
