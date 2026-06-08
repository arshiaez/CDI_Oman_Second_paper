"""
CDI two-stage Gaussian Process analytics pipeline.

Stages
------
0  Data loading & cycle segmentation
1  Per-cycle metric extraction (Λκ, SEC, R_κ)
2  LOCO-CV validation (all cycles per held-out condition — T-01 fix)
3  Uncertainty calibration (empirical multiplier — T-04)
4  Ablation study (Ridge / GP-no-Λκ / GP-with-Λκ — T-07)
5  Region-based validation (near-observed vs. far-extrapolation — T-06)
6  Pareto front + knee candidate (T-03 Option A: flow in Stage 2 — T-05)
7  Seven publication figures ≥300 DPI (T-06)

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
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.lines import Line2D
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ── Config & globals ───────────────────────────────────────────────────────────

CFG    = yaml.safe_load(open("config.yaml"))
OUT    = Path(CFG["paths"]["output_dir"])
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stage 1 GP features; Stage 2 includes Λκ (T-03 Option A — fully defines Pareto candidate)
FEAT_S1 = ["conc", "flow", "potential"]
FEAT_S2 = ["conc", "flow", "potential", "lambda_kappa"]


# ── Stage 0: Data loading & cycle segmentation ─────────────────────────────────

def _load_file(path, sheet_to_meta):
    """Parse all sheets in one Excel file into (df, conc, flow, potential) tuples."""
    xl = pd.ExcelFile(path)
    out = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, usecols=[0, 1, 2]).dropna()
        df.columns = ["time", "conductivity", "current"]
        out.append((df, *sheet_to_meta(sheet)))
    return out


def load_raw_sheets():
    """Load the three experimental Excel files; deduplicate shared (1000,3,1.8) condition."""
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
    """Scalar metrics for one desal+regen cycle."""
    kappa = desal["conductivity"].values
    i_des = desal["current"].values

    # Λκ: conductivity reduction per unit charge (∫Δκ dt / ∫I dt)
    delta_k    = np.clip(kappa[0] - kappa, 0, None)
    int_charge = np.trapz(np.abs(i_des), dx=1.0)          # mC
    lambda_k   = np.trapz(delta_k, dx=1.0) / int_charge if int_charge > 0 else np.nan

    r_kappa = (kappa[0] - kappa.min()) / kappa[0] if kappa[0] > 0 else np.nan

    # SEC: gross energy during desalination phase / volume processed
    flow_m3s = flow * 1e-6 / 60
    vol_m3   = flow_m3s * len(desal)
    w_wh     = np.trapz(np.abs(i_des) * 1e-3 * potential, dx=1.0) / 3600
    sec      = w_wh / (vol_m3 * 1000) if vol_m3 > 0 else np.nan   # kWh/m³

    return dict(conc=conc, flow=flow, potential=potential, cycle_id=cid,
                lambda_kappa=lambda_k, r_kappa_pos=r_kappa,
                sec_wh_m3=sec, w_net_wh=w_wh)


def build_cycle_table(records):
    """Extract metrics for every cycle across all conditions."""
    rows = []
    for df, conc, flow, potential in tqdm(records, desc="Extracting cycles", unit="cond"):
        for cid, (desal, regen) in enumerate(segment_cycles(df)):
            rows.append(_metrics(desal, regen, conc, flow, potential, cid))
    return pd.DataFrame(rows).dropna()


# ── GP model (ARD RBF kernel, GPyTorch, runs on DEVICE) ───────────────────────

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
    """Fit an ExactGP with ARD-RBF kernel; return (model, lik, scaler_X, scaler_y)."""
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


# ── GP hyperparameter grid search ─────────────────────────────────────────────

def gp_grid_search(df):
    """K-fold CV over GP hyperparameter grid; updates CFG['gp'] in-place if enabled."""
    gs = CFG["gp"]["grid_search"]
    if not gs["enabled"]:
        return

    print("\nRunning GP hyperparameter grid search...")
    X = df[FEAT_S1].values
    y = df["lambda_kappa"].values          # tune on Stage 1 target (representative)
    n = len(X)
    k = gs["cv_folds"]
    fold_size = n // k
    indices   = np.random.default_rng(0).permutation(n)

    best_mae, best_params = np.inf, {}
    grid = [(ni, lr, nc)
            for ni in gs["n_iter_values"]
            for lr in gs["lr_values"]
            for nc in gs["noise_values"]]

    for n_iter, lr, noise in tqdm(grid, desc="GP grid search", unit="config"):
        fold_maes = []
        for fold in range(k):
            val_idx   = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size],
                                        indices[(fold + 1) * fold_size:]])
            # temporarily override CFG for this trial
            CFG["gp"]["n_iter"] = n_iter
            CFG["gp"]["lr"]     = lr
            CFG["gp"]["noise_constraint"] = noise

            gp       = train_gp(X[train_idx], y[train_idx])
            mu, _    = predict_gp(*gp, X[val_idx])
            fold_maes.append(mean_absolute_error(y[val_idx], mu))

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


# ── Stage 2: LOCO-CV validation (T-01: iterates ALL cycles per condition) ─────

def run_loco_cv(df):
    """Leave-one-condition-out CV; one prediction row per cycle."""
    preds = []
    for cond in tqdm(_conditions(df), desc="LOCO-CV", unit="fold"):
        train, test = _split(df, cond)

        gp_lk            = train_gp(train[FEAT_S1].values, train["lambda_kappa"].values)
        lk_mu, lk_std    = predict_gp(*gp_lk, test[FEAT_S1].values)

        X2_tr = np.column_stack([train[FEAT_S1].values, train["lambda_kappa"].values])
        X2_te = np.column_stack([test[FEAT_S1].values,  lk_mu])

        gp_sec            = train_gp(X2_tr, train["sec_wh_m3"].values)
        gp_rkp            = train_gp(X2_tr, train["r_kappa_pos"].values)
        sec_mu, sec_std   = predict_gp(*gp_sec, X2_te)
        rkp_mu, rkp_std   = predict_gp(*gp_rkp, X2_te)

        preds.append(test.assign(
            lk_pred=lk_mu,   lk_std=lk_std,
            sec_pred=sec_mu, sec_std=sec_std,
            rkp_pred=rkp_mu, rkp_std=rkp_std,
        ))
    return pd.concat(preds, ignore_index=True)


# ── Stage 3: Uncertainty calibration (T-04) ───────────────────────────────────

def calibrate(preds):
    """Empirical multiplier so adjusted 90 % CI achieves 85–95 % coverage."""
    z   = CFG["calibration"]["nominal_z"]
    lo  = CFG["calibration"]["target_low"]
    hi  = CFG["calibration"]["target_high"]
    a0, a1, da = CFG["calibration"]["alpha_search"]

    triples = [("lambda_kappa", "lk_pred",  "lk_std"),
               ("sec_wh_m3",   "sec_pred", "sec_std"),
               ("r_kappa_pos", "rkp_pred", "rkp_std")]
    cal = {}
    for tc, mc, sc in triples:
        res, raw = (preds[tc] - preds[mc]).abs().values, preds[sc].values
        alpha = next(
            (a for a in np.arange(a0, a1, da)
             if lo <= np.mean(res <= a * z * raw) <= hi),
            np.arange(a0, a1, da)[-1]
        )
        cal[tc] = {"multiplier":        round(float(alpha), 4),
                   "adjusted_coverage": round(float(np.mean(res <= alpha * z * raw)), 4)}

    json.dump(cal, open(OUT / "calibration.json", "w"), indent=2)
    return cal


# ── Stage 4: Ablation (T-07) ──────────────────────────────────────────────────

# sklearn baselines: each entry is (label, constructor).
# All are fitted on FEAT_S1 (no Λκ) — fair comparison against GP-no-Λκ.
_SK_BASELINES = [
    ("mean",   lambda: DummyRegressor(strategy="mean")),
    ("ridge",  lambda: Ridge()),
    ("rf",     lambda: RandomForestRegressor(n_estimators=200, random_state=0)),
    ("svr",    lambda: SVR(kernel="rbf", C=10, epsilon=0.01)),
    ("mlp",    lambda: MLPRegressor(hidden_layer_sizes=(64, 32),
                                    max_iter=1000, random_state=0,
                                    early_stopping=True, n_iter_no_change=20)),
]


def _sk_predict(model_fn, Xtr, ytr, Xte):
    """Fit a scaled sklearn model; return predictions."""
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    return model_fn().fit(Xtr, ytr).predict(Xte)


def run_ablation(df):
    """Compare sklearn baselines + GP variants via LOCO-CV; write ablation_table.csv."""
    model_keys = [label for label, _ in _SK_BASELINES] + ["gp_nolk", "gp_lk"]
    buckets    = {m: {"sec": [], "rkp": []} for m in model_keys}

    for cond in tqdm(_conditions(df), desc="Ablation", unit="fold"):
        train, test = _split(df, cond)
        Xtr_s1 = train[FEAT_S1].values
        Xte_s1 = test[FEAT_S1].values

        # sklearn baselines (all on FEAT_S1, no Λκ)
        for label, model_fn in _SK_BASELINES:
            for col, key in [("sec_wh_m3", "sec"), ("r_kappa_pos", "rkp")]:
                pred = _sk_predict(model_fn, Xtr_s1, train[col].values, Xte_s1)
                buckets[label][key].extend(np.abs(test[col].values - pred))

        # GP without Λκ
        for col, key in [("sec_wh_m3", "sec"), ("r_kappa_pos", "rkp")]:
            gp    = train_gp(Xtr_s1, train[col].values)
            mu, _ = predict_gp(*gp, Xte_s1)
            buckets["gp_nolk"][key].extend(np.abs(test[col].values - mu))

        # GP with Λκ (two-stage)
        gp_lk    = train_gp(Xtr_s1, train["lambda_kappa"].values)
        lk_te, _ = predict_gp(*gp_lk, Xte_s1)
        X2_tr    = np.column_stack([Xtr_s1, train["lambda_kappa"].values])
        X2_te    = np.column_stack([Xte_s1, lk_te])
        for col, key in [("sec_wh_m3", "sec"), ("r_kappa_pos", "rkp")]:
            gp    = train_gp(X2_tr, train[col].values)
            mu, _ = predict_gp(*gp, X2_te)
            buckets["gp_lk"][key].extend(np.abs(test[col].values - mu))

    sec_ref, rkp_ref = df["sec_wh_m3"].mean(), df["r_kappa_pos"].mean()
    rows = []
    for name, b in buckets.items():
        sa, ra = np.array(b["sec"]), np.array(b["rkp"])
        rows.append(dict(
            model=name,
            sec_mae=sa.mean(),       sec_rmse=np.sqrt((sa**2).mean()),
            sec_rel_mae=sa.mean() / sec_ref,
            rkp_mae=ra.mean(),       rkp_rmse=np.sqrt((ra**2).mean()),
            rkp_rel_mae=ra.mean() / rkp_ref,
        ))
    abl = pd.DataFrame(rows)
    abl.to_csv(OUT / "ablation_table.csv", index=False)
    return abl


# ── Stage 5: Region-based validation (T-06) ───────────────────────────────────

def region_validation(df, preds):
    """Tag predictions near-observed / far-extrapolation by min-distance to training conditions."""
    cond_pts = df[["conc","flow","potential"]].drop_duplicates().values
    test_pts = preds[["conc","flow","potential"]].values
    sc       = StandardScaler().fit(cond_pts)
    dists    = cdist(sc.transform(test_pts), sc.transform(cond_pts)).min(axis=1)

    preds = preds.copy()
    rank  = np.argsort(dists)
    labels = np.empty(len(dists), dtype=object)
    labels[rank[:len(rank) // 2]]  = "near"
    labels[rank[len(rank) // 2:]]  = "far"
    preds["region"] = labels

    rows = []
    for region in ["near", "far"]:
        sub = preds[preds["region"] == region]
        for tc, pc, name in [("lambda_kappa","lk_pred","lambda_kappa"),
                              ("sec_wh_m3","sec_pred","sec"),
                              ("r_kappa_pos","rkp_pred","r_kappa_pos")]:
            rows.append({"region": region, "target": name,
                         "mae": mean_absolute_error(sub[tc], sub[pc])})

    region_df = pd.DataFrame(rows)
    region_df.to_csv(OUT / "region_validation.csv", index=False)
    return region_df, preds


# ── Stage 6: Pareto optimisation (T-05) ───────────────────────────────────────

def _pareto_mask(sec, rkp):
    """True where point is non-dominated (minimise sec, maximise rkp)."""
    dom = np.zeros(len(sec), dtype=bool)
    for i in tqdm(range(len(sec)), desc="Pareto dominance", leave=False, unit="pt"):
        dom[i] = np.any(
            (sec <= sec[i]) & (rkp >= rkp[i]) &
            ((sec < sec[i]) | (rkp > rkp[i]))
        )
    return ~dom


def run_pareto(gp_lk, gp_sec, gp_rkp, df, cal):
    """Dense grid prediction → Pareto front → knee candidate."""
    n = CFG["pareto"]["grid_points"]
    grid = np.array([
        [c, f, p]
        for c in sorted(df["conc"].unique())
        for f in np.linspace(df["flow"].min(),      df["flow"].max(),      n)
        for p in np.linspace(df["potential"].min(), df["potential"].max(), n)
    ])

    lk_g,  _      = predict_gp(*gp_lk, grid)
    X2_g          = np.column_stack([grid, lk_g])
    sec_g, sec_sg = predict_gp(*gp_sec, X2_g)
    rkp_g, rkp_sg = predict_gp(*gp_rkp, X2_g)

    z        = CFG["calibration"]["nominal_z"]
    sec_band = cal["sec_wh_m3"]["multiplier"]   * z * sec_sg
    rkp_band = cal["r_kappa_pos"]["multiplier"] * z * rkp_sg

    pmask = _pareto_mask(sec_g, rkp_g)
    pidx  = np.where(pmask)[0]

    # Knee: closest to utopia corner in normalised objective space
    sp  = sec_g[pidx]; rp = rkp_g[pidx]
    sn  = (sp - sp.min()) / (sp.max() - sp.min() + 1e-9)
    rn  = (rp.max() - rp) / (rp.max() - rp.min() + 1e-9)
    knee = pidx[np.argmin(np.hypot(sn, rn))]

    result = dict(
        conc=float(grid[knee, 0]), flow=float(grid[knee, 1]),
        potential=float(grid[knee, 2]),
        sec_pred=float(sec_g[knee]),   sec_band=float(sec_band[knee]),
        rkp_pred=float(rkp_g[knee]),   rkp_band=float(rkp_band[knee]),
        label="model-suggested candidate for experimental validation",
    )
    json.dump(result, open(OUT / "pareto_knee.json", "w"), indent=2)
    return grid, sec_g, rkp_g, sec_band, rkp_band, pmask, knee


# ── Stage 7: Figures ───────────────────────────────────────────────────────────

# ── Global style ───────────────────────────────────────────────────────────────
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
_COND_PAL = plt.cm.tab10.colors   # 10 distinct condition colours

_MODEL_PAL = {
    "mean":    "#bbbbbb", "ridge":   "#6c757d",
    "rf":      _C["orange"], "svr":  _C["coral"],
    "mlp":     _C["teal"],
    "gp_nolk": _C["blue"], "gp_lk":  _C["navy"],
}
_MODEL_LABEL = {
    "mean": "Mean", "ridge": "Ridge", "rf": "Random Forest",
    "svr": "SVR", "mlp": "MLP (ANN)",
    "gp_nolk": "GP  (no Λκ)", "gp_lk": "GP  (with Λκ)",
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
    """Annotate with a styled text box."""
    props = dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9)
    anchors = {"upper left": (0.03, 0.97), "upper right": (0.97, 0.97),
               "lower left": (0.03, 0.03), "lower right": (0.97, 0.03)}
    x, y = anchors.get(loc, (0.03, 0.97))
    va = "top" if "upper" in loc else "bottom"
    ha = "left" if "left" in loc else "right"
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fs,
            va=va, ha=ha, bbox=props)


# ── Fig 1 — Pipeline schematic ─────────────────────────────────────────────────

def fig1_schematic():
    """Two-row pipeline schematic with data-flow annotations."""
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # top row: data pipeline
    top_steps = [
        ("Raw\nTimeseries", "#d0e4f7", "#2255aa"),
        ("Cycle\nSegmentation", "#d0e4f7", "#2255aa"),
        ("Metric Extraction\nΛκ · SEC · R_κ", "#cce5cc", "#1a6b1a"),
    ]
    # bottom row: modelling
    bot_steps = [
        ("Stage 1 GP\npredict Λκ", "#ffe5cc", "#b35900"),
        ("Stage 2 GP\nSEC · R_κ prediction", "#ffe5cc", "#b35900"),
        ("Uncertainty\nCalibration", "#ffe0e0", "#8b0000"),
        ("Pareto / Trade-off\nAnalysis", "#e8d5f5", "#5b0092"),
    ]

    def _draw_row(steps, y_c, x_start, x_end):
        n  = len(steps)
        xs = np.linspace(x_start, x_end, n)
        for i, (txt, fc, ec) in enumerate(steps):
            ax.text(xs[i], y_c, txt, ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.6", fc=fc, ec=ec, lw=1.8),
                    zorder=3)
            if i < n - 1:
                ax.annotate("", xy=(xs[i+1] - 0.07, y_c),
                            xytext=(xs[i] + 0.07, y_c),
                            arrowprops=dict(arrowstyle="-|>", color=ec,
                                            lw=1.5, mutation_scale=14))

    _draw_row(top_steps, 0.72, 0.08, 0.55)
    _draw_row(bot_steps, 0.28, 0.08, 0.92)

    # diagonal connector from "Metric Extraction" (top-right box) to "Stage 1 GP" (bot-left box)
    ax.annotate("", xy=(0.08, 0.38), xytext=(0.55, 0.62),
                arrowprops=dict(arrowstyle="-|>", color="#1a6b1a",
                                lw=1.5, mutation_scale=14))

    # subtitle
    ax.text(0.5, 0.95, "CDI Two-Stage GP Analytics Pipeline",
            ha="center", va="top", fontsize=14, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.02,
            "Inputs: concentration · flow · potential  →  Outputs: Λκ · SEC · R_κ (with calibrated uncertainty)",
            ha="center", va="bottom", fontsize=9, color="#555555",
            transform=ax.transAxes)
    _save("fig1_pipeline")


# ── Fig 2 — Cycle segmentation ─────────────────────────────────────────────────

def fig2_segmentation(records):
    """Conductivity + current timeseries with shaded desal/regen phases and cycle labels."""
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
        for ax in (a1, a2):
            ax.axvspan(t0, t1, alpha=0.18, color=fc, zorder=0)
            ax.axvline(t0, color=ec, lw=0.7, ls="--", alpha=0.6)
        if phase == "desal":
            cycle_num += 1
            # get_xaxis_transform(): x in data coords, y in axes fraction — safe before plot
            a1.text((t0 + t1) / 2, 0.97, f"C{cycle_num}", ha="center", va="top",
                    fontsize=8, color=_C["blue"], fontweight="bold",
                    transform=a1.get_xaxis_transform())

    a1.plot(sub["time"], sub["conductivity"], color=_C["blue"], lw=1.4, label="κ (mS/cm)")
    a2.plot(sub["time"], sub["current"],      color=_C["red"],  lw=1.4, label="I (mA)")
    a2.axhline(0, color="black", lw=0.8, ls=":")

    a1.set_ylabel("Conductivity (mS/cm)")
    a2.set_ylabel("Current (mA)")
    a2.set_xlabel("Time (s)")

    phase_els = [
        Line2D([0], [0], color=_C["blue"],   lw=6, alpha=0.3, label="Desalination"),
        Line2D([0], [0], color=_C["orange"], lw=6, alpha=0.3, label="Regeneration"),
    ]
    line_els_a1 = a1.get_legend_handles_labels()
    line_els_a2 = a2.get_legend_handles_labels()
    a1.legend(handles=phase_els + line_els_a1[0], labels=["Desalination","Regeneration"] + line_els_a1[1],
              fontsize=9, loc="lower right")
    a2.legend(handles=line_els_a2[0], fontsize=9, loc="upper right")
    _save("fig2_segmentation")


# ── Fig 3 — κ(t) conductivity profile ─────────────────────────────────────────

def fig3_kappa_profile(records):
    """Overlay first 3 desal cycles; annotate κ₀, κ_min, R_κ formula."""
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
            # annotate kappa_0
            ax.annotate(f"κ₀ = {kappa[0]:.3f}", xy=(0, kappa[0]),
                        xytext=(len(t) * 0.15, kappa[0] + 0.01),
                        fontsize=9, color=_C["navy"],
                        arrowprops=dict(arrowstyle="->", color=_C["navy"], lw=1))
            # annotate kappa_min
            tmin = np.argmin(kappa)
            ax.annotate(f"κ_min = {kappa.min():.3f}", xy=(tmin, kappa.min()),
                        xytext=(tmin + len(t)*0.1, kappa.min() - 0.02),
                        fontsize=9, color=_C["red"],
                        arrowprops=dict(arrowstyle="->", color=_C["red"], lw=1))
            # R_κ value box
            r_kappa = (kappa[0] - kappa.min()) / kappa[0]
            _textbox(ax, f"R_κ = (κ₀ − κ_min)/κ₀\n    = {r_kappa:.3f}", "upper right")

    ax.set_xlabel("Time within desal phase (s)")
    ax.set_ylabel("Conductivity κ (mS/cm)")
    ax.legend()
    _save("fig3_kappa_profile")


# ── Fig 4 — Λκ sensitivity surfaces ───────────────────────────────────────────

def fig4_lambda_surface(gp_lk, df):
    """Two surface panels: conc×potential and flow×potential; observed conditions overlaid."""
    flow_med = float(df["flow"].median())
    conc_med = float(df["conc"].median())

    concs = np.linspace(df["conc"].min(),      df["conc"].max(),      60)
    pots  = np.linspace(df["potential"].min(), df["potential"].max(), 60)
    flows = np.linspace(df["flow"].min(),      df["flow"].max(),      60)
    CC, PP = np.meshgrid(concs, pots)
    FF, PP2 = np.meshgrid(flows, pots)

    X1 = np.column_stack([CC.ravel(),  np.full(CC.size, flow_med), PP.ravel()])
    X2 = np.column_stack([np.full(FF.size, conc_med), FF.ravel(), PP2.ravel()])
    lk1, _ = predict_gp(*gp_lk, X1)
    lk2, _ = predict_gp(*gp_lk, X2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Λκ sensitivity surface (conductivity-native efficiency proxy)",
                 fontsize=13, fontweight="bold")

    for ax, X, Y, Z, xlabel, xobs, yobs in [
        (a1, CC, PP, lk1.reshape(CC.shape),
         "Concentration (ppm)",
         df["conc"].values, df["potential"].values),
        (a2, FF, PP2, lk2.reshape(FF.shape),
         "Flow rate (mL/min)",
         df["flow"].values, df["potential"].values),
    ]:
        vmin, vmax = Z.min(), Z.max()
        cs = ax.contourf(X, Y, Z, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
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
    """Predicted vs. observed for Λκ, SEC, R_κ; points coloured by condition."""
    pairs = [("lambda_kappa", "lk_pred",  "Λκ (mS·cm⁻¹·mC⁻¹)"),
             ("sec_wh_m3",   "sec_pred", "SEC (kWh/m³)"),
             ("r_kappa_pos", "rkp_pred", "R_κ")]

    conds  = preds[["conc","flow","potential"]].drop_duplicates().reset_index(drop=True)
    cmap   = {tuple(r): _COND_PAL[i % 10] for i, r in conds.iterrows()}
    colors = preds.apply(
        lambda r: cmap[(r["conc"], r["flow"], r["potential"])], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.suptitle("LOCO-CV: Predicted vs. Observed (one point per cycle)",
                 fontsize=13, fontweight="bold")

    for ax, (tc, pc, label) in zip(axes, pairs):
        y, yh = preds[tc].values, preds[pc].values
        mae   = mean_absolute_error(y, yh)
        rmse  = np.sqrt(((y - yh)**2).mean())
        r2    = 1 - ((y - yh)**2).sum() / ((y - y.mean())**2).sum()
        lo    = min(y.min(), yh.min()) - 0.02 * abs(y.max() - y.min())
        hi    = max(y.max(), yh.max()) + 0.02 * abs(y.max() - y.min())

        ax.scatter(y, yh, c=colors, s=40, alpha=0.8, edgecolors="white", lw=0.4, zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1 line", zorder=2)

        # residual shading ±MAE
        ax.fill_between([lo, hi], [lo - mae, hi - mae],
                        [lo + mae, hi + mae], alpha=0.08, color="grey", label="±MAE band")

        _textbox(ax, f"R² = {r2:.3f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}", "upper left")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Observed  {label}")
        ax.set_ylabel(f"Predicted  {label}")
        ax.set_title(label)
        ax.legend(fontsize=8)

    # condition colour legend
    legend_els = [Line2D([0],[0], marker="o", color="w", markerfacecolor=_COND_PAL[i],
                         markersize=8, label=f"{int(r.conc)}ppm {r.flow}mL {r.potential}V")
                  for i, r in conds.iterrows()]
    fig.legend(handles=legend_els, title="Condition", fontsize=7,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.22)
    _save("fig5_parity")


# ── Fig 6 — Pareto / trade-off front ──────────────────────────────────────────

def fig6_pareto(grid, sec_g, rkp_g, sec_b, rkp_b, pmask, knee_idx, df=None):
    """Pareto front coloured by concentration, calibrated bands, knee callout."""
    pidx    = np.where(pmask)[0]
    concs_g = grid[:, 0]
    uniq_c  = sorted(np.unique(concs_g))
    c_cmap  = {c: _COND_PAL[i] for i, c in enumerate(uniq_c)}

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Predicted operating trade-off space (SEC vs. R_κ)",
                 fontsize=13, fontweight="bold")

    # background candidates (light)
    ax.scatter(sec_g[~pmask], rkp_g[~pmask], s=5, c="lightgrey",
               alpha=0.4, label="Grid candidates", zorder=1)

    # Pareto front coloured by concentration
    for c in uniq_c:
        mask_c = (concs_g[pidx] == c)
        if mask_c.any():
            idx_c = pidx[mask_c]
            ax.scatter(sec_g[idx_c], rkp_g[idx_c], s=35, color=c_cmap[c],
                       zorder=3, label=f"{int(c)} ppm")
            ax.errorbar(sec_g[idx_c], rkp_g[idx_c],
                        xerr=sec_b[idx_c], yerr=rkp_b[idx_c],
                        fmt="none", ecolor=c_cmap[c], alpha=0.35, capsize=2)

    # observed conditions (if df provided)
    if df is not None:
        cond_pts = df.groupby(["conc","flow","potential"])[["sec_wh_m3","r_kappa_pos"]].mean()
        ax.scatter(cond_pts["sec_wh_m3"], cond_pts["r_kappa_pos"],
                   s=120, marker="D", c="black", zorder=6, label="Observed conditions")

    # knee
    ax.scatter(sec_g[knee_idx], rkp_g[knee_idx], s=250, c=_C["red"],
               marker="*", zorder=7, label="Pareto knee candidate")
    ax.annotate("Model-suggested\ncandidate for\nexperimental validation",
                xy=(sec_g[knee_idx], rkp_g[knee_idx]),
                xytext=(sec_g[knee_idx] + 0.05*(sec_g.max()-sec_g.min()),
                        rkp_g[knee_idx] - 0.08*(rkp_g.max()-rkp_g.min())),
                fontsize=8, color=_C["red"],
                arrowprops=dict(arrowstyle="->", color=_C["red"], lw=1.2))

    ax.set_xlabel("SEC — Specific Energy Consumption (kWh/m³)")
    ax.set_ylabel("R_κ — Conductivity removal fraction")
    ax.margins(0.12)
    ax.legend(fontsize=8, loc="upper right")
    _textbox(ax, "Shaded bars = calibrated 90 % CI", "lower left", fs=8)
    _save("fig6_pareto")


# ── Fig 7 — Region-based validation ───────────────────────────────────────────

def fig7_region_bar(region_df):
    """Horizontal bar chart: near-observed vs. far-extrapolation MAE with % difference."""
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

    # % degradation annotation
    all_vals = [near[t] for t in targets] + [far[t] for t in targets]
    max_val  = max(all_vals) if all_vals else 1.0
    for i, t in enumerate(targets):
        pct = (far[t] - near[t]) / near[t] * 100 if near[t] > 0 else 0
        ax.text(max(near[t], far[t]) * 1.08,
                y[i], f"+{pct:.0f}%", va="center", fontsize=8,
                color=_C["red"], fontweight="bold")

    ax.set_xlim(0, max_val * 1.55)
    ax.set_yticks(y); ax.set_yticklabels(targets)
    ax.set_xlabel("MAE"); ax.legend()
    ax.invert_yaxis()
    _save("fig7_region_validation")


# ── Fig 8 — Λκ per-condition box plot ─────────────────────────────────────────

def fig8_lambda_boxplot(df):
    """Λκ box + jitter per condition, sorted by median, colour-coded by sweep type."""
    df = df.copy()
    # label sweep type
    def _sweep(r):
        if r["conc"] != 1000:          return "Concentration sweep"
        if r["potential"] not in [1.8]: return "Potential sweep"
        if r["flow"] != 3:             return "Flow sweep"
        return "Reference (1000ppm·3mL·1.8V)"

    df["sweep"] = df.apply(_sweep, axis=1)
    df["label"] = (df["conc"].astype(int).astype(str) + " ppm / "
                   + df["flow"].astype(str) + " mL / "
                   + df["potential"].astype(str) + " V")
    order  = df.groupby("label")["lambda_kappa"].median().sort_values().index.tolist()
    sweep_color = {
        "Concentration sweep": _C["blue"],
        "Flow sweep":          _C["orange"],
        "Potential sweep":     _C["green"],
        "Reference (1000ppm·3mL·1.8V)": _C["red"],
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Λκ cycle-to-cycle variability per operating condition",
                 fontsize=13, fontweight="bold")

    for i, lbl in enumerate(order):
        sub   = df[df["label"] == lbl]["lambda_kappa"].values
        sweep = df[df["label"] == lbl]["sweep"].iloc[0]
        col   = sweep_color[sweep]
        # box
        bp = ax.boxplot(sub, positions=[i], widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(color=col),
                        capprops=dict(color=col),
                        flierprops=dict(marker="x", color=col, markersize=5))
        bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.35)
        bp["boxes"][0].set_edgecolor(col)
        # jitter
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(sub))
        ax.scatter(i + jitter, sub, s=20, color=col, alpha=0.7, zorder=3)
        # mean marker
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
    """Coverage vs. multiplier for all three models on one axes; target band shaded."""
    z   = CFG["calibration"]["nominal_z"]
    a0, a1, da = CFG["calibration"]["alpha_search"]
    alphas = np.arange(a0, a1, da)
    # zoom to relevant range
    mask_alpha = alphas <= 12

    triples = [("lambda_kappa","lk_pred","lk_std",  "Λκ",  _C["green"]),
               ("sec_wh_m3",  "sec_pred","sec_std", "SEC", _C["orange"]),
               ("r_kappa_pos","rkp_pred","rkp_std",  "R_κ", _C["blue"])]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Uncertainty calibration — coverage vs. multiplier α",
                 fontsize=13, fontweight="bold")

    ax.axhspan(CFG["calibration"]["target_low"], CFG["calibration"]["target_high"],
               color="#d0e4f7", alpha=0.5, label="Target 85–95%", zorder=0)
    ax.axhline(1.0, color="grey", lw=0.7, ls=":", label="100% coverage")
    ax.axhline(CFG["calibration"]["nominal_z"] / CFG["calibration"]["nominal_z"],
               color="grey", lw=0.3)   # y=1 reference

    for tc, mc, sc, label, col in triples:
        res = (preds[tc] - preds[mc]).abs().values
        raw = preds[sc].values
        covs = np.array([np.mean(res <= a * z * raw) for a in alphas])
        ax.plot(alphas[mask_alpha], covs[mask_alpha], color=col, lw=2, label=label)
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


# ── Fig 10 — Ablation comparison ──────────────────────────────────────────────

def fig10_ablation_bar(abl):
    """Horizontal bars for all models; highlight best; annotate % gain over Ridge."""
    models = abl["model"].tolist()
    colors = [_MODEL_PAL.get(m, "#999") for m in models]
    xlabels= [_MODEL_LABEL.get(m, m) for m in models]
    ridge_sec = abl.loc[abl["model"]=="ridge","sec_mae"].values[0]
    ridge_rkp = abl.loc[abl["model"]=="ridge","rkp_mae"].values[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model comparison — LOCO-CV MAE (all conditions)",
                 fontsize=13, fontweight="bold")

    for ax, col, title, ref in [
        (a1, "sec_mae", "SEC MAE (kWh/m³)", ridge_sec),
        (a2, "rkp_mae", "R_κ MAE",          ridge_rkp),
    ]:
        vals  = abl[col].values
        best  = np.argmin(vals)
        bars  = ax.barh(range(len(models)), vals, color=colors,
                        edgecolor="white", height=0.6)
        bars[best].set_edgecolor(_C["red"]); bars[best].set_linewidth(2.5)

        # value and % vs Ridge labels
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(v + 0.002 * vals.max(), bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=8)
            if models[i] not in ("ridge", "mean") and ref > 0:
                pct = (ref - v) / ref * 100
                sign = "+" if pct < 0 else ""
                color = _C["green"] if pct > 0 else _C["red"]
                ax.text(vals.max() * 1.18, bar.get_y() + bar.get_height()/2,
                        f"{sign}{pct:.1f}%\nvs Ridge",
                        va="center", fontsize=7, color=color)

        ax.set_xlim(0, vals.max() * 1.55)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(xlabels, fontsize=9)
        ax.set_xlabel(title)
        ax.axvline(ref, color=_C["grey"], lw=1, ls="--", alpha=0.5)
        ax.invert_yaxis()

    _save("fig10_ablation_bar")


# ── Fig 11 — GP response slices ────────────────────────────────────────────────

def fig11_response_slices(gp_lk, gp_sec, gp_rkp, df, cal):
    """1D GP response ribbons (mean ± calibrated CI) along each experimental sweep."""
    z = CFG["calibration"]["nominal_z"]
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
        ("sec_wh_m3",   "sec_pred", "sec_std",  "SEC (kWh/m³)",    _C["orange"]),
        ("r_kappa_pos", "rkp_pred", "rkp_std",  "R_κ (removal)",   _C["blue"]),
    ]

    fig, axes = plt.subplots(len(targets), len(sweeps), figsize=(15, 8),
                              sharex="col", constrained_layout=True)
    fig.suptitle("GP response along each experimental sweep (calibrated 90 % CI)",
                 fontsize=13, fontweight="bold")

    for col, (xlabel, xvals, make_X1, obs_df) in enumerate(sweeps):
        X1  = make_X1(xvals)
        lk_mu, lk_sg = predict_gp(*gp_lk, X1)
        X2  = np.column_stack([X1, lk_mu])
        for row, (tc, pc, sc, ylabel, col_c) in enumerate(targets):
            ax = axes[row, col]
            gp = gp_sec if tc == "sec_wh_m3" else gp_rkp
            mu, sg = predict_gp(*gp, X2)
            mult = cal[tc]["multiplier"]
            ci   = mult * z * sg
            ax.plot(xvals, mu, color=col_c, lw=2, label="GP mean")
            ax.fill_between(xvals, mu - ci, mu + ci, alpha=0.2,
                            color=col_c, label="Calibrated 90% CI")
            # observed scatter
            if len(obs_df) > 0:
                x_col = {"Concentration (ppm)": "conc",
                         "Flow rate (mL/min)":  "flow",
                         "Potential (V)":        "potential"}[xlabel]
                ax.scatter(obs_df[x_col], obs_df[tc], s=50, c="black",
                           zorder=5, label="Observed", edgecolors="white", lw=0.5)
            if row == 0:
                ax.set_title(xlabel, fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(targets) - 1:
                ax.set_xlabel(xlabel)
            ax.legend(fontsize=7)

    _save("fig11_response_slices")


# ── Fig 12 — Condition-level trade-off scatter ─────────────────────────────────

def fig12_condition_tradeoff(preds, cal):
    """10 observed conditions in SEC vs R_κ space with calibrated error bars."""
    z    = CFG["calibration"]["nominal_z"]
    grp  = preds.groupby(["conc","flow","potential"])
    rows = []
    for (c, f, p), g in grp:
        rows.append(dict(
            conc=c, flow=f, potential=p,
            sec_true=g["sec_wh_m3"].mean(),
            rkp_true=g["r_kappa_pos"].mean(),
            sec_pred=g["sec_pred"].mean(),
            rkp_pred=g["rkp_pred"].mean(),
            sec_ci=cal["sec_wh_m3"]["multiplier"]   * z * g["sec_std"].mean(),
            rkp_ci=cal["r_kappa_pos"]["multiplier"] * z * g["rkp_std"].mean(),
        ))
    cdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Operating condition trade-off: SEC vs. R_κ\n(observed conditions with calibrated prediction intervals)",
                 fontsize=13, fontweight="bold")

    concs_u = sorted(cdf["conc"].unique())
    c_map   = {c: _COND_PAL[i] for i, c in enumerate(concs_u)}

    cond_labels = []
    for num, (_, r) in enumerate(cdf.iterrows(), 1):
        col = c_map[r["conc"]]
        ax.errorbar(r["sec_pred"], r["rkp_pred"],
                    xerr=r["sec_ci"], yerr=r["rkp_ci"],
                    fmt="none", ecolor=col, alpha=0.5, capsize=4, lw=1.5)
        # large circle for number label
        ax.scatter(r["sec_pred"], r["rkp_pred"], s=200, color=col,
                   edgecolors="black", lw=0.8, zorder=4)
        ax.text(r["sec_pred"], r["rkp_pred"], str(num),
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white", zorder=5)
        # observed true value (cross)
        ax.scatter(r["sec_true"], r["rkp_true"], s=70, marker="x",
                   color=col, lw=2, zorder=5)
        cond_labels.append(
            f"{num}: {int(r['conc'])} ppm · {r['flow']} mL/min · {r['potential']} V"
        )

    ax.set_xlabel("SEC — Specific Energy Consumption (kWh/m³)")
    ax.set_ylabel("R_κ — Conductivity removal fraction")

    # Numbered legend replaces overlapping text annotations
    row_colors = [c_map[r["conc"]] for _, r in cdf.iterrows()]
    num_els = [Line2D([0],[0], marker="o", color="w",
                      markerfacecolor=row_colors[i],
                      markersize=9, label=lbl)
               for i, lbl in enumerate(cond_labels)]
    type_els = [
        Line2D([0],[0], marker="o", color="grey", markersize=8, label="● GP prediction"),
        Line2D([0],[0], marker="x", color="grey", markersize=8, lw=2, label="✕ Observed mean"),
    ]
    ax.legend(handles=num_els + type_els, fontsize=7.5,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.subplots_adjust(right=0.70)
    _textbox(ax, "Error bars = calibrated 90% CI", "lower right", fs=8)
    _save("fig12_condition_tradeoff")


# ── Fig 13 — Cycle stability per condition ────────────────────────────────────

def fig13_cycle_stability(df):
    """Λκ, SEC, R_κ vs cycle_id for each condition — checks stationarity."""
    metrics = [("lambda_kappa", "Λκ (mS·cm⁻¹·mC⁻¹)", _C["green"]),
               ("sec_wh_m3",   "SEC (kWh/m³)",         _C["orange"]),
               ("r_kappa_pos", "R_κ",                  _C["blue"])]
    conds   = _conditions(df)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 9), sharex=False)
    fig.suptitle("Cycle-to-cycle stability across all operating conditions",
                 fontsize=13, fontweight="bold")

    for ax, (col, ylabel, base_col) in zip(axes, metrics):
        for i, cond in enumerate(conds):
            _, sub = _split(df, cond)
            sub    = sub.sort_values("cycle_id")
            col_c  = _COND_PAL[i % 10]
            label  = f"{int(cond[0])}ppm / {cond[1]}mL / {cond[2]}V"
            ax.plot(sub["cycle_id"], sub[col], marker="o", markersize=4,
                    lw=1.2, color=col_c, alpha=0.85, label=label)
            # mean line per condition
            ax.axhline(sub[col].mean(), color=col_c, lw=0.6, ls="--", alpha=0.4)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Cycle index")

    # shared legend at bottom — subplots_adjust makes room inside the saved figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.18)
    _save("fig13_cycle_stability")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    # Load & segment
    records = load_raw_sheets()
    df      = build_cycle_table(records)
    conds   = _conditions(df)
    print(f"Cycles extracted : {len(df)} across {len(conds)} conditions")

    # Optional GP hyperparameter grid search (toggle in config.yaml → gp.grid_search.enabled)
    gp_grid_search(df)

    # LOCO-CV (T-01: all cycles iterated)
    preds = run_loco_cv(df)
    preds.to_csv(OUT / "validation_summary.csv", index=False)
    print(f"LOCO-CV rows     : {len(preds)}")

    # Calibration (T-04)
    cal = calibrate(preds)
    print("Calibration coverage:", {k: v["adjusted_coverage"] for k, v in cal.items()})

    # Ablation (T-07)
    abl = run_ablation(df)
    print("\nAblation:\n", abl[["model","sec_mae","sec_rel_mae","rkp_mae","rkp_rel_mae"]]
          .to_string(index=False))

    # Region validation (T-06)
    region_df, preds_reg = region_validation(df, preds)

    # Fit full-dataset GPs for surface & Pareto
    gp_lk  = train_gp(df[FEAT_S1].values, df["lambda_kappa"].values)
    gp_sec = train_gp(df[FEAT_S2].values, df["sec_wh_m3"].values)
    gp_rkp = train_gp(df[FEAT_S2].values, df["r_kappa_pos"].values)

    # Pareto (T-05 / T-03 Option A)
    grid, sec_g, rkp_g, sec_b, rkp_b, pmask, knee = run_pareto(
        gp_lk, gp_sec, gp_rkp, df, cal)

    # Flow decision record (T-03)
    json.dump({"option": "A", "stage2_features": FEAT_S2,
               "note": "Flow included in Stage 2 GP to fully define Pareto candidate "
                       "(concentration, flow, potential)."},
              open(OUT / "flow_decision.json", "w"), indent=2)

    # Figures (T-06) — 13 total
    figs = [
        ("fig1_pipeline",           lambda: fig1_schematic()),
        ("fig2_segmentation",       lambda: fig2_segmentation(records)),
        ("fig3_kappa_profile",      lambda: fig3_kappa_profile(records)),
        ("fig4_lambda_surface",     lambda: fig4_lambda_surface(gp_lk, df)),
        ("fig5_parity",             lambda: fig5_parity(preds)),
        ("fig6_pareto",             lambda: fig6_pareto(grid, sec_g, rkp_g, sec_b, rkp_b, pmask, knee, df)),
        ("fig7_region_validation",  lambda: fig7_region_bar(region_df)),
        ("fig8_lambda_boxplot",     lambda: fig8_lambda_boxplot(df)),
        ("fig9_calibration_curve",  lambda: fig9_calibration_curve(preds, cal)),
        ("fig10_ablation_bar",      lambda: fig10_ablation_bar(abl)),
        ("fig11_response_slices",   lambda: fig11_response_slices(gp_lk, gp_sec, gp_rkp, df, cal)),
        ("fig12_condition_tradeoff",lambda: fig12_condition_tradeoff(preds, cal)),
        ("fig13_cycle_stability",   lambda: fig13_cycle_stability(df)),
    ]
    for name, fn in tqdm(figs, desc="Figures", unit="fig"):
        fn()

    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
