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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cdist

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
    for df, conc, flow, potential in records:
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
    for _ in range(CFG["gp"]["n_iter"]):
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
    for cond in _conditions(df):
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

def run_ablation(df):
    """Compare Ridge / GP-no-Λκ / GP-with-Λκ via LOCO-CV; write ablation_table.csv."""
    buckets = {m: {"sec": [], "rkp": []} for m in ["ridge", "gp_nolk", "gp_lk"]}

    for cond in _conditions(df):
        train, test = _split(df, cond)

        # Ridge baseline (3-feature, no Λκ)
        for col, key in [("sec_wh_m3", "sec"), ("r_kappa_pos", "rkp")]:
            sc  = StandardScaler()
            Xtr = sc.fit_transform(train[FEAT_S1].values)
            Xte = sc.transform(test[FEAT_S1].values)
            err = np.abs(test[col].values - Ridge().fit(Xtr, train[col].values).predict(Xte))
            buckets["ridge"][key].extend(err)

        # GP without Λκ
        for col, key in [("sec_wh_m3", "sec"), ("r_kappa_pos", "rkp")]:
            gp      = train_gp(train[FEAT_S1].values, train[col].values)
            mu, _   = predict_gp(*gp, test[FEAT_S1].values)
            buckets["gp_nolk"][key].extend(np.abs(test[col].values - mu))

        # GP with Λκ (two-stage)
        gp_lk      = train_gp(train[FEAT_S1].values, train["lambda_kappa"].values)
        lk_te, _   = predict_gp(*gp_lk, test[FEAT_S1].values)
        X2_tr      = np.column_stack([train[FEAT_S1].values, train["lambda_kappa"].values])
        X2_te      = np.column_stack([test[FEAT_S1].values,  lk_te])
        for col, key in [("sec_wh_m3", "sec"), ("r_kappa_pos", "rkp")]:
            gp      = train_gp(X2_tr, train[col].values)
            mu, _   = predict_gp(*gp, X2_te)
            buckets["gp_lk"][key].extend(np.abs(test[col].values - mu))

    sec_ref, rkp_ref = df["sec_wh_m3"].mean(), df["r_kappa_pos"].mean()
    rows = []
    for name, b in buckets.items():
        sa, ra = np.array(b["sec"]), np.array(b["rkp"])
        rows.append(dict(
            model=name,
            sec_mae=sa.mean(),  sec_rmse=np.sqrt((sa**2).mean()),
            sec_rel_mae=sa.mean() / sec_ref,
            rkp_mae=ra.mean(),  rkp_rmse=np.sqrt((ra**2).mean()),
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

    preds    = preds.copy()
    preds["region"] = np.where(dists <= np.median(dists), "near", "far")

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
    for i in range(len(sec)):
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

def _save(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.{CFG['figures']['format']}",
                dpi=CFG["figures"]["dpi"], bbox_inches="tight")
    plt.close()


def fig1_schematic():
    """Pipeline flow diagram."""
    steps = ["Raw\nTimeseries", "Cycle\nSegmentation",
             "Metric Extraction\n(Λκ · SEC · R_κ)",
             "Stage 1 GP\npredict Λκ", "Stage 2 GP\nSEC · R_κ", "Pareto\nFront"]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis("off")
    for i, s in enumerate(steps):
        x = i / (n - 1)
        ax.text(x, 0.5, s, ha="center", va="center", transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.5",
                                      fc="#d0e4f7", ec="#2255aa", lw=1.5))
        if i < n - 1:
            ax.annotate("", xy=((i + 0.82) / (n-1), 0.5),
                        xytext=((i + 0.18) / (n-1), 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#2255aa", lw=1.5))
    _save("fig1_pipeline")


def fig2_segmentation(records):
    """Current + conductivity vs. time with phase boundaries annotated."""
    df, *_ = records[0]
    sub = df[df["time"] <= df["time"].iloc[0] + 2500].copy()
    chg = np.where(np.diff(np.sign(sub["current"].values)) != 0)[0]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    a1.plot(sub["time"], sub["conductivity"], color="#1f77b4")
    a2.plot(sub["time"], sub["current"],      color="#d62728")
    a2.axhline(0, color="k", lw=0.5, ls=":")
    for idx in chg:
        a1.axvline(sub["time"].iloc[idx], color="grey", lw=0.8, ls="--", alpha=0.5)
        a2.axvline(sub["time"].iloc[idx], color="grey", lw=0.8, ls="--", alpha=0.5)
    a1.set_ylabel("Conductivity (mS/cm)")
    a2.set_ylabel("Current (mA)"); a2.set_xlabel("Time (s)")
    _save("fig2_segmentation")


def fig3_kappa_profile(records):
    """κ(t) for a representative desal cycle with R_κ area shaded."""
    df, *_ = records[0]
    desal, _ = segment_cycles(df)[0]
    kappa = desal["conductivity"].values
    t = np.arange(len(kappa))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, kappa, color="#1f77b4", lw=1.5, label="κ(t)")
    ax.fill_between(t, kappa.min(), kappa, alpha=0.2, color="#1f77b4",
                    label="R_κ shaded area")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Conductivity (mS/cm)")
    ax.legend()
    _save("fig3_kappa_profile")


def fig4_lambda_surface(gp_lk, df):
    """Λκ sensitivity surface: concentration × potential at median flow."""
    flow_med = float(df["flow"].median())
    concs = np.linspace(df["conc"].min(),      df["conc"].max(),      50)
    pots  = np.linspace(df["potential"].min(), df["potential"].max(), 50)
    CC, PP = np.meshgrid(concs, pots)
    X_g    = np.column_stack([CC.ravel(), np.full(CC.size, flow_med), PP.ravel()])
    lk_mu, _ = predict_gp(*gp_lk, X_g)

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(CC, PP, lk_mu.reshape(CC.shape), levels=20, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="Λκ (mS·cm⁻¹·mC⁻¹)")
    ax.set_xlabel("Concentration (ppm)"); ax.set_ylabel("Potential (V)")
    ax.set_title(f"Flow fixed at {flow_med} mL/min (median)")
    _save("fig4_lambda_surface")


def fig5_parity(preds):
    """Prediction vs. ground-truth scatter for Λκ, SEC, R_κ."""
    pairs = [("lambda_kappa", "lk_pred",  "Λκ (mS·cm⁻¹·mC⁻¹)"),
             ("sec_wh_m3",   "sec_pred", "SEC (kWh/m³)"),
             ("r_kappa_pos", "rkp_pred", "R_κ")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (tc, pc, label) in zip(axes, pairs):
        y, yh = preds[tc].values, preds[pc].values
        mae   = mean_absolute_error(y, yh)
        ss    = ((y - y.mean()) ** 2).sum()
        r2    = 1 - ((y - yh) ** 2).sum() / ss
        lo, hi = min(y.min(), yh.min()), max(y.max(), yh.max())
        ax.scatter(y, yh, s=25, alpha=0.7, color="#1f77b4", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"Observed {label}"); ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"R²={r2:.3f}   MAE={mae:.4f}")
    _save("fig5_parity")


def fig6_pareto(grid, sec_g, rkp_g, sec_b, rkp_b, pmask, knee_idx):
    """Pareto front with calibrated uncertainty bands and knee candidate marked."""
    pidx = np.where(pmask)[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sec_g[~pmask], rkp_g[~pmask], s=4, c="lightgrey", label="Grid candidates")
    ax.scatter(sec_g[pidx],   rkp_g[pidx],   s=20, c="#1f77b4",  label="Pareto front")
    ax.errorbar(sec_g[pidx], rkp_g[pidx],
                xerr=sec_b[pidx], yerr=rkp_b[pidx],
                fmt="none", ecolor="#1f77b4", alpha=0.4, capsize=2)
    ax.scatter(sec_g[knee_idx], rkp_g[knee_idx], s=160, c="#d62728", zorder=5,
               marker="*", label="Pareto knee candidate")
    ax.set_xlabel("SEC (kWh/m³)"); ax.set_ylabel("Conductivity removal R_κ")
    ax.legend(fontsize=8)
    _save("fig6_pareto")


def fig7_region_bar(region_df):
    """Grouped bar chart: near-observed vs. far-extrapolation MAE per target."""
    targets = region_df["target"].unique()
    x, w    = np.arange(len(targets)), 0.35
    near    = region_df[region_df["region"] == "near"].set_index("target")["mae"]
    far     = region_df[region_df["region"] == "far"].set_index("target")["mae"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, [near[t] for t in targets], w, color="#1f77b4", label="Near-observed")
    ax.bar(x + w/2, [far[t]  for t in targets], w, color="#d62728", label="Far-extrapolation")
    ax.set_xticks(x); ax.set_xticklabels(targets, rotation=10)
    ax.set_ylabel("MAE"); ax.legend()
    _save("fig7_region_validation")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    # Load & segment
    records = load_raw_sheets()
    df      = build_cycle_table(records)
    conds   = _conditions(df)
    print(f"Cycles extracted : {len(df)} across {len(conds)} conditions")

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

    # Figures (T-06)
    fig1_schematic()
    fig2_segmentation(records)
    fig3_kappa_profile(records)
    fig4_lambda_surface(gp_lk, df)
    fig5_parity(preds)
    fig6_pareto(grid, sec_g, rkp_g, sec_b, rkp_b, pmask, knee)
    fig7_region_bar(region_df)

    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
