---
name: project-cdi-framework
description: "Core project context — pipeline implementation status, dataset structure, 10 operating conditions, resolved tasks, and current known issues"
metadata:
  type: project
---

CDI (Capacitive Deionization) analytics pipeline for Q1 journal publication (Desalination or Separation and Purification Technology).

**Why:** Replace fixed charge-efficiency assumptions with a learned conductivity proxy Λκ. Paper is a methodology contribution; with only 10 unique conditions, it cannot claim optimisation results.

**How to apply:** Every suggestion must respect the sparse-data constraint. The ablation improvement is the central quantitative claim. The Pareto knee is always a "model-suggested candidate for experimental validation", never an "optimal solution".

## Environments

- Local Windows: `D:\Work\Oman\CDI\CDI_Sceond_paper\CDI_Oman_Second_paper\`
- Lightning AI studio: `/teamspace/studios/this_studio/CDI_Oman_Second_paper/`
- Config path must be resolved relative to the script (`Path(__file__).parent`) not CWD — the FileNotFoundError for `config.yaml` occurs when running from a different directory.

## Files Written This Session

- `cdi_pipeline.py` — full pipeline (~600 lines)
- `config.yaml` — all hyperparameters (GP, calibration, Pareto grid, figure DPI/format)
- `requirements.txt` — pinned dependencies; PyTorch commented out (install separately via pytorch.org)

## Dataset (`Dataset/`)

- `1000-2500-5000-7500 ppm.xlsx` — vary concentration (1000/2500/5000/7500 ppm), flow=3 mL/min, potential=1.8 V
- `vary flow rate.xlsx` — vary flow (2/3/4/5 mL/min), conc=1000 ppm, potential=1.8 V
- `Vary potential-1000ppm.xlsx` — vary potential (0.9/1.2/1.5/1.8 V), conc=1000 ppm, flow=3 mL/min; has junk trailing columns — stripped via `usecols=[0,1,2]`
- `Combined_CDI_DATA.xlsx` — pre-computed cycle-level summaries; `consentration` column has a typo; not used directly in pipeline

**10 unique (concentration ppm, flow mL/min, potential V) conditions:**
(1000,2,1.8), (1000,3,0.9), (1000,3,1.2), (1000,3,1.5), (1000,3,1.8), (1000,4,1.8), (1000,5,1.8), (2500,3,1.8), (5000,3,1.8), (7500,3,1.8)

**Actual cycle count from pipeline run: 89 desal cycles** (previously estimated ~78; the raw timeseries yield more cycles than the Combined file suggests).

The (1000,3,1.8) condition appears in all three raw files — pipeline deduplicates via a `seen` set, keeping the first occurrence (from the concentration file).

## Pipeline Stages (implemented)

0. Data loading + cycle segmentation — current sign detection; iterates ALL cycles (T-01 fix applied)
1. Cycle metric extraction: Λκ = ∫Δκ dt / ∫|I| dt, SEC = gross desal energy / volume, R_kappa_pos
2. LOCO-CV — one prediction row per cycle; 89 rows confirmed
3. Uncertainty calibration — empirical multiplier sweep; **T-04 resolved** (see science file)
4. Ablation — 7 models (see below)
5. Region-based validation — rank-split (not median-split; median-split caused empty-group crash)
6. Pareto front + knee candidate — T-03 Option A (flow included in Stage 2)
7. 10 publication figures (fig1–fig10)

## Ablation Models (7 total)

All sklearn baselines use FEAT_S1 = (conc, flow, potential); no Λκ — fair comparison:

| Model | Notes |
|---|---|
| Mean | Trivial floor |
| Ridge | Linear baseline |
| Random Forest | n_estimators=200 |
| SVR | RBF kernel, C=10 |
| MLP/ANN | (64,32) hidden, early stopping |
| GP no-Λκ | GPyTorch, ARD-RBF, FEAT_S1 only |
| GP with-Λκ | Two-stage; FEAT_S2 = FEAT_S1 + Λκ |

**Current finding:** GP-with-Λκ is underperforming Ridge in the ablation (sec_mae=0.149 vs ridge=0.102). Suspected cause: Stage 1 prediction error for Λκ propagates into Stage 2 and compounds. Suggested fix: increase GP n_iter (500) and decrease lr (0.01) in config.yaml and rerun.

## Figures (10 total, ≥300 DPI)

| File | Content |
|---|---|
| fig1_pipeline | Pipeline schematic |
| fig2_segmentation | Current + conductivity vs. time, phase boundaries |
| fig3_kappa_profile | κ(t) representative cycle, R_κ area shaded |
| fig4_lambda_surface | Λκ contour: concentration × potential at median flow |
| fig5_parity | Predicted vs. observed scatter (Λκ, SEC, R_κ) with R² and MAE |
| fig6_pareto | Pareto front, calibrated bands, knee marked |
| fig7_region_validation | Near-observed vs. far-extrapolation MAE bar chart |
| fig8_lambda_boxplot | Λκ box plot per condition — cycle-to-cycle variability |
| fig9_calibration_curve | Coverage vs. multiplier α for all three models |
| fig10_ablation_bar | All 7 models compared on SEC MAE and R_κ MAE |

## Pareto Front — Replacement Recommendation

With only 10 conditions, the dense-grid Pareto surface is largely extrapolated and uncertain. Recommended replacement (not yet implemented):
- **Left panel:** SEC vs. R_κ scatter of the 10 observed conditions with calibrated GP uncertainty ellipses
- **Right panel:** R_κ and SEC vs. potential (1D slice, most interesting sweep) with GP ribbon and observed-condition dots

This is more defensible for a methodology paper and avoids overclaiming optimisation.

## Known Bug in Old Pipeline (`Old/` — ignore entirely)

`run_validation()` and `run_ablation()` saved only cycle index [0] per held-out condition → ~10 data points instead of ~89. All metrics from `Old/cdi_output/` are invalid. Never reference or port from `Old/`.
