---
name: project-cdi-framework
description: "Core project context — what the CDI pipeline is, its publication target, dataset structure, 10 operating conditions, and the two-stage GP architecture"
metadata: 
  node_type: memory
  type: project
  originSessionId: 18db3b1a-1489-4578-a515-94b9f7309e3e
---

CDI (Capacitive Deionization) analytics pipeline for Q1 journal publication (Desalination or Separation and Purification Technology). Working directory: `C:\work\Oman\CDI`.

**Why:** Replace fixed charge-efficiency assumptions with a learned conductivity proxy Λκ. Paper is framed as a methodology contribution; with only 10 unique conditions, it cannot claim optimisation results.

**How to apply:** Every suggestion should respect the sparse-data constraint. Quantitative claims must be limited to the ~17–21% ablation improvement. The Pareto knee is always a "model-suggested candidate for experimental validation", never an "optimal solution".

## Dataset (`Dataset/`)
- `1000-2500-5000-7500 ppm.xlsx` — vary concentration (1000/2500/5000/7500 ppm), flow=3 mL/min, potential=1.8 V
- `vary flow rate.xlsx` — vary flow (2/3/4/5 mL/min), conc=1000 ppm, potential=1.8 V
- `Vary potential-1000ppm.xlsx` — vary potential (0.9/1.2/1.5/1.8 V), conc=1000 ppm, flow=3 mL/min
- `Combined_CDI_DATA.xlsx` — pre-computed cycle-level summaries (single `Data` sheet)

Raw sheets: `Time (s)`, `Conductivity (mS/cm)`, `Current (mA)` per second.

**10 unique (concentration ppm, flow mL/min, potential V) conditions:**
(1000,2,1.8), (1000,3,0.9), (1000,3,1.2), (1000,3,1.5), (1000,3,1.8), (1000,4,1.8), (1000,5,1.8), (2500,3,1.8), (5000,3,1.8), (7500,3,1.8)

Cycle counts: 4–18 desal cycles per condition; ~78 total desal cycles.

## Pipeline Stages
0. Data loading + cycle segmentation (detect desal/regen phases from current sign)
1. Cycle metric extraction: Λκ, SEC, R_kappa_pos, Δc, Removal%, W_net, ENAS
2. Stage 1 GP: predict Λκ from (conc, flow, potential)
3. Stage 2 GP: predict SEC and R_kappa_pos from (conc, flow, potential, Λκ)
4. LOCO-CV validation (Leave-One-Condition-Out) — **must iterate all ~78 cycles, not just 1 per condition**
5. Uncertainty calibration — empirical multiplier, target 85–95% adjusted coverage
6. Ablation: Ridge vs GP-without-Λκ vs GP-with-Λκ
7. Region-based validation: near-observed vs far-extrapolation MAE
8. Pareto front (SEC minimize, R_kappa_pos maximize), knee identification
9. 7 publication-ready figures (≥300 DPI)

## Known Bug in Old Pipeline (`Old/` — ignore)
`run_validation()` and `run_ablation()` saved only the first cycle per held-out condition → ~10 data points instead of ~78. All metrics from `Old/cdi_output/` are invalid.
