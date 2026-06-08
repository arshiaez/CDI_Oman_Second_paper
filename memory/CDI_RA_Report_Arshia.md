---
name: cdi-ra-report
description: Supervisor task directive — 8 tasks (T-01 to T-08), priorities, required deliverables, and language rules for the CDI paper
metadata:
  type: project
---

## Paper Target
Q1 — Desalination or Separation and Purification Technology.
Frame as: *"A reproducible, cycle-resolved, uncertainty-aware CDI analytics framework that replaces fixed charge-efficiency assumptions with a learned conductivity proxy."*

## Task List

### 🔴 CRITICAL (blocking)

**T-01 — Fix the Validation Loop**
- `run_validation()` and `run_ablation()` currently save only cycle index [0] per held-out condition → ~10 points instead of ~78.
- Fix: iterate over **all cycles** in each held-out condition.
- Output must have one prediction row per cycle + condition-level summary (mean ± std).
- Confirm n_cycles ≈ 78 before proceeding.

**T-02 — Regenerate All Results** *(after T-01)*
- Rerun Stage 1 (Λκ), Stage 2 (energy + performance), ablation, region-based validation, uncertainty calibration.
- Produce old-vs-new comparison table for supervisor sign-off.

### 🟠 IMPORTANT (required for scientific validity)

**T-03 — Resolve Flow Variable Handling in Stage 2**
- Flow was dropped from Stage 2 without justification — reviewers will flag this.
- **Option A (preferred):** Include flow in Stage 2 → fully defined Pareto point (conc, flow, potential).
- **Option B:** Keep it out — document ARD sensitivity for flow, state the fixed/median value used, write a one-paragraph justification.
- Either way: the final Pareto candidate must report all three operating variables.

**T-04 — Fix Uncertainty Calibration**
- Current coverage: 0% (performance), 80% (energy) vs. 90% target — not reportable.
- Diagnose LOCO-based calibration failure; apply empirical multiplier to all three models (Λκ, energy, performance).
- Target: 85–95% adjusted coverage on all three models before any uncertainty band appears in a figure.

**T-05 — Complete the Pareto Candidate** *(after T-03 and T-04)*
- Current knee `(1000 ppm, 0.947 V)` is missing its flow coordinate and calibrated uncertainty bands.
- Extract full knee: `(concentration, flow, potential)` + predicted objectives + calibrated uncertainty.
- Replace all "optimal" language (see Language Rules below).

### 🟡 RECOMMENDED (publication quality)

**T-06 — 7 Publication-Ready Figures (≥300 DPI, after T-01–T-05)**
1. Pipeline schematic: raw data → segmentation → cycle metrics → Stage 1 → Stage 2 → Pareto
2. Cycle segmentation: current + conductivity vs. time, phase boundaries annotated
3. Conductivity removal profile: κ(t) for a representative cycle, R_kappa area shaded
4. Λκ sensitivity surface: Λκ vs. concentration and potential (flow at median)
5. Prediction vs. ground truth scatter: one panel each for Λκ, energy, performance; annotate R² and MAE
6. Pareto front: SEC vs. R_kappa_pos with calibrated bands, near-observed points, knee marked
7. Region-based validation: grouped bar chart, near-observed vs. far-extrapolation MAE

**T-07 — Consolidate Ablation Table**
- One table: rows = Ridge / GP without Λκ / GP with Λκ; columns = Energy MAE/RMSE/Rel.MAE and Performance MAE/RMSE/Rel.MAE.
- Highlight % improvement of full model over each baseline.
- The ~17% energy and ~21% performance improvements are the core quantitative claim.

**T-08 — Dataset Characterisation + Λκ Framing**
- Dataset summary table: sweep type, variable range, conditions per sweep, QC-passing cycles, total cycles.
- Add to docs: *"Sparse experimental design — 10 unique physical conditions. Results are condition-level trends, not high-density optimisation."*
- Λκ clarification: (a) NOT physical charge efficiency; (b) IS conductivity reduction per unit charge from available sensors; (c) valid as relative comparator within same system; (d) cannot convert to absolute moles without Cout calibration.

## Language Rules

| Remove | Replace with |
|---|---|
| optimal solution / optimal operating point | model-suggested candidate for experimental validation |
| maximum removal | highest predicted conductivity removal in near-observed region |
| true efficiency | conductivity-based efficiency proxy (Λκ) |
| proven global optimum | Pareto knee candidate — subject to experimental confirmation |
| charge efficiency | conductivity-native efficiency proxy |
