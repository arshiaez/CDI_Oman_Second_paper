---
name: project-cdi-science
description: "Scientific definitions, Λκ framing, forbidden language, resolved task decisions, and calibration status"
metadata:
  type: project
---

## Λκ (Lambda-kappa) — Critical Definition

Λκ is **NOT** physical charge efficiency. It is a **conductivity-native proxy**: conductivity reduction per unit charge derived purely from available sensors. Formula: `∫Δκ dt / ∫|I| dt` (Δκ clipped to zero to discard noise artifacts).

- Valid as a relative comparator within the same CDI system
- Cannot be converted to absolute moles removed without Cout calibration
- The novelty: replacing a fixed charge-efficiency assumption with this learned proxy

**Why this matters:** Reviewers will scrutinise this claim. Calling it "charge efficiency" invites rejection. The framing paragraph in Methods/Introduction is load-bearing for perceived novelty.

## SEC Computation (implementation decision)

SEC = gross energy during desalination phase / volume processed:
- `W_wh = ∫|I_des| × V dt / 3600` (Wh)
- `vol_m3 = flow_m3s × len(desal_phase)`
- `SEC = W_wh / (vol_m3 × 1000)` (kWh/m³)

This matches the Combined_CDI_DATA.xlsx reference values. The regeneration phase is excluded (assumed open/short circuit — no voltage work recovered).

## Forbidden Language (replace before any drafting)

| Remove | Replace with |
|---|---|
| optimal solution / optimal operating point | model-suggested candidate for experimental validation |
| maximum removal | highest predicted conductivity removal in near-observed region |
| true efficiency | conductivity-based efficiency proxy (Λκ) |
| proven global optimum | Pareto knee candidate — subject to experimental confirmation |
| charge efficiency | conductivity-native efficiency proxy |

## Sparse Data Constraint

10 unique physical conditions, 89 total desal cycles. This is a **methodology paper**. Results must be framed as condition-level trends, not high-density optimisation. Claiming global optimisation results will invite rejection.

## Resolved Decisions

### T-03 — Flow in Stage 2: RESOLVED → Option A

Flow is included in Stage 2 GP features: `FEAT_S2 = [conc, flow, potential, lambda_kappa]`.
This fully defines the Pareto candidate (conc, flow, potential) and removes the reviewer inconsistency.
Recorded in `cdi_output/flow_decision.json`.

### T-04 — Uncertainty Calibration: RESOLVED

Calibration achieved from first run:
- Λκ adjusted coverage: **86.5%** ✓
- SEC adjusted coverage: **85.4%** ✓
- R_κ adjusted coverage: **86.5%** ✓

All within 85–95% target. Multipliers and coverages stored in `cdi_output/calibration.json`.

## Open Issue — Ablation Inversion

GP-with-Λκ is currently worse than Ridge in LOCO-CV:
- Ridge: sec_mae=0.102, rkp_mae=0.053
- GP-no-Λκ: sec_mae=0.124, rkp_mae=0.057
- GP-with-Λκ: sec_mae=0.149, rkp_mae=0.078

**Suspected cause:** Stage 1 Λκ prediction error propagates into Stage 2 as a noisy feature, degrading Stage 2 performance. This is error propagation in the two-stage design.

**Suggested fix:** In `config.yaml`, set `n_iter: 500` and `lr: 0.01` then rerun. If inversion persists, consider using true Λκ as Stage 2 feature during cross-validation (oracle upper bound) to diagnose whether the issue is Stage 1 quality or Stage 2 design.

## Calibration Implementation

Empirical multiplier α found by linear sweep `np.arange(0.05, 20.0, 0.05)` — first α where adjusted coverage lands in [85%, 95%]. Search bounds are configurable in `config.yaml` under `calibration.alpha_search`.

## GP Architecture

- Kernel: ARD-RBF (`RBFKernel(ard_num_dims=d)`) wrapped in `ScaleKernel`
- Mean: `ConstantMean`
- Noise: `GaussianLikelihood` with lower bound `noise_constraint` from config
- Optimiser: Adam, `n_iter` steps at `lr` learning rate
- Inference: `fast_pred_var` on GPU (CUDA confirmed available)
- Scalers: `StandardScaler` on both X and y before GP fitting; predictions inverse-transformed back to original scale
