---
name: project-cdi-science
description: "Scientific definitions, Λκ framing, forbidden language, and key constraints for the CDI paper"
metadata: 
  node_type: memory
  type: project
  originSessionId: 18db3b1a-1489-4578-a515-94b9f7309e3e
---

## Λκ (Lambda-kappa) — Critical Definition

Λκ is **NOT** physical charge efficiency. It is a **conductivity-native proxy**: conductivity reduction per unit charge derived purely from available sensors (conductivity meter + ammeter). Formula: `∫Δκ dt / ∫I dt`.

- Valid as a relative comparator within the same CDI system
- Cannot be converted to absolute moles removed without Cout calibration
- The novelty: replacing a fixed charge-efficiency assumption with this learned proxy

**Why this matters:** Reviewers will scrutinise this claim. Calling it "charge efficiency" invites rejection. The framing paragraph in Methods/Introduction is load-bearing for perceived novelty.

## Forbidden Language (replace before any drafting)

| Remove | Replace with |
|---|---|
| optimal solution / optimal operating point | model-suggested candidate for experimental validation |
| maximum removal | highest predicted conductivity removal in near-observed region |
| true efficiency | conductivity-based efficiency proxy (Λκ) |
| proven global optimum | Pareto knee candidate — subject to experimental confirmation |
| charge efficiency | conductivity-native efficiency proxy |

## Sparse Data Constraint

10 unique physical conditions, repeated cycle measurements. This is a **methodology paper**. Results must be framed as condition-level trends, not high-density optimisation. Claiming global optimisation results will invite rejection.

## Stage 2 Flow Handling (T-03 — open decision)

Flow was dropped from Stage 2 in the old pipeline without justification. Two options:
- **Option A (preferred):** Include flow in Stage 2 → fully defined Pareto candidate (conc, flow, potential)
- **Option B (acceptable):** Keep it out, but document ARD sensitivity values for flow, state the fixed/median flow used, write a one-paragraph justification

Either way: the final Pareto knee must report all three operating variables.

## Calibration Target

All three GP models (Λκ, SEC, R_kappa_pos) must achieve 85–95% adjusted coverage on the nominal 90% credible interval before any uncertainty bands appear in figures.
