"""
Unit and integration tests for cdi_pipeline.py.

Run with:  pytest tests/test_pipeline.py -v
Tests are designed to be fast (synthetic data, no real Excel files).
"""

import sys
import os
import types
import unittest.mock as mock

# ── Mock heavy GPU libraries before importing the pipeline ─────────────────────
# These are only available in the project's GPU environment; the tests don't
# exercise GP training, so stubs are sufficient for import to succeed.
def _make_stub(name):
    m = types.ModuleType(name)
    m.__spec__ = mock.MagicMock()
    return m

_HEAVY = [
    "torch", "torch.optim",
    "gpytorch", "gpytorch.models", "gpytorch.likelihoods", "gpytorch.kernels",
    "gpytorch.means", "gpytorch.distributions", "gpytorch.mlls",
    "gpytorch.constraints", "gpytorch.settings",
    "matplotlib", "matplotlib.pyplot", "matplotlib.lines",
    "matplotlib.cm", "matplotlib.colors",
    "tqdm", "tqdm.auto",
]
for _mod in _HEAVY:
    if _mod not in sys.modules:
        sys.modules[_mod] = _make_stub(_mod)

# Wire sub-module attributes onto parent stubs (Python does this for real packages)
def _wire(parent, child_attr, child_mod_key):
    p = sys.modules.get(parent)
    if p is not None and not hasattr(p, child_attr):
        setattr(p, child_attr, sys.modules.get(child_mod_key, _make_stub(child_mod_key)))

for _p, _c, _k in [
    ("gpytorch", "models",        "gpytorch.models"),
    ("gpytorch", "likelihoods",   "gpytorch.likelihoods"),
    ("gpytorch", "kernels",       "gpytorch.kernels"),
    ("gpytorch", "means",         "gpytorch.means"),
    ("gpytorch", "distributions", "gpytorch.distributions"),
    ("gpytorch", "mlls",          "gpytorch.mlls"),
    ("gpytorch", "constraints",   "gpytorch.constraints"),
    ("gpytorch", "settings",      "gpytorch.settings"),
    ("torch",    "optim",         "torch.optim"),
    ("matplotlib", "pyplot",      "matplotlib.pyplot"),
    ("matplotlib", "lines",       "matplotlib.lines"),
    ("matplotlib", "cm",          "matplotlib.cm"),
]:
    _wire(_p, _c, _k)

# gpytorch class stubs needed for subclassing _ExactGP
_gp_models = sys.modules["gpytorch.models"]
_gp_models.ExactGP = type("ExactGP", (), {
    "__init__": lambda self, *a, **kw: None,
    "forward":  lambda self, x: None,
    "train":    lambda self, *a: self,
    "eval":     lambda self, *a: self,
    "parameters": lambda self: iter([]),
    "to":       lambda self, *a, **kw: self,
})
sys.modules["gpytorch.likelihoods"].GaussianLikelihood = mock.MagicMock(
    return_value=mock.MagicMock(train=lambda *a: None, eval=lambda *a: None,
                                to=lambda *a, **kw: mock.MagicMock()))
sys.modules["gpytorch.kernels"].RBFKernel     = mock.MagicMock
sys.modules["gpytorch.kernels"].ScaleKernel   = mock.MagicMock
sys.modules["gpytorch.means"].ConstantMean    = mock.MagicMock
sys.modules["gpytorch.mlls"].ExactMarginalLogLikelihood = mock.MagicMock
sys.modules["gpytorch.constraints"].GreaterThan = mock.MagicMock
sys.modules["gpytorch.distributions"].MultivariateNormal = mock.MagicMock

# Minimal torch shims
_torch = sys.modules["torch"]
_torch.device   = lambda x: x
_torch.tensor   = mock.MagicMock()
_torch.no_grad  = mock.MagicMock(return_value=mock.MagicMock(
    __enter__=lambda s, *a: s, __exit__=lambda s, *a: None))
_torch.cuda     = mock.MagicMock()
_torch.cuda.is_available = lambda: False
_torch.float32  = None
_torch.Tensor   = type("Tensor", (), {})   # scipy checks torch.Tensor at import

# tqdm shim: just iterate
import functools
sys.modules["tqdm"].tqdm = lambda it, *a, **kw: it

# matplotlib shims for module-level rcParams and plt usage
import numpy as _np
class _RCParams(dict):
    def update(self, *a, **kw):
        pass

_mpl = sys.modules["matplotlib"]
_mpl.use      = lambda *a, **kw: None
_mpl.rcParams = _RCParams()
_plt = sys.modules["matplotlib.pyplot"]
_plt.subplots = mock.MagicMock(return_value=(mock.MagicMock(), mock.MagicMock()))
_plt.figure   = mock.MagicMock()
_plt.close    = mock.MagicMock()
_plt.savefig  = mock.MagicMock()
_plt.cm       = mock.MagicMock()
_plt.rcParams = _RCParams()
sys.modules["matplotlib.lines"].Line2D = mock.MagicMock()

import numpy as np
import pandas as pd
import pytest

# NumPy < 2.0 compat: production code uses np.trapezoid (added in NumPy 2.0)
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

# Allow importing the pipeline from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Minimal mock of config so the import doesn't fail without config.yaml ──
_CFG = {
    "paths": {"dataset_dir": "Dataset", "output_dir": "cdi_output_test"},
    "data": {
        "concentration_file": "dummy.xlsx",
        "flow_file": "dummy.xlsx",
        "potential_file": "dummy.xlsx",
        "sheet_meta": {
            "concentration": {"fixed_flow": 3, "fixed_potential": 1.8},
            "flow":          {"fixed_conc": 1000, "fixed_potential": 1.8},
            "potential":     {"fixed_conc": 1000, "fixed_flow": 3},
        },
    },
    "gp":          {"n_iter": 5, "lr": 0.05, "noise_constraint": 1e-4,
                    "grid_search": {"enabled": False}},
    "calibration": {"nominal_z": 1.645, "target_low": 0.85, "target_high": 0.95,
                    "alpha_search": [0.05, 20.0, 0.05], "nested": False},
    "pareto":      {"grid_points": 5, "sec_model": "two_stage_gp",
                    "r_kappa_model": "two_stage_gp"},
    "figures":     {"dpi": 72, "format": "png"},
    "physics":     {"epsilon": 1e-9, "lk_mc_samples": 5,
                    "pareto_constraints": {"r_kappa_min": 0.0, "sec_max": 100.0}},
    "prediction_mode": "retrospective",
}

_real_open = open

def _open_intercept(path, *args, **kwargs):
    if str(path) == "config.yaml":
        return mock.mock_open(read_data="")()
    return _real_open(path, *args, **kwargs)

with mock.patch("builtins.open", side_effect=_open_intercept), \
     mock.patch("yaml.safe_load", return_value=_CFG), \
     mock.patch("pathlib.Path.mkdir"):
    import cdi_pipeline as pipe

from pathlib import Path
pipe.CFG = _CFG
pipe.OUT = Path("cdi_output_test")
pipe.EPS = 1e-9
os.makedirs("cdi_output_test", exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_timeseries(n_cycles=3, duration=60, rng=None):
    """Synthetic desal+regen timeseries."""
    if rng is None:
        rng = np.random.default_rng(0)
    t, kappa, cur = [], [], []
    tick = 0
    for _ in range(n_cycles):
        for phase, sign in [("desal", 1), ("regen", -1)]:
            for k in range(duration):
                t.append(float(tick)); tick += 1
                kappa.append(2.0 - sign * 0.5 * (k / duration))
                cur.append(sign * (10.0 + rng.normal(0, 0.1)))
    return pd.DataFrame({"time": t, "conductivity": kappa, "current": cur})


def _make_cycle_df(n_conds=4, n_cycles_each=3):
    """Synthetic cycle-level DataFrame with the full column set."""
    rng  = np.random.default_rng(42)
    rows = []
    conds = [(1000,3,1.8),(2500,3,1.8),(1000,2,1.8),(1000,3,1.2)][:n_conds]
    for (conc, flow, pot) in conds:
        for cid in range(n_cycles_each):
            rows.append(dict(
                conc=conc, flow=flow, potential=pot, cycle_id=cid,
                duration_s=60.0,
                charge_c=rng.uniform(0.5, 2.0),
                energy_kwh=rng.uniform(1e-5, 5e-5),
                volume_m3=rng.uniform(1e-5, 5e-5),
                kappa_0=rng.uniform(1.5, 3.0),
                kappa_min=rng.uniform(0.5, 1.5),
                delta_kappa_peak=rng.uniform(0.3, 1.0),
                delta_kappa_integral=rng.uniform(10, 50),
                lambda_kappa=rng.uniform(0.001, 0.01),
                r_kappa_pos=rng.uniform(0.1, 0.9),
                sec_wh_m3=rng.uniform(0.01, 0.5),
                w_net_wh=rng.uniform(0.01, 0.05),
                initial_conductivity=rng.uniform(1.5, 3.0),
                initial_current=rng.uniform(5.0, 15.0),
                initial_resistance=rng.uniform(50, 200),
                inverse_flow_proxy=1.0 / (flow * 1e-6 / 60.0),
            ))
    return pd.DataFrame(rows)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCycleSegmentation:
    def test_correct_number_of_cycles(self):
        ts     = _make_timeseries(n_cycles=3)
        cycles = pipe.segment_cycles(ts)
        assert len(cycles) == 3, f"Expected 3 cycles, got {len(cycles)}"

    def test_desal_current_positive(self):
        ts     = _make_timeseries(n_cycles=2)
        cycles = pipe.segment_cycles(ts)
        for desal, regen in cycles:
            assert desal["current"].mean() > 0
            assert regen["current"].mean() < 0

    def test_single_phase_returns_empty(self):
        t = np.arange(100, dtype=float)
        df = pd.DataFrame({"time": t, "conductivity": np.ones(100)*2.0,
                           "current": np.ones(100)*10.0})
        cycles = pipe.segment_cycles(df)
        assert len(cycles) == 0


class TestMetricsExtraction:
    def setup_method(self):
        self.ts     = _make_timeseries(n_cycles=1)
        self.cycles = pipe.segment_cycles(self.ts)
        assert len(self.cycles) == 1
        self.desal, self.regen = self.cycles[0]

    def test_returns_dict(self):
        m = pipe._metrics(self.desal, self.regen, 1000, 3.0, 1.8, 0)
        assert m is not None and isinstance(m, dict)

    def test_sec_positive(self):
        m = pipe._metrics(self.desal, self.regen, 1000, 3.0, 1.8, 0)
        assert m["sec_wh_m3"] > 0

    def test_r_kappa_bounded(self):
        m = pipe._metrics(self.desal, self.regen, 1000, 3.0, 1.8, 0)
        assert 0.0 <= m["r_kappa_pos"] <= 1.0

    def test_lambda_kappa_positive(self):
        m = pipe._metrics(self.desal, self.regen, 1000, 3.0, 1.8, 0)
        assert m["lambda_kappa"] > 0

    def test_inverse_flow_proxy_present(self):
        m = pipe._metrics(self.desal, self.regen, 1000, 3.0, 1.8, 0)
        assert "inverse_flow_proxy" in m
        assert "residence_time" not in m

    def test_short_desal_returns_none(self):
        short = self.desal.iloc[:1]
        assert pipe._metrics(short, self.regen, 1000, 3.0, 1.8, 0) is None


class TestTransforms:
    def test_logit_sigmoid_roundtrip(self):
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(pipe._sigmoid(pipe._logit(x)), x, atol=1e-6)

    def test_sigmoid_bounded(self):
        z = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        s = pipe._sigmoid(z)
        # Strict bounds only hold for finite z away from ±∞
        assert np.all(s > 0) and np.all(s < 1)

    def test_logit_clips_extremes(self):
        # Should not produce inf
        x = np.array([0.0, 1.0])
        result = pipe._logit(x)
        assert np.all(np.isfinite(result))


class TestViolationRate:
    def test_all_valid(self):
        assert pipe._violation_rate(np.array([0.1, 0.5, 0.9]), lo=0.0, hi=1.0) == 0.0

    def test_all_violated(self):
        assert pipe._violation_rate(np.array([-0.1, 1.1]), lo=0.0, hi=1.0) == 1.0

    def test_half_violated(self):
        v = pipe._violation_rate(np.array([-0.1, 0.5]), lo=0.0)
        assert abs(v - 0.5) < 1e-9


class TestOOFChargePredictions:
    def test_no_nan_in_output(self):
        df   = _make_cycle_df(n_conds=4, n_cycles_each=3)
        oof  = pipe._oof_charge_predictions(df)
        assert len(oof) == len(df)
        assert np.all(np.isfinite(oof))

    def test_single_condition_fallback(self):
        df   = _make_cycle_df(n_conds=1, n_cycles_each=5)
        oof  = pipe._oof_charge_predictions(df)
        expected = np.log(df["charge_c"].values + pipe.EPS)
        np.testing.assert_allclose(oof, expected)


class TestConditionLevelMetrics:
    def test_one_row_per_condition(self):
        preds = _make_cycle_df(n_conds=4, n_cycles_each=3).copy()
        # Add prediction columns
        preds["lk_pred"]  = preds["lambda_kappa"]  + np.random.default_rng(0).normal(0, 0.001, len(preds))
        preds["sec_pred"] = preds["sec_wh_m3"]      + np.random.default_rng(1).normal(0, 0.01,  len(preds))
        preds["rkp_pred"] = preds["r_kappa_pos"]    + np.random.default_rng(2).normal(0, 0.01,  len(preds))
        cdf = pipe.condition_level_metrics(preds)
        assert len(cdf) == 4

    def test_mae_non_negative(self):
        preds = _make_cycle_df(n_conds=3, n_cycles_each=4).copy()
        preds["lk_pred"]  = preds["lambda_kappa"]
        preds["sec_pred"] = preds["sec_wh_m3"]
        preds["rkp_pred"] = preds["r_kappa_pos"]
        cdf = pipe.condition_level_metrics(preds)
        for col in ["lk_mae", "sec_mae", "rkp_mae"]:
            assert (cdf[col] >= 0).all()


class TestFeatureSets:
    def test_feat_s1_is_operating(self):
        assert pipe.FEAT_S1 is pipe.FEAT_OPERATING

    def test_feat_phys_is_precycle(self):
        assert pipe.FEAT_PHYS is pipe.FEAT_PRECYCLE

    def test_inverse_flow_proxy_in_precycle(self):
        assert "inverse_flow_proxy" in pipe.FEAT_PRECYCLE
        assert "residence_time" not in pipe.FEAT_PRECYCLE


class TestAblationModelOrder:
    def test_14_models(self):
        assert len(pipe._ABL_MODEL_ORDER) == 14

    def test_oof_suffix_in_names(self):
        assert "physics_sec_ridge_residual_oof" in pipe._ABL_MODEL_ORDER
        assert "physics_sec_gp_residual_oof"    in pipe._ABL_MODEL_ORDER

    def test_new_r_kappa_baselines(self):
        assert "ridge_clipped_r_kappa" in pipe._ABL_MODEL_ORDER
        assert "ridge_logit_r_kappa"   in pipe._ABL_MODEL_ORDER

    def test_bounded_r_kappa_gp_renamed(self):
        assert "bounded_r_kappa_gp" in pipe._ABL_MODEL_ORDER
        assert "bounded_r_kappa"    not in pipe._ABL_MODEL_ORDER
