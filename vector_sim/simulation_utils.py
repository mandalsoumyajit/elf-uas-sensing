"""Shared experiment settings, provenance, models and paired bootstrap summaries."""

import argparse, json, hashlib, platform, sys
from pathlib import Path
import numpy as np
import scipy, sklearn, matplotlib
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

ROOT = Path(__file__).resolve().parent


def arguments(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=Path, default=ROOT / "simulation_results" / "v2")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke run; use a separate output directory",
    )
    a = p.parse_args()
    if a.quick and a.out.resolve() == (ROOT / "simulation_results" / "v2").resolve():
        a.out = ROOT / "simulation_results" / "smoke"
    a.out.mkdir(parents=True, exist_ok=True)
    return a


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def provenance(out, study, settings):
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")
    }
    save_json(
        out / f"{study}_provenance.json",
        {
            "study": study,
            "settings": settings,
            "source_sha256": sources,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    )


def make_model(kind, seed):
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=1
        )
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                max_iter=1200,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
            ),
        ),
        transformer=StandardScaler(),
    )


def rmse_deg(y, p):
    return float(np.rad2deg(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2))))


def paired_summary(y, p, reference=None, seed=42, draws=2000):
    """Resample independent training repetitions, then paired test observations.

    Inputs [training_repeat,test_sample]. Three repetitions give exploratory
    intervals, not a strong characterization of training-population variance.
    """
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    repeats, n = y.shape
    vals = []
    for _ in range(draws):
        r = rng.integers(repeats, size=repeats)
        i = rng.integers(n, size=(repeats, n))
        yt, pt = y[r[:, None], i], p[r[:, None], i]
        value = rmse_deg(yt, pt)
        if reference is not None:
            value = rmse_deg(yt, np.asarray(reference)[r[:, None], i]) - value
        vals.append(value)
    central = (
        rmse_deg(y, p) if reference is None else rmse_deg(y, reference) - rmse_deg(y, p)
    )
    return {
        "estimate_deg": central,
        "ci95_deg": np.quantile(vals, [0.025, 0.975]).tolist(),
    }
