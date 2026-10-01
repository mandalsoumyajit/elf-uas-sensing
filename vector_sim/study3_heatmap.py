"""Study 3: inclination error maps; compare learned estimators, not physical limits."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import plotstyle
from forward_model import (
    QuadrotorConfig,
    tilt_magnitude,
    DEFAULT_FS,
    DEFAULT_DURATION,
    DEFAULT_BW,
)
from dataset import build_dataset, make_sample
from simulation_utils import arguments, provenance, make_model, save_json


def main():
    args = arguments(__doc__)
    plotstyle.apply()
    ntrain, ncell, nside, repeats = (
        (800, 12, 11, 3) if not args.quick else (60, 2, 3, 1)
    )
    grid = np.linspace(-40, 40, nside)
    cfg = QuadrotorConfig()
    snrs = [20, 60] if not args.quick else [20]
    result = {}
    summary = {}
    for snr in snrs:
        rng = np.random.default_rng(9999)
        X = []
        y = []
        meta = []
        for pitch in grid:
            for roll in grid:
                for _ in range(ncell):
                    r, p = np.deg2rad([roll, pitch])
                    yaw = rng.uniform(-np.pi, np.pi)
                    R = rng.uniform(1, 4)
                    f, m = make_sample(
                        cfg,
                        r,
                        p,
                        yaw,
                        R,
                        15,
                        snr,
                        DEFAULT_FS,
                        DEFAULT_DURATION,
                        DEFAULT_BW,
                        rng,
                    )
                    X.append(f["vector"])
                    y.append(tilt_magnitude(r, p))
                    meta.append(m)
        X, y = np.array(X), np.array(y)
        for kind in ["mlp", "rf"]:
            predictions = []
            for repeat in range(repeats):
                Xtr, ytr, _ = build_dataset(ntrain, snr, 1234 + repeat)
                model = make_model(kind, repeat).fit(Xtr["vector"], ytr)
                predictions.append(model.predict(X))
            errors = np.rad2deg(np.array(predictions) - y)
            maps = errors.reshape(repeats, nside, nside, ncell)
            err = np.sqrt(np.mean(maps**2, axis=(0, 3)))
            bias = np.mean(maps, axis=(0, 3))
            key = f"{kind}_{snr}dB"
            result[key + "_predictions"] = predictions
            result[key + "_rmse"] = err
            result[key + "_bias"] = bias
            summary[key] = {
                "pooled_rmse_deg": float(np.sqrt(np.mean(errors**2))),
                "center_rmse_deg": float(err[nside // 2, nside // 2]),
                "min_cell_rmse_deg": float(err.min()),
                "max_cell_rmse_deg": float(err.max()),
                "bias_range_deg": [float(bias.min()), float(bias.max())],
            }
            print(key, summary[key], flush=True)
        result[f"test_X_{snr}"] = X
        result[f"test_y_{snr}"] = y
        result[f"test_meta_{snr}"] = meta
    np.savez_compressed(args.out / "study3_results.npz", grid_deg=grid, **result)
    save_json(args.out / "study3_summary.json", summary)
    provenance(
        args.out,
        "study3",
        {
            "n_train": ntrain,
            "n_per_cell": ncell,
            "n_grid": nside,
            "training_repeats": repeats,
            "snr_db": snrs,
            "models": ["MLP", "random forest"],
            "target": "true body-z inclination",
            "test_seed": 9999,
            "train_seeds": [1234 + i for i in range(repeats)],
            "fs": 100000,
            "record_s": 0.2,
            "note": "60 dB tests estimator behavior; neither map proves physical identifiability",
        },
    )
    fig, axs = plt.subplots(
        len(snrs), 2, figsize=(7.16, 3 * len(snrs)), squeeze=False, layout="constrained"
    )
    roll, pitch = np.meshgrid(grid, grid)
    tilt = np.rad2deg(tilt_magnitude(np.deg2rad(roll), np.deg2rad(pitch)))
    vmax = max(
        float(result[f"{k}_{s}dB_rmse"].max()) for s in snrs for k in ["mlp", "rf"]
    )
    for i, snr in enumerate(snrs):
        for j, kind in enumerate(["mlp", "rf"]):
            ax = axs[i, j]
            im = ax.pcolormesh(
                grid,
                grid,
                result[f"{kind}_{snr}dB_rmse"],
                shading="nearest",
                vmin=0,
                vmax=vmax,
                cmap="viridis",
            )
            ax.contour(
                roll,
                pitch,
                tilt,
                levels=[10, 20, 30, 40],
                colors="white",
                linewidths=0.5,
                linestyles="--",
            )
            ax.set(
                xlabel="Roll (deg)",
                ylabel="Pitch (deg)",
                title=f"{kind.upper()}: {snr:+d} dB conditioned SNR",
                aspect="equal",
            )
            fig.colorbar(im, ax=ax, label="Inclination RMSE (deg)", shrink=0.8)
    plotstyle.save(fig, "fig_study3_heatmap", args.out)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
