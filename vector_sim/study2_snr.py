"""Study 2: corrected inclination estimation with true one-loop baselines.
Known fundamental band, ideal calibrated axes, conditioned Gaussian-draw SNR.
"""

import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import plotstyle
from dataset import build_dataset
from simulation_utils import (
    arguments,
    provenance,
    make_model,
    paired_summary,
    save_json,
    rmse_deg,
)

MODELS = {
    "vector_mlp": ("vector", "mlp"),
    "x_loop_mlp": ("x", "mlp"),
    "y_loop_mlp": ("y", "mlp"),
    "z_loop_mlp": ("z", "mlp"),
    "total_power_mlp": ("total", "mlp"),
    "vector_rf": ("vector", "rf"),
}


def main():
    args = arguments(__doc__)
    plotstyle.apply()
    snrs = [-5, 0, 5, 10, 15, 20, 25, 30] if not args.quick else [0, 20]
    ntrain, ntest, repeats = (500, 300, 3) if not args.quick else (60, 30, 1)
    provenance(
        args.out,
        "study2",
        {
            "snrs": snrs,
            "n_train": ntrain,
            "n_test": ntest,
            "repeats": repeats,
            "target": "arccos(cos(roll)*cos(pitch))",
            "features": "one 100 Hz band around 675.5 Hz",
            "noise": "per-record conditioned total-vector post-filter SNR",
            "emi_nT": 0,
            "fs": 100000,
            "record_s": 0.2,
            "roll_pitch_deg": [-45, 45],
            "yaw_deg": [-180, 180],
            "horizontal_range_m": [1, 4],
            "azimuth_deg": [-15, 15],
            "height_m": 0.3,
        },
    )
    summary = {}
    for snr in snrs:
        y_all = []
        preds = {k: [] for k in [*MODELS, "mean"]}
        for repeat in range(repeats):
            start = time.perf_counter()
            X, y, meta = build_dataset(ntrain, snr, 1000 + repeat)
            Xt, yt, mt = build_dataset(ntest, snr, 7000 + repeat)
            values = {
                "train_target_rad": y,
                "test_target_rad": yt,
                "train_meta": meta,
                "test_meta": mt,
                **{"train_feature_" + k: v for k, v in X.items()},
                **{"test_feature_" + k: v for k, v in Xt.items()},
            }
            for name, (feature, kind) in MODELS.items():
                model = make_model(kind, repeat).fit(X[feature], y)
                pred = model.predict(Xt[feature])
                preds[name].append(pred)
                values["pred_" + name] = pred
            mean = np.full_like(yt, y.mean())
            preds["mean"].append(mean)
            values["pred_mean"] = mean
            y_all.append(yt)
            np.savez_compressed(
                args.out / f"study2_snr{snr:+d}_repeat{repeat}.npz", **values
            )
            print(
                f"SNR {snr:+d}, repeat {repeat}: vector={rmse_deg(yt, preds['vector_mlp'][-1]):.2f} deg; elapsed {time.perf_counter() - start:.1f}s",
                flush=True,
            )
        yy = np.array(y_all)
        pp = {k: np.array(v) for k, v in preds.items()}
        summary[str(snr)] = {k: paired_summary(yy, p) for k, p in pp.items()}
        summary[str(snr)]["vector_advantage_vs_z_loop"] = paired_summary(
            yy, pp["vector_mlp"], pp["z_loop_mlp"]
        )
        summary[str(snr)]["vector_advantage_vs_mean"] = paired_summary(
            yy, pp["vector_mlp"], pp["mean"]
        )
        save_json(args.out / "study2_summary.json", summary)
    fig, ax = plt.subplots(figsize=(7.16, 3.7), layout="constrained")
    labels = {
        "vector_mlp": "Vector MLP",
        "vector_rf": "Vector random forest",
        "x_loop_mlp": "x loop MLP",
        "y_loop_mlp": "y loop MLP",
        "z_loop_mlp": "z loop MLP",
        "total_power_mlp": "Total 3-axis power MLP",
        "mean": "Training mean",
    }
    for k, label in labels.items():
        est = np.array([summary[str(s)][k]["estimate_deg"] for s in snrs])
        ci = np.array([summary[str(s)][k]["ci95_deg"] for s in snrs])
        # Bootstrap estimates need not be symmetrically located; plot CI as segments.
        line = ax.plot(snrs, est, marker="o", markersize=3, label=label)[0]
        ax.vlines(snrs, ci[:, 0], ci[:, 1], color=line.get_color(), alpha=0.6)
    ax.set(
        xlabel="Conditioned total-vector in-band SNR (dB)",
        ylabel="Inclination RMSE (deg)",
        title="Independent training repeats; exploratory 95% bootstrap intervals",
    )
    ax.legend(ncol=2, fontsize=7)
    ax.grid(alpha=0.3)
    plotstyle.save(fig, "fig_study2_snr", args.out)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
