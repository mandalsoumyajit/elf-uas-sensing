"""Study 4: conditional 3-D position inversion at explicit receiver noise ASDs.
Known attitude, frequencies, rotor geometry and moment ratio are supplied to
estimator. Unknown phase/amplitude gains are fitted across nodes. Not joint
pose estimation and not experimental localization. Historical PCA is diagnostic.
"""

import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import plotstyle
from forward_model import (
    QuadrotorConfig,
    simulate_array,
    noise_sigma_from_asd,
    DEFAULT_FS,
)
from features import bandpass, estimate_snr_db
from localization import (
    extract_fundamental_phasors,
    estimate_position,
    pca_line_position,
)
from simulation_utils import arguments, provenance, save_json


def summarize(errors, nside, ncell, seed):
    E = np.asarray(errors).reshape(nside * nside, ncell)
    cells = np.sqrt(np.mean(E**2, axis=1))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        sampled = E[np.arange(len(E))[:, None], rng.integers(ncell, size=E.shape)]
        draws.append(
            [
                np.sqrt(np.mean(sampled**2)),
                np.median(np.sqrt(np.mean(sampled**2, axis=1))),
            ]
        )
    return {
        "pooled_rmse_m": float(np.sqrt(np.mean(E**2))),
        "median_cell_rmse_m": float(np.median(cells)),
        "mean_cell_rmse_m": float(cells.mean()),
        "worst_cell_rmse_m": float(cells.max()),
        "best_cell_rmse_m": float(cells.min()),
        "error_p95_m": float(np.quantile(E, 0.95)),
        "fraction_cells_below_50cm": float(np.mean(cells < 0.5)),
        "pooled_rmse_ci95_m": np.quantile(
            np.array(draws)[:, 0], [0.025, 0.975]
        ).tolist(),
        "median_cell_rmse_ci95_m": np.quantile(
            np.array(draws)[:, 1], [0.025, 0.975]
        ).tolist(),
    }


def main():
    args = arguments(__doc__)
    plotstyle.apply()
    nside, ncell = (8, 8) if not args.quick else (3, 2)
    grid = np.linspace(0.5, 4.5, nside)
    nodes = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0]])
    cfg = QuadrotorConfig()
    fs = DEFAULT_FS
    center = cfg.feature_center_hz()
    asds = [0.0, 0.06338219172543927, 10.0]
    rng = np.random.default_rng(42)
    records = {
        str(a): {
            k: []
            for k in [
                "estimate",
                "pca_estimate",
                "snr_true",
                "snr_estimated",
                "success",
                "boundary",
                "residual",
                "seconds",
            ]
        }
        for a in asds
    }
    truth = []
    attitudes = []
    for ix, x in enumerate(grid):
        for y in grid:
            for _ in range(ncell):
                pos = np.array([x, y, rng.uniform(0.5, 1.5)])
                attitude = np.r_[
                    rng.uniform(-np.pi / 6, np.pi / 6, 2), rng.uniform(-np.pi, np.pi)
                ]
                B = simulate_array(cfg, pos, attitude, nodes, rng=rng)
                Bf = bandpass(B, fs, center - 50, center + 50)
                unit_noise = rng.normal(size=B.shape)
                unit_nf = bandpass(unit_noise, fs, center - 50, center + 50)
                truth.append(pos)
                attitudes.append(attitude)
                for asd in asds:
                    sigma = noise_sigma_from_asd(asd, fs)
                    noisy = B + sigma * unit_noise
                    start = time.perf_counter()
                    Y = extract_fundamental_phasors(noisy, fs, cfg)
                    estimate, info = estimate_position(Y, attitude, nodes, cfg)
                    elapsed = time.perf_counter() - start
                    pca = pca_line_position(Bf + sigma * unit_nf, nodes)
                    true_snr = (
                        10
                        * np.log10(
                            np.mean(Bf**2, axis=(1, 2))
                            / (sigma**2 * np.mean(unit_nf**2, axis=(1, 2)))
                        )
                        if asd
                        else np.full(4, np.inf)
                    )
                    r = records[str(asd)]
                    for k, v in [
                        ("estimate", estimate),
                        ("pca_estimate", pca),
                        ("snr_true", true_snr),
                        (
                            "snr_estimated",
                            [estimate_snr_db(b, fs, center) for b in noisy],
                        ),
                        ("success", info["success"]),
                        ("boundary", info["at_boundary"]),
                        ("residual", info["residual"]),
                        ("seconds", elapsed),
                    ]:
                        r[k].append(v)
        print(f"Position grid row {ix + 1}/{nside} done", flush=True)
    truth = np.array(truth)
    summary = {}
    maps = []
    for asd in asds:
        r = {k: np.array(v) for k, v in records[str(asd)].items()}
        err3 = np.linalg.norm(r["estimate"] - truth, axis=1)
        err2 = np.linalg.norm(r["estimate"][:, :2] - truth[:, :2], axis=1)
        pcaerr = np.linalg.norm(r["pca_estimate"][:, :2] - truth[:, :2], axis=1)
        summary[str(asd)] = {
            "position_3d": summarize(err3, nside, ncell, 10),
            "horizontal_2d": summarize(err2, nside, ncell, 11),
            "historical_pca_horizontal": summarize(pcaerr, nside, ncell, 12),
            "optimizer_success_fraction": float(r["success"].mean()),
            "boundary_fraction": float(r["boundary"].mean()),
            "median_inversion_seconds": float(np.median(r["seconds"])),
            "noise_sample_sigma_pT": float(noise_sigma_from_asd(asd, fs) * 1e12),
        }
        if asd:
            summary[str(asd)]["true_per_node_snr_min_median_max_db"] = np.quantile(
                r["snr_true"], [0, 0.5, 1]
            ).tolist()
        np.savez_compressed(
            args.out / f"study4_asd_{asd:g}.npz",
            truth=truth,
            attitude=attitudes,
            nodes=nodes,
            grid=grid,
            asd_pT_sqrtHz=asd,
            errors_3d=err3,
            errors_2d=err2,
            **r,
        )
        maps.append(np.sqrt(np.mean(err2.reshape(nside, nside, ncell) ** 2, axis=-1)))
    save_json(args.out / "study4_summary.json", summary)
    provenance(
        args.out,
        "study4",
        {
            "n_grid": nside,
            "n_per_cell": ncell,
            "seed": 42,
            "ASD_pT_sqrtHz": asds,
            "height_m": [0.5, 1.5],
            "roll_pitch_deg": [-30, 30],
            "yaw_deg": [-180, 180],
            "fs": 100000,
            "record_s": 0.2,
            "known_to_estimator": [
                "attitude",
                "motor frequencies",
                "rotor geometry",
                "axial/rotating moment ratio",
                "sensor positions",
            ],
            "unknowns": ["position", "shared complex amplitude/phase per rotor"],
            "position_bounds_m": [[0, 0, 0.25], [5, 5, 2.5]],
            "truth_used_for_initialization": False,
            "noise": "one-sided per-axis ASD; same source and paired noise draws across ASD scenarios",
            "CI": "within each fixed grid point, resample eight independent records; not deployment uncertainty",
            "PCA": "unweighted infinite-line diagnostic only; no DoA claim or sign search",
        },
    )
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 3.1), layout="constrained")
    vmax = max(float(m.max()) for m in maps)
    for ax, asd, m in zip(axs, asds, maps):
        im = ax.pcolormesh(
            grid, grid, m.T, shading="nearest", vmin=0, vmax=vmax, cmap="viridis"
        )
        ax.scatter(
            nodes[:, 0], nodes[:, 1], marker="s", c="white", edgecolors="black", s=18
        )
        ax.set(
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{asd:.3g} pT/sqrt(Hz)",
            aspect="equal",
        )
    fig.colorbar(im, ax=axs, label="Horizontal position RMSE (m)", shrink=0.7)
    fig.suptitle("Dipole-model inversion: known attitude and source model", fontsize=10)
    plotstyle.save(fig, "fig_study4_localization", args.out)
    plt.close(fig)
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
