"""Study 1: test approximate range stability across geometry and motor parameters."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import plotstyle
from forward_model import (
    QuadrotorConfig,
    simulate_B_timeseries,
    add_sensor_noise,
    DEFAULT_FS,
)
from features import all_features
from simulation_utils import arguments, provenance, save_json


def main():
    args = arguments(__doc__)
    plotstyle.apply()
    rng = np.random.default_rng(71)
    ranges = np.geomspace(0.5, 10, 24 if not args.quick else 6)
    count = 64 if not args.quick else 5
    ratios = []
    point = []
    parameters = []
    for case in range(count):
        roll, pitch = rng.uniform(-np.pi / 4, np.pi / 4, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        az = rng.uniform(-np.pi, np.pi)
        el = rng.uniform(0, np.pi / 4)
        direction = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        phases = rng.uniform(0, 2 * np.pi, 4)
        ratio = 10 ** rng.uniform(-1, 1)
        motor_freqs = np.array([95, 98.5, 92, 101]) * rng.uniform(0.85, 1.15)
        cfg = QuadrotorConfig(m_rotating=2e-3 * ratio, motor_freqs_hz=motor_freqs)
        point_cfg = QuadrotorConfig(
            arm_length=0, m_rotating=2e-3 * ratio, motor_freqs_hz=motor_freqs
        )
        center = cfg.feature_center_hz()
        bw = float(np.ptp(motor_freqs * 7) + 30)
        rows = []
        prows = []
        for r in ranges:
            for model, dest in [(cfg, rows), (point_cfg, prows)]:
                _, B = simulate_B_timeseries(
                    model, r * direction, roll, pitch, yaw, phase_offsets=phases
                )
                dest.append(all_features(B, DEFAULT_FS, center, bw)["vector"][:3])
        ratios.append(rows)
        point.append(prows)
        parameters.append([roll, pitch, yaw, az, el, ratio, *motor_freqs, *phases])
    ratios, point = np.asarray(ratios), np.asarray(point)
    deviation = np.max(np.abs(ratios - ratios[:, -1:, :]), axis=-1)
    point_deviation = np.max(np.abs(point - point[:, -1:, :]), axis=-1)
    cfg = QuadrotorConfig()
    phases = np.arange(4)
    pitches = np.linspace(-45, 45, 31)
    power = []
    pitch_ratio = []
    for pitch in pitches:
        _, B = simulate_B_timeseries(
            cfg, (2, 0.5, 0.3), 0, np.deg2rad(pitch), 0, phase_offsets=phases
        )
        feat = all_features(B, DEFAULT_FS, cfg.feature_center_hz())
        power.append([feat[k][0] for k in ["x", "y", "z", "total"]])
        pitch_ratio.append(feat["vector"][:3])
    power = np.array(power)
    clean_power = []
    noisy_ratios = []
    clean_ratios = []
    for r in ranges:
        _, B = simulate_B_timeseries(
            cfg, (r, 0, 0), 0, np.pi / 6, 0, phase_offsets=phases
        )
        feat = all_features(B, DEFAULT_FS, cfg.feature_center_hz())
        clean_power.append(feat["total"][0])
        clean_ratios.append(feat["vector"][:3])
        noisy_ratios.append(
            all_features(
                add_sensor_noise(B, DEFAULT_FS, 10, rng),
                DEFAULT_FS,
                cfg.feature_center_hz(),
            )["vector"][:3]
        )
    clean_power = np.array(clean_power)
    np.savez_compressed(
        args.out / "study1_results.npz",
        ranges=ranges,
        ratios=ratios,
        point_ratios=point,
        parameters=parameters,
        pitches=pitches,
        pitch_ratios=pitch_ratio,
        pitch_log_power=power,
        clean_log_power=clean_power,
        clean_ratios=clean_ratios,
        noisy_ratios=noisy_ratios,
    )
    summary = {
        "cases": count,
        "range_min_m": float(ranges[0]),
        "range_max_m": float(ranges[-1]),
        "max_absolute_ratio_deviation_from_10m": float(deviation.max()),
        "point_source_max_deviation": float(point_deviation.max()),
        "median_case_max_ratio_deviation": float(np.median(deviation.max(axis=1))),
        "max_deviation_ranges_ge_1m": float(deviation[:, ranges >= 1].max()),
        "pitch_power_span_db": dict(
            zip(["x", "y", "z", "total"], (10 * np.ptp(power, axis=0)).tolist())
        ),
        "clean_power_drop_db": float(10 * (clean_power[0] - clean_power[-1])),
    }
    save_json(args.out / "study1_summary.json", summary)
    provenance(
        args.out,
        "study1",
        {
            "seed": 71,
            "cases": count,
            "range_definition": "radial distance with fixed bearing per case",
            "phase_policy": "same four phases along each range curve",
            "motor_rotation_to_axial_ratio": [0.1, 10],
            "RPM_scale": [0.85, 1.15],
            "elevation_deg": [0, 45],
            "roll_pitch_deg": [-45, 45],
            "fs": 100000,
            "duration_s": 0.2,
            "fixed_noise_example_ASD_pT_sqrtHz": 10,
        },
    )
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.6), layout="constrained")
    for i, k in enumerate(["x loop", "y loop", "z loop", "Total 3-axis"]):
        axs[0, 0].plot(
            pitches, 10 * (power[:, i] - power[len(pitches) // 2, i]), label=k
        )
    axs[0, 0].set(
        xlabel="Pitch (deg)",
        ylabel="Power relative to level (dB)",
        title="(a) Even one loop can depend on attitude",
    )
    axs[0, 0].legend(fontsize=6)
    q = np.quantile(deviation, [0.1, 0.5, 0.9], axis=0)
    axs[0, 1].fill_between(ranges, q[0], q[2], alpha=0.2, label="10th-90th percentile")
    axs[0, 1].plot(ranges, q[1], label="Median case")
    axs[0, 1].plot(ranges, deviation.max(axis=0), ls="--", label="Worst case")
    axs[0, 1].set(
        xscale="log",
        xlabel="Radial range (m)",
        ylabel="Max axis-ratio change from 10 m",
        title="(b) Separated rotors: approximate stability",
    )
    axs[0, 1].legend(fontsize=6)
    axs[1, 0].plot(
        ranges, 10 * (clean_power - clean_power[0]), label="Separated-rotor simulation"
    )
    axs[1, 0].plot(
        ranges,
        -60 * np.log10(ranges / ranges[0]),
        ls="--",
        label="Common dipole reference",
    )
    axs[1, 0].set(
        xscale="log",
        xlabel="Radial range (m)",
        ylabel="Total power change (dB)",
        title="(c) Clean fixed-attitude range sweep",
    )
    axs[1, 0].legend(fontsize=6)
    for i, k in enumerate("xyz"):
        axs[1, 1].plot(ranges, np.asarray(noisy_ratios)[:, i], marker=".", label=k)
    axs[1, 1].axhline(1 / 3, color="gray", ls="--")
    axs[1, 1].set(
        xscale="log",
        xlabel="Radial range (m)",
        ylabel="Measured axis-power ratio",
        title="(d) Fixed 10 pT/sqrt(Hz) receiver scenario",
    )
    axs[1, 1].legend(fontsize=6)
    for ax in axs.flat:
        ax.grid(alpha=0.25)
    plotstyle.save(fig, "fig_study1_sweeps", args.out)
    plt.close(fig)
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
