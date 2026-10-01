"""Sensitivity checks, separate from validation on measured or unseen drones."""

import numpy as np
from threadpoolctl import threadpool_limits
from forward_model import QuadrotorConfig, simulate_array, add_sensor_noise, DEFAULT_FS
from dataset import build_dataset
from localization import extract_fundamental_phasors, estimate_position
from simulation_utils import (
    arguments,
    provenance,
    make_model,
    paired_summary,
    save_json,
)


def main():
    args = arguments(__doc__)
    repeats, ntrain, ntest = (3, 500, 300) if not args.quick else (1, 60, 30)
    scenarios = {
        "nominal": (QuadrotorConfig(), 0.0),
        "rotating_moment_half": (QuadrotorConfig(m_rotating=1.5e-4), 0.0),
        "rotating_moment_double": (QuadrotorConfig(m_rotating=6e-4), 0.0),
        "axial_at_half_electrical_frequency": (
            QuadrotorConfig(axial_frequency_ratio=0.5),
            0.0,
        ),
        "motor_speed_80percent": (
            QuadrotorConfig(motor_freqs_hz=np.array([95, 98.5, 92, 101]) * 0.8),
            0.0,
        ),
        "motor_speed_120percent": (
            QuadrotorConfig(motor_freqs_hz=np.array([95, 98.5, 92, 101]) * 1.2),
            0.0,
        ),
        "toy_mains_5nT": (QuadrotorConfig(), 5.0),
    }
    summary = {}
    predictions = {}
    for repeat in range(repeats):
        Xtr, ytr, _ = build_dataset(ntrain, 20, 1000 + repeat)
        models = {
            k: make_model(k, repeat).fit(Xtr["vector"], ytr) for k in ["mlp", "rf"]
        }
        for name, (cfg, emi) in scenarios.items():
            X, y, meta = build_dataset(ntest, 20, 9300 + repeat, cfg, emi_nT=emi)
            row = {"y": y, "meta": meta, "features": X["vector"]}
            for kind, model in models.items():
                row[kind] = model.predict(X["vector"])
            predictions.setdefault(name, []).append(row)
            np.savez_compressed(args.out / f"study5_{name}_repeat{repeat}.npz", **row)
        print(
            f"Orientation sensitivity repeat {repeat + 1}/{repeats} complete",
            flush=True,
        )
    for name, rows in predictions.items():
        y = np.array([row["y"] for row in rows])
        summary[name] = {
            k: paired_summary(y, np.array([row[k] for row in rows]))
            for k in ["mlp", "rf"]
        }
    # Separate position sensitivity: matched positions/phases/noise in each case.
    nodes = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0]])
    npos = 80 if not args.quick else 6
    rng = np.random.default_rng(701)
    nominal = QuadrotorConfig()
    position = {}
    poses = [
        (
            rng.uniform([0.5, 0.5, 0.5], [4.5, 4.5, 1.5]),
            rng.uniform([-0.5, -0.5, -np.pi], [0.5, 0.5, np.pi]),
        )
        for _ in range(npos)
    ]
    for case in [
        "nominal",
        "attitude_offset_5deg",
        "rotating_moment_double",
        "arm_length_plus_2cm",
    ]:
        source = (
            QuadrotorConfig(m_rotating=6e-4)
            if case == "rotating_moment_double"
            else QuadrotorConfig(arm_length=0.14)
            if case == "arm_length_plus_2cm"
            else nominal
        )
        estimates = []
        truth = []
        for i, (pos, attitude) in enumerate(poses):
            r = np.random.default_rng(8000 + i)
            B = simulate_array(source, pos, attitude, nodes, rng=r)
            B = add_sensor_noise(B, DEFAULT_FS, 0.06338219172543927, r)
            Y = extract_fundamental_phasors(B, DEFAULT_FS, nominal)
            supplied_attitude = (
                attitude + np.deg2rad([5, 5, 0])
                if case == "attitude_offset_5deg"
                else attitude
            )
            pred, _ = estimate_position(Y, supplied_attitude, nodes, nominal)
            estimates.append(pred)
            truth.append(pos)
        errors = np.linalg.norm(np.array(estimates) - truth, axis=1)
        position[case] = {
            "pooled_3d_rmse_m": float(np.sqrt(np.mean(errors**2))),
            "median_3d_error_m": float(np.median(errors)),
            "error_p95_m": float(np.quantile(errors, 0.95)),
        }
        np.savez_compressed(
            args.out / f"study5_position_{case}.npz",
            truth=truth,
            predictions=estimates,
            errors=errors,
        )
        print("Position sensitivity", case, position[case], flush=True)
    save_json(
        args.out / "study5_summary.json",
        {"orientation": summary, "conditional_position": position},
    )
    provenance(
        args.out,
        "study5",
        {
            "n_train": ntrain,
            "n_test": ntest,
            "repeats": repeats,
            "snr_db": 20,
            "training": "nominal model only",
            "test_motor_band": "oracle center adjusted for true motor frequencies",
            "EMI": "explicit 60/180/300/420 Hz toy wave; adds interference after SNR conditioning",
            "position_samples": npos,
            "position_ASD_pT_sqrtHz": 0.06338219172543927,
            "position_seed": 701,
            "position_attitude_offset_deg": [5, 5, 0],
            "limitations": "parameter scenarios only; not measured cross-drone or urban validation",
        },
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
