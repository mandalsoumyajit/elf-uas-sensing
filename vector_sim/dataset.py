"""Orientation datasets with conditioned in-band SNR, not fixed receiver ASD.
A white-noise draw is scaled by its realized filtered power across ALL axes.
The scaling conditions on SNR; absolute-ASD studies use add_sensor_noise.
Fixed-bearing range mapping: +20 dB at 1 m -> -8.63 dB at 3 m.
"""

import numpy as np
from forward_model import (
    QuadrotorConfig,
    simulate_B_timeseries,
    tilt_magnitude,
    add_power_line_emi,
    DEFAULT_FS,
    DEFAULT_DURATION,
    DEFAULT_BW,
)
from features import bandpass, all_features


def add_snr_controlled_noise(B, fs, snr_db, f_center, bw_hz=DEFAULT_BW, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if np.isposinf(snr_db):
        return B.copy()
    Bf = bandpass(B, fs, f_center - bw_hz / 2, f_center + bw_hz / 2)
    N = rng.normal(size=B.shape)
    Nf = bandpass(N, fs, f_center - bw_hz / 2, f_center + bw_hz / 2)
    sp, npower = np.mean(Bf**2), np.mean(Nf**2)
    if sp <= 0:
        raise ValueError("Cannot condition SNR of zero signal")
    return B + N * np.sqrt(sp / npower / 10 ** (snr_db / 10))


def make_sample(
    cfg, roll, pitch, yaw, R, az_cone_deg, snr_db, fs, dur, bw, rng, emi_nT=0.0
):
    az = rng.uniform(-np.deg2rad(az_cone_deg), np.deg2rad(az_cone_deg))
    pos = (R * np.cos(az), R * np.sin(az), 0.3)
    _, B = simulate_B_timeseries(cfg, pos, roll, pitch, yaw, fs, dur, rng=rng)
    B = add_snr_controlled_noise(B, fs, snr_db, cfg.feature_center_hz(), bw, rng)
    if emi_nT:
        B = add_power_line_emi(B, fs, emi_nT, rng)
    return all_features(B, fs, cfg.feature_center_hz(), bw), np.array(
        [roll, pitch, yaw, *pos]
    )


def build_dataset(
    n_samples,
    snr_db,
    seed=0,
    cfg=None,
    emi_nT=0.0,
    fs=DEFAULT_FS,
    dur=DEFAULT_DURATION,
    bw=DEFAULT_BW,
    range_min=1.0,
    range_max=4.0,
    az_cone_deg=15.0,
):
    cfg = QuadrotorConfig() if cfg is None else cfg
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ["vector", "total", "x", "y", "z"]}
    y, meta = [], []
    for _ in range(n_samples):
        roll, pitch = rng.uniform(-np.pi / 4, np.pi / 4, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        R = rng.uniform(range_min, range_max)
        feat, m = make_sample(
            cfg, roll, pitch, yaw, R, az_cone_deg, snr_db, fs, dur, bw, rng, emi_nT
        )
        for key in rows:
            rows[key].append(feat[key])
        y.append(tilt_magnitude(roll, pitch))
        meta.append(m)
    return (
        {key: np.asarray(v) for key, v in rows.items()},
        np.asarray(y),
        np.asarray(meta),
    )


def build_orientation_dataset(cfg, n_samples, snr_db, seed=0, **kwargs):
    """Compatibility wrapper: scalar is now z loop; target is true tilt."""
    X, y, meta = build_dataset(n_samples, snr_db, seed, cfg, **kwargs)
    return X["vector"], X["z"], y, np.linalg.norm(meta[:, 3:5], axis=1)
