"""Calibrated B features; normalized covariance is only approximately range-stable.
Default: one band covering four motor fundamentals. No invented harmonics.
"""

from functools import lru_cache
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt


@lru_cache(maxsize=64)
def _filter(fs, f_low, f_high, order):
    if not 0 < f_low < f_high < fs / 2:
        raise ValueError("Band must lie strictly inside (0, Nyquist)")
    return butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")


def bandpass(B, fs, f_low, f_high, order=4):
    """Offline zero-phase filter along time; not real-time processing."""
    return sosfiltfilt(_filter(fs, f_low, f_high, order), B, axis=-1)


def component_psd(B, fs):
    f, P = welch(np.atleast_2d(B), fs=fs, nperseg=min(8192, B.shape[-1]), axis=-1)
    return f, P.sum(axis=0)


def detect_motor_harmonic(
    B, fs, f_search_low=200.0, f_search_high=5000.0, power_line_hz=60.0, notch_hz=5.0
):
    """Dominant line and peak/background ratio, not integrated SNR."""
    f, P = component_psd(B, fs)
    mask = (f >= f_search_low) & (f <= f_search_high)
    if notch_hz > 0:
        for h in range(1, int(f_search_high / power_line_hz) + 1):
            mask &= np.abs(f - h * power_line_hz) >= notch_hz
    if not mask.any() or P[mask].max() == 0:
        return 0.0, float("-inf")
    idx = np.flatnonzero(mask)[np.argmax(P[mask])]
    return float(f[idx]), float(
        10 * np.log10(P[idx] / max(np.median(P[mask]), np.finfo(float).tiny))
    )


def covariance_features(Bf):
    """Scale-invariant normalized second moments. Zero axes have zero correlation."""
    scale = np.max(np.abs(Bf))
    if scale == 0:
        return np.array([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0])
    X = Bf / scale
    C = X @ X.T / X.shape[-1]
    P = np.diag(C)
    correlations = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        den = np.sqrt(P[i]) * np.sqrt(P[j])
        correlations.append(np.clip(C[i, j] / den, -1.0, 1.0) if den > 0 else 0.0)
    return np.r_[P / P.sum(), correlations]


def vector_features_known_f(B, fs, f_center, bw_hz=100.0, n_harmonics=1):
    stats = [
        covariance_features(
            bandpass(B, fs, h * f_center - bw_hz / 2, h * f_center + bw_hz / 2)
        )
        for h in range(1, n_harmonics + 1)
    ]
    return np.r_[np.concatenate([s[:3] for s in stats]), stats[0][3:]]


def log_power(B):
    return float(
        np.log10(
            max(np.mean(np.sum(np.atleast_2d(B) ** 2, axis=0)), np.finfo(float).tiny)
        )
    )


def scalar_features_known_f(B, fs, f_center, bw_hz=100.0, n_harmonics=1, axis=2):
    """Fixed one-axis loop, ideal calibrated input-referred B; default z."""
    return np.array(
        [
            log_power(
                bandpass(
                    B[axis : axis + 1],
                    fs,
                    h * f_center - bw_hz / 2,
                    h * f_center + bw_hz / 2,
                )
            )
            for h in range(1, n_harmonics + 1)
        ]
    )


def total_power_features_known_f(B, fs, f_center, bw_hz=100.0):
    return np.array(
        [log_power(bandpass(B, fs, f_center - bw_hz / 2, f_center + bw_hz / 2))]
    )


def all_features(B, fs, f_center, bw_hz=100.0):
    Bf = bandpass(B, fs, f_center - bw_hz / 2, f_center + bw_hz / 2)
    return {
        "vector": covariance_features(Bf),
        "total": np.array([log_power(Bf)]),
        **{axis: np.array([log_power(Bf[i : i + 1])]) for i, axis in enumerate("xyz")},
    }


def estimate_snr_db(B, fs, f_center, bw=100.0):
    """Local mean-noise PSD excess, summed over axes. Not a localization oracle."""
    f, P = component_psd(B, fs)
    inside = np.abs(f - f_center) <= bw / 2
    outside = (np.abs(f - f_center) >= 2 * bw) & (np.abs(f - f_center) <= 4 * bw)
    if not inside.any() or not outside.any():
        return float("nan")
    noise = np.mean(P[outside])
    if noise <= 0:
        return float("inf") if P[inside].sum() else float("nan")
    ratio = np.mean(P[inside]) / noise - 1
    return float(10 * np.log10(ratio)) if ratio > 0 else float("-inf")


def vector_features_autodetect(B, fs, bw_hz=100.0, n_harmonics=1):
    f, _ = detect_motor_harmonic(B, fs)
    if f <= 0:
        return np.r_[np.tile([1 / 3] * 3, n_harmonics), [0.0] * 3]
    return vector_features_known_f(B, fs, f, bw_hz, n_harmonics)
