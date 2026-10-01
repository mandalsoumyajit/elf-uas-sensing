"""Position-only dipole inversion, conditional on known attitude and source model.

Four frequency-resolved complex field vectors share one unknown complex gain
per motor across all nodes. Eliminate these gains by variable projection and
fit position in a declared search volume. This is not joint attitude recovery.
"""

from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares
from forward_model import R_body_to_world, dipole_tensor


@lru_cache(maxsize=16)
def _tone_projector(fs, n, electrical_freqs, pwm_freq, include_pwm):
    fundamentals = np.asarray(electrical_freqs)
    freqs = list(fundamentals)
    if include_pwm:
        for h in (1, 2):
            for f in fundamentals:
                freqs.extend([abs(h * pwm_freq - f), h * pwm_freq + f])
    freqs.extend([60.0, 180.0, 300.0, 420.0])
    if max(freqs) >= fs / 2:
        raise ValueError("Tone design aliases")
    if len(np.unique(np.round(freqs, 8))) != len(freqs):
        raise ValueError(
            "Distinct motor/sideband frequencies required for tone separation"
        )
    t = np.arange(n) / fs
    D = np.column_stack(
        [np.ones(n)]
        + [
            col
            for f in freqs
            for col in (np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t))
        ]
    )
    # Keep only fundamental coefficient rows; nuisance tones remove leakage.
    return np.linalg.solve(D.T @ D, D.T)[1:9]


def extract_fundamental_phasors(B, fs, cfg):
    """B[node,axis,time] -> complex phasors[motor,node,axis].

    Electrical frequencies are known here. Other modeled PWM tones, DC and
    four mains harmonics are nuisance regressors, not rectified-magnitude PSDs.
    """
    if cfg.axial_frequency_ratio != 1:
        raise ValueError("Position inversion assumes axial frequency equals f_e")
    P = _tone_projector(
        fs,
        B.shape[-1],
        tuple(cfg.motor_freqs_hz * cfg.pole_pairs),
        cfg.pwm_freq_hz,
        bool(cfg.pwm_depth),
    )
    coeff = np.einsum("ft,nat->fna", P, B, optimize=True)
    return coeff[::2] - 1j * coeff[1::2]


def unit_phasors(position, attitude, nodes, cfg):
    R = R_body_to_world(*attitude)
    rotors = cfg.rotor_positions_body() @ R.T + position
    T = dipole_tensor(np.asarray(nodes)[None, :, :] - rotors[:, None, :])
    body = np.column_stack(
        [
            np.full(4, cfg.m_rotating),
            -1j * cfg.spin_dirs * cfg.m_rotating,
            -1j * cfg.spin_dirs * cfg.m_axial,
        ]
    )
    m = body @ R.T
    return np.einsum("knij,kj->kni", T, m)


def projected_residual(position, Y, attitude, nodes, cfg):
    G = unit_phasors(position, attitude, nodes, cfg).reshape(4, -1)
    y = Y.reshape(4, -1)
    gains = np.sum(G.conj() * y, axis=1) / np.sum(np.abs(G) ** 2, axis=1)
    r = (G * gains[:, None] - y).ravel()
    # Data-derived constant scaling helps optimizer without changing objective.
    scale = max(np.linalg.norm(y), np.finfo(float).tiny)
    return np.r_[r.real, r.imag] / scale


def estimate_position(
    Y, attitude, nodes, cfg, bounds=((0.0, 0.0, 0.25), (5.0, 5.0, 2.5)), max_nfev=100
):
    """Deterministic starts from search volume, never initialized at truth.

    Known attitude is an explicit input. Spatially independent receiver noise
    is assumed equal on each node/axis; no ground-truth SNR weighting is used.
    """
    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    candidates = [
        lo + (hi - lo) * np.array([x, y, z])
        for x in (0.2, 0.5, 0.8)
        for y in (0.2, 0.5, 0.8)
        for z in (0.25, 0.75)
    ]
    ranked = sorted(
        candidates,
        key=lambda x: np.sum(projected_residual(x, Y, attitude, nodes, cfg) ** 2),
    )
    fits = [
        least_squares(
            projected_residual,
            x,
            args=(Y, attitude, nodes, cfg),
            bounds=(lo, hi),
            max_nfev=max_nfev,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        for x in ranked[:3]
    ]
    best = min(fits, key=lambda result: result.cost)
    return best.x, {
        "residual": float(2 * best.cost),
        "success": bool(best.success),
        "nfev": sum(f.nfev for f in fits),
        "at_boundary": bool(np.any(best.x - lo < 1e-3) | np.any(hi - best.x < 1e-3)),
    }


def pca_line_position(B_filtered, nodes):
    """Unweighted historical PCA-line diagnostic; NOT a valid general DoA."""
    dirs = np.array([np.linalg.eigh(np.cov(b))[1][:, -1] for b in B_filtered])
    P = np.eye(3)[None] - dirs[:, :, None] * dirs[:, None, :]
    return np.linalg.lstsq(P.sum(axis=0), np.einsum("nij,nj->i", P, nodes), rcond=None)[
        0
    ]
