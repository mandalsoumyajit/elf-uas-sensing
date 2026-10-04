"""Measurement-informed source and noise models for the vector-node simulations (v7: front-end calibrated).

v7 (2026-10-04) uses the grid data referred to field through the documented analog front end
(review/FRONTEND_CALIBRATION_2026-10-04.md): harmonic ratios, rotor residual and noise shape below replace the v6
values (5th/7th at -21/-25 dB, rotor residual 3.6e-3 A m^2, noise slope f^-1.3), which assumed a flat gain.

Source (5-inch FPV class; drone_source.py nominal, calibrated where the grid data allow):
  - four motors, arm 0.11 m, 7 pole pairs, hover f_e = 1074 Hz (drone_source), per-motor speed offsets
    (-3, 0, +3.5, +7 %) and a common operating-point scale U[0.92, 1.25] (measured lines 1.07-1.39 kHz)
  - frequency wander per motor: slow OU (sd 15 Hz, tau 3.5 s) + fast OU (sd 5 Hz, tau 0.1 s)
    (measured on the grid data: sd 9-28 Hz, tau 1.2-9.5 s, rms change 2.5-14 Hz over 0.2 s)
  - at f_e: body-axial linear moment m_a = 1.06e-3 A m^2 (phase-lead loop, drone_source nominal) and a
    rotating transverse moment m_r = m_a / 1.44 (measured median axial/rotating ratio, grid V2 fit)
  - 5th and 7th harmonics of the axial term at -8.4 and -10.1 dB (drone_source nominal, calibrated on the grid:
    measured 5th/1st -9..+2 dB, 7th/1st -18..-9 dB on three antennas)
  - rotor residual rotating at f_mech, 2.0e-4 A m^2 (drone_source nominal; grid bound ~1.7-2.5e-4 A m^2)
Noise (per node and axis, mutually independent: measured inter-antenna coherence <~0.03):
  - coloured Gaussian background with ASD n(f) = level * max((f/1100)^-0.6, 0.55) (calibrated indoor shape:
    f^-0.6 to ~3 kHz, flat above), front-end high-pass at 300 Hz; level = ASD at 1.1 kHz
  - residual mains comb (50.05 Hz harmonics) after the synchronous comb: each harmonic +10 dB above the
    floor in a 1 Hz bin
  - impulsive bursts (Class A-like): Poisson 15 /s, 0.5 ms Gaussian-windowed bursts, Gamma = 10
    (impulsive power 1/Gamma of the in-band Gaussian power; channel A/C-like fits)
Processing: phase-referenced demodulation of each motor line with its tracked phase (the grid experiment
shows a near node can supply it); tracking is declared feasible for a motor if its strongest node has
SNR >= -1.5 dB in 20 Hz (measured single-node lock threshold).
"""
import numpy as np
from forward_model import R_body_to_world, dipole_tensor, tilt_magnitude

FS = 20000.0
F50 = 50.05


class Src:
    arm = 0.11
    pole_pairs = 7
    f_e0 = 1074.3
    offsets = np.array([-0.030, 0.0, 0.035, 0.070])
    spin = np.array([1, -1, 1, -1])
    m_axial = 1.06e-3
    beta = 1.44
    harm = {5: 10 ** (-8.4 / 20), 7: 10 ** (-10.1 / 20)}
    m_rotor = 2.0e-4
    wander = ((15.0, 3.5), (5.0, 0.1))

    @property
    def m_rot(self):
        return self.m_axial / self.beta

    def rotor_body(self):
        a = self.arm
        return np.array([[a, 0, 0], [0, a, 0], [-a, 0, 0], [0, -a, 0]])


def ou(n, fs, sd, tau, rng, step=0.005):
    """Ornstein-Uhlenbeck path sampled at fs (generated on a coarse grid and linearly interpolated)."""
    m = int(np.ceil(n / fs / step)) + 2
    a = np.exp(-step / tau)
    w = rng.normal(size=m) * sd * np.sqrt(1 - a * a)
    x = np.empty(m); x[0] = rng.normal() * sd
    for i in range(1, m):
        x[i] = a * x[i - 1] + w[i]
    return np.interp(np.arange(n) / fs, np.arange(m) * step, x)


def tracks(n, rng, src=None, scale=None, fs=FS):
    """Instantaneous electrical frequency f[k, t] and phase psi[k, t] of the four motors."""
    src = src or Src()
    scale = rng.uniform(0.92, 1.25) if scale is None else scale
    f0 = src.f_e0 * scale * (1 + src.offsets)
    f = f0[:, None] + sum(np.array([ou(n, fs, sd, tau, rng) for _ in range(4)]) for sd, tau in src.wander)
    psi = 2 * np.pi * np.cumsum(f, axis=1) / fs + rng.uniform(0, 2 * np.pi, (4, 1))
    return f, psi


def moments_world(psi, att, src, rng):
    """m[k, 3, t] in world coordinates."""
    R = R_body_to_world(*att)
    s = src.spin[:, None]
    ax = src.m_axial * np.sin(psi)
    for h, a in src.harm.items():
        ax = ax + a * src.m_axial * np.sin(h * psi + rng.uniform(0, 2 * np.pi, (4, 1)))
    pr = psi / src.pole_pairs + rng.uniform(0, 2 * np.pi, (4, 1))
    mx = src.m_rot * np.cos(psi) + src.m_rotor * np.cos(pr)
    my = src.m_rot * s * np.sin(psi) + src.m_rotor * s * np.sin(pr)
    body = np.stack([mx, my, ax], axis=1)                       # k, 3, t
    return np.einsum('ij,kjt->kit', R, body)


def simulate(pos, att, nodes, T, rng, src=None, fs=FS, scale=None):
    """B[node, 3, t] (tesla) and the motor phases psi[k, t]."""
    src = src or Src()
    n = int(round(T * fs))
    f, psi = tracks(n, rng, src, scale, fs)
    R = R_body_to_world(*att)
    rotors = src.rotor_body() @ R.T + np.asarray(pos)
    D = dipole_tensor(np.asarray(nodes)[:, None, :] - rotors[None, :, :])    # node, k, 3, 3
    m = moments_world(psi, att, src, rng)
    return np.einsum('nkij,kjt->nit', D, m, optimize=True), psi, f


def asd(f, level):
    f = np.maximum(np.asarray(f, float), 1.0)
    return level * np.maximum((f / 1100.0) ** -0.6, 0.55) / np.sqrt(1 + (300.0 / f) ** 4)


def noise(shape, level, rng, fs=FS, mains=True, impulsive=True, gamma=10.0, rate=15.0):
    """Measured-statistics background: shape = (..., n). level = ASD at 1.1 kHz (T/rtHz)."""
    n = shape[-1]
    f = np.fft.rfftfreq(n, 1 / fs)
    X = np.fft.rfft(rng.normal(size=shape), axis=-1) * asd(f, level) * np.sqrt(fs / 2)
    x = np.fft.irfft(X, n, axis=-1)
    t = np.arange(n) / fs
    if mains:
        h = np.arange(1, int(fs / 2 / F50))
        amp = asd(h * F50, level) * np.sqrt(20.0)
        ph = rng.uniform(0, 2 * np.pi, shape[:-1] + (len(h),))
        x = x + np.einsum('h,...ht->...t', amp, np.cos(2 * np.pi * F50 * h[:, None] * t + ph[..., None]), optimize=True)
    if impulsive:
        band = (f > 300) & (f < 9000)
        pg = np.sum(asd(f[band], level) ** 2) * (f[1] - f[0])            # in-band Gaussian power
        L = int(0.002 * fs); w = np.exp(-0.5 * ((np.arange(L) - L / 2) / (0.00025 * fs)) ** 2)
        e1 = np.sum(w ** 2)
        flat = x.reshape(-1, n)
        for row in flat:
            k = rng.poisson(rate * n / fs)
            if k == 0:
                continue
            amp = np.sqrt(pg / gamma * n / (rate * n / fs) / e1)                # mean impulsive power = pg / gamma
            for s in rng.integers(0, max(1, n - L), k):
                row[s:s + L] += amp * rng.exponential() ** 0.5 * w * rng.normal(size=L)
        x = flat.reshape(shape)
    return x


def demod(B, psi):
    """Complex phasor of each motor line on every channel: Y[k, ...] = (2/n) sum B e^{-j psi_k}."""
    E = np.exp(-1j * psi)
    return 2 * np.einsum('...t,kt->k...', B, E, optimize=True) / B.shape[-1]


def phasor_noise_var(level, f):
    """Variance E|w|^2 of a demodulated phasor per channel for unit window: 2 n(f)^2 / T (divide by T)."""
    return 2 * asd(f, level) ** 2


def tracking_ok(Yclean, f_mean, level, scalar_normals=None, thr_db=-1.5, bw=20.0):
    """Per motor: strongest node's line SNR in a 20 Hz tracking band >= threshold (measured lock threshold)."""
    if scalar_normals is None:
        p = 0.5 * np.sum(np.abs(Yclean) ** 2, axis=-1)                          # k, node (sum over axes)
    else:
        p = 0.5 * np.abs(np.einsum('kna,na->kn', Yclean, scalar_normals)) ** 2
    snr = p / (asd(f_mean, level)[:, None] ** 2 * bw)
    return 10 * np.log10(np.max(snr, axis=1)) >= thr_db
