"""Quasistatic point-rotor model v2. Parameters are hypotheses, not measured fits."""

import numpy as np

MU0 = 4 * np.pi * 1e-7
DEFAULT_FS, DEFAULT_DURATION, DEFAULT_BW = 100_000.0, 0.2, 100.0


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_body_to_world(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def tilt_magnitude(roll, pitch):
    """Inclination of body +z from world +z, radians, not Euler norm."""
    return np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1))


def snr_at_range(snr_ref_db, r, r_ref):
    """Common-dipole, fixed-bearing, fixed-noise approximation."""
    if np.any(np.asarray(r) <= 0) or r_ref <= 0:
        raise ValueError("Ranges must be positive")
    return snr_ref_db - 60 * np.log10(np.asarray(r) / r_ref)


class QuadrotorConfig:
    def __init__(
        self,
        arm_length=0.12,
        motor_freqs_hz=None,
        pole_pairs=7,
        pwm_freq_hz=16e3,
        m_rotating=3e-4,
        m_axial=2e-3,
        pwm_depth=0.5,
        axial_frequency_ratio=1.0,
    ):
        self.arm_length = float(arm_length)
        self.motor_freqs_hz = np.asarray(
            [95, 98.5, 92, 101] if motor_freqs_hz is None else motor_freqs_hz,
            dtype=float,
        )
        if self.motor_freqs_hz.shape != (4,) or np.any(self.motor_freqs_hz <= 0):
            raise ValueError("Four positive motor frequencies required")
        self.spin_dirs = np.array([1, -1, 1, -1])
        self.pole_pairs = pole_pairs
        self.pwm_freq_hz = pwm_freq_hz
        self.m_rotating = m_rotating
        self.m_axial = m_axial
        self.pwm_depth = pwm_depth
        self.axial_frequency_ratio = axial_frequency_ratio

    def rotor_positions_body(self):
        a = self.arm_length
        return np.array([[a, 0, 0], [0, a, 0], [-a, 0, 0], [0, -a, 0]])

    def motor_axis_body(self):
        return np.array([0.0, 0.0, 1.0])

    def electrical_fundamental_hz(self, motor_idx=0):
        return float(self.motor_freqs_hz[motor_idx] * self.pole_pairs)

    def feature_center_hz(self):
        f = self.motor_freqs_hz * self.pole_pairs
        return float((f.min() + f.max()) / 2)


def dipole_tensor(displacement):
    """B = tensor @ m; source-to-sensor displacement and SI units."""
    r = np.asarray(displacement, dtype=float)
    norm = np.linalg.norm(r, axis=-1)
    if np.any(norm <= 0):
        raise ValueError("Point sensor cannot coincide with point rotor")
    u = r / norm[..., None]
    return (
        1e-7
        * (3 * u[..., :, None] * u[..., None, :] - np.eye(3))
        / norm[..., None, None] ** 3
    )


def simulate_B_timeseries(
    cfg,
    drone_pos_world,
    roll,
    pitch,
    yaw,
    fs=DEFAULT_FS,
    duration_s=DEFAULT_DURATION,
    sensor_pos_world=(0, 0, 0),
    rng=None,
    phase_offsets=None,
):
    """Return time and B[3,time]. Reuse phases across nodes of one array.

    Sampling must resolve all included PWM sidebands. Both moment components
    rotate rigidly with the body. axial_frequency_ratio is a sensitivity knob.
    """
    fmax = np.max(cfg.motor_freqs_hz * cfg.pole_pairs) * max(
        1, cfg.axial_frequency_ratio
    )
    if cfg.pwm_depth:
        fmax += 2 * cfg.pwm_freq_hz
    if fs <= 2 * fmax:
        raise ValueError(f"fs={fs:g} must exceed {2 * fmax:g} Hz to avoid aliasing")
    rng = np.random.default_rng() if rng is None else rng
    phases = (
        rng.uniform(0, 2 * np.pi, 4)
        if phase_offsets is None
        else np.asarray(phase_offsets)
    )
    if phases.shape != (4,):
        raise ValueError("Four shared rotor phases required")
    t = np.arange(int(round(fs * duration_s))) / fs
    R = R_body_to_world(roll, pitch, yaw)
    rotors = cfg.rotor_positions_body() @ R.T + np.asarray(drone_pos_world)
    tensors = dipole_tensor(np.asarray(sensor_pos_world) - rotors)
    phase = (2 * np.pi * cfg.motor_freqs_hz * cfg.pole_pairs * cfg.spin_dirs)[
        :, None
    ] * t + phases[:, None]
    pwm = 1 + cfg.pwm_depth * (
        np.cos(2 * np.pi * cfg.pwm_freq_hz * t)
        + 0.3 * np.cos(4 * np.pi * cfg.pwm_freq_hz * t)
    )
    moments = (
        np.stack(
            [
                cfg.m_rotating * np.cos(phase),
                cfg.m_rotating * np.sin(phase),
                cfg.m_axial * np.sin(cfg.axial_frequency_ratio * phase),
            ],
            axis=1,
        )
        * pwm
    )
    B = np.einsum("kij,kjt->it", tensors @ R, moments, optimize=True)
    return t, B


def simulate_array(cfg, drone_pos_world, attitude, nodes, rng=None, **kwargs):
    rng = np.random.default_rng() if rng is None else rng
    phases = rng.uniform(0, 2 * np.pi, 4)
    return np.asarray(
        [
            simulate_B_timeseries(
                cfg,
                drone_pos_world,
                *attitude,
                sensor_pos_world=n,
                phase_offsets=phases,
                **kwargs,
            )[1]
            for n in nodes
        ]
    )


def noise_sigma_from_asd(noise_pT_per_rtHz, fs):
    if noise_pT_per_rtHz < 0 or fs <= 0:
        raise ValueError("Nonnegative ASD and positive sample rate required")
    return noise_pT_per_rtHz * 1e-12 * np.sqrt(fs / 2)


def add_sensor_noise(B, fs, noise_pT_per_rtHz=10.0, rng=None):
    """Fixed one-sided per-axis ASD. 10 pT/sqrt(Hz) is a scenario, not a measurement."""
    rng = np.random.default_rng() if rng is None else rng
    return B + rng.normal(0, noise_sigma_from_asd(noise_pT_per_rtHz, fs), B.shape)


def power_line_waveform(n, fs, rng, amp_nT=5.0, power_line_hz=60.0):
    t = np.arange(n) / fs
    phase = rng.uniform(0, 2 * np.pi)
    return (
        amp_nT
        * 1e-9
        * sum(
            a * np.cos(2 * np.pi * power_line_hz * h * t + h * phase)
            for h, a in [(1, 1), (3, 0.3), (5, 0.15), (7, 0.08)]
        )
    )


def add_power_line_emi(B, fs, amp_nT=5.0, rng=None):
    """Shared waveform, unequal channel coupling; toy EMI, not measured urban noise."""
    rng = np.random.default_rng() if rng is None else rng
    wave = power_line_waveform(B.shape[-1], fs, rng, amp_nT)
    coupling = rng.uniform(0.5, 1.0, size=B.shape[:-1])
    return B + coupling[..., None] * wave
