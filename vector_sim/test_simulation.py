"""Regression checks against independent physical identities and known signals."""

import unittest
import numpy as np
from forward_model import (
    QuadrotorConfig,
    simulate_B_timeseries,
    snr_at_range,
    noise_sigma_from_asd,
    add_sensor_noise,
    rot_z,
    simulate_array,
    tilt_magnitude,
    dipole_tensor,
    DEFAULT_FS,
)
from features import (
    covariance_features,
    scalar_features_known_f,
    bandpass,
    component_psd,
    detect_motor_harmonic,
)
from dataset import add_snr_controlled_noise
from localization import extract_fundamental_phasors, estimate_position


class PhysicsTests(unittest.TestCase):
    def test_axial_frequency_and_alias_guard(self):
        cfg = QuadrotorConfig(motor_freqs_hz=[95] * 4, m_rotating=0, pwm_depth=0)
        t, B = simulate_B_timeseries(
            cfg, (2, 0, 0), 0, 0, 0, fs=10000, duration_s=2, phase_offsets=np.zeros(4)
        )
        f = np.fft.rfftfreq(len(t), 1 / 10000)
        self.assertAlmostEqual(f[np.argmax(abs(np.fft.rfft(B[2])))], 665)
        with self.assertRaises(ValueError):
            simulate_B_timeseries(QuadrotorConfig(), (2, 0, 0), 0, 0, 0, fs=50000)

    def test_point_dipole_scaling_and_snr(self):
        cfg = QuadrotorConfig(arm_length=0, pwm_depth=0)
        args = dict(roll=0.2, pitch=0.3, yaw=0.7, fs=10000, phase_offsets=np.arange(4))
        _, b1 = simulate_B_timeseries(cfg, (1, 0, 0), **args)
        _, b3 = simulate_B_timeseries(cfg, (3, 0, 0), **args)
        np.testing.assert_allclose(b1, b3 * 27, rtol=1e-12, atol=1e-25)
        self.assertAlmostEqual(snr_at_range(20, 3, 1), -8.627275283, places=7)

    def test_correlations_are_amplitude_invariant(self):
        t = np.arange(10000) / 10000
        x = np.cos(2 * np.pi * 665 * t)
        B = np.array([x, -2 * x, 3 * x]) * 1e-12
        f = covariance_features(B)
        np.testing.assert_allclose(f[:3], [1 / 14, 4 / 14, 9 / 14])
        np.testing.assert_allclose(f[3:], [-1, 1, -1])
        np.testing.assert_allclose(f, covariance_features(B * 1e-12), atol=1e-12)
        np.testing.assert_allclose(
            covariance_features(np.zeros_like(B)), [1 / 3] * 3 + [0] * 3
        )

    def test_single_loop_is_projection(self):
        t = np.arange(10000) / 10000
        B = (
            np.array(
                [
                    np.cos(2 * np.pi * 665 * t),
                    2 * np.cos(2 * np.pi * 665 * t),
                    4 * np.cos(2 * np.pi * 665 * t),
                ]
            )
            * 1e-12
        )
        a = scalar_features_known_f(B, 10000, 665, axis=0)
        b = scalar_features_known_f(B, 10000, 665, axis=2)
        self.assertAlmostEqual(float((b - a)[0]), np.log10(16), places=10)

    def test_conditioned_snr_in_actual_filter(self):
        cfg = QuadrotorConfig(pwm_depth=0)
        _, B = simulate_B_timeseries(cfg, (2, 0, 0.3), 0.2, 0.3, 0.4, fs=10000)
        for snr in (-5, 0, 20, 40):
            noisy = add_snr_controlled_noise(
                B, 10000, snr, cfg.feature_center_hz(), rng=np.random.default_rng(22)
            )
            S = bandpass(B, 10000, 625.5, 725.5)
            N = bandpass(noisy - B, 10000, 625.5, 725.5)
            self.assertAlmostEqual(
                10 * np.log10(np.sum(S**2) / np.sum(N**2)), snr, places=8
            )

    def test_asd_units_and_spectral_detector(self):
        fs = 10000
        self.assertAlmostEqual(
            noise_sigma_from_asd(10, fs) * 1e12, 10 * np.sqrt(fs / 2)
        )
        rng = np.random.default_rng(52)
        N = add_sensor_noise(np.zeros((3, 200000)), fs, 10, rng)
        f, P = component_psd(N, fs)
        self.assertAlmostEqual(
            float(np.mean(P[10:-10]) / (3 * (10e-12) ** 2)), 1, delta=0.025
        )
        t = np.arange(20000) / fs
        B = np.array(
            [np.cos(2 * np.pi * 665 * t), np.sin(2 * np.pi * 665 * t), np.zeros_like(t)]
        )
        peak, _ = detect_motor_harmonic(B, fs, notch_hz=0)
        self.assertAlmostEqual(peak, 665, delta=2)

    def test_rotation_covariance_and_shared_source(self):
        cfg = QuadrotorConfig(pwm_depth=0)
        pos = np.array([2.0, 0.5, 0.3])
        node = np.array([-0.2, 0.3, 0.1])
        Q = rot_z(0.6)
        args = dict(fs=10000, phase_offsets=np.arange(4))
        _, B = simulate_B_timeseries(
            cfg, pos, 0.2, 0.3, 0.4, sensor_pos_world=node, **args
        )
        _, BQ = simulate_B_timeseries(
            cfg, Q @ pos, 0.2, 0.3, 1.0, sensor_pos_world=Q @ node, **args
        )
        np.testing.assert_allclose(BQ, Q @ B, rtol=1e-10, atol=1e-24)
        A = simulate_array(
            cfg,
            pos,
            (0.2, 0.3, 0.4),
            [node, node],
            fs=10000,
            rng=np.random.default_rng(3),
        )
        np.testing.assert_array_equal(A[0], A[1])

    def test_tilt_and_polarization_counterexample(self):
        self.assertAlmostEqual(np.rad2deg(tilt_magnitude(np.pi / 4, np.pi / 4)), 60)
        B = dipole_tensor([1.0, 0, 0]) @ np.array([0.0, 0, 1.0])
        self.assertEqual(B[0], 0)
        self.assertLess(B[2], 0)

    def test_conditional_position_inversion(self):
        cfg = QuadrotorConfig()
        nodes = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0]])
        for pos, attitude in [
            ([1.7, 3.1, 0.8], [0, 0, 0]),
            ([3.7, 1.2, 1.5], [0.3, -0.4, 0.8]),
        ]:
            B = simulate_array(cfg, pos, attitude, nodes, rng=np.random.default_rng(25))
            Y = extract_fundamental_phasors(B, DEFAULT_FS, cfg)
            p, info = estimate_position(Y, attitude, nodes, cfg)
            np.testing.assert_allclose(p, pos, atol=2e-4)
            self.assertTrue(info["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
