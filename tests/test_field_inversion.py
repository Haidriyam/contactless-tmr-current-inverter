import unittest
import numpy as np
from metrology.biot_savart_engine import BiotSavartEngine, SensorGeometry
from metrology.regularized_solver import RegularizedFieldInverter


class TestContactlessFieldInversion(unittest.TestCase):

    def setUp(self):
        self.geo = SensorGeometry(ring_radius_m=0.035, sensor_count=8)
        self.engine = BiotSavartEngine(self.geo)
        self.inverter = RegularizedFieldInverter(lambda_reg=1e-12)

        # Standard 3-phase conductor bundle geometry (10 mm radius pitch from core centre)
        pitch = 0.010
        self.conductors = [
            (pitch * np.cos(0.0), pitch * np.sin(0.0)),                      # Phase A
            (pitch * np.cos(2 * np.pi / 3), pitch * np.sin(2 * np.pi / 3)),  # Phase B
            (pitch * np.cos(4 * np.pi / 3), pitch * np.sin(4 * np.pi / 3)),  # Phase C
        ]
        self.a_matrix = self.engine.build_coupling_matrix(self.conductors)

    def test_forward_coupling_matrix_dimensions(self):
        # 8 sensors x 3 conductors
        self.assertEqual(self.a_matrix.shape, (8, 3))
        # Condition number must be well-posed (< 1000) for regularized inversion
        cond = np.linalg.cond(self.a_matrix)
        self.assertLess(cond, 50.0)

    def test_balanced_three_phase_current_reconstruction(self):
        np.random.seed(42)
        # 3-phase currents: 100 A peak, 120 deg apart
        t_phase = 0.05
        omega = 2.0 * np.pi * 50.0
        i_true = np.array([
            100.0 * np.sin(omega * t_phase),
            100.0 * np.sin(omega * t_phase - 2 * np.pi / 3),
            100.0 * np.sin(omega * t_phase + 2 * np.pi / 3),
        ], dtype=np.float64)

        # Compute forward magnetic field at all 8 sensors
        b_clean = self.engine.forward_simulate(self.a_matrix, i_true)

        # Add 1% white Gaussian noise (simulating sensor analog front-end noise)
        noise = np.random.normal(0, 0.01 * np.max(np.abs(b_clean)), size=b_clean.shape)
        b_noisy = b_clean + noise

        # Execute inverse spatial reconstruction
        out = self.inverter.reconstruct_currents(self.a_matrix, b_noisy)
        i_est = out["estimated_currents_a"]

        # Assert max absolute current error is under 2.0 Amperes (< 2.0%)
        error_norm = np.max(np.abs(i_true - i_est))
        self.assertLess(error_norm, 2.0)
        self.assertLess(out["residual_norm_tesla"], 1e-4)

    def test_dimension_mismatch_exception(self):
        with self.assertRaises(ValueError):
            self.inverter.reconstruct_currents(self.a_matrix, np.array([1e-5, 2e-5]))


if __name__ == "__main__":
    unittest.main()