"""
Tikhonov-Regularized Spatial Inversion Engine.
Reconstructs multi-phase currents from non-invasive peripheral magnetic field readings.
"""
from typing import Dict
import numpy as np


class RegularizedFieldInverter:
    def __init__(self, lambda_reg: float = 1e-12):
        self.lambda_reg = lambda_reg

    def reconstruct_currents(
        self,
        a_matrix: np.ndarray,
        b_measured_tesla: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Reconstructs conductor current vector I given sensor observations B:
        I = (A^T * A + lambda * Identity)^(-1) * A^T * B
        """
        m, n = a_matrix.shape
        if len(b_measured_tesla) != m:
            raise ValueError(
                f"Dimension mismatch: A matrix has {m} rows, but received "
                f"{len(b_measured_tesla)} measurements."
            )

        # Compute regularized normal equations
        at_a = a_matrix.T @ a_matrix
        reg_term = self.lambda_reg * np.eye(n, dtype=np.float64)
        rhs = a_matrix.T @ b_measured_tesla

        currents_estimated = np.linalg.solve(at_a + reg_term, rhs)

        # Residual reconstruction error in field space
        b_reprojected = a_matrix @ currents_estimated
        residual_norm = float(np.linalg.norm(b_measured_tesla - b_reprojected))

        return {
            "estimated_currents_a": currents_estimated,
            "residual_norm_tesla": residual_norm,
        }