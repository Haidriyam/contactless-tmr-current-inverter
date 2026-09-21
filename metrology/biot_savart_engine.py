"""
Two-Dimensional Biot-Savart Magnetostatic Mapping Engine.
Constructs geometric cross-coupling matrices between conductor cores and sensor rings.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class SensorGeometry:
    ring_radius_m: float = 0.035          # 35 mm radius sensor ring
    sensor_count: int = 8                 # 8 TMR sensors equally spaced
    mu_0: float = 4.0 * np.pi * 1e-7      # Magnetic permeability of free space


class BiotSavartEngine:
    def __init__(self, geometry: SensorGeometry = SensorGeometry()):
        self.geo = geometry
        self.sensor_positions = self._initialize_sensor_array()

    def _initialize_sensor_array(self) -> np.ndarray:
        """Positions of M sensors around a circular fixture: [x, y, theta_tangent]."""
        angles = np.linspace(0, 2 * np.pi, self.geo.sensor_count, endpoint=False)
        sensors = []
        for phi in angles:
            x = self.geo.ring_radius_m * np.cos(phi)
            y = self.geo.ring_radius_m * np.sin(phi)
            # Tangential unit vector is perpendicular to radial vector: [-sin(phi), cos(phi)]
            sensors.append([x, y, phi + np.pi / 2.0])
        return np.array(sensors, dtype=np.float64)

    def build_coupling_matrix(self, conductor_positions: List[Tuple[float, float]]) -> np.ndarray:
        """
        Build the forward matrix A (M sensors x N conductors) relating:
        B_tangential = A * I
        """
        num_conductors = len(conductor_positions)
        a_matrix = np.zeros((self.geo.sensor_count, num_conductors), dtype=np.float64)

        for m_idx in range(self.geo.sensor_count):
            sx, sy, s_tangent_angle = self.sensor_positions[m_idx]
            # Unit tangent vector for sensor sensitivity axis
            u_t = np.array([np.cos(s_tangent_angle), np.sin(s_tangent_angle)], dtype=np.float64)

            for n_idx, (cx, cy) in enumerate(conductor_positions):
                # Vector from conductor to sensor
                rx = sx - cx
                ry = sy - cy
                dist_sq = rx**2 + ry**2
                dist = np.sqrt(dist_sq)

                if dist < 1e-4:
                    raise ValueError(f"Sensor {m_idx} coincides with conductor {n_idx}.")

                # Biot-Savart unit vector direction for current in +z direction: (-ry, rx) / dist
                b_dir = np.array([-ry / dist, rx / dist], dtype=np.float64)
                # Field magnitude per Ampere: (mu_0) / (2 * pi * dist)
                b_per_amp = self.geo.mu_0 / (2.0 * np.pi * dist)

                # Project onto the directional sensitivity axis of the TMR sensor
                a_matrix[m_idx, n_idx] = b_per_amp * np.dot(b_dir, u_t)

        return a_matrix

    def forward_simulate(self, a_matrix: np.ndarray, currents: np.ndarray) -> np.ndarray:
        """Simulate true magnetic field readings across sensor array."""
        return a_matrix @ currents