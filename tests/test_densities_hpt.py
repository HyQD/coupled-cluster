import pytest

import numpy as np


class LiLaser:
    def __init__(self, E_max=0.1, omega=1):
        self.E_max = E_max
        self.omega = omega
        self.cycle = 2 * np.pi / self.omega

    def envelope(self, t):
        if 0 <= t <= self.cycle:
            return t / self.cycle * self.E_max
        elif self.cycle <= t <= 2 * self.cycle:
            return self.E_max
        elif 2 * self.cycle <= t <= 3 * self.cycle:
            return (3 - t / self.cycle) * self.E_max

        return 0

    def __call__(self, t):
        return self.envelope(t) * np.sin(self.omega * t)


@pytest.fixture
def ho_system():
    n = 2
    l = 30

    grid_length = 8
    num_grid_points = 801
    omega_0 = 1
    omega_l = 1

    polarization_vector = np.zeros(1)
    polarization_vector[0] = 1

    system = ODQD(n, l, grid_length, num_grid_points)
    system.setup_system(potential=ODQD.HOPotential(omega=omega_0))
    system.set_time_evolution_operator(
        LaserField(
            LiLaser(omega=omega_l), polarization_vector=polarization_vector
        )
    )

    return system
