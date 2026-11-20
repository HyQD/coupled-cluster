#!/usr/bin/env python
"""Check same-time OATDCC transformations with different orbital states."""

import numpy as np

from coupled_cluster import OATDCCD
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction


def transformed_integrals(system, state, solver, time):
    amplitudes = solver.amp_template.from_array(state)
    h_prime = system.transform_one_body_elements(
        system.h_t(time), amplitudes.C, amplitudes.C_tilde
    )
    u_prime = system.transform_two_body_elements(
        system.u, amplitudes.C, amplitudes.C_tilde
    )
    f_prime = system.construct_fock_matrix(h_prime, u_prime)
    return h_prime, u_prime, f_prime


def errors(solver, expected):
    actual = solver.h_prime, solver.u_prime, solver.f_prime
    return tuple(np.linalg.norm(a - e) for a, e in zip(actual, expected))


def main():
    system = construct_pyscf_system_rhf("he", basis="cc-pvdz")
    polarization = np.zeros(3)
    polarization[2] = 1.0
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            lambda time: 1.0, polarization_vector=polarization
        )
    )

    assert system.has_one_body_time_evolution_operator
    assert not system.has_two_body_time_evolution_operator

    solver = OATDCCD(system)
    first_state = solver.amp_template.asarray()
    time = 0.25
    solver.update_hamiltonian(time, y=first_state)

    second_state = first_state.copy()
    second_amplitudes = solver.amp_template.from_array(second_state)
    C = np.eye(system.l, dtype=second_state.dtype)
    C[0, 1] = 0.2
    second_amplitudes.C[:] = C
    second_amplitudes.C_tilde[:] = np.linalg.inv(C)

    expected = transformed_integrals(system, second_state, solver, time)

    solver.update_hamiltonian(time, y=second_state)
    same_time_errors = errors(solver, expected)

    labels = "h_prime", "u_prime", "f_prime"
    for label, error in zip(labels, same_time_errors):
        print(f"{label:7s}: same_time_error={error:.12e}")

    assert max(same_time_errors) < 1e-12


if __name__ == "__main__":
    main()
