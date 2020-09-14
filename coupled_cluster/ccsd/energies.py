from coupled_cluster.cc_helper import compute_reference_energy
from coupled_cluster.ccsd.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.ccd.energies import (
    compute_lagrangian_functional as ccd_functional,
)


def compute_ccsd_ground_state_energy(f, u, t_1, t_2, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_ground_state_energy_correction(
        f, u, t_1, t_2, o, v, np=np
    )

    return energy


def compute_ground_state_energy_correction(f, u, t_1, t_2, o, v, np):
    """

    f^{i}_{a} t^{a}_{i}
    + (0.25) u^{ij}_{ab} t^{ab}_{ij}
    + (0.5) t^{ij}_{ab} t^{a}_{i} t^{b}_{j}

    """

    energy = np.tensordot(f[o, v], t_1, axes=((0, 1), (1, 0)))  # ia, ai ->
    energy += (0.25) * np.tensordot(
        u[o, o, v, v], t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )  # ijab, abij ->
    term = (0.5) * np.tensordot(
        u[o, o, v, v], t_1, axes=((0, 2), (1, 0))
    )  # ijab, ai -> jb
    energy += np.tensordot(term, t_1, axes=((0, 1), (1, 0)))
    return energy


def compute_time_dependent_energy(f, u, t_1, t_2, l_1, l_2, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += lagrangian_functional(f, u, t_1, t_2, l_1, l_2, o, v, np=np)

    return energy


def lagrangian_functional(f, u, t_1, t_2, l_1, l_2, o, v, np, test=False):

    energy = np.tensordot(f[v, o], l_1, axes=((0, 1), (1, 0)))
    energy += np.tensordot(f[o, v], t_1, axes=((0, 1), (1, 0)))

    term = np.tensordot(f[v, v], l_1, axes=((0), (1)))  # bi
    energy += np.tensordot(term, t_1, axes=((0, 1), (0, 1)))
    term = np.tensordot(l_1, t_2, axes=((0, 1), (3, 1)))  # ai
    energy += np.tensordot(f[o, v], term, axes=((0, 1), (1, 0)))

    term = (0.5) * np.tensordot(
        t_2, u[o, o, v, o], axes=((1, 2, 3), (2, 0, 1))
    )  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0)))
    term = (0.5) * np.tensordot(l_1, t_2, axes=((0), (2)))  # abcj
    energy += np.tensordot(
        term, u[v, o, v, v], axes=((0, 1, 2, 3), (0, 2, 3, 1))
    )
    term = (0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbk
    energy += np.tensordot(
        term, u[v, o, o, o], axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )
    term = (0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # jabc
    energy += np.tensordot(
        term, u[v, v, v, o], axes=((0, 1, 2, 3), (3, 0, 1, 2))
    )

    term = (-1) * np.tensordot(f[o, o], l_1, axes=((1), (0)))  # ja
    energy += np.tensordot(term, t_1, axes=((0, 1), (1, 0)))
    term = (-1) * np.tensordot(t_1, u[v, o, v, o], axes=((0, 1), (2, 1)))  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0)))

    term = (-0.5) * np.tensordot(
        t_1, u[o, o, v, v], axes=((0, 1), (3, 0))
    )  # ja
    energy += np.tensordot(t_1, term, axes=((0, 1), (1, 0)))

    term = np.tensordot(t_1, u[o, o, v, o], axes=((0, 1), (2, 1)))  # ji
    term = np.tensordot(t_1, term, axes=((1), (0)))  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0)))
    term = np.tensordot(t_1, u[v, o, v, v], axes=((0, 1), (3, 1)))  # ab
    term = np.tensordot(t_1, term, axes=((0), (1)))  # ia
    energy += np.tensordot(l_1, term, axes=((0, 1), (0, 1)))
    term = np.tensordot(t_2, u[o, o, v, v], axes=((1, 3), (3, 1)))  # aijb
    term = np.tensordot(t_1, term, axes=((0, 1), (3, 2)))  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0)))
    term = np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbk
    term = np.tensordot(term, t_1, axes=((0), (1)))  # jbkc
    energy += np.tensordot(
        term, u[v, o, v, o], axes=((0, 1, 2, 3), (3, 0, 1, 2))
    )

    term = (-1) * np.tensordot(l_1, t_1, axes=((0), (1)))  # ba
    term = np.tensordot(f[o, v], term, axes=((1), (1)))  # ib
    energy += np.tensordot(term, t_1, axes=((0, 1), (1, 0)))
    term = (-1) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbk
    term = np.tensordot(term, t_2, axes=((0, 2), (2, 0)))  # jkcl
    energy += np.tensordot(
        term, u[o, o, v, o], axes=((0, 1, 2, 3), (3, 0, 2, 1))
    )
    term = (-1) * np.tensordot(l_2, t_1, axes=((0), (1)))  # jabc
    term = np.tensordot(term, t_2, axes=((0, 1), (2, 0)))  # bcdk
    energy += np.tensordot(
        term, u[v, o, v, v], axes=((0, 1, 2, 3), (0, 2, 3, 1))
    )

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # kbca
    term = np.tensordot(f[o, v], term, axes=((1), (3)))  # ikbc
    energy += np.tensordot(term, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # jkci
    term = np.tensordot(f[o, v], term, axes=((0), (3)))  # ajkc
    energy += np.tensordot(term, t_2, axes=((0, 1, 2, 3), (0, 2, 3, 1)))
    term = (-0.5) * np.tensordot(l_1, t_1, axes=((1), (0)))  # ij
    term = np.tensordot(term, t_2, axes=((0), (2)))  # jbck
    energy += np.tensordot(
        term, u[o, o, v, v], axes=((0, 1, 2, 3), (0, 2, 3, 1))
    )
    term = (-0.5) * np.tensordot(l_1, t_1, axes=((0), (1)))  # ab
    term = np.tensordot(term, t_2, axes=((0), (0)))  # bcjk
    energy += np.tensordot(
        term, u[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )
    term = (-0.5) * np.tensordot(l_2, t_2, axes=((0, 2, 3), (2, 0, 1)))  # jl
    term = np.tensordot(term, u[o, o, v, o], axes=((0, 1), (3, 1)))  # kc
    energy += np.tensordot(t_1, term, axes=((0, 1), (1, 0)))
    term = (-0.5) * np.tensordot(l_2, t_2, axes=((0, 1, 2), (2, 3, 0)))  # bd
    term = np.tensordot(term, u[v, o, v, v], axes=((0, 1), (0, 3)))  # kc
    energy += np.tensordot(t_1, term, axes=((0, 1), (1, 0)))

    term = (-0.25) * np.tensordot(t_1, u[o, o, o, o], axes=((1), (0)))  # blij
    term = np.tensordot(t_1, term, axes=((1), (1)))  # abij
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    term = (-0.25) * np.tensordot(t_1, u[v, v, v, v], axes=((0), (3)))  # iabc
    term = np.tensordot(t_1, term, axes=((0), (3)))  # jiab
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (1, 0, 2, 3)))
    term = (0.25) * np.tensordot(
        t_2, u[v, o, v, v], axes=((0, 1), (2, 3))
    )  # ijbk
    term = np.tensordot(t_1, term, axes=((1), (3)))  # aijb
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (1, 2, 0, 3)))
    term = (0.25) * np.tensordot(
        t_2, u[o, o, v, o], axes=((2, 3), (0, 1))
    )  # abcj
    term = np.tensordot(t_1, term, axes=((0), (2)))  # iabj
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (0, 3, 1, 2)))

    term = (-1) * np.tensordot(t_1, u[o, o, v, v], axes=((0), (3)))  # ijkb
    term = np.tensordot(t_1, term, axes=((0, 1), (3, 1)))  # ik
    term = np.tensordot(t_1, term, axes=((1), (1)))  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0)))
    term = (-1) * np.tensordot(
        t_2, u[o, o, v, v], axes=((1, 3), (3, 1))
    )  # bjkc
    term = np.tensordot(t_1, term, axes=((0), (3)))  # ibjk
    term = np.tensordot(t_1, term, axes=((1), (3)))  # aibj
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (1, 3, 0, 2)))

    term = (-0.5) * np.tensordot(t_1, u[v, o, v, v], axes=((0), (3)))  # ibkc
    term = np.tensordot(t_1, term, axes=((0), (3)))  # jibk
    term = np.tensordot(t_1, term, axes=((1), (3)))  # ajib
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (2, 1, 0, 3)))
    term = (-0.5) * np.tensordot(
        t_1, u[o, o, v, v], axes=((0, 1), (2, 1))
    )  # kd
    term = np.tensordot(t_1, term, axes=((1), (0)))  # ad
    term = np.tensordot(l_2, term, axes=((2), (0)))  # ijbd
    energy += np.tensordot(term, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    term = np.tensordot(term, t_1, axes=((0), (1)))  # jlkc
    energy += np.tensordot(
        term, u[o, o, v, o], axes=((0, 1, 2, 3), (3, 1, 0, 2))
    )
    term = (-0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # jabc
    term = np.tensordot(term, t_2, axes=((0, 1, 2), (2, 0, 1)))  # cl
    term = np.tensordot(term, u[o, o, v, v], axes=((0, 1), (2, 1)))  # kd
    energy += np.tensordot(t_1, term, axes=((0, 1), (1, 0)))

    term = (-0.125) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    term = np.tensordot(term, t_2, axes=((0, 1), (2, 3)))  # lkcd
    energy += np.tensordot(
        term, u[o, o, v, v], axes=((0, 1, 2, 3), (1, 0, 2, 3))
    )
    term = (-0.125) * np.tensordot(l_2, t_1, axes=((1), (1)))  # iabc
    term = np.tensordot(term, t_1, axes=((0), (1)))  # abcd
    term = np.tensordot(term, t_2, axes=((0, 1), (0, 1)))  # cdkl
    energy += np.tensordot(
        term, u[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )

    term = (0.25) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    term = np.tensordot(term, t_1, axes=((1), (1)))  # ilkc
    term = np.tensordot(term, t_1, axes=((0), (1)))  # lkcd
    energy += np.tensordot(
        term, u[o, o, v, v], axes=((0, 1, 2, 3), (1, 0, 2, 3))
    )

    if not test:
        energy += ccd_functional(f, u, t_2, l_2, o, v, np=np)

    return energy
