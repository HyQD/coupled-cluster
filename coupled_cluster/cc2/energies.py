from coupled_cluster.cc_helper import compute_reference_energy


def compute_cc2_ground_state_energy(f, u, t_1, t_2, o, v, np):
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


def compute_time_dependent_energy(
    f, f_transform, u_transform, t_1, t_2, l_1, l_2, o, v, np
):
    #    energy = compute_reference_energy(f, u, o, v, np=np)
    energy = lagrangian_functional(
        f, f_transform, u_transform, t_1, t_2, l_1, l_2, o, v, np=np
    )

    return energy


def lagrangian_functional(
    f, f_transform, u_transform, t_1, t_2, l_1, l_2, o, v, np
):

    energy = np.tensordot(
        f_transform[v, o], l_1, axes=((0, 1), (1, 0))
    )  # l_1*f

    term = np.tensordot(l_1, t_2, axes=((0, 1), (3, 1)))  # ai
    energy += np.tensordot(
        f_transform[o, v], term, axes=((0, 1), (1, 0))
    )  # l1*t2*f

    term = (0.5) * np.tensordot(
        t_2, u_transform[o, o, v, o], axes=((1, 2, 3), (2, 0, 1))
    )  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0)))  # l1*t2*u

    term = (0.5) * np.tensordot(l_1, t_2, axes=((0), (2)))  # abcj
    energy += np.tensordot(
        term, u_transform[v, o, v, v], axes=((0, 1, 2, 3), (0, 2, 3, 1))
    )  # u*l1*t2

    energy += 0.25 * np.tensordot(
        l_2, u_transform[v, v, o, o], axes=((0, 1, 2, 3), (2, 3, 0, 1))  # l2*u
    )

    energy += 0.25 * np.tensordot(
        t_2, u_transform[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1))  # t2*u
    )

    temp_dckl = 0.5 * np.tensordot(t_2, f[o, o], axes=((3), (0)))
    energy += np.tensordot(
        l_2, temp_dckl, axes=((0, 1, 2, 3), (3, 2, 0, 1))
    )  # f*t2*l2

    temp_dclk = 0.5 * np.tensordot(f[v, v], t_2, axes=((1), (0)))

    energy += np.tensordot(
        temp_dclk, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )  # f*t2*l2

    return energy
