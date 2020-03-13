from coupled_cluster.cc_helper import compute_reference_energy


def compute_ccs_ground_state_energy(f, u, t, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_ccs_ground_state_energy_correction(f, u, t, o, v, np=np)

    return energy


def compute_ccs_ground_state_energy_correction(f, u, t, o, v, np):
    energy = np.tensordot(f[o, v], t, axes=((0, 1), (1, 0)))

    term_ja = -0.5 * np.tensordot(t, u[o, o, v, v], axes=((0, 1), (3, 0)))
    energy += np.tensordot(t, term_ja, axes=((0, 1), (1, 0)))

    return energy


def compute_time_dependent_energy(f, u, t_1, l_1, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_lagrangian_functional(f, u, t_1, l_1, o, v, np=np)

    return energy


def compute_lagrangian_functional(f, u, t_1, l_1, o, v, np):
    # L <- f^{a}_{b} l^{i}_{a} t^{b}_{i}
    energy = np.tensordot(l_1, np.dot(f[v, v], t_1), axes=((0, 1), (1, 0)))

    # L <- f^{a}_{i} l^{i}_{a}
    energy += np.trace(np.dot(f[v, o], l_1))

    # L <- -f^{i}_{a} l^{j}_{b} t^{a}_{j} t^{b}_{i}
    energy -= np.trace(
        np.dot(np.dot(f[o, v], t_1), np.dot(l_1, t_1))  # ij  # ji
    )

    # L <- f^{i}_{a} t^{a}_{i}
    energy += np.trace(np.dot(f[o, v], t_1))

    # L <- -f^{j}_{i} l^{i}_{a} t^{a}_{j}
    energy -= np.tensordot(
        f[o, o], np.dot(l_1, t_1), axes=((0, 1), (1, 0))  # ji  # ij
    )

    # L <- l^{i}_{a} t^{a}_{j} t^{b}_{k} u^{jk}_{bi}
    energy += np.trace(
        np.dot(
            np.tensordot(
                u[o, o, v, o],  # jkbi
                np.dot(l_1, t_1),  # ij
                axes=((0, 3), (1, 0)),
            ),  # kb
            t_1,  # bk
        )
    )

    # L <- -l^{i}_{a} t^{a}_{k} t^{b}_{j} t^{c}_{i} u^{jk}_{bc}
    energy -= np.trace(
        np.dot(
            np.dot(l_1, t_1),  # ik
            np.dot(
                np.tensordot(
                    t_1, u[o, o, v, v], axes=((0, 1), (2, 0))  # bj  # jkbc
                ),  # kc
                t_1,  # ci
            ),  # ki
        )
    )

    # L <- l^{i}_{a} t^{b}_{i} t^{c}_{j} u^{aj}_{bc}
    energy += np.trace(
        np.dot(
            np.dot(
                l_1,  # ia
                np.tensordot(
                    t_1, u[v, o, v, v], axes=((0, 1), (3, 1))  # cj  # ajbc
                ),  # ab
            ),  # ib
            t_1,  # bi
        )
    )

    # L <- -l^{i}_{a} t^{b}_{j} u^{aj}_{bi}
    energy -= np.trace(
        np.dot(
            np.tensordot(
                l_1, u[v, o, v, o], axes=((0, 1), (3, 0))  # ia  # ajbi
            ),  # jb
            t_1,  # bj
        )
    )

    # L <- -0.5 t^{a}_{j} t^{b}_{i} u^{ij}_{ab}
    energy -= 0.5 * np.trace(
        np.dot(
            np.tensordot(
                t_1, u[o, o, v, v], axes=((0, 1), (3, 0))  # bi  # ijab
            ),  # ja
            t_1,  # aj
        )
    )

    return energy
