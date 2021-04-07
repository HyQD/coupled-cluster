import numpy as np
from coupled_cluster.ccsd.ccsd import CCSD
from coupled_cluster.mix import AlphaMixer, DIIS


def test_one_body_density(zanghellini_system):

    zang_ccsd = CCSD(zanghellini_system, verbose=True, mixer=AlphaMixer)

    t_kwargs = dict(theta=0.8)
    zang_ccsd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=t_kwargs)

    rho_est = zang_ccsd.compute_one_body_density_matrix()

    h = zanghellini_system.h
    o = zanghellini_system.o
    v = zanghellini_system.v
    n = zanghellini_system.n

    (
        t_0,
        t_1,
        t_2,
        l_1,
        l_2,
    ) = zang_ccsd.get_amplitudes(get_t_0=True).unpack()

    rho_qp = np.zeros_like(h)

    rho_qp.fill(0)

    rho_qp[v, v] += np.dot(t_1, l_1)
    rho_qp[v, v] += 0.5 * np.tensordot(t_2, l_2, axes=((1, 2, 3), (3, 0, 1)))

    np.testing.assert_allclose(rho_qp[v, v], rho_est[v, v])

    rho_qp[o, v] += l_1

    np.testing.assert_allclose(rho_qp[o, v], rho_est[o, v])

    rho_qp[o, o] += np.eye(n)
    rho_qp[o, o] -= np.dot(l_1, t_1)
    rho_qp[o, o] += 0.5 * np.tensordot(l_2, t_2, axes=((1, 2, 3), (2, 0, 1)))

    np.testing.assert_allclose(rho_qp[o, o], rho_est[o, o])

    rho_qp[v, o] += t_1
    rho_qp[v, o] += np.tensordot(
        l_1, t_2 - np.einsum("bi, aj -> abij", t_1, t_1), axes=((0, 1), (3, 1))
    )
    rho_qp[v, o] += 0.5 * np.einsum(
        "bi, kjcb, ackj -> ai", t_1, l_2, t_2, optimize=True
    )
    rho_qp[v, o] -= 0.5 * np.einsum(
        "aj, kjcb, cbki -> ai", t_1, l_2, t_2, optimize=True
    )

    np.testing.assert_allclose(rho_qp[v, o], rho_est[v, o])

    np.testing.assert_allclose(rho_qp, rho_est, atol=1e-7)


def test_v_o_term(zanghellini_system):

    zang_ccsd = CCSD(zanghellini_system, verbose=True, mixer=AlphaMixer)

    t_kwargs = dict(theta=0.8)
    zang_ccsd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=t_kwargs)

    (
        t_0,
        t_1,
        t_2,
        l_1,
        l_2,
    ) = zang_ccsd.get_amplitudes(get_t_0=True).unpack()

    term_1 = np.tensordot(
        l_1, t_2 - np.einsum("bi, aj -> abij", t_1, t_1), axes=((0, 1), (3, 1))
    )

    term = t_2 - np.einsum("bi, aj->abij", t_1, t_1)
    term_2 = np.tensordot(l_1, term, axes=((0, 1), (3, 1)))

    np.testing.assert_allclose(term_1, term_2)

    term_1 = 0.5 * np.einsum(
        "bi, kjcb, ackj -> ai", t_1, l_2, t_2, optimize=True
    )

    term = 0.5 * np.tensordot(t_1, l_2, axes=((0), (3)))
    term_2 = np.tensordot(
        term, t_2, axes=((1, 2, 3), (2, 3, 1))
    ).transpose()  # ia->ai

    np.testing.assert_allclose(term_1, term_2)

    term_1 = -0.5 * np.einsum(
        "aj, kjcb, cbki -> ai", t_1, l_2, t_2, optimize=True
    )

    term = -(0.5) * np.tensordot(t_1, l_2, axes=((1), (1)))  # akcb
    term_2 = np.tensordot(term, t_2, axes=((1, 2, 3), (2, 0, 1)))  # ai

    np.testing.assert_allclose(term_1, term_2)
