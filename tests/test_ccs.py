import numpy as np

from coupled_cluster.ccs.rhs_t import (
    add_s1_t,
    add_s3a_t,
    add_s3b_t,
    add_s3c_t,
    add_s5a_t,
    add_s5b_t,
    add_s5c_t,
    add_s6_t,
)

from coupled_cluster.ccs.rhs_l import (
    add_s1_l,
    add_s2a_l,
    add_s2b_l,
    add_s3a_l,
    add_s3b_l,
    add_s5a_l,
    add_s5b_l,
    add_s5c_l,
    add_s5d_l,
    add_s8a_l,
    add_s8b_l,
    add_s11f_l,
    add_s11g_l,
    add_s11h_l,
)


def test_add_s1_t(large_system_ccs):
    t_1, l_1, rs = large_system_ccs
    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(t_1)
    add_s1_t(f, o, v, out, np=np)

    np.testing.assert_allclose(out, f[v, o], atol=1e-10)


def test_add_s3a_t(large_system_ccs):
    t_1, l_1, rs = large_system_ccs
    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(t_1)
    add_s3a_t(f, t_1, o, v, out, np=np)
    out_e = np.einsum("ac, ci->ai", f[v, v], t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3b_t(large_system_ccs):
    t_1, l_1, rs = large_system_ccs
    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(t_1)
    add_s3b_t(f, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("ki, ak->ai", f[o, o], t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3c_t(large_system_ccs):
    t_1, l_1, rs = large_system_ccs
    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(t_1)
    add_s3c_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum("akic, ck->ai", u[v, o, o, v], t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5a_t(large_system_ccs):
    t_1, l_1, rs = large_system_ccs
    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(t_1)
    add_s5a_t(f, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("kc, ci, ak->ai", f[o, v], t_1, t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5b_t(large_system_ccs):
    t_1, l_1, rs = large_system_ccs
    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(t_1)
    add_s5b_t(u, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "akcd, ci, dk->ai", u[v, o, v, v], t_1, t_1, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5c_t(large_system_ccs):
    t_1, l_1, rs = large_system_ccs
    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(t_1)
    add_s5c_t(u, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klic, ak, cl->ai", u[o, o, o, v], t_1, t_1, optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s6_t(large_system_ccs):
    t_1, l_1, rs = large_system_ccs
    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(t_1)
    add_s6_t(u, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "klcd, ci, ak, dl->ai", u[o, o, v, v], t_1, t_1, t_1
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s1_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s1_l(f, o, v, out, np)
    out_e = f[o, v]

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s2a_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s2a_l(f, l_1, o, v, out, np=np)
    out_e = np.einsum("ba, ib->ia", f[v, v], l_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s2b_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s2b_l(f, l_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("ij, ja->ia", f[o, o], l_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3a_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s3a_l(u, l_1, o, v, out, np=np)
    out_e = np.einsum("jb, ibaj->ia", l_1, u[o, v, v, o], optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s3b_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s3b_l(u, t_1, o, v, out, np=np)
    out_e = np.einsum("bj, jiba->ia", t_1, u[o, o, v, v], optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5a_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s5a_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ib, cj, bjac->ia", l_1, t_1, u[v, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5b_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s5b_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "ja, bk, ikbj->ia", l_1, t_1, u[o, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5c_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s5c_l(u, l_1, t_1, o, v, out, np=np)
    out_e = np.einsum(
        "jb, cj, ibac->ia", l_1, t_1, u[o, v, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s5d_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s5d_l(u, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "jb, bk, ikaj->ia", l_1, t_1, u[o, o, v, o], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s8a_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s8a_l(f, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("ib, ja, bj->ia", f[o, v], l_1, t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s8b_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    f = rs.f
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s8b_l(f, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum("ja, ib, bj->ia", f[o, v], l_1, t_1, optimize=True)

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11f_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s11f_l(u, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ib, bj, ck, jkac->ia", l_1, t_1, t_1, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11g_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s11g_l(u, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "ja, bj, ck, ikbc->ia", l_1, t_1, t_1, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)


def test_add_s11h_l(large_system_ccs):
    t_1, l_1, rs = large_system_ccs

    u = rs.u
    o = rs.o
    v = rs.v

    out = np.zeros_like(l_1)
    add_s11h_l(u, l_1, t_1, o, v, out, np=np)
    out_e = (-1) * np.einsum(
        "jb, bk, cj, ikac->ia", l_1, t_1, t_1, u[o, o, v, v], optimize=True
    )

    np.testing.assert_allclose(out, out_e, atol=1e-10)
