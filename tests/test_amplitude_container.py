import numpy as np
from coupled_cluster.cc_helper import AmplitudeContainer, OACCVector


def test_addition_single_amp(large_system_ccd):
    t, l, cs = large_system_ccd

    container = AmplitudeContainer(t=t, l=l)
    k = 10
    new_container = container + k
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t + k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l + k, new_l[0], atol=1e-10)

    new_container = container + [[t], [l]]
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t + t, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l + l, new_l[0], atol=1e-10)

    new_container = container + container
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t + t, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l + l, new_l[0], atol=1e-10)


def test_multiplication_single_amp(large_system_ccd):
    t, l, cs = large_system_ccd

    container = AmplitudeContainer(t=t, l=l)
    k = 10
    new_container = container * k
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t * k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l * k, new_l[0], atol=1e-10)

    new_container = container * [[t], [l]]
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t * t, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l * l, new_l[0], atol=1e-10)

    new_container = container * container
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t * t, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l * l, new_l[0], atol=1e-10)


def test_addition_double_amp(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    container = AmplitudeContainer(t=[t_1, t_2], l=[l_1, l_2])
    k = 10
    new_container = container + k
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + k, new_l[1], atol=1e-10)

    new_container = container + [[t_1, t_2], [l_1, l_2]]
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)

    new_container = container + container
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)


def test_multiplication_double_amp(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    container = AmplitudeContainer(t=[t_1, t_2], l=[l_1, l_2])
    k = 10
    new_container = container * k
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * k, new_l[1], atol=1e-10)

    new_container = container * [[t_1, t_2], [l_1, l_2]]
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)

    new_container = container * container
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)


def test_addition_double_amp_oaccvector(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    C = np.random.random((cs.l, cs.l)).astype(t_2.dtype)
    C_tilde = np.random.random((cs.l, cs.l)).astype(t_2.dtype)

    container = OACCVector(t=[t_1, t_2], l=[l_1, l_2], C=C, C_tilde=C_tilde)
    k = 10
    new_container = container + k
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + k, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C + k, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde + k, new_C_tilde, atol=1e-10)

    new_container = container + [[t_1, t_2], [l_1, l_2], C, C_tilde]
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C + C, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde + C_tilde, new_C_tilde, atol=1e-10)

    new_container = container + container
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C + C, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde + C_tilde, new_C_tilde, atol=1e-10)


def test_multiplication_double_amp_oaccvector(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    C = np.random.random((cs.l, cs.l)).astype(t_2.dtype)
    C_tilde = np.random.random((cs.l, cs.l)).astype(t_2.dtype)

    container = OACCVector(t=[t_1, t_2], l=[l_1, l_2], C=C, C_tilde=C_tilde)
    k = 10
    new_container = container * k
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * k, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C * k, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde * k, new_C_tilde, atol=1e-10)

    new_container = container * [[t_1, t_2], [l_1, l_2], C, C_tilde]
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C * C, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde * C_tilde, new_C_tilde, atol=1e-10)

    new_container = container * container
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C * C, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde * C_tilde, new_C_tilde, atol=1e-10)
