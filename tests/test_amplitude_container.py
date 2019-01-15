import numpy as np
from coupled_cluster.time_propagator import AmplitudeContainer


def test_addition_single_amp(large_system_ccd):
    t, l, cs = large_system_ccd

    container = AmplitudeContainer(l=l, t=t)
    k = 10
    new_container = container + k
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t + k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l + k, new_l[0], atol=1e-10)

    new_container = container + [[l], [t]]
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t + t, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l + l, new_l[0], atol=1e-10)

    new_container = container + container
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t + t, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l + l, new_l[0], atol=1e-10)


def test_multiplication_single_amp(large_system_ccd):
    t, l, cs = large_system_ccd

    container = AmplitudeContainer(l=l, t=t)
    k = 10
    new_container = container * k
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t * k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l * k, new_l[0], atol=1e-10)

    new_container = container * [[l], [t]]
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t * t, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l * l, new_l[0], atol=1e-10)

    new_container = container * container
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t * t, new_t[0], atol=1e-10)
    np.testing.assert_allclose(l * l, new_l[0], atol=1e-10)


def test_addition_double_amp(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    container = AmplitudeContainer(l=[l_1, l_2], t=[t_1, t_2])
    k = 10
    new_container = container + k
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + k, new_l[1], atol=1e-10)

    new_container = container + [[l_1, l_2], [t_1, t_2]]
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)

    new_container = container + container
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 + t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)


def test_addition_double_amp(large_system_ccsd):
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    container = AmplitudeContainer(l=[l_1, l_2], t=[t_1, t_2])
    k = 10
    new_container = container * k
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * k, new_l[1], atol=1e-10)

    new_container = container * [[l_1, l_2], [t_1, t_2]]
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)

    new_container = container * container
    new_l, new_t = new_container

    assert type(new_container) == type(container)
    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_1 * t_1, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)
