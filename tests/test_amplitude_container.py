import numpy as np
from coupled_cluster.cc_helper import AmplitudeContainer, OACCVector


def test_addition_single_amp(large_system_ccd):
    t_0 = np.array([np.random.random()])
    t_2, l, cs = large_system_ccd
    t = [t_0, t_2]

    container = AmplitudeContainer(t=t, l=l, np=np)
    k = 10
    new_container = container + k
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l + k, new_l[0], atol=1e-10)

    new_container = container + [t, [l]]
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l + l, new_l[0], atol=1e-10)

    new_container = container + container
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l + l, new_l[0], atol=1e-10)


def test_multiplication_single_amp(large_system_ccd):
    t_0 = np.array([np.random.random()])
    t_2, l, cs = large_system_ccd
    t = [t_0, t_2]

    container = AmplitudeContainer(t=t, l=l, np=np)
    k = 10
    new_container = container * k
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l * k, new_l[0], atol=1e-10)

    new_container = container * [[*t], [l]]
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l * l, new_l[0], atol=1e-10)

    new_container = container * container
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[1], atol=1e-10)
    np.testing.assert_allclose(l * l, new_l[0], atol=1e-10)


def test_addition_double_amp(large_system_ccsd):
    t_0 = np.array([np.random.random()])
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    container = AmplitudeContainer(t=[t_0, t_1, t_2], l=[l_1, l_2], np=np)
    k = 10
    new_container = container + k
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 + k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 + k, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 + k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + k, new_l[1], atol=1e-10)

    new_container = container + [[t_0, t_1, t_2], [l_1, l_2]]
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 + t_1, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)

    new_container = container + container
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 + t_1, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)


def test_multiplication_double_amp(large_system_ccsd):
    t_0 = np.array([np.random.random()])
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    container = AmplitudeContainer(t=[t_0, t_1, t_2], l=[l_1, l_2], np=np)
    k = 10
    new_container = container * k
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 * k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 * k, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 * k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * k, new_l[1], atol=1e-10)

    new_container = container * [[t_0, t_1, t_2], [l_1, l_2]]
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 * t_1, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)

    new_container = container * container
    new_t, new_l = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 * t_1, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)


def test_addition_double_amp_oaccvector(large_system_ccsd):
    t_0 = np.array([np.random.random()])
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    C = np.random.random((cs.l, cs.l)).astype(t_2.dtype)
    C_tilde = np.random.random((cs.l, cs.l)).astype(t_2.dtype)

    container = OACCVector(
        t=[t_0, t_1, t_2], l=[l_1, l_2], C=C, C_tilde=C_tilde, np=np
    )
    k = 10
    new_container = container + k
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 + k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 + k, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 + k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + k, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C + k, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde + k, new_C_tilde, atol=1e-10)

    new_container = container + [[t_0, t_1, t_2], [l_1, l_2], C, C_tilde]
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 + t_1, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C + C, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde + C_tilde, new_C_tilde, atol=1e-10)

    new_container = container + container
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 + t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 + t_1, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 + t_2, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 + l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 + l_2, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C + C, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde + C_tilde, new_C_tilde, atol=1e-10)


def test_multiplication_double_amp_oaccvector(large_system_ccsd):
    t_0 = np.array([np.random.random()])
    t_1, t_2, l_1, l_2, cs = large_system_ccsd
    C = np.random.random((cs.l, cs.l)).astype(t_2.dtype)
    C_tilde = np.random.random((cs.l, cs.l)).astype(t_2.dtype)

    container = OACCVector(
        t=[t_0, t_1, t_2], l=[l_1, l_2], C=C, C_tilde=C_tilde, np=np
    )
    k = 10
    new_container = container * k
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * k, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 * k, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 * k, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 * k, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * k, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C * k, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde * k, new_C_tilde, atol=1e-10)

    new_container = container * [[t_0, t_1, t_2], [l_1, l_2], C, C_tilde]
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 * t_1, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C * C, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde * C_tilde, new_C_tilde, atol=1e-10)

    new_container = container * container
    new_t, new_l, new_C, new_C_tilde = new_container

    assert type(new_container) == type(container)
    np.testing.assert_allclose(t_0 * t_0, new_t[0], atol=1e-10)
    np.testing.assert_allclose(t_1 * t_1, new_t[1], atol=1e-10)
    np.testing.assert_allclose(t_2 * t_2, new_t[2], atol=1e-10)
    np.testing.assert_allclose(l_1 * l_1, new_l[0], atol=1e-10)
    np.testing.assert_allclose(l_2 * l_2, new_l[1], atol=1e-10)
    np.testing.assert_allclose(C * C, new_C, atol=1e-10)
    np.testing.assert_allclose(C_tilde * C_tilde, new_C_tilde, atol=1e-10)


def test_amplitude_divide_and_join(large_system_ccsd):
    t_0 = np.array([np.random.random()])
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    amp_container = AmplitudeContainer([t_0, t_1, t_2], [l_1, l_2], np=np)
    concat_amp = amp_container.asarray()

    i = 0
    for amp in amp_container.unpack():

        np.testing.assert_allclose(amp.ravel(), concat_amp[i : i + amp.size])

        i += amp.size

    some_number = np.random.rand()
    concat_amp += some_number
    amp_container = amp_container + some_number
    new_amp_container = AmplitudeContainer.from_array(amp_container, concat_amp)

    for amp, amp_e in zip(amp_container.unpack(), new_amp_container.unpack()):
        np.testing.assert_allclose(amp, amp_e)


def test_splat(large_system_ccsd):
    t_0 = np.array([np.random.random()])
    t_1, t_2, l_1, l_2, cs = large_system_ccsd

    amp_container = AmplitudeContainer([t_0, t_1, t_2], [l_1, l_2], np=np)
    t_sp, l_sp = amp_container

    for amp, amp_e in zip(amp_container.unpack(), [*t_sp, *l_sp]):
        np.testing.assert_allclose(amp, amp_e)
