import numpy as np

from coupled_cluster import TDCCS, TDCCD, TDCCSD, TDRCCSD, OATDCCD
from coupled_cluster.cc_helper import AmplitudeContainer, OACCVector

from quantum_systems import (
    RandomBasisSet,
    SpatialOrbitalSystem,
)


def test_amp_template():
    l, dim = 20, 1
    n = 4
    m = l - n
    l_spat = l // 2
    n_spat = n // 2
    m_spat = m // 2
    k = 8
    m_trunc = k - n
    rbs = RandomBasisSet(l_spat, dim, includes_spin=False)

    sos = SpatialOrbitalSystem(n, rbs)
    gos = sos.construct_general_orbital_system()

    C_mat = np.eye(l)

    # sizes
    t_0 = 1
    t_1 = n * m
    t_2 = n * n * m * m
    l_1 = m * n
    l_2 = m * m * n * n
    C = l * l
    t_2_trunc = m_trunc * m_trunc * n * n
    l_2_trunc = n * n * m_trunc * m_trunc
    C_trunc = l * k

    assert TDCCS(gos)._amp_template.asarray().size == t_0 + t_1 + l_1
    assert TDCCD(gos)._amp_template.asarray().size == t_0 + t_2 + l_2
    assert (
        TDCCSD(gos)._amp_template.asarray().size == t_0 + t_1 + l_1 + t_2 + l_2
    )
    assert (
        OATDCCD(gos, C=C_mat)._amp_template.asarray().size
        == t_0 + t_2 + l_2 + 2 * C
    )
    assert (
        OATDCCD(gos, C=C_mat[:, :k])._amp_template.asarray().size
        == t_0 + t_2_trunc + l_2_trunc + 2 * C_trunc
    )

    assert (
        TDRCCSD(sos)._amp_template.asarray().size
        == t_0 + t_1 // 4 + l_1 // 4 + t_2 // 16 + l_2 // 16
    )


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


def test_construct_container_from_array():
    truncation = "CCSD"

    n, m = 4, 10

    t_1_shape = (m, n)
    t_2_shape = (m, m, n, n)

    t_0 = RandomBasisSet.get_random_elements((1,), np)
    t_1 = RandomBasisSet.get_random_elements(t_1_shape, np)
    t_2 = RandomBasisSet.get_random_elements(t_2_shape, np)

    l_1 = RandomBasisSet.get_random_elements(t_1_shape[::-1], np)
    l_2 = RandomBasisSet.get_random_elements(t_2_shape[::-1], np)

    arr = np.concatenate(
        [t_0.ravel(), t_1.ravel(), t_2.ravel(), l_1.ravel(), l_2.ravel()]
    )

    amps = AmplitudeContainer.construct_container_from_array(
        arr, truncation, n, m, np
    )

    np.testing.assert_allclose(t_0, amps.t[0])
    np.testing.assert_allclose(t_1, amps.t[1])
    np.testing.assert_allclose(t_2, amps.t[2])
    np.testing.assert_allclose(l_1, amps.l[0])
    np.testing.assert_allclose(l_2, amps.l[1])
