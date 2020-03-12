import numpy as np

from coupled_cluster import TDCCS, TDCCD, TDCCSD, TDRCCSD, OATDCCD
from quantum_systems import (
    RandomBasisSet,
    GeneralOrbitalSystem,
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


if __name__ == "__main__":
    test_amp_template()
