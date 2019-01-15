class TimeDependentHFCC:
    """Class combining the time-dependent Hartree-Fock (TDHF) method with the
    time-dependent Coupled Cluster (TDCC) method. This combination uses TDHF to
    evolve the orbitals in time and TDCC to evolve the amplitudes in time. This
    alleviates some of the stress put on the amplitudes in pure TDCC if the
    overlap with the reference state goes to zero.
    """

    def __init__(self, system, tdhf, tdcc):
        self.system = system
        self.tdhf = tdhf
        self.tdcc = tdcc

    def step(self, dt):
        """Runge-Kutta 4 scheme for orbitals and amplitudes"""
        pass
