from quantum_systems.sampler import SampleCollector, Sampler


class TDCCObservableSampler(Sampler):
    energy_key = "energy"
    dipole_keys = {
        0: "dipole_moment_x",
        1: "dipole_moment_y",
        2: "dipole_moment_z",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.energy = self.np.zeros(self.num_samples, dtype=self.np.complex128)

        self.dim = self.system.dipole_moment.shape[0]
        self.dipole_moment = self.np.zeros(
            (self.dim, self.num_samples), dtype=self.np.complex128
        )

    def sample(self, step):
        self.energy[step] = self.solver.compute_energy()

        rho_qp = self.solver.compute_one_body_density_matrix()

        for i in range(self.dim):
            dipole = self.system.dipole_moment[i]
            self.dipole_moment[i, step] = self.np.trace(rho_qp @ dipole)

    def dump(self, samples):
        samples[self.energy_key] = self.energy

        for i in range(self.dim):
            samples[self.dipole_keys[i]] = self.dipole_moment[i]

        return samples


class OATDCCObservableSampler(TDCCObservableSampler):
    def sample(self, step):
        self.energy[step] = self.solver.compute_energy()

        t, l, C, C_tilde = self.solver.amplitudes
        rho_qp = self.solver.compute_one_body_density_matrix()

        for i in range(self.dim):
            dipole = C_tilde @ self.system.dipole_moment[i] @ C
            self.dipole_moment[i, step] = self.np.trace(rho_qp @ dipole)


class TDCCAmplitudeSampler(Sampler):
    amp_keys = {2: "1", 4: "2", 6: "3"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_amps = len(self.solver.amplitudes.l)
        self.norm_t = self.np.zeros((self.num_amps, self.num_samples))
        self.norm_l = self.np.zeros_like(self.norm_t)

        self.phase_right = self.np.zeros(
            self.num_samples, dtype=self.np.complex128
        )
        self.phase_left = self.np.zeros_like(self.phase_right)

    def sample(self, step):
        t = self.solver.amplitudes.t
        t_0 = t[0]
        t = t[1:]

        l = self.solver.amplitudes.l

        for i in range(self.num_amps):
            self.norm_t[i, step] = self.np.linalg.norm(t[i])
            self.norm_l[i, step] = self.np.linalg.norm(l[i])

        self.phase_right[step] = self.solver.compute_right_phase()
        self.phase_left[step] = self.solver.compute_left_phase()

    def dump(self, samples):
        t = self.solver.amplitudes.t
        t_0 = t[0]
        t = t[1:]

        l = self.solver.amplitudes.l

        for i in range(self.num_amps):
            samples["norm_t" + self.amp_keys[len(t[i].shape)]] = self.norm_t[i]
            samples["norm_l" + self.amp_keys[len(l[i].shape)]] = self.norm_l[i]

        samples["phase_right"] = self.phase_right
        samples["phase_left"] = self.phase_left

        return samples


# Create alias for the OA amplitude sampler
OATDCCAmplitudeSampler = TDCCAmplitudeSampler


class OATDCCDiagnosticsSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        np = self.np

        self.diff_rho_hermitian = np.zeros(self.num_samples)
        self.diff_biorthogonal = np.zeros(self.num_samples)
        self.diff_ctilde_cdagger = np.zeros(self.num_samples)
        self.cond_Aiajb = np.zeros(self.num_samples)
        self.det_Aiajb = np.zeros(self.num_samples)
        self.natural_occupation_numbers = np.zeros(
            (self.num_samples, self.system.l), dtype=np.complex128
        )
        self.occupation_numbers = np.zeros(
            (self.num_samples, self.system.l), dtype=np.complex128
        )

        self.ht_eps = np.zeros(
            (self.num_samples, self.system.l), dtype=np.complex128
        )

    def sample(self, step):
        from coupled_cluster.ccd.p_space_equations import compute_A_ibaj

        np = self.np

        rho_qp = self.solver.compute_one_body_density_matrix()
        self.diff_rho_hermitian[step] = np.linalg.norm(rho_qp - rho_qp.conj().T)
        self.occupation_numbers[step] = np.diagonal(rho_qp)

        Docc = rho_qp[self.system.o, self.system.o]
        Dvirt = rho_qp[self.system.v, self.system.v]

        eps_occ, Cocc = np.linalg.eig(Docc)
        eps_virt, Cvirt = np.linalg.eig(Dvirt)

        self.natural_occupation_numbers[step, 0 : self.system.n] = eps_occ
        self.natural_occupation_numbers[step, self.system.n :] = eps_virt

        t, l, C, Ctilde = self.solver.amplitudes
        self.diff_biorthogonal[step] = np.linalg.norm(
            C @ Ctilde - np.complex128(np.eye(C.shape[1]))
        )
        self.diff_ctilde_cdagger[step] = np.linalg.norm(Ctilde - C.conj().T)

        A_ibaj = compute_A_ibaj(rho_qp, self.system.o, self.system.v, np=np)
        A_iajb = A_ibaj.transpose(0, 2, 3, 1)
        A_iajb = A_iajb.reshape(
            self.system.o.stop * (self.system.v.stop - self.system.o.stop),
            self.system.o.stop * (self.system.v.stop - self.system.o.stop),
        )

        self.cond_Aiajb[step] = np.linalg.cond(A_iajb)
        self.det_Aiajb[step] = np.abs(np.linalg.det(A_iajb))

    def dump(self, samples):
        samples["diff_rho_hermitian"] = self.diff_rho_hermitian
        samples["diff_biorthogonal"] = self.diff_biorthogonal
        samples["diff_ctilde_cdagger"] = self.diff_ctilde_cdagger
        samples["cond_Aiajb"] = self.cond_Aiajb
        samples["det_Aiajb"] = self.det_Aiajb
        samples["natural_occupation_numbers"] = self.natural_occupation_numbers
        samples["occupation_numbers"] = self.occupation_numbers
        samples["ht_eps"] = self.ht_eps

        return samples


class TDCCSampleAll(SampleCollector):
    def __init__(self, solver, num_samples, np):
        super().__init__(
            [
                TDCCObservableSampler(solver, num_samples, np),
                TDCCAmplitudeSampler(solver, num_samples, np),
            ],
            np=np,
        )


class OATDCCSampleAll(SampleCollector):
    def __init__(self, solver, num_samples, np):
        super().__init__(
            [
                OATDCCObservableSampler(solver, num_samples, np),
                OATDCCAmplitudeSampler(solver, num_samples, np),
                OATDCCDiagnosticsSampler(solver, num_samples, np),
            ],
            np=np,
        )
