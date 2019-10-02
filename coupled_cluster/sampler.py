import os

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

        self.energy = self.np.zeros(self.num_samples, dtype=self.system.h.dtype)

        self.dim = self.system.dipole_moment.shape[0]
        self.dipole_moment = self.np.zeros(
            (self.dim, self.num_samples), dtype=self.system.h.dtype
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
            samples[dipole_keys[i]] = self.dipole_moment[i]

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
        # TODO: Include phase

    def sample(self, step):
        t = self.solver.amplitudes.t
        t_0 = t[0]
        t = t[1:]

        l = self.solver.amplitudes.l

        for i in range(self.num_amps):
            self.norm_t[i, step] = self.np.linalg.norm(t[i])
            self.norm_l[i, step] = self.np.linalg.norm(l[i])

    def dump(self, samples):
        t = self.solver.amplitudes.t
        t_0 = t[0]
        t = t[1:]

        l = self.solver.amplitudes.l

        for i in range(self.num_amps):
            samples["norm_t" + amp_keys[len(t[i].shape)]] = self.norm_t[i]
            samples["norm_l" + amp_keys[len(l[i].shape)]] = self.norm_l[i]
