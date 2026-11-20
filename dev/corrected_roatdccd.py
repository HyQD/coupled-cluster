"""Development ROATDCCD variants for investigating Hamiltonian caching."""

from coupled_cluster.rccd import ROATDCCD


class CorrectedROATDCCD(ROATDCCD):
    """Cache laboratory integrals by time and transform them for every state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_lab_hamiltonian_time = None
        self.lab_hamiltonian_updates = 0
        self.orbital_transform_updates = 0

    def _resolve_orbitals(self, y, C, C_tilde):
        if y is not None:
            _, _, C, C_tilde = self._amp_template.from_array(y)
        elif C is None or C_tilde is None:
            raise ValueError(
                "either the amplitude array or both C and C_tilde are required"
            )
        return C, C_tilde

    def _update_lab_hamiltonian(self, current_time):
        if self.system.has_one_body_time_evolution_operator:
            self.h = self.system.h_t(current_time)
        if self.system.has_two_body_time_evolution_operator:
            self.u = self.system.u_t(current_time)
        self.lab_hamiltonian_updates += 1

    def _transform_hamiltonian(self, C, C_tilde):
        self.h_prime = self.system.transform_one_body_elements(
            self.h, C, C_tilde
        )
        self.u_prime = self.system.transform_two_body_elements(
            self.u, C, C_tilde
        )
        self.f_prime = self.system.construct_fock_matrix(
            self.h_prime, self.u_prime
        )
        self.orbital_transform_updates += 1

    def update_hamiltonian(self, current_time, y=None, C=None, C_tilde=None):
        C, C_tilde = self._resolve_orbitals(y, C, C_tilde)

        if self._last_lab_hamiltonian_time != current_time:
            self._update_lab_hamiltonian(current_time)
            self._last_lab_hamiltonian_time = current_time

        # These transformations depend on the current orbital state even when
        # current_time is unchanged.
        self._transform_hamiltonian(C, C_tilde)


class StaleROATDCCD(CorrectedROATDCCD):
    """Preserve the old time-only early return for comparison runs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hamiltonian_update_calls = 0
        self.stale_cache_hits = 0
        self.stale_cache_misses = 0

    def update_hamiltonian(self, current_time, y=None, C=None, C_tilde=None):
        self.hamiltonian_update_calls += 1
        if self.last_timestep == current_time:
            self.stale_cache_hits += 1
            return

        self.stale_cache_misses += 1
        self.last_timestep = current_time
        C, C_tilde = self._resolve_orbitals(y, C, C_tilde)
        self._update_lab_hamiltonian(current_time)
        self._last_lab_hamiltonian_time = current_time
        self._transform_hamiltonian(C, C_tilde)


class UncachedROATDCCD(CorrectedROATDCCD):
    """Recompute both laboratory and transformed integrals on every call."""

    def update_hamiltonian(self, current_time, y=None, C=None, C_tilde=None):
        C, C_tilde = self._resolve_orbitals(y, C, C_tilde)
        self._update_lab_hamiltonian(current_time)
        self._last_lab_hamiltonian_time = current_time
        self._transform_hamiltonian(C, C_tilde)
