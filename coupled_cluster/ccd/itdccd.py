from . import TDCCD
from ..cc_helper import AmplitudeContainer


class ITDCCD(TDCCD):
    """Performs imaginary time propagation for ground state calculations."""

    def __call__(self, prev_amp, current_time):
        o, v = self.system.o, self.system.v

        prev_amp = AmplitudeContainer.from_array(self._amplitudes, prev_amp)
        t_old, l_old = prev_amp

        self.update_hamiltonian(current_time, prev_amp)

        # Remove phase from t-amplitude list
        t_old = t_old[1:]

        t_new = [
            -rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = -self.rhs_t_0_amplitude(
            self.f, self.u, *t_old, self.o, self.v, np=self.np
        )
        t_new = [t_0_new, *t_new]

        l_new = [
            -rhs_l_func(self.f, self.u, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        self.last_timestep = current_time

        return AmplitudeContainer(t=t_new, l=l_new, np=self.np).asarray()
