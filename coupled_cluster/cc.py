import abc
import tqdm
import warnings

from coupled_cluster.cc_helper import (
    AmplitudeContainer,
    compute_reference_energy,
    compute_particle_density,
)
from coupled_cluster.mix import AlphaMixer, DIIS
from coupled_cluster.integrators import SimpleRosenbrock


class CoupledCluster(metaclass=abc.ABCMeta):
    """Coupled Cluster Abstract class
    
    Abstract base class defining the skeleton of a 
    Coupled Cluster ground state solver class.

    Parameters
    ----------
    system : QuantumSystems
        Quantum systems class instance
    mixer : AlphaMixer
        AlpaMixer object
    verbose : bool
        Prints iterations for ground state computation if True
    """

    def __init__(
        self,
        system,
        mixer=DIIS,
        integrator=SimpleRosenbrock,
        verbose=False,
        np=None,
    ):
        if np is None:
            import numpy as np

        self.np = np

        self.system = system
        self.verbose = verbose
        self.mixer = mixer
        self.integrator = integrator(np=self.np)

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        self._amplitudes = None

        self.o, self.v = self.system.o, self.system.v

    def get_amplitudes(self, get_t_0=False):
        """Getter for amplitudes

        Parameters
        ----------
        get_t_0 : bool
            Returns amplitude at t=0 if True

        Returns
        -------
        AmplitudeContainer
            Amplitudes in AmplitudeContainer object
        """

        if get_t_0:
            return AmplitudeContainer(
                t=[self.np.array([0]), *self._get_t_copy()],
                l=self._get_l_copy(),
                np=self.np,
            )

        return AmplitudeContainer(
            t=self._get_t_copy(), l=self._get_l_copy(), np=self.np
        )

    @abc.abstractmethod
    def _get_t_copy(self):
        pass

    @abc.abstractmethod
    def _get_l_copy(self):
        pass

    @abc.abstractmethod
    def compute_energy(self):
        pass

    @abc.abstractmethod
    def compute_one_body_density_matrix(self):
        pass

    @abc.abstractmethod
    def compute_two_body_density_matrix(self):
        pass

    @abc.abstractmethod
    def compute_t_amplitudes(self):
        pass

    @abc.abstractmethod
    def compute_l_amplitudes(self):
        pass

    @abc.abstractmethod
    def setup_l_mixer(self, **kwargs):
        pass

    @abc.abstractmethod
    def setup_t_mixer(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_l_residuals(self):
        pass

    @abc.abstractmethod
    def compute_t_residuals(self):
        pass

    def compute_particle_density(self):
        """Computes one-particle density

        Returns
        -------
        np.array
            Particle density        
        """
        np = self.np

        rho_qp = self.compute_one_body_density_matrix()

        if np.abs(np.trace(rho_qp) - self.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.n)
            warnings.warn(warn)

        rho = compute_particle_density(
            rho_qp, self.system.bra_spf, self.system.spf, np=np
        )

        return rho

    def compute_reference_energy(self):
        """Computes reference energy

        Returns
        -------
        np.array
            Reference energy
        """

        return compute_reference_energy(
            self.f, self.u, self.o, self.v, np=self.np
        )

    def compute_ground_state(
        self, t_args=[], t_kwargs={}, l_args=[], l_kwargs={}
    ):
        """Compute ground state energy
        """
        self.iterate_t_amplitudes(*t_args, **t_kwargs)
        self.iterate_l_amplitudes(*l_args, **l_kwargs)

    def iterate_l_amplitudes(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):
        np = self.np

        if not np in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.setup_l_mixer(**mixer_kwargs)

        for i in range(max_iterations):
            self.compute_l_amplitudes()
            residuals = self.compute_l_residuals()

            if self.verbose:
                print(f"Iteration: {i}\tResiduals (l): {residuals}")

            if all(res < tol for res in residuals):
                break

    def iterate_t_amplitudes(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):
        np = self.np

        if not np in mixer_kwargs:
            mixer_kwargs["np"] = np

        self.setup_t_mixer(**mixer_kwargs)

        for i in range(max_iterations):
            self.compute_t_amplitudes()
            residuals = self.compute_t_residuals()

            if self.verbose:
                print(f"Iteration: {i}\tResiduals (t): {residuals}")

            if all(res < tol for res in residuals):
                break

    def propagate_to_ground_state(
        self, t_args=[], t_kwargs={}, l_args=[], l_kwargs={}
    ):
        """Propagate in complex time to ground state
        """
        self.propagate_t_amplitudes(*t_args, **t_kwargs)
        self.propagate_l_amplitudes(*t_args, **t_kwargs)

    def setup_t_integrator(self, **kwargs):
        if self.t_integrator is None:
            self.t_integrator = self.integrator.set_rhs(self.call_t)

        if isinstance(self.t_integrator, SimpleRosenbrock):
            self.t_integrator.set_rhs_der(self.t_rhs_der)

    def setup_l_integrator(self, **kwargs):
        if self.l_integrator is None:
            self.l_integrator = self.integrator.set_rhs(self.call_l)

        if isinstance(self.l_integrator, SimpleRosenbrock):
            self.l_integrator.set_rhs_der(self.l_rhs_der)

    def call_t(self, prev_amp, current_time):
        """Like __call__ for tdcc
        
        Unlike call, it only returns the function for tau and there is 
        no imaginary factor or t0 phase amplitude. 
        Used for complex time propagation. current_time is a dummy to 
        conform to integrators
        """
        o, v = self.system.o, self.system.v

        t_old = self.t_shape(prev_amp)

        t_new = [
            -rhs_t_func(self.f, self.u, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        return self.u_flat(*t_new)

    def call_l(self, prev_amp, current_time):
        """Like __call__ for tdcc
        
        Unlike call, it only returns the function for lambda and there is 
        no imaginary factor or t0 phase amplitude. 
        Used for complex time propagation. current_time is a dummy to 
        conform to integrators
        """
        o, v = self.system.o, self.system.v

        l_old = self.l_shape(prev_amp)

        l_new = [
            -rhs_l_func(self.f, self.u, *self.get_t_amps(), *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        return self.u_flat(*l_new)

    def propagate_t_amplitudes(
        self, eps=0.005, max_steps=100, tol=1e-6, max_dt=1, **integrate_kwargs
    ):
        """Propagate t amplitudes in complex time to convergence. 
        """

        np = self.np

        if not np in integrate_kwargs:
            integrate_kwargs["np"] = np

        self.setup_t_integrator(**integrate_kwargs)

        dt = np.sqrt(eps)

        prev_amp = self.get_zero_vec()
        prev_error = self.get_zero_vec()

        for i in range(max_steps):

            amp_vec = self.t_integrator.step(self.get_t_flat(),0,dt)
            error_vec = self.t_integrator.get_residual()

            self.update_t_and_rhs(amp_vec, error_vec)

            residuals = self.compute_t_residuals()

            if self.verbose:
                print(
                    f"Iteration: {i:{3}}\t delta t: {dt:.4f}\tResiduals (t): "
                    f"{residuals[0]:#.8g}  {residuals[1]:#.8g}"
                )

            if all(res < tol for res in residuals):
                break

            dt_norm = np.linalg.norm(amp_vec - prev_amp)
            rhs_norm = np.linalg.norm(error_vec - prev_error)

            dt = np.sqrt(2 * eps * dt_norm / rhs_norm)
            if dt > max_dt:
                dt = max_dt

            prev_amp = amp_vec
            prev_error = error_vec


    def propagate_l_amplitudes(
        self, eps=0.005, max_steps=100, tol=1e-6, max_dt=1, **integrate_kwargs
    ):
        """Propagate t amplitudes in complex time to convergence. 
        """

        np = self.np

        if not np in integrate_kwargs:
            integrate_kwargs["np"] = np

        self.setup_l_integrator(**integrate_kwargs)

        dt = np.sqrt(eps)

        prev_amp = self.get_zero_vec()
        prev_error = self.get_zero_vec()

        for i in range(max_steps):

            amp_vec = self.l_integrator.step(self.get_l_flat(),0,dt)
            error_vec = self.l_integrator.get_residual()

            self.update_l_and_rhs(amp_vec, error_vec)

            residuals = self.compute_l_residuals()

            if self.verbose:
                print(f"Iteration: {i:{3}}\t delta t: {dt:.4f}\tResiduals (l): "\
                      f"{residuals[0]:#.8g}  {residuals[1]:#.8g}")

            if all(res < tol for res in residuals):
                break

            dl_norm  = np.linalg.norm(amp_vec - prev_amp)
            rhs_norm = np.linalg.norm(error_vec - prev_error)
            
            dt = np.sqrt(2*eps*dl_norm/rhs_norm)
            if(dt > max_dt):
                dt = max_dt

            prev_amp = amp_vec
            prev_error = error_vec

