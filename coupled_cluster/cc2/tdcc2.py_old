from coupled_cluster.cc2.tdcc import TimeDependentCoupledCluster
from coupled_cluster.cc2.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.cc2.rhs_l import (
    compute_l_1_amplitudes,
    compute_l_2_amplitudes,
)
from coupled_cluster.cc2 import CC2
from coupled_cluster.cc2.energies import (
    compute_time_dependent_energy,
    compute_cc2_ground_state_energy,
)
from coupled_cluster.cc2.density_matrices import (
    compute_one_body_density_matrix,
)
from coupled_cluster.cc2.time_dependent_overlap import (
    compute_time_dependent_overlap)
#New
from coupled_cluster.cc_helper import (
     AmplitudeContainer)


class TDCC2(TimeDependentCoupledCluster):
    def __init__(self, *args, **kwargs):
        super().__init__(CC2, *args, **kwargs)
        
        #print(help(TDCC2))
        print("The arguments in TDCC2 init")
        print(args[0])
        
        for keyword, value in kwargs.items():
             print("next key word arg")
             print(f'{keyword}={value}')

        system = args[0] 
        self.n = system.n
        self.m = system.m
       
        self.f_0 = self.f.copy()
        self.h_0 = self.h.copy()
        self.u_0 = self.u.copy()

        self.f_t1_transformed, self.u_t1_transformed  = self.cc.T1_transform_integrals()

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array(
            [compute_cc2_ground_state_energy(*args, **kwargs)]
        )

    def rhs_t_amplitudes(self):
        yield compute_t_1_amplitudes
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_1_amplitudes
        yield compute_l_2_amplitudes

    def left_reference_overlap(self):
        np = self.np

        t_0, t_1, t_2, l_1, l_2 = self._amplitudes.unpack()

        temp = np.einsum("ai, bj -> abij", t_1, t_1)
        temp -= temp.swapaxes(2, 3)
        temp -= temp.swapaxes(0, 1)

        return (
            1
            - 0.25 * np.tensordot(l_2, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
            - np.trace(l_1 @ t_1)
            + 0.125 * np.tensordot(l_2, temp, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
        )

    def compute_energy(self):

#Use transformed or not transformed f and u here? Used to be self.f and self.u 

        t_0, t_1, t_2, l_1, l_2 = self._amplitudes.unpack()

        return compute_time_dependent_energy(
            self.f_t1_transformed,
            self.u_t1_transformed,
            t_1,
            t_2,
            l_1,
            l_2,
            self.system.o,
            self.system.v,
            np=self.np,
        )

    def compute_one_body_density_matrix(self):
        t_0, t_1, t_2, l_1, l_2 = self._amplitudes.unpack()
        return compute_one_body_density_matrix(
            t_1, t_2, l_1, l_2, self.o, self.v, np=self.np
        )

    # TODO: Implement this?
    def compute_two_body_density_matrix(self):
        pass

    def compute_time_dependent_overlap(self, use_old=False):
        t_0, t_1, t_2, l_1, l_2 = self._amplitudes.unpack()

        return compute_time_dependent_overlap(
            self.cc.t_1,
            self.cc.t_2,
            self.cc.l_1,
            self.cc.l_2,
            t_0,
            t_1,
            t_2,
            l_1,
            l_2,
            np=self.np,
            use_old=use_old,
        )

    def __call__(self, prev_amp, current_time):

        o, v = self.system.o, self.system.v

        prev_amp = AmplitudeContainer.from_array(self._amplitudes, prev_amp)
        t_old, l_old = prev_amp

        self.update_hamiltonian(current_time, prev_amp)

        #Remove phase from t-amplitude list
        t_old = t_old[1:]
        #print("t_old")
        #print(t_old[1].shape)
        #print("f")
        #print(self.f)
        #print("u")
        #print(self.u)

        t_new = [
            -1j * rhs_t_func(self.f, self.f_t1_transformed, self.u_t1_transformed, *t_old, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = -1j * self.rhs_t_0_amplitude(
            self.f, self.u, *t_old, self.o, self.v, np=self.np
        )
        t_new = [t_0_new, *t_new]


        l_new = [
            1j * rhs_l_func(self.f, self.f_t1_transformed, self.u_t1_transformed, *t_old, *l_old, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        self.last_timestep = current_time

        return AmplitudeContainer(t=t_new, l=l_new, np=self.np).asarray()
