import numpy as np

from ..cc_helper import OACCVector
from . import OATDCCD


class OATDCCD_dvr(OATDCCD):
    def __init__(self, system, C=None, C_tilde=None, eps_reg=1e-6):
        """Assumes a system with dvr basis functions, with u saved in a
        sparse.COO-format, and converts it to an internal hacked version with
        u saved in a 2d numpy array. 
        """
        assert system._basis_set.u_repr == "2d"
        super().__init__(system, C=C, C_tilde=C_tilde, eps_reg=eps_reg)

    def initialize_arrays(self, l, l_prime):
        np = self.np

        if self.has_non_zero_Q_space:
            self.W = np.zeros((l, l_prime, l_prime), dtype=np.complex128)
            self.u_quart_ket = np.zeros(
                (l, l_prime, l_prime, l_prime), dtype=np.complex128
            )
            self.u_quart_bra = np.zeros(
                (l_prime, l_prime, l, l_prime), dtype=np.complex128
            )
        else:
            self.W = None
            self.u_quart_ket = None
            self.u_quart_bra = None

        self.h_prime = np.zeros((l_prime, l_prime), dtype=np.complex128)
        self.u_prime = np.zeros(
            (l_prime, l_prime, l_prime, l_prime), dtype=np.complex128
        )
        self.f_prime = np.zeros((l_prime, l_prime), dtype=np.complex128)

    @staticmethod
    def construct_mean_field_operator(u, C, C_tilde, np, **kwargs):
        """Constructs the mean field operator W from the (non-antisymmetric) coloumb
        integrals by converting one of the symmetry-reduced indices to the new basis
        given by C & C_tilde:
            W^{ar}_{s} = \sum_{g} u^{ag} \tilde C^r_g C^g_s.
        Here a and g are indices in the original basis, while r and s are indicies
        in the new basis
        """
        return np.einsum("rg,gs,ag->ars", C_tilde, C, u, optimize=True, **kwargs)

    @staticmethod
    def construct_W_partial_ket(W_ars, C, C_tilde, np, **kwargs):
        """Constructs an intermediate coloumb integral matrices with one unconverted
        index in the ket. For use in both the construction of the fully converted
        integrals and in the Q-space ket equations

        Parameters:
        W_ars : numpy array 
            shape (l, l_prime, l_prime)
        C : numpy array 
            shape (l, l_prime)
        C_tilde : numpy array 
            shape (l_prime, l)
        out_values : numpy array
            shapes (l, l_prime, l_prime, l_prime)
        **kwargs : supports out (see np.einsum)
        """
        return np.einsum("aq,ars->arqs", C, W_ars, **kwargs)

    @staticmethod
    def construct_W_partial_bra(W_ars, C, C_tilde, np, **kwargs):
        """Constructs an intermediate coloumb integral matrices with one unconverted
        index in the bra. For use in the Q-space bra equations

        Parameters:
        W_ars : numpy array 
            shape (l, l_prime, l_prime)
        C : numpy array 
            shape (l, l_prime)
        C_tilde : numpy array 
            shape (l_prime, l)
        out_values : numpy array
            shapes (l_prime, l_prime, l, l_prime) 
        **kwargs : supports out (see np.einsum)
        """
        return np.einsum("pa,ars->pras", C_tilde, W_ars, **kwargs)
