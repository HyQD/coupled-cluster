class AmplitudeContainer:
    """Container for Amplitude functions

    Parameters
    ----------
    t : list, tuple, set
        Tau amplitudes
    l : list, tuple, set
        Lambda amplitude
    np : module
        Matrix library to be used, e.g., numpy, cupy, etc.
    """

    def __init__(self, t, l, np):
        self.np = np
        self.n = 0

        if type(t) not in [list, tuple, set]:
            t = [t]

        self._t = t

        for _t in self._t:
            self.n += _t.size

        if type(l) not in [list, tuple, set]:
            l = [l]

        self._l = l

        for _l in self._l:
            self.n += _l.size

    @property
    def t(self):
        return self._t

    @property
    def l(self):
        return self._l

    def __add__(self, k):
        # Check if k is a constant to be added to all l- and t-amplitudes
        if type(k) not in [list, tuple, set, type(self)]:
            new_t = [t + k for t in self._t]
            new_l = [l + k for l in self._l]

            return AmplitudeContainer(new_t, new_l, np=self.np)

        # Assuming that k = [k_t, k_l], a list where each element should be
        # added to each amplitude in the l- and t-lists.
        k_t, k_l = k
        new_t = [t + _k_t for t, _k_t in zip(self._t, k_t)]
        new_l = [l + _k_l for l, _k_l in zip(self._l, k_l)]

        return AmplitudeContainer(new_t, new_l, np=self.np)

    def __radd__(self, k):
        return self.__add__(k)

    def __mul__(self, k):
        # Check if k is a constant to be multiplied with all l- and t-amplitudes
        if type(k) not in [list, tuple, set, type(self)]:
            new_t = [t * k for t in self._t]
            new_l = [l * k for l in self._l]

            return AmplitudeContainer(new_t, new_l, np=self.np)

        # Assuming that k = [k_t, k_l], a list where each element should be
        # mulitplied with each amplitude in the l- and t-lists.
        k_t, k_l = k
        new_t = [t * _k_t for t, _k_t in zip(self._t, k_t)]
        new_l = [l * _k_l for l, _k_l in zip(self._l, k_l)]

        return AmplitudeContainer(new_t, new_l, np=self.np)

    def __rmul__(self, k):
        return self.__mul__(k)

    def __iter__(self):
        yield self._t
        yield self._l

    def unpack(self):
        """
        :rtype: Iterator
        """
        yield from self._t
        yield from self._l

    def asarray(self):
        """Returns amplitudes as numpy array

        Returns
        -------
        np.array
            Amplitude vector
        """
        np = self.np

        amp_vec = np.zeros(self.n)
        start_index = 0
        stop_index = 0

        for amp in self.unpack():
            start_index = stop_index
            stop_index += amp.size

            try:
                amp_vec[start_index:stop_index] += amp.ravel()
            except TypeError:
                amp_vec = amp_vec.astype(amp.dtype)
                amp_vec[start_index:stop_index] += amp.ravel()

        return amp_vec

    def zeros_like(self):
        np = self.np

        args = []
        for amps in self:
            inner = []

            if type(amps) == list:
                inner = [np.zeros_like(amp) for amp in amps]
            else:
                inner = np.zeros_like(amps)

            args.append(inner)

        return type(self)(*args, np=np)

    def from_array(self, arr):
        np = self.np

        args = []
        start_index = 0
        stop_index = 0

        for amps in self:
            inner = []

            if type(amps) == list:
                for amp in amps:
                    start_index = stop_index
                    stop_index += amp.size

                    inner.append(arr[start_index:stop_index].reshape(amp.shape))
            else:
                start_index = stop_index
                stop_index += amps.size
                inner = arr[start_index:stop_index].reshape(amps.shape)

            args.append(inner)

        return type(self)(*args, np=np)

    def residuals(self):
        return [
            [np.linalg.norm(t) for t in self.t],
            [np.linalg.norm(l) for l in self.l],
        ]


class OACCVector(AmplitudeContainer):
    """Container for OA amplitudes

    This is a container for the amplitudes, t and l, and the orbital
    transformation coefficients C and C_tilde.

    Parameters
    ----------
    t: ?
        Tau amplitudes
    l: ?
        Lambda amplitude
    C: ?
        RHS coefficient matrix
    C_tilde: ?
        LHS coefficient matrix
    """

    def __init__(self, t, l, C, C_tilde, np):
        super().__init__(t=t, l=l, np=np)

        self._C = C
        self._C_tilde = C_tilde

        self.n += self._C.size
        self.n += self._C_tilde.size

    @property
    def C(self):
        return self._C

    @property
    def C_tilde(self):
        return self._C_tilde

    def __add__(self, k):
        # Check if k is a constant to be added to all l- and t-amplitudes and
        # coefficients.
        if type(k) not in [list, tuple, set, type(self)]:
            new_t = [t + k for t in self._t]
            new_l = [l + k for l in self._l]
            new_C = self._C + k
            new_C_tilde = self._C_tilde + k

            return OACCVector(new_t, new_l, new_C, new_C_tilde, np=self.np)

        # Assuming that k = [k_t, k_l, k_C, k_C_tilde], a list where each
        # element should be added to each amplitude in the l- and t-lists and
        # each of the coefficients.
        k_t, k_l, k_C, k_C_tilde = k
        new_t = [t + _k_t for t, _k_t in zip(self._t, k_t)]
        new_l = [l + _k_l for l, _k_l in zip(self._l, k_l)]
        new_C = self._C + k_C
        new_C_tilde = self._C_tilde + k_C_tilde

        return OACCVector(new_t, new_l, new_C, new_C_tilde, np=self.np)

    def __mul__(self, k):
        # Check if k is a constant to be multiplied with all l- and t-amplitudes
        # and coefficients.
        if type(k) not in [list, tuple, set, type(self)]:
            new_t = [t * k for t in self._t]
            new_l = [l * k for l in self._l]
            new_C = self._C * k
            new_C_tilde = self._C_tilde * k

            return OACCVector(new_t, new_l, new_C, new_C_tilde, np=self.np)

        # Assuming that k = [k_t, k_l, k_C, k_C_tilde], a list where each
        # element should be multiplied with each amplitude in the l- and t-lists
        # and each of the coefficients.
        k_t, k_l, k_C, k_C_tilde = k
        new_t = [t * _k_t for t, _k_t in zip(self._t, k_t)]
        new_l = [l * _k_l for l, _k_l in zip(self._l, k_l)]
        new_C = self._C * k_C
        new_C_tilde = self._C_tilde * k_C_tilde

        return OACCVector(new_t, new_l, new_C, new_C_tilde, np=self.np)

    def __iter__(self):
        yield self._t
        yield self._l
        yield self._C
        yield self._C_tilde

    def unpack(self):
        yield from super().unpack()
        yield self._C
        yield self._C_tilde


def compute_reference_energy(f, u, o, v, np):
    """
    Computes reference energy

    Parameters
    ----------
    f: np.array
        One-particle operator (Fock matrix)
    u: np.array
        Two-particle operator
    o: Slice
        Occupied orbitals
    v: Slice
        Virtual orbitals
    np: Module
        Matrix library

    Returns
    -------
    np.float
        Reference energy
    """
    return np.trace(f[o, o]) - 0.5 * np.trace(
        np.trace(u[o, o, o, o], axis1=1, axis2=3)
    )


def construct_d_t_1_matrix(f, o, v, np):
    f_diag = np.diag(f)
    d_t_1 = f_diag[o] - f_diag[v].reshape(-1, 1)

    return d_t_1


def construct_d_t_2_matrix(f, o, v, np):
    f_diag = np.diag(f)
    d_t_2 = (
        f_diag[o]
        + f_diag[o].reshape(-1, 1)
        - f_diag[v].reshape(-1, 1, 1)
        - f_diag[v].reshape(-1, 1, 1, 1)
    )

    return d_t_2
