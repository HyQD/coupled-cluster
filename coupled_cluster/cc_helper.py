class AmplitudeContainer:
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
        yield from self._t
        yield from self._l

    def asarray(self):
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

    @staticmethod
    def zeros_like(u):
        np = u.np

        args = []
        for amps in u:
            inner = []

            if type(amps) == list:
                inner = [np.zeros_like(amp) for amp in amps]
            else:
                inner = np.zeros_like(amps)

            args.append(inner)

        return type(u)(*args, np=np)

    @staticmethod
    def from_array(u, arr):
        np = u.np

        args = []
        start_index = 0
        stop_index = 0

        for amps in u:
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

        return type(u)(*args, np=np)


class OACCVector(AmplitudeContainer):
    """This is a container for the amplitudes, t and l, and the orbital
    transformation coefficients C and C_tilde.
    """

    def __init__(self, t, l, C, C_tilde, np):
        super().__init__(t=t, l=l, np=np)

        self._C = C
        self._C_tilde = C_tilde

        self.n += self._C.size
        self.n += self._C_tilde.size

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
    return np.trace(f[o, o]) - 0.5 * np.trace(
        np.trace(u[o, o, o, o], axis1=1, axis2=3)
    )


def compute_spin_reduced_one_body_density_matrix(rho_qp):
    return rho_qp[::2, ::2] + rho_qp[1::2, 1::2]


def compute_particle_density(rho_qp, spf, np):
    rho = np.zeros(spf.shape[1:], dtype=spf.dtype)
    spf_slice = slice(0, spf.shape[0])

    for _i in np.ndindex(rho.shape):
        i = (spf_slice, *_i)
        rho[_i] += np.dot(spf[i].conj(), np.dot(rho_qp, spf[i]))

    return rho


def remove_diagonal_in_matrix(matrix, np):
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag, 0)

    return off_diag
