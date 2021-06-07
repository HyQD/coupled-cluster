TRUNCATION_CODES = {"S": 1, "D": 2, "T": 3, "Q": 4, "5": 5, "6": 6}
INV_TRUNCATION_CODES = {v: k for k, v in TRUNCATION_CODES.items()}


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

    @staticmethod
    def construct_amplitude_template(truncation, n, m, np, dtype=complex):
        r"""Constructs an empty ``AmplitudeContainer`` with the correct shapes,
        for conversion between arrays and amplitudes.

        Parameters
        ----------
        truncation : str
            String of the form ``CCXYZ...`` where ``XYZ...`` specifies the
            coupled-cluster truncation. For example ``CCSD`` for the singles-
            and doubles-truncation.
        n : int
            Number of occupied orbitals.
        m : int
            Number of virtual oritals.
        np : module
            Array module, often NumPy.
        dtype : type
            Data type for the elements in the amplitude arrays. Default is
            ``complex``.

        Returns
        -------
        AmplitudeContainer
            An instatiated ``AmplitudeContainer`` with the necessary amplitudes
            for the specified truncation level.


        >>> import numpy as np
        >>> n, m = 4, 6
        >>> amps = AmplitudeContainer.construct_amplitude_template(
        ...     "CCSD", n, m, np, dtype=complex
        ... )
        >>> amps.t[0].shape
        (1,)
        >>> amps.t[1].shape
        (6, 4)
        >>> amps.t[2].shape
        (6, 6, 4, 4)
        >>> amps.l[0].shape
        (4, 6)
        >>> amps.l[1].shape
        (4, 4, 6, 6)
        >>> amps.l[1].dtype
        dtype('complex128')
        """

        levels = [TRUNCATION_CODES[c] for c in truncation[2:]]

        # start with t_0
        t = [np.array([0], dtype=dtype)]
        l = []

        for level in levels:
            shape = level * [m] + level * [n]
            t.append(np.zeros(shape, dtype=dtype))
            l.append(np.zeros(shape[::-1], dtype=dtype))
        return AmplitudeContainer(t=t, l=l, np=np)

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

    @staticmethod
    def construct_container_from_array(
        arr,
        truncation,
        n,
        m,
        np,
    ):
        r"""Method constructing an ``AmplitudeContainer`` from an array.  This
        function assumes that the elements in the array are the flattened
        amplitudes sorted as [t_0, t_1, t_2, ..., l_1, l_2, ...] in a long flat
        array.

        Parameters
        ----------
        arr : np.array
            Array with the flattened amplitudes.
        truncation : str
            String of the form ``CCXYZ...`` where ``XYZ...`` specifies the
            coupled-cluster truncation. For example ``CCSD`` for the singles-
            and doubles-truncation.
        n : int
            Number of occupied orbitals.
        m : int
            Number of virtual oritals.
        np : module
            Array module, often NumPy.

        Returns
        -------
        AmplitudeContainer
            A filled ``AmplitudeContainer`` built from ``arr``.

        >>> import numpy as np
        >>> n, m = 4, 6
        >>> amps = AmplitudeContainer.construct_amplitude_template(
        ...     "CCSD", n, m, np, dtype=complex
        ... )
        >>> arr = amps.asarray()
        >>> amps_2 = AmplitudeContainer.construct_container_from_array(
        ...     arr, "CCSD", n, m, np
        ... )
        >>> for t, t_2 in zip(amps.t, amps_2.t):
        ...     assert np.linalg.norm(t - t_2) < 1e-12
        >>> for l, l_2 in zip(amps.l, amps_2.l):
        ...     assert np.linalg.norm(l - l_2) < 1e-12
        """

        import math

        # Pick out t_0
        t = [np.array([arr[0]])]
        l = []

        assert (len(arr) - 1) % 2 == 0

        start_t_index = 1
        start_l_index = int((len(arr) - 1) / 2) + 1
        levels = [TRUNCATION_CODES[c] for c in truncation[2:]]

        for level in levels:
            shape = level * [m] + level * [n]
            stop_t_index = start_t_index + math.prod(shape)
            stop_l_index = start_l_index + math.prod(shape)
            t.append(arr[start_t_index:stop_t_index].reshape(shape))
            l.append(arr[start_l_index:stop_l_index].reshape(shape[::-1]))
            start_t_index = stop_t_index
            start_l_index = stop_l_index

        return AmplitudeContainer(t=t, l=l, np=np)

    def from_array(self, arr):
        levels = [int(len(t.shape) / 2) for t in self._t[1:]]
        truncation = "CC" + "".join(
            [INV_TRUNCATION_CODES[level] for level in levels]
        )

        return self.construct_container_from_array(
            arr, truncation, self._t[1].shape[-1], self._t[1].shape[0], self.np
        )

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

    @staticmethod
    def construct_amplitude_template(
        truncation, n_prime, m_prime, l, np, dtype=complex
    ):
        r"""Constructs an empty ``OACCVector`` with the correct shapes, for
        conversion between arrays and amplitudes and coefficients.
        This construction supports variable sizes for the untransformed atomic
        orbital basis and the orbital optimized basis.

        Parameters
        ----------
        truncation : str
            String of the form ``CCXYZ...`` where ``XYZ...`` specifies the
            coupled-cluster truncation. For example ``CCSD`` for the singles-
            and doubles-truncation.
        n_prime : int
            Number of occupied optimized orbitals.
        m_prime : int
            Number of virtual optimized oritals.
        l : int
            Number of atomic orbitals, i.e., the untransformed basis.
        np : module
            Array module, often NumPy.
        dtype : type
            Data type for the elements in the amplitude arrays. Default is
            ``complex``.

        Returns
        -------
        OACCVector
            An instatiated ``OACCVector`` with the necessary amplitudes and
            coefficient matrices for the specified truncation level.

        See Also
        --------
        AmplitudeContainer


        >>> import numpy as np
        >>> n_prime, m_prime, l = 4, 6, 20
        >>> oa_vec = OACCVector.construct_amplitude_template(
        ...     "CCSD", n_prime, m_prime, l, np, dtype=complex
        ... )
        >>> oa_vec.t[0].shape
        (1,)
        >>> oa_vec.t[1].shape
        (6, 4)
        >>> oa_vec.t[2].shape
        (6, 6, 4, 4)
        >>> oa_vec.l[0].shape
        (4, 6)
        >>> oa_vec.l[1].shape
        (4, 4, 6, 6)
        >>> oa_vec.C.shape
        (20, 10)
        >>> oa_vec.C.shape == oa_vec.C_tilde.T.shape
        True
        >>> oa_vec.l[1].dtype
        dtype('complex128')
        """

        amps = AmplitudeContainer.construct_amplitude_template(
            truncation, n_prime, m_prime, np, dtype=dtype
        )

        shape = (l, l) if n_prime + m_prime == l else (l, n_prime + m_prime)

        C = np.zeros(shape, dtype=dtype)
        C_tilde = C.T.copy()

        return OACCVector(*amps, C, C_tilde, np=np)

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
