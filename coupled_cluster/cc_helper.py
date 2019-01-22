class AmplitudeContainer:
    def __init__(self, t, l):
        if type(t) not in [list, tuple, set]:
            t = [t]

        self._t = t

        if type(l) not in [list, tuple, set]:
            l = [l]

        self._l = l

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

            return AmplitudeContainer(new_t, new_l)

        # Assuming that k = [k_t, k_l], a list where each element should be
        # added to each amplitude in the l- and t-lists.
        k_t, k_l = k
        new_t = [t + _k_t for t, _k_t in zip(self._t, k_t)]
        new_l = [l + _k_l for l, _k_l in zip(self._l, k_l)]

        return AmplitudeContainer(new_t, new_l)

    def __radd__(self, k):
        return self.__add__(k)

    def __mul__(self, k):
        # Check if k is a constant to be multiplied with all l- and t-amplitudes
        if type(k) not in [list, tuple, set, type(self)]:
            new_t = [t * k for t in self._t]
            new_l = [l * k for l in self._l]

            return AmplitudeContainer(new_t, new_l)

        # Assuming that k = [k_t, k_l], a list where each element should be
        # mulitplied with each amplitude in the l- and t-lists.
        k_t, k_l = k
        new_t = [t * _k_t for t, _k_t in zip(self._t, k_t)]
        new_l = [l * _k_l for l, _k_l in zip(self._l, k_l)]

        return AmplitudeContainer(new_t, new_l)

    def __rmul__(self, k):
        return self.__mul__(k)

    def __iter__(self):
        yield self._t
        yield self._l

    def unpack(self):
        yield from self._t
        yield from self._l
