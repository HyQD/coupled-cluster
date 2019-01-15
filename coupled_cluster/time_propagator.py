class AmplitudeContainer:
    def __init__(self, l, t):
        if type(l) not in [list, tuple, set]:
            l = [l]

        self.l = l

        if type(t) not in [list, tuple, set]:
            t = [t]

        self.t = t

    def __add__(self, k):
        # Check if k is a constant to be added to all l- and t-amplitudes
        if type(k) not in [list, tuple, set, type(self)]:
            new_l = [l + k for l in self.l]
            new_t = [t + k for t in self.t]

            return AmplitudeContainer(new_l, new_t)

        # Assuming that k = [k_l, k_t], a list where each element should be
        # added to each amplitude in the l- and t-lists.
        k_l, k_t = k
        new_l = [l + _k_l for l, _k_l in zip(self.l, k_l)]
        new_t = [t + _k_t for t, _k_t in zip(self.t, k_t)]

        return AmplitudeContainer(new_l, new_t)

    def __mul__(self, k):
        # Check if k is a constant to be multiplied with all l- and t-amplitudes
        if type(k) not in [list, tuple, set, type(self)]:
            new_l = [l * k for l in self.l]
            new_t = [t * k for t in self.t]

            return AmplitudeContainer(new_l, new_t)

        # Assuming that k = [k_l, k_t], a list where each element should be
        # mulitplied with each amplitude in the l- and t-lists.
        k_l, k_t = k
        new_l = [l * _k_l for l, _k_l in zip(self.l, k_l)]
        new_t = [t * _k_t for t, _k_t in zip(self.t, k_t)]

        return AmplitudeContainer(new_l, new_t)

    def __iter__(self):
        yield self.l
        yield self.t


class TimePropagator:
    def __init__(self):
        pass
