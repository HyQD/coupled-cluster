import abc


class Integrator(metaclass=abc.ABCMeta):
    def __init__(self, rhs, np):
        self.np = np

        if not callable(rhs):
            rhs = lambda u, t: rhs

        self.rhs = rhs

    @abc.abstractmethod
    def step(self):
        pass


class RungeKutta4(Integrator):
    def step(self, u, t, dt):
        f = self.rhs

        K1 = dt * f(u, t)
        K2 = dt * f(u + 0.5 * K1, t + 0.5 * dt)
        K3 = dt * f(u + 0.5 * K2, t + 0.5 * dt)
        K4 = dt * f(u + K3, t + dt)
        u_new = u + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)

        return u_new
