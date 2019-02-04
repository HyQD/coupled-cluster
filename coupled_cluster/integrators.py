import abc


class Integrator(metaclass=abc.ABCMeta):
    def __init__(self, rhs, np):
        self.np = np

        if not callable(rhs):
            rhs = lambda u, t: rhs

        self.rhs = rhs
        self.rhs_evals = 0

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

        self.rhs_evals += 4

        return u_new


class GaussIntegrator(Integrator):
    """
    Simple implemenation of a Gauss integrator,
    order 4 and 6 (s=2 and 3)."""

    def __init__(self, rhs, y0, t0, h, np, s=2, maxit=20, eps=1e-14, mu=1.75):
        assert maxit > 0

        super().__init__(rhs, np)

        np = self.np

        self.s = s
        self.h = h
        self.maxit = maxit
        self.eps = eps

        self.y0 = np.array(y0, dtype=complex)
        self.n = len(y0)
        self.t = t0
        self.y = np.array(self.y0, dtype=complex)
        self.y_prev = np.zeros(len(self.y0))

        self.F = np.zeros((self.n, self.s))
        self.Z = np.zeros((1, self.n, self.s), dtype=complex)

        self.a, self.b, self.c = gauss_tableau(self.s)

        # Compute starting guess method data
        self.mu = mu  # parameter to method B.

    def eval_rhs(self, y, t):
        self.rhs_evals += 1

        return self.rhs(y, t)

    def Z_solve(self, y, Z0):
        """
        Solve the problem Z = h*f(y + Z)*a^T by fix point iterations
        Use Z0 as initial guess, and maxit iterations, and residual norm
        tolerance eps.
        """
        Z = Z0

        converged = False

        for j in range(self.maxit):
            F = np.zeros((self.n, self.s), dtype=complex)
            for i in range(self.s):
                F[:, i] = self.eval_rhs(
                    self.y + Z[:, i], self.t + self.h * self.c[i]
                )

            Z_new = self.h * np.matmul(F, self.a.transpose())
            R = Z - Z_new
            Z = Z_new

            if np.linalg.norm(R) < self.eps:
                converged = True
                break

        assert converged, "Z iterations did not converge! Residual = " + str(
            np.linalg.norm(R)
        )

        return Z, F

    def step(self):
        """ Do a time step. Return (y,t) at next time step. """

        # Predict solution Z of nonlinear equations
        # Note that a fixed 8th order predictor is implemented

        # Compute interpolating polynomial w(t), that interpolates (t_{n-1},
        # y_{n-1}) and the points (t_{n-1}+c_i*h,Y_{n-1,i}).

        t_vec = (self.t - self.h) + np.append([0], self.h * self.c)
        t_vec2 = self.t + self.h * self.c

        W = np.zeros((self.n, self.s + 1), dtype=complex)
        W[:, 0] = self.y_prev

        for i in range(self.s):
            W[:, i + 1] = self.y_prev + self.Z[0, :, i]

        Y0 = barycentric_interpolate(t_vec, W.transpose(), t_vec2).transpose()

        # Save as initial guess Z0
        Z0 = np.zeros((self.n, self.s), dtype=complex)
        for i in range(self.s):
            Z0[:, i] = Y0[:, i] - self.y

        # Solve nonlinear equations
        Z_new, self.F = self.Z_solve(self.y, Z0)

        # Store solution for next predictor step
        self.Z[1:, :, :] = self.Z[:-1, :, :]
        self.Z[0, :, :] = np.array(Z_new)

        # save previous vector
        self.y_prev = np.array(self.y)

        # Make a step using Gauss method
        for i in range(self.s):
            self.y += self.h * self.b[i] * self.F[:, i]

        self.t += self.h

        return (self.y, self.t)
