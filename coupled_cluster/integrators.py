import abc
from scipy.interpolate import barycentric_interpolate


class Integrator(metaclass=abc.ABCMeta):
    """Integrator Abstract Parent Class

    Sub-class must impement step-method
    """

    def __init__(self, np):
        self.np = np

    def set_rhs(self, rhs):
        if not callable(rhs):
            rhs = lambda u, t: rhs

        self.rhs = rhs
        self.rhs_evals = 0

        return self

    @abc.abstractmethod
    def step(self, u, t, dt):
        pass


class RungeKutta4(Integrator):
    """The Classical Runge-Kutta Method

    """

    def step(self, u, t, dt):
        """One integration step

        Parameters
        ----------
        u : np.array
            Flattened array of amplitudes    
        t : float
            Current time step
        dt : float
            Time step size (h)
        
        Returns
        -------
        np.array
            New flattened array of amplitudes after integration
        """

        f = self.rhs

        K1 = dt * f(u, t)
        K2 = dt * f(u + 0.5 * K1, t + 0.5 * dt)
        K3 = dt * f(u + 0.5 * K2, t + 0.5 * dt)
        K4 = dt * f(u + K3, t + dt)
        u_new = u + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)

        self.rhs_evals += 4

        return u_new


class SimpleRosenbrock(Integrator):
    """The Most Simple Rosenbrock method

    Similar to implicit Euler, but K1 = f(u + dt*K1) is 
    not solved exactly. Instead, the newton iteration is 
    terminated after the first step. This can be approximated 
    further by using an approximate derivative of f in the 
    Newton solver.

    """

    def set_rhs_der(self, rhs_der, diagonal=True):
        """Set derivative of rhs

        In practice, calculating the full Jacobian of rhs will be too expensive to do in CC,
        especially since it has to be inverted, so a diagonal approximation will be used. 
        """

        if not callable(rhs_der):
            rhs = lambda u, t: rhs_der

        self.rhs_der = rhs_der

    def step(self, u, t, dt):
        """One integration step

        Parameters
        ----------
        u : np.array
            Flattened array of amplitudes    
        t : float
            Current time step
        dt : float
            Time step size (h)
        
        Returns
        -------
        np.array
            New flattened array of amplitudes after integration
        """

        f = self.rhs

        K1 = dt * f(u, t)
        K2 = dt * f(u + K1, t + dt)

        if diag:
            u_new = u + dt * (K1 + (K2 - K1) / (1 - dt * rhs_der(u, t)))
        else:
            inv_mat = np.linalg.inv(np.identity(len(a)) - dt * rhs_der(u, t))
            u_new = u + dt * (K1 + inv_mat * (K2 - K1))

        self.rhs_evals += 2

        return u_new


class GaussIntegrator(Integrator):
    """Gaussian Quadrature
    
    Simple implementation of a symplectic Gauss integrator,
    order 4 and 6 (s=2 and s=3).

    Note, this is a modified code recieved from Simen Kvaal and
    Thomas Bondo Pedersen.
    
    Parameters
    ----------
    np : module
        Matrix / Linear algebra library; eg numpy, cupy
    s : int
        order = 2 * s
    maxit : int
        Maximum number of iterations
    eps : float
        Tolerance parameter; e.g. 1e-8
    """

    def __init__(self, np, s=2, maxit=20, eps=1e-14):
        assert maxit > 0

        super().__init__(np)

        self.s = s
        self.maxit = maxit
        self.eps = eps

        self.y = None
        self.y_prev = None

        self.a, self.b, self.c = gauss_tableau(self.s, np=self.np)

    def eval_rhs(self, y, t):
        self.rhs_evals += 1

        return self.rhs(y, t)

    def Z_solve(self, y, Z0, t, dt):
        """Solver method

        Solves the problem Z = h*f(y + Z)*a^T by fix point iterations. 
        Use Z0 as initial guess, and maxit iterations, and residual norm tolerance eps.
        
        Parameters
        ----------
        y : np.array
            Flattened array of amplitudes ??
        Z0 : np.array
            Initial guess
        t : float
            Current time step
        dt : float
            Time step length
        """

        np = self.np

        Z = Z0
        converged = False

        for j in range(self.maxit):
            F = np.zeros((self.n, self.s), dtype=np.complex128)
            for i in range(self.s):
                F[:, i] = self.eval_rhs(y + Z[:, i], t + dt * self.c[i])

            Z_new = dt * np.matmul(F, self.a.transpose())
            R = Z - Z_new
            Z = Z_new

            if np.linalg.norm(R) < self.eps:
                converged = True
                break

        assert converged, "Z iterations did not converge! Residual = " + str(
            np.linalg.norm(R)
        )

        return Z, F

    def step(self, u, t, dt):
        """One integration step

        Parameters
        ----------
        y : np.array
            Flattened array of amplitudes ??
        Z0 : np.array
            Initial guess
        t : float
            Current time step
        dt : float
            Time step length

        Returns
        -------
        np.array
            New flattened array of amplitudes after integration step.
        """

        # Predict solution Z of nonlinear equations
        # Note that a fixed 8th order predictor is implemented

        # Compute interpolating polynomial w(t), that interpolates (t_{n-1},
        # y_{n-1}) and the points (t_{n-1}+c_i*h,Y_{n-1,i}).

        np = self.np

        self.y = u.astype(np.complex128)
        self.n = len(self.y)

        if self.y_prev is None:
            self.y_prev = np.zeros_like(self.y)
            self.y_prev += self.y
            self.Z = np.zeros((1, self.n, self.s), dtype=np.complex128)

        t_vec = (t - dt) + np.append([0], dt * self.c)
        t_vec2 = t + dt * self.c

        W = np.zeros((self.n, self.s + 1), dtype=np.complex128)
        W[:, 0] += self.y_prev

        for i in range(self.s):
            W[:, i + 1] = self.y_prev + self.Z[0, :, i]

        Y0 = barycentric_interpolate(t_vec, W.transpose(), t_vec2).transpose()

        # Save as initial guess Z0
        Z0 = np.zeros((self.n, self.s), dtype=np.complex128)
        for i in range(self.s):
            Z0[:, i] = Y0[:, i] - self.y

        # Solve nonlinear equations
        Z_new, self.F = self.Z_solve(self.y, Z0, t, dt)

        # Store solution for next predictor step
        self.Z[1:, :, :] = self.Z[:-1, :, :]
        self.Z[0, :, :] = np.array(Z_new)

        # save previous vector
        self.y_prev = np.array(self.y)

        # Make a step using Gauss method
        for i in range(self.s):
            self.y += dt * self.b[i] * self.F[:, i]

        return self.y


def gauleg(n, np):
    """Compute weights and abscissa for Gauss-Legendre quadrature.

    Adapted from an old MATLAB code from 2011 by S. Kvaal.

    Usage: x,w = gauleg(n)

    Uses Golub-Welsh algorithm (Golub & Welsh, Mathematics of Computation, Vol.
    23, No. 106, (Apr., 1969), pp. 221-230)

    In the algorithm, one computes a tridiagonal matrix T, whose elements are
    given by the recurrence coefficients of the orthogonal polynomials one
    wishes to compute Gauss-quadrature rules from.  Thus, gauleg is easily
    adaptable to other orthogonal polynomials.
    """
    nn = np.arange(1, n + 1)

    a = np.sqrt((2 * nn - 1) * (2 * nn + 1)) / nn
    b = 0 * nn
    temp = (2 * nn + 1) / (2 * nn - 3)
    temp = (
        np.abs(temp) + temp
    ) / 2  # hack to remove negative indices, which are not
    # used but still give a runtime warning in np.sqrt
    c = (nn - 1) / nn * np.sqrt(temp)

    alpha = -b / a
    beta = np.sqrt(c[1:] / (a[:-1] * a[1:]))

    mu0 = 2

    J = np.diag(beta, -1) + np.diag(alpha) + np.diag(beta, 1)
    v, u = np.linalg.eig(J)
    j = np.argsort(v)
    w = mu0 * u[0, :] ** 2
    w = w[j]
    x = v[j]

    return x, w


def lagpol(c, j, np):
    """Compute Lagrange interpolation polynomial.

    Usage:  p = lagpol(c,j)

    Given a vector of collocation points c[i], compute the j'th
    Lagrange interpolation polynomial.

    Returns a np.poly1d object.
    """

    r = np.delete(c, j)
    a = np.prod(c[j] - r)
    p = np.poly1d(r, r=True) / a

    return p


def gauss_tableau(s, np):
    """Compute Butcher Tableau of s-stage Gauss integrator.

    Usage a,b,c = gauss_tableau(s)
    """

    # compute collocation points and weights
    c, b = gauleg(s, np=np)
    c = (c + 1) / 2
    b = b / 2

    # compute a matrix
    a = np.zeros((s, s))
    for j in range(s):
        p = np.polyint(lagpol(c, j, np=np))
        for i in range(s):
            a[i, j] = np.polyval(p, c[i]) - np.polyval(p, 0)

    return a, b, c
