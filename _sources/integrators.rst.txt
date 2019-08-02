Integrators
============

Any integrator can be used as long as it is built by inhereting
from the abstract base class ``Integrator``.

.. autoclass:: coupled_cluster.integrators.Integrator
    :members:

There are already some integrators readily available for your
convenience.

Runge-Kutta
-----------

This is an implementation of the classical Runge-Kutta integrator
scheme (RK4).

.. math:: y_{n + 1} &= y_n  + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4) \\
    t_{n+1} &= t_n + h

for `n=1, 2, 3, 4, ...` where

.. math:: k_1 &= h f(t_n, y_n), \\
        k_2 &= h f(t_n + \frac{h}{2}, y_n + \frac{k_1}{2}), \\
        k_3 &= h f(t_n + \frac{h}{2}, y_2 + \frac{k_2}{2}), \\
        k_4 &= h f(t_n + h, y_n + k_3).

Here, :math:`y_{n+1}`is the RK4 approximation of :math:`y(t_{n+1})`, and the
next value is determined by the present value plus the weighted average of
four increments, where each increment is the product of the size of
the interval :math:`h` and an estimated slope specified by function :math:`f`.

.. autoclass:: coupled_cluster.integrators.RungeKutta4
    :members:


The Gaussian Quadrature
-----------------------

Symplectic integrator. Adaptive and smart.

See `Simen and Pedersen (2018) <https://arxiv.org/abs/1812.04393v1>`_.

.. autoclass:: coupled_cluster.integrators.GaussIntegrator
    :members:
