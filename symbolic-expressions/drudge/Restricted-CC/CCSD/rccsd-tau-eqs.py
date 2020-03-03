#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational, symbols
from drudge import RestrictedPartHoleDrudge, Stopwatch

# Environment setting up.
conf = SparkConf().setAppName("rccsd")
ctx = SparkContext(conf=conf)
dr = RestrictedPartHoleDrudge(ctx)
dr.full_simplify = False

p = dr.names
e_ = p.e_
a, b, c, d = p.V_dumms[:4]
i, j, k, l = p.O_dumms[:4]

t = IndexedBase("t")
rhs = IndexedBase("rhs")
cluster = dr.einst(
    t[a, i] * e_[a, i] + Rational(1, 2) * t[a, b, i, j] * e_[a, i] * e_[b, j]
)

dr.set_n_body_base(t, 2)
cluster = cluster.simplify()
cluster.cache()

#### Similarity transform of the Hamiltonian
stopwatch = Stopwatch()

curr = dr.ham
h_bar = dr.ham


for order in range(4):
    curr = (curr | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator order {}".format(order + 1), curr)
    h_bar += curr
    continue

# In[ ]:
h_bar = h_bar.simplify()
h_bar.repartition(cache=True)
stopwatch.tock("H-bar assembly", h_bar)

# vev = vacuum expectation value
en_eqn = h_bar.eval_fermi_vev().simplify()
stopwatch.tock("Energy equation", en_eqn)
dr.wick_parallel = 1

Y1 = Rational(1, 2) * e_[i, a]
Y2 = Rational(1, 3) * e_[i, a] * e_[j, b] + Rational(1, 6) * e_[j, a] * e_[i, b]

# ### Find equations and print to file
t1_eqn = dr.define(rhs[a, i], (Y1 * h_bar).eval_fermi_vev().simplify())
t2_eqn = dr.define(rhs[a, b, i, j], (Y2 * h_bar).eval_fermi_vev().simplify())
# stopwatch.tock('T2 equation', t2_eqn)

working_eqn1 = [t1_eqn]
working_eqn2 = [t2_eqn]

with dr.report("rCCSD.html", "restricted CCSD theory") as rep:
    rep.add("Energy equation", en_eqn)
    rep.add(content=t1_eqn, description="t1 amplitude equation")
    rep.add(content=t2_eqn, description="t2 amplitude equation")

from gristmill import EinsumPrinter

printer = EinsumPrinter()

with open("rccsd_tau1_equation.txt", "w") as outfile:
    outfile.write(printer.doprint(working_eqn1))

with open("rccsd_tau2_equation.txt", "w") as outfile:
    outfile.write(printer.doprint(working_eqn2))
