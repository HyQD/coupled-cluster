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

X1 = e_[a, i]
X2 = e_[a, i] * e_[b, j]

Y1 = Rational(1, 2) * e_[i, a]
Y2 = Rational(1, 3) * e_[i, a] * e_[j, b] + Rational(1, 6) * e_[j, a] * e_[i, b]

t = IndexedBase("t")
rhs = IndexedBase("rhs")
cluster = dr.einst(
    t[a, i] * e_[a, i] + Rational(1, 2) * t[a, b, i, j] * e_[a, i] * e_[b, j]
)
dr.set_n_body_base(t, 2)
cluster = cluster.simplify()
cluster.cache()

lambd = IndexedBase("l")
lambd_op = dr.einst(lambd[i, a] * Y1 + lambd[i, j, a, b] * Y2)
dr.set_n_body_base(lambd, 2)
lambd_op = lambd_op.simplify()
lambd_op.cache()

#### Similarity transform of the Hamiltonian
stopwatch = Stopwatch()

curr1 = dr.ham | X1
curr2 = dr.ham | X2
sim_HX1 = dr.ham | X1
sim_HX2 = dr.ham | X2
for order in range(4):
    curr1 = (curr1 | cluster).simplify() * Rational(1, order + 1)
    curr2 = (curr2 | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator1 order {}".format(order + 1), curr1)
    stopwatch.tock("Commutator2 order {}".format(order + 1), curr2)
    sim_HX1 += curr1
    sim_HX2 += curr2
    continue


# In[ ]:
sim_HX1 = sim_HX1.simplify()
sim_HX1.repartition(cache=True)
stopwatch.tock("[H,X1]-bar assembly", sim_HX1)

sim_HX2 = sim_HX2.simplify()
sim_HX2.repartition(cache=True)
stopwatch.tock("[H,X2]-bar assembly", sim_HX2)

# ### Find equations and print to file
l1_eqn = dr.define(
    rhs[i, a],
    sim_HX1.eval_fermi_vev().simplify()
    + (lambd_op * sim_HX1).eval_fermi_vev().simplify(),
)
l2_eqn = dr.define(
    rhs[i, j, a, b],
    sim_HX2.eval_fermi_vev().simplify()
    + (lambd_op * sim_HX2).eval_fermi_vev().simplify(),
)
# stopwatch.tock('l2-eqn', l2_eqn)

with dr.report("CCSD lambda equation.html", "CCSD lambda equation") as rep:
    rep.add(content=l1_eqn, description="l1 amplitude equation")
    rep.add(content=l2_eqn, description="l2 amplitude equation")


from gristmill import EinsumPrinter

printer = EinsumPrinter()

working_eqn1 = [l1_eqn]
working_eqn2 = [l2_eqn]

with open("rccsd_lambda1_equation.txt", "w") as outfile:
    outfile.write(printer.doprint(working_eqn1))

with open("rccsd_lambda2_equation.txt", "w") as outfile:
    outfile.write(printer.doprint(working_eqn2))
