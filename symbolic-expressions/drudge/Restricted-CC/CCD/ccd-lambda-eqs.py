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

X2 = e_[a, i] * e_[b, j]
Y2 = Rational(1, 3) * e_[i, a] * e_[j, b] + Rational(1, 6) * e_[j, a] * e_[i, b]

t = IndexedBase("t")
cluster = dr.einst(
    # t[a, i] * e_[a, i] +
    Rational(1, 2)
    * t[a, b, i, j]
    * e_[a, i]
    * e_[b, j]
)
dr.set_n_body_base(t, 2)
cluster = cluster.simplify()
cluster.cache()

lambd = IndexedBase("l")
lambd_op = dr.einst(lambd[i, j, a, b] * Y2)
dr.set_n_body_base(lambd, 2)
lambd_op = lambd_op.simplify()
lambd_op.cache()

#### Similarity transform of the Hamiltonian
stopwatch = Stopwatch()

curr = dr.ham | X2
sim_HX2 = dr.ham | X2
for order in range(4):
    curr = (curr | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator order {}".format(order + 1), curr)
    sim_HX2 += curr
    continue


# In[ ]:
sim_HX2 = sim_HX2.simplify()
sim_HX2.repartition(cache=True)
stopwatch.tock("[H,X2]-bar assembly", sim_HX2)

# ### Find equations and print to file
l2_eqn = sim_HX2.eval_fermi_vev().simplify()
l2_eqn += (lambd_op * sim_HX2).eval_fermi_vev().simplify()
stopwatch.tock("l2-eqn", l2_eqn)

with dr.report("CCD lambda equation.html", "CCD lambda equation") as rep:
    rep.add("l2 equation", l2_eqn)


from gristmill import get_flop_cost

working_eqn = [dr.define(lambd[i, j, a, b], l2_eqn).simplify()]
cost = get_flop_cost(working_eqn, leading=True)
print(cost)


from gristmill import EinsumPrinter

printer = EinsumPrinter()

with open("rccd_lambda_equations.txt", "w") as outfile:
    outfile.write(printer.doprint(working_eqn))
