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

proj_doubles = (
    Rational(1, 3) * e_[i, a] * e_[j, b] + Rational(1, 6) * e_[j, a] * e_[i, b]
)

# ### Find equations and print to file
t2_eqn = (proj_doubles * h_bar).eval_fermi_vev().simplify()
stopwatch.tock("T2 equation", t2_eqn)

with dr.report("rCCD.html", "restricted CCD theory") as rep:
    rep.add("Energy equation", en_eqn)
    rep.add("T2-amplitude equation", t2_eqn)

"""
from gristmill import get_flop_cost
working_eqn = [
    dr.define(t[a,b,i,j], t2_eqn).simplify()
]
cost = get_flop_cost(working_eqn, leading=True)
cost


from gristmill import EinsumPrinter
printer = EinsumPrinter()

with open('rccd_equations.txt', 'w') as outfile:
    outfile.write(printer.doprint(working_eqn))
"""
