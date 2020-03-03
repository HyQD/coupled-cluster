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

Y1 = Rational(1, 2) * e_[i, a]
Y2 = Rational(1, 3) * e_[i, a] * e_[j, b] + Rational(1, 6) * e_[j, a] * e_[i, b]

t = IndexedBase("t")
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

curr1 = e_[i, j]
curr2 = e_[i, a]
curr3 = e_[a, i]
curr4 = e_[a, b]


sim_Eij = e_[i, j]
sim_Eia = e_[i, a]
sim_Eai = e_[a, i]
sim_Eab = e_[a, b]

for order in range(4):
    curr1 = (curr1 | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator1 order {}".format(order + 1), curr1)
    curr2 = (curr2 | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator2 order {}".format(order + 1), curr2)
    curr3 = (curr3 | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator3 order {}".format(order + 1), curr3)
    curr4 = (curr4 | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator4 order {}".format(order + 1), curr4)
    sim_Eij += curr1
    sim_Eia += curr2
    sim_Eai += curr3
    sim_Eab += curr4
    continue


# In[ ]:
sim_Eij = sim_Eij.simplify()
sim_Eij.repartition(cache=True)
stopwatch.tock("E_ij-bar assembly", sim_Eij)

sim_Eia = sim_Eia.simplify()
sim_Eia.repartition(cache=True)
stopwatch.tock("E_ia-bar assembly", sim_Eia)

sim_Eai = sim_Eai.simplify()
sim_Eai.repartition(cache=True)
stopwatch.tock("E_ai-bar assembly", sim_Eai)

sim_Eab = sim_Eab.simplify()
sim_Eab.repartition(cache=True)
stopwatch.tock("E_ab-bar assembly", sim_Eab)

# ### Find equations and print to file
rho_ij = sim_Eij.eval_fermi_vev().simplify()
rho_ij += (lambd_op * sim_Eij).eval_fermi_vev().simplify()
stopwatch.tock("rho_ij", rho_ij)

rho_ia = sim_Eia.eval_fermi_vev().simplify()
rho_ia += (lambd_op * sim_Eia).eval_fermi_vev().simplify()
stopwatch.tock("rho_ia", rho_ia)

rho_ai = sim_Eai.eval_fermi_vev().simplify()
rho_ai += (lambd_op * sim_Eai).eval_fermi_vev().simplify()
stopwatch.tock("rho_ai", rho_ai)

rho_ab = sim_Eab.eval_fermi_vev().simplify()
rho_ab += (lambd_op * sim_Eab).eval_fermi_vev().simplify()
stopwatch.tock("rho_ab", rho_ab)

with dr.report("CCSD-density-matrices.html", "CCSD density matrices") as rep:
    rep.add("rho_ij", rho_ij)
    rep.add("rho_ia", rho_ia)
    rep.add("rho_ai", rho_ai)
    rep.add("rho_ab", rho_ab)


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
