from coupled_cluster.rccsd.utils import ndot
from coupled_cluster.rccsd.l_intermediates_psi4 import *
from coupled_cluster.rccsd.t_intermediates_psi4 import *
from coupled_cluster.rccsd.cc_hbar import *


class RCCSDIntermediates:
    def __init__(self, system):
        self.system = system
        self.o, self.v = system.o, system.v
        self.Loovv = build_Loovv(system.u, system.o, system.v)
        self.Lvovv = build_Lvovv(system.u, system.o, system.v)
        self.Looov = build_Looov(system.u, system.o, system.v)

    def update_intermediates(self, f, u, t1, t2):
        Loovv = self.Loovv
        Lvovv = self.Lvovv
        Looov = self.Looov
        o, v = self.o, self.v

        ### Build OEI intermediates
        self.Fae = build_Fae(f, u, t1, t2, o, v)
        self.Fmi = build_Fmi(f, u, t1, t2, o, v)
        self.Fme = build_Fme(f, u, t1, o, v)

        self.Wmnij = build_Wmnij(u, t1, t2, o, v)
        self.Wmbej = build_Wmbej(u, t1, t2, o, v)
        self.Wmbje = build_Wmbje(u, t1, t2, o, v)
        self.Zmbij = build_Zmbij(u, t1, t2, o, v)

        self.Hoo = build_Hoo(f, Looov, Loovv, t1, t2, o, v)
        self.Hov = build_Hov(f, Loovv, t1, o, v)
        self.Hvv = build_Hvv(f, Lvovv, Loovv, t1, t2, o, v)
        self.Hovvo = build_Hovvo(u, Loovv, t1, t2, o, v)
        self.Hovov = build_Hovov(u, t1, t2, o, v)
        self.Hvvvo = build_Hvvvo(f, u, Loovv, Lvovv, t1, t2, o, v)
        self.Hovoo = build_Hovoo(f, u, Loovv, Looov, t1, t2, o, v)
        self.Hvovv = build_Hvovv(u, t1, o, v)
        self.Hooov = build_Hooov(u, t1, o, v)

        self.Hoooo = build_Hoooo(u, t1, t2, o, v)
        self.Hvvvv = build_Hvvvv(u, t1, t2, o, v)
