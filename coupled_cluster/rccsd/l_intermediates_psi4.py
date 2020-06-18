from coupled_cluster.rccsd.utils import ndot


def build_Goo(t2, l2):
    Goo = 0
    Goo += ndot("abmj,ijab->mi", t2, l2)
    return Goo


def build_Gvv(t2, l2):
    Gvv = 0
    Gvv -= ndot("ijab,ebij->ae", l2, t2)
    return Gvv
