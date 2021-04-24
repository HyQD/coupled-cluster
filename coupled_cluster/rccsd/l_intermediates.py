__authors__ = "Ashutosh Kumar"
__credits__ = [
    "T. D. Crawford",
    "Daniel G. A. Smith",
    "Lori A. Burns",
    "Ashutosh Kumar",
]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-17"


def build_Goo(t2, l2, np):
    Goo = 0
    Goo += np.einsum("abmj,ijab->mi", t2, l2)
    return Goo


def build_Gvv(t2, l2, np):
    Gvv = 0
    Gvv -= np.einsum("ijab,ebij->ae", l2, t2)
    return Gvv
