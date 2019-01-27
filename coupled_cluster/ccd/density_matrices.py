def compute_one_body_density_matrix(t, l, o, v, np, rho=None):
    if rho is None:
        rho = np.zeros((v.stop, v.stop), dtype=t.dtype)

    rho.fill(0)
    rho[o, o] += np.eye(o.stop)
    rho[o, o] -= 0.5 * np.tensordot(l, t, axes=((0, 2, 3), (2, 0, 1)))

    rho[v, v] += 0.5 * np.tensordot(t, l, axes=((1, 2, 3), (3, 0, 1)))

    return rho


"""
Two body density matrices from Kvaal(2012). 
t2 is misshaped so it has to passed in as t2.transpose(2,3,0,1).
The final twobody density matrix should satisfy (which it does in my test implementation):
    np.einsum('pqpq->',Dpqrs) = N(N-1)
where N is the number of electrons.
"""

def compute_two_body_density_matrix(t2, l2, occ, virt, np, rho_pqrs=None):

    if rho_pqrs is None:
        rho_pqrs = np.zeros((virt.stop, virt.stop,virt.stop,virt.stop), dtype=t2.dtype)

    rho_pqrs.fill(0)
    rho_pqrs[occ,occ,occ,occ] = rho_ijkl(l2,t2,np)
    rho_pqrs[virt,virt,virt,virt] = rho_abcd(l2,t2,np)
    rho_pqrs[occ,virt,occ,virt] = rho_iajb(l2,t2,np)
    rho_pqrs[occ,virt,virt,occ] = -rho_pqrs[occ,virt,occ,virt].transpose(0,1,3,2)
    rho_pqrs[virt,occ,occ,virt] = -rho_pqrs[occ,virt,occ,virt].transpose(1,0,2,3)
    rho_pqrs[virt,occ,virt,occ] = rho_pqrs[occ,virt,occ,virt].transpose(1,0,3,2)
    rho_pqrs[occ,occ,virt,virt] = rho_ijab(l2,t2,np)
    rho_pqrs[virt,virt,occ,occ] = rho_abij(l2,t2,np)
    return rho_pqrs

def rho_ijkl(l2,t2,np):
    """
    Compute rho_{ij}^{kl}
    """
    delta_ij  = np.eye(l2.shape[0],dtype=np.complex128)
    rho_ijkl  = np.einsum('ik,jl->ijkl',delta_ij,delta_ij,optimize=True)
    rho_ijkl -= rho_ijkl.swapaxes(0,1)
    Pijkl     = 0.5*np.einsum('ik,lmcd,jmcd->ijkl',delta_ij,l2,t2,optimize=True)
    rho_ijkl -= Pijkl
    rho_ijkl += Pijkl.swapaxes(0,1)
    rho_ijkl += Pijkl.swapaxes(2,3)
    rho_ijkl -= Pijkl.swapaxes(0,1).swapaxes(2,3)
    rho_ijkl += 0.5*np.einsum('klcd,ijcd->ijkl',l2,t2,optimize=True)
    return rho_ijkl

def rho_abcd(l2,t2,np):
    """
    Compute rho_{ab}^{cd}
    """
    rho_abcd = 0.5*np.einsum('ijab,ijcd->abcd',l2,t2,optimize=True)
    return rho_abcd

def rho_iajb(l2,t2,np):
    """
    Compute rho_{ia}^{jb}
    """
    rho_iajb = 0.5*np.einsum('ij,klac,klbc->iajb',np.eye(l2.shape[0]),l2,t2,optimize=True)
    rho_iajb -= np.einsum('jkac,ikbc->iajb',l2,t2)
    return rho_iajb

def rho_abij(l2,t2,np):
    """
    Compute rho_{ab}^{ij}
    """
    return l2.transpose(2,3,0,1).copy()

def rho_ijab(l2,t2,np):
    """
    Compute rho_{ij}^{ab}
    """
    rho_ijab = -0.5*np.einsum('klcd,ijac,klbd->ijab',l2,t2,t2,optimize=True)
    rho_ijab += rho_ijab.swapaxes(2,3)
    Pij = np.einsum('klcd,ikac,jlbd->ijab',l2,t2,t2,optimize=True)
    Pij += 0.5*np.einsum('klcd,ilab,jkcd->ijab',l2,t2,t2,optimize=True)
    rho_ijab += Pij
    rho_ijab -= Pij.swapaxes(0,1)
    rho_ijab += 0.25*np.einsum('klcd,klab,ijcd->ijab',l2,t2,t2,optimize=True)
    rho_ijab += t2
    return rho_ijab
