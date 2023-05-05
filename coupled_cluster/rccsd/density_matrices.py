from opt_einsum import contract


def compute_one_body_density_matrix(t1, t2, l1, l2, o, v, np, out=None):
    nocc = o.stop
    nvirt = v.stop - nocc

    rho = np.zeros((nocc + nvirt, nocc + nvirt), dtype=t1.dtype)

    rho[o, o] += 2 * np.eye(nocc)
    rho[o, o] -= contract("kjab,baik->ji", l2, t2)
    rho[o, o] -= contract("ja,ai->ji", l1, t1)

    rho[v, o] = 2 * t1
    rho[v, o] += 2 * contract("jb,abij->ai", l1, t2)
    rho[v, o] -= contract("jb,abji->ai", l1, t2)

    rho[v, o] -= contract("jb,aj,bi->ai", l1, t1, t1, optimize=True)
    rho[v, o] -= contract("ak,jkcb,bcij->ai", t1, l2, t2, optimize=True)
    rho[v, o] -= contract("ci,jkcb,abjk->ai", t1, l2, t2, optimize=True)

    rho[o, v] = l1

    rho[v, v] = contract("ia,bi->ba", l1, t1)
    rho[v, v] += contract("ijac,bcij->ba", l2, t2)

    return rho


def compute_two_body_density_matrix(t1, t2, l1, l2, o, v, np, out=None):
    if out is None:
        out = np.zeros((v.stop, v.stop, v.stop, v.stop), dtype=t1.dtype)
    out.fill(0)

    add_rho_klij(t1, t2, l1, l2, o, v, out, np)

    add_rho_kaij(t1, t2, l1, l2, o, v, out, np)
    add_rho_akij(t1, t2, l1, l2, o, v, out, np)
    add_rho_jkai(t1, t2, l1, l2, o, v, out, np)
    add_rho_jkia(t1, t2, l1, l2, o, v, out, np)

    add_rho_ijab(t1, t2, l1, l2, o, v, out, np)
    add_rho_abij(t1, t2, l1, l2, o, v, out, np)

    add_rho_jbia(t1, t2, l1, l2, o, v, out, np)
    add_rho_bjai(t1, t2, l1, l2, o, v, out, np)

    add_rho_bjia(t1, t2, l1, l2, o, v, out, np)
    add_rho_jbai(t1, t2, l1, l2, o, v, out, np)

    add_rho_bcai(t1, t2, l1, l2, o, v, out, np)
    add_rho_bcia(t1, t2, l1, l2, o, v, out, np)
    add_rho_ciab(t1, t2, l1, l2, o, v, out, np)
    add_rho_icab(t1, t2, l1, l2, o, v, out, np)

    add_rho_cdab(t1, t2, l1, l2, o, v, out, np)

    return out


def add_rho_klij(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[o, o, o, o] -= 2 * contract("il, jk->klij", delta, delta)

    out[o, o, o, o] += 4 * contract("ik, jl->klij", delta, delta)

    out[o, o, o, o] += contract("il, mkab,abmj->klij", delta, l2, t2)

    out[o, o, o, o] += contract("jk, mlab,baim->klij", delta, l2, t2)

    out[o, o, o, o] -= 2 * contract("jl, mkab,baim->klij", delta, l2, t2)

    out[o, o, o, o] -= 2 * contract("ik, mlab,abmj->klij", delta, l2, t2)

    out[o, o, o, o] += contract("il, ka,aj->klij", delta, l1, t1)

    out[o, o, o, o] += contract("jk, la,ai->klij", delta, l1, t1)

    out[o, o, o, o] -= 2 * contract("jl, ka,ai->klij", delta, l1, t1)

    out[o, o, o, o] -= 2 * contract("ik, la,aj->klij", delta, l1, t1)

    out[o, o, o, o] += contract("klab,abij->klij", l2, t2)

    out[o, o, o, o] += contract("ai,bj,klab->klij", t1, t1, l2)


def add_rho_kaij(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[o, v, o, o] += contract("ai,lkbc,bclj->kaij", t1, l2, t2)

    out[o, v, o, o] += contract("al,lkbc,cbij->kaij", t1, l2, t2)

    out[o, v, o, o] += contract("bj,lkbc,acli->kaij", t1, l2, t2)

    out[o, v, o, o] += contract("ci,lkbc,ablj->kaij", t1, l2, t2)

    out[o, v, o, o] += contract("cj,lkbc,abil->kaij", t1, l2, t2)

    out[o, v, o, o] -= 2 * contract("aj,lkbc,cbil->kaij", t1, l2, t2)

    out[o, v, o, o] -= 2 * contract("ci,lkbc,abjl->kaij", t1, l2, t2)

    out[o, v, o, o] += contract("al,bj,ci,lkbc->kaij", t1, t1, t1, l2)

    out[o, v, o, o] += contract("kb,abij->kaij", l1, t2)

    out[o, v, o, o] -= 2 * contract("kb,abji->kaij", l1, t2)

    out[o, v, o, o] += contract("kb,ai,bj->kaij", l1, t1, t1)

    out[o, v, o, o] -= 2 * contract("kb,aj,bi->kaij", l1, t1, t1)

    out[o, v, o, o] += contract("jk, lb,abli->kaij", delta, l1, t2)

    out[o, v, o, o] -= 2 * contract("jk, lb,abil->kaij", delta, l1, t2)

    out[o, v, o, o] -= 2 * contract("ik, lb,ablj->kaij", delta, l1, t2)

    out[o, v, o, o] += 4 * contract("ik, lb,abjl->kaij", delta, l1, t2)

    out[o, v, o, o] += contract("jk, lb,al,bi->kaij", delta, l1, t1, t1)

    out[o, v, o, o] -= 2 * contract("ik, lb,al,bj->kaij", delta, l1, t1, t1)

    out[o, v, o, o] -= 2 * contract("jk, ai->kaij", delta, t1)

    out[o, v, o, o] += 4 * contract("ik, aj->kaij", delta, t1)

    out[o, v, o, o] += contract("jk, al,lmbc,bcim->kaij", delta, t1, l2, t2)

    out[o, v, o, o] += contract("jk, bi,lmbc,aclm->kaij", delta, t1, l2, t2)

    out[o, v, o, o] -= 2 * contract("ik, al,lmcb,bcmj->kaij", delta, t1, l2, t2)

    out[o, v, o, o] -= 2 * contract("ik, bj,lmbc,aclm->kaij", delta, t1, l2, t2)


def add_rho_akij(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[v, o, o, o] += contract("aj,lkbc,cbil->akij", t1, l2, t2)

    out[v, o, o, o] += contract("al,lkbc,bcij->akij", t1, l2, t2)

    out[v, o, o, o] += contract("bi,lkbc,aclj->akij", t1, l2, t2)

    out[v, o, o, o] += contract("ci,lkbc,abjl->akij", t1, l2, t2)

    out[v, o, o, o] += contract("cj,lkbc,abli->akij", t1, l2, t2)

    out[v, o, o, o] -= 2 * contract("ai,lkbc,bclj->akij", t1, l2, t2)

    out[v, o, o, o] -= 2 * contract("cj,lkbc,abil->akij", t1, l2, t2)

    out[v, o, o, o] += contract("al,bi,cj,lkbc->akij", t1, t1, t1, l2)

    out[v, o, o, o] += contract("kb,abji->akij", l1, t2)

    out[v, o, o, o] -= 2 * contract("kb,abij->akij", l1, t2)

    out[v, o, o, o] += contract("kb,aj,bi->akij", l1, t1, t1)

    out[v, o, o, o] -= 2 * contract("kb,ai,bj->akij", l1, t1, t1)

    out[v, o, o, o] += contract("ik, lb,ablj->akij", delta, l1, t2)

    out[v, o, o, o] -= 2 * contract("ik, lb,abjl->akij", delta, l1, t2)

    out[v, o, o, o] -= 2 * contract("jk, lb,abli->akij", delta, l1, t2)

    out[v, o, o, o] += 4 * contract("jk, lb,abil->akij", delta, l1, t2)

    out[v, o, o, o] += contract("ik, lb,al,bj->akij", delta, l1, t1, t1)

    out[v, o, o, o] -= 2 * contract("jk, lb,al,bi->akij", delta, l1, t1, t1)

    out[v, o, o, o] -= 2 * contract("ik, aj->akij", delta, t1)

    out[v, o, o, o] += 4 * contract("jk, ai->akij", delta, t1)

    out[v, o, o, o] += contract("ik, al,lmcb,bcmj->akij", delta, t1, l2, t2)

    out[v, o, o, o] += contract("ik, bj,lmbc,aclm->akij", delta, t1, l2, t2)

    out[v, o, o, o] -= 2 * contract("jk, al,lmbc,bcim->akij", delta, t1, l2, t2)

    out[v, o, o, o] -= 2 * contract("jk, bi,lmbc,aclm->akij", delta, t1, l2, t2)


def add_rho_jkai(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[o, o, v, o] -= contract("ij, ka->jkai", delta, l1)

    out[o, o, v, o] += 2 * contract("ik, ja->jkai", delta, l1)

    out[o, o, v, o] -= contract("bi,jkab->jkai", t1, l2)


def add_rho_jkia(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[o, o, o, v] -= contract("ik, ja->jkia", delta, l1)

    out[o, o, o, v] += 2 * contract("ij, ka->jkia", delta, l1)

    out[o, o, o, v] -= contract("bi,jkba->jkia", t1, l2)


def add_rho_ijab(t1, t2, l1, l2, o, v, out, np):
    out[o, o, v, v] += l2


def add_rho_abij(t1, t2, l1, l2, o, v, out, np):
    out[v, v, o, o] -= 2 * contract("abji->abij", t2)

    out[v, v, o, o] += 4 * contract("abij->abij", t2)

    out[v, v, o, o] -= 2 * contract("aj,bi->abij", t1, t1)

    out[v, v, o, o] += 4 * contract("ai,bj->abij", t1, t1)

    out[v, v, o, o] += contract("kc,aj,bcki->abij", l1, t1, t2)

    out[v, v, o, o] += contract("kc,ak,bcij->abij", l1, t1, t2)

    out[v, v, o, o] += contract("kc,bi,ackj->abij", l1, t1, t2)

    out[v, v, o, o] += contract("kc,bk,acji->abij", l1, t1, t2)

    out[v, v, o, o] += contract("kc,ci,abjk->abij", l1, t1, t2)

    out[v, v, o, o] += contract("kc,cj,abki->abij", l1, t1, t2)

    out[v, v, o, o] -= 2 * contract("kc,ai,bckj->abij", l1, t1, t2)

    out[v, v, o, o] -= 2 * contract("kc,aj,bcik->abij", l1, t1, t2)

    out[v, v, o, o] -= 2 * contract("kc,ak,bcji->abij", l1, t1, t2)

    out[v, v, o, o] -= 2 * contract("kc,bi,acjk->abij", l1, t1, t2)

    out[v, v, o, o] -= 2 * contract("kc,bj,acki->abij", l1, t1, t2)

    out[v, v, o, o] -= 2 * contract("kc,bk,acij->abij", l1, t1, t2)

    out[v, v, o, o] -= 2 * contract("kc,ci,abkj->abij", l1, t1, t2)

    out[v, v, o, o] -= 2 * contract("kc,cj,abik->abij", l1, t1, t2)

    out[v, v, o, o] += 4 * contract("kc,ai,bcjk->abij", l1, t1, t2)

    out[v, v, o, o] += 4 * contract("kc,bj,acik->abij", l1, t1, t2)

    out[v, v, o, o] += contract("kc,aj,bk,ci->abij", l1, t1, t1, t1)

    out[v, v, o, o] += contract("kc,ak,bi,cj->abij", l1, t1, t1, t1)

    out[v, v, o, o] -= 2 * contract("kc,ai,bk,cj->abij", l1, t1, t1, t1)

    out[v, v, o, o] -= 2 * contract("kc,ak,bj,ci->abij", l1, t1, t1, t1)

    out[v, v, o, o] += contract("klcd,abjk,cdil->abij", l2, t2, t2)

    out[v, v, o, o] += contract("klcd,abkl,cdij->abij", l2, t2, t2)

    out[v, v, o, o] += contract("klcd,acji,bdkl->abij", l2, t2, t2)

    out[v, v, o, o] += contract("klcd,acjk,bdli->abij", l2, t2, t2)

    out[v, v, o, o] += contract("klcd,acki,bdlj->abij", l2, t2, t2)

    out[v, v, o, o] += contract("klcd,ackj,bdil->abij", l2, t2, t2)

    out[v, v, o, o] += contract("kldc,abki,cdlj->abij", l2, t2, t2)

    out[v, v, o, o] += contract("kldc,ackj,bdli->abij", l2, t2, t2)

    out[v, v, o, o] += contract("kldc,ackl,bdij->abij", l2, t2, t2)

    out[v, v, o, o] -= 2 * contract("klcd,abkj,cdil->abij", l2, t2, t2)

    out[v, v, o, o] -= 2 * contract("klcd,acij,bdkl->abij", l2, t2, t2)

    out[v, v, o, o] -= 2 * contract("klcd,acik,bdlj->abij", l2, t2, t2)

    out[v, v, o, o] -= 2 * contract("klcd,acjk,bdil->abij", l2, t2, t2)

    out[v, v, o, o] -= 2 * contract("klcd,acki,bdjl->abij", l2, t2, t2)

    out[v, v, o, o] -= 2 * contract("kldc,abik,cdlj->abij", l2, t2, t2)

    out[v, v, o, o] -= 2 * contract("kldc,ackl,bdji->abij", l2, t2, t2)

    out[v, v, o, o] += 4 * contract("klcd,acik,bdjl->abij", l2, t2, t2)

    out[v, v, o, o] += contract("aj,bk,klcd,cdil->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("aj,ci,klcd,bdkl->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("ak,bi,kldc,cdlj->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("ak,bl,klcd,cdij->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("ak,ci,klcd,bdlj->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("ak,cj,klcd,bdil->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("ak,cj,kldc,bdli->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("bi,cj,klcd,adkl->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("bk,ci,klcd,adjl->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("bk,ci,kldc,adlj->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("bk,cj,klcd,adli->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("ci,dj,klcd,abkl->abij", t1, t1, l2, t2)

    out[v, v, o, o] -= 2 * contract("ai,bk,kldc,cdlj->abij", t1, t1, l2, t2)

    out[v, v, o, o] -= 2 * contract("ai,cj,klcd,bdkl->abij", t1, t1, l2, t2)

    out[v, v, o, o] -= 2 * contract("ak,bj,klcd,cdil->abij", t1, t1, l2, t2)

    out[v, v, o, o] -= 2 * contract("ak,ci,klcd,bdjl->abij", t1, t1, l2, t2)

    out[v, v, o, o] -= 2 * contract("bj,ci,klcd,adkl->abij", t1, t1, l2, t2)

    out[v, v, o, o] -= 2 * contract("bk,cj,klcd,adil->abij", t1, t1, l2, t2)

    out[v, v, o, o] += contract("ak,bl,ci,dj,klcd->abij", t1, t1, t1, t1, l2)


def add_rho_jbia(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[o, v, o, v] -= contract("ja,bi->jbia", l1, t1)

    out[o, v, o, v] += 2 * contract("ij, klac,bckl->jbia", delta, l2, t2)

    out[o, v, o, v] += 2 * contract("ij, ka,bk->jbia", delta, l1, t1)

    out[o, v, o, v] -= contract("kjac,bcki->jbia", l2, t2)

    out[o, v, o, v] -= contract("kjca,bcik->jbia", l2, t2)

    out[o, v, o, v] -= contract("bk,ci,kjac->jbia", t1, t1, l2)


def add_rho_bjai(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[v, o, v, o] -= contract("ja,bi->bjai", l1, t1)

    out[v, o, v, o] += 2 * contract("ij, klac,bckl->bjai", delta, l2, t2)

    out[v, o, v, o] += 2 * contract("ij, ka,bk->bjai", delta, l1, t1)

    out[v, o, v, o] -= contract("kjac,bcki->bjai", l2, t2)

    out[v, o, v, o] -= contract("kjca,bcik->bjai", l2, t2)

    out[v, o, v, o] -= contract("bk,ci,kjac->bjai", t1, t1, l2)


def add_rho_bjia(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[v, o, o, v] += 2 * contract("ja,bi->bjia", l1, t1)

    out[v, o, o, v] -= contract("ij, klac,bckl->bjia", delta, l2, t2)

    out[v, o, o, v] -= contract("ij, ka,bk->bjia", delta, l1, t1)

    out[v, o, o, v] -= contract("kjca,bcki->bjia", l2, t2)

    out[v, o, o, v] += 2 * contract("kjca,bcik->bjia", l2, t2)

    out[v, o, o, v] -= contract("bk,ci,kjca->bjia", t1, t1, l2)


def add_rho_jbai(t1, t2, l1, l2, o, v, out, np):
    delta = np.eye(o.stop)

    out[o, v, v, o] += 2 * contract("ja,bi->jbai", l1, t1)

    out[o, v, v, o] -= contract("ij, klac,bckl->jbai", delta, l2, t2)

    out[o, v, v, o] -= contract("ij, ka,bk->jbai", delta, l1, t1)

    out[o, v, v, o] -= contract("kjca,bcki->jbai", l2, t2)

    out[o, v, v, o] += 2 * contract("kjca,bcik->jbai", l2, t2)

    out[o, v, v, o] -= contract("bk,ci,kjca->jbai", t1, t1, l2)


def add_rho_bcai(t1, t2, l1, l2, o, v, out, np):
    out[v, v, v, o] -= contract("bi,jkad,cdjk->bcai", t1, l2, t2)

    out[v, v, v, o] -= contract("bj,jkad,cdki->bcai", t1, l2, t2)

    out[v, v, v, o] -= contract("cj,jkad,bdik->bcai", t1, l2, t2)

    out[v, v, v, o] -= contract("ck,jkad,bdji->bcai", t1, l2, t2)

    out[v, v, v, o] -= contract("di,jkad,bcjk->bcai", t1, l2, t2)

    out[v, v, v, o] += 2 * contract("bj,jkad,cdik->bcai", t1, l2, t2)

    out[v, v, v, o] += 2 * contract("ci,jkad,bdjk->bcai", t1, l2, t2)

    out[v, v, v, o] -= contract("bj,ck,di,jkad->bcai", t1, t1, t1, l2)

    out[v, v, v, o] -= contract("ja,bcij->bcai", l1, t2)

    out[v, v, v, o] += 2 * contract("ja,bcji->bcai", l1, t2)

    out[v, v, v, o] -= contract("ja,bi,cj->bcai", l1, t1, t1)

    out[v, v, v, o] += 2 * contract("ja,bj,ci->bcai", l1, t1, t1)


def add_rho_bcia(t1, t2, l1, l2, o, v, out, np):
    out[v, v, o, v] -= contract("bj,jkad,cdik->bcia", t1, l2, t2)

    out[v, v, o, v] -= contract("bk,jkad,cdji->bcia", t1, l2, t2)

    out[v, v, o, v] -= contract("ci,jkad,bdjk->bcia", t1, l2, t2)

    out[v, v, o, v] -= contract("cj,jkad,bdki->bcia", t1, l2, t2)

    out[v, v, o, v] -= contract("di,jkad,bckj->bcia", t1, l2, t2)

    out[v, v, o, v] += 2 * contract("bi,jkad,cdjk->bcia", t1, l2, t2)

    out[v, v, o, v] += 2 * contract("cj,jkad,bdik->bcia", t1, l2, t2)

    out[v, v, o, v] -= contract("bk,cj,di,jkad->bcia", t1, t1, t1, l2)

    out[v, v, o, v] -= contract("ja,bcji->bcia", l1, t2)

    out[v, v, o, v] += 2 * contract("ja,bcij->bcia", l1, t2)

    out[v, v, o, v] -= contract("ja,bj,ci->bcia", l1, t1, t1)

    out[v, v, o, v] += 2 * contract("ja,bi,cj->bcia", l1, t1, t1)


def add_rho_icab(t1, t2, l1, l2, o, v, out, np):
    out[o, v, v, v] += contract("cj,ijab->icab", t1, l2)


def add_rho_ciab(t1, t2, l1, l2, o, v, out, np):
    out[v, o, v, v] += contract("cj,ijba->ciab", t1, l2)


def add_rho_cdab(t1, t2, l1, l2, o, v, out, np):
    out[v, v, v, v] += contract("ijab,cdij->cdab", l2, t2)

    out[v, v, v, v] += contract("ci,dj,ijab->cdab", t1, t1, l2)
