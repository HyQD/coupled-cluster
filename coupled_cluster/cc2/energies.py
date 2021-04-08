from coupled_cluster.cc_helper import compute_reference_energy

def compute_cc2_ground_state_energy(f, u, t_1, t_2, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += compute_ground_state_energy_correction(
        f, u, t_1, t_2, o, v, np=np
    )

    return energy


def compute_ground_state_energy_correction(f, u, t_1, t_2, o, v, np):
    """

        f^{i}_{a} t^{a}_{i} 
        + (0.25) u^{ij}_{ab} t^{ab}_{ij}
        + (0.5) t^{ij}_{ab} t^{a}_{i} t^{b}_{j}

    """

    energy = np.tensordot(f[o, v], t_1, axes=((0, 1), (1, 0)))  # ia, ai ->
    energy += (0.25) * np.tensordot(
        u[o, o, v, v], t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))
    )  # ijab, abij ->
    term = (0.5) * np.tensordot(
        u[o, o, v, v], t_1, axes=((0, 2), (1, 0))
    )  # ijab, ai -> jb
    energy += np.tensordot(term, t_1, axes=((0, 1), (1, 0)))
    return energy


def compute_time_dependent_energy(f, u, t_1, t_2, l_1, l_2, o, v, np):
    energy = compute_reference_energy(f, u, o, v, np=np)
    energy += lagrangian_functional(f, u, t_1, t_2, l_1, l_2, o, v, np=np)

    return energy


def lagrangian_functional(f, f_transform, u_transform, t_1, t_2, l_1, l_2, o, v, np):
   
    energy = np.tensordot(f_transform[v, o], l_1, axes=((0, 1), (1, 0)))         #Stays the same

    term = np.tensordot(l_1, t_2, axes=((0, 1), (3, 1)))  # ai
    energy += np.tensordot(f_transform[o, v], term, axes=((0, 1), (1, 0)))       #Stays the same

    term = (0.5) * np.tensordot(
        t_2, u_transform[o, o, v, o], axes=((1, 2, 3), (2, 0, 1))
    )  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0)))                     #Stays the same
 
    term = (0.5) * np.tensordot(l_1, t_2, axes=((0), (2)))  # abcj
    energy += np.tensordot(
        term, u_transform[v, o, v, v], axes=((0, 1, 2, 3), (0, 2, 3, 1))         #Stays the same
  

    ##########################         CCD        ################################

    energy += 0.25 * np.tensordot(
        l_2, u_transform[v, v, o, o], axes=((0, 1, 2, 3), (2, 3, 0, 1)) #l2*u
    )

    energy += 0.25 * np.tensordot(
        t_2, u_transform[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1)) #l2*u 
    )

    temp_ldem = np.tensordot(l_2, t_2, axes=((1, 3), (2, 1)))
    energy += np.tensordot(
        temp_ldem, u[v, o, v, o], axes=((0, 1, 2, 3), (3, 0, 2, 1))   #u*t2*l2
    )

    temp_dckl = 0.5 * np.tensordot(t_2, f[o, o], axes=((3), (0)))
    energy += np.tensordot(l_2, temp_dckl, axes=((0, 1, 2, 3), (3, 2, 0, 1))) #f*t2*l2 Stays


    temp_dclk = 0.5 * np.tensordot(f[v, v], t_2, axes=((1), (0)))
    
    energy += np.tensordot(temp_dclk, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #f*t2*l2 Stays
    energy += np.tensordot(temp_dclk, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #f*t2*l2 Stays

    temp_dclk = 0.125 * np.tensordot(u[v, v, v, v], t_2, axes=((2, 3), (0, 1))) 
  #  energy += np.tensordot(temp_dclk, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #u*t2*l2

    temp_clnf = 0.5 * np.tensordot(t_2, u[o, o, v, v], axes=((0, 3), (2, 0)))
    temp_kdnf = np.tensordot(l_2, temp_clnf, axes=((0, 3), (1, 0)))
  #  energy += np.tensordot(t_2, temp_kdnf, axes=((0, 1, 2, 3), (1, 3, 0, 2))) #u*t2*l2*t2  Maybe


    temp_cf = -0.25 * np.tensordot(
        t_2, u[o, o, v, v], axes=((0, 2, 3), (2, 0, 1))
    )
    temp_lkdf = np.tensordot(l_2, temp_cf, axes=((3), (0)))
  #  energy += np.tensordot(temp_lkdf, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #t2*u*l2*t2 Maybe

    temp_ln = -0.125 * np.tensordot(
        t_2, u[o, o, v, v], axes=((0, 1, 3), (2, 3, 0))
    )
    temp_dckl = np.tensordot(t_2, temp_ln, axes=((3), (1)))
  #  energy += np.tensordot(l_2, temp_dckl, axes=((0, 1, 2, 3), (3, 2, 0, 1))) #t2*u*t2*l2 Maybe

    temp_km = -0.125 * np.tensordot(l_2, t_2, axes=((0, 2, 3), (2, 0, 1))) 
    temp_knef = np.tensordot(temp_km, u[o, o, v, v], axes=((1), (0)))
  #  energy += np.tensordot(temp_knef, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #l2*t2*u*t2 Maybe

    if o.stop >= v.stop // 2:
        temp_dcef = 0.0625 * np.tensordot(
            t_2, u[o, o, v, v], axes=((2, 3), (0, 1))
        )
        temp_efdc = np.tensordot(t_2, l_2, axes=((2, 3), (0, 1)))
  #      energy += np.tensordot(
  #          temp_efdc, temp_dcef, axes=((0, 1, 2, 3), (2, 3, 0, 1)) #l2*t2*u
  #      )
    else:
        temp_lkmn = 0.0625 * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))
        temp_mnlk = np.tensordot(u[o, o, v, v], t_2, axes=((2, 3), (0, 1)))
 #       energy += np.tensordot(
 #           temp_lkmn, temp_mnlk, axes=((0, 1, 2, 3), (2, 3, 0, 1)) #l2*t2*u
 #       )

    return energy
