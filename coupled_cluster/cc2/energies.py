from coupled_cluster.cc_helper import compute_reference_energy
from coupled_cluster.cc2.rhs_t import (
    compute_t_1_amplitudes,
    compute_t_2_amplitudes,
)
from coupled_cluster.ccd.energies import (
    compute_lagrangian_functional as ccd_functional,
)


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

    ### Testing the Lagrangian####

    epsilon = 1e-6
    y = np.random.rand(len(t_1))
    y = y/np.linalg.norm(y)
    y = epsilon*y

    #For t_1
    t_1_x = t_1
    t_1_y = t_1
    t_1_x[:,0] = t_1[:,0] + epsilon*y
    t_1_y[:,0] = t_1[:,0] - epsilon*y 

    #For t_2

    y = np.random.rand(len(t_2))
    y = y/np.linalg.norm(y)
    y = epsilon*y

    t_2_x = t_2
    t_2_y = t_2
    t_2_x[:,0,0,0] = t_2[:,0,0,0] + epsilon*y
    t_2_y[:,0,0,0] = t_2[:,0,0,0] - epsilon*y 

  
    test = (lagrangian_functional(f, u, t_1_x, t_2, l_1, l_2, o, v, np=np)-lagrangian_functional(f, u, t_1_y, t_2, l_1, l_2, o, v, np=np))/epsilon

#    test = (f(x + epsilon*y) - f(x - epsilon*y))/epsilon 

    print("The value of the Lagrangian test, small change in t_1")
    print(test)

    test = (lagrangian_functional(f, u, t_1, t_2_x, l_1, l_2, o, v, np=np)-lagrangian_functional(f, u, t_1, t_2_y, l_1, l_2, o, v, np=np))/epsilon

#    test = (f(x + epsilon*y) - f(x - epsilon*y))/epsilon 

    print("The value of the Lagrangian test, small change in t_2")
    print(test)
    return energy


def lagrangian_functional(f, u, t_1, t_2, l_1, l_2, o, v, np, test=False):

    energy = np.tensordot(f[v, o], l_1, axes=((0, 1), (1, 0))) #Check
    energy += np.tensordot(f[o, v], t_1, axes=((0, 1), (1, 0))) #Check

    term = np.tensordot(f[v, v], l_1, axes=((0), (1)))  # bi #Check
    energy += np.tensordot(term, t_1, axes=((0, 1), (0, 1))) #Check
    term = np.tensordot(l_1, t_2, axes=((0, 1), (3, 1)))  # ai #Check
    energy += np.tensordot(f[o, v], term, axes=((0, 1), (1, 0))) #Check

    term = (0.5) * np.tensordot(
        t_2, u[o, o, v, o], axes=((1, 2, 3), (2, 0, 1))
    )  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0))) #check
    term = (0.5) * np.tensordot(l_1, t_2, axes=((0), (2)))  # abcj
    energy += np.tensordot(
        term, u[v, o, v, v], axes=((0, 1, 2, 3), (0, 2, 3, 1)) #Check
    )
    term = (0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbk
    energy += np.tensordot(
        term, u[v, o, o, o], axes=((0, 1, 2, 3), (2, 3, 0, 1)) #Check
    )
    term = (0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # jabc
    energy += np.tensordot(
        term, u[v, v, v, o], axes=((0, 1, 2, 3), (3, 0, 1, 2)) #Check
    )

    term = (-1) * np.tensordot(f[o, o], l_1, axes=((1), (0)))  # ja
    energy += np.tensordot(term, t_1, axes=((0, 1), (1, 0))) #Check
    term = (-1) * np.tensordot(t_1, u[v, o, v, o], axes=((0, 1), (2, 1)))  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0))) #Check

    term = (-0.5) * np.tensordot(
        t_1, u[o, o, v, v], axes=((0, 1), (3, 0))
    )  # ja
    energy += np.tensordot(t_1, term, axes=((0, 1), (1, 0))) #Check

    term = np.tensordot(t_1, u[o, o, v, o], axes=((0, 1), (2, 1)))  # ji 
    term = np.tensordot(t_1, term, axes=((1), (0)))  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0))) #Check
    term = np.tensordot(t_1, u[v, o, v, v], axes=((0, 1), (3, 1)))  # ab
    term = np.tensordot(t_1, term, axes=((0), (1)))  # ia
    energy += np.tensordot(l_1, term, axes=((0, 1), (0, 1)))#Check
    term = np.tensordot(t_2, u[o, o, v, v], axes=((1, 3), (3, 1)))  # aijb
    term = np.tensordot(t_1, term, axes=((0, 1), (3, 2)))  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0))) #Check
    term = np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbk
    term = np.tensordot(term, t_1, axes=((0), (1)))  # jbkc
    energy += np.tensordot(
        term, u[v, o, v, o], axes=((0, 1, 2, 3), (3, 0, 1, 2)) #Check
    )

    term = (-1) * np.tensordot(l_1, t_1, axes=((0), (1)))  # ba 
    term = np.tensordot(f[o, v], term, axes=((1), (1)))  # ib
    energy += np.tensordot(term, t_1, axes=((0, 1), (1, 0))) #Check
    #term = (-1) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbk
    #term = np.tensordot(term, t_2, axes=((0, 2), (2, 0)))  # jkcl
    #energy += np.tensordot(
    #    term, u[o, o, v, o], axes=((0, 1, 2, 3), (3, 0, 2, 1))
    #)
   # term = (-1) * np.tensordot(l_2, t_1, axes=((0), (1)))  # jabc
   # term = np.tensordot(term, t_2, axes=((0, 1), (2, 0)))  # bcdk
   # energy += np.tensordot(
   #     term, u[v, o, v, v], axes=((0, 1, 2, 3), (0, 2, 3, 1))
   # )

    term = (-0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # kbca
    term = np.tensordot(f[o, v], term, axes=((1), (3)))  # ikbc
    energy += np.tensordot(term, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #Check
    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # jkci
    term = np.tensordot(f[o, v], term, axes=((0), (3)))  # ajkc
    energy += np.tensordot(term, t_2, axes=((0, 1, 2, 3), (0, 2, 3, 1)))#Check
    term = (-0.5) * np.tensordot(l_1, t_1, axes=((1), (0)))  # ij
    term = np.tensordot(term, t_2, axes=((0), (2)))  # jbck
    energy += np.tensordot(
        term, u[o, o, v, v], axes=((0, 1, 2, 3), (0, 2, 3, 1)) #Check
    )
    term = (-0.5) * np.tensordot(l_1, t_1, axes=((0), (1)))  # ab
    term = np.tensordot(term, t_2, axes=((0), (0)))  # bcjk
    energy += np.tensordot(
        term, u[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1)) #Check
    )
    #term = (-0.5) * np.tensordot(l_2, t_2, axes=((0, 2, 3), (2, 0, 1)))  # jl
    #term = np.tensordot(term, u[o, o, v, o], axes=((0, 1), (3, 1)))  # kc
    #energy += np.tensordot(t_1, term, axes=((0, 1), (1, 0)))
    #term = (-0.5) * np.tensordot(l_2, t_2, axes=((0, 1, 2), (2, 3, 0)))  # bd
    #term = np.tensordot(term, u[v, o, v, v], axes=((0, 1), (0, 3)))  # kc
    #energy += np.tensordot(t_1, term, axes=((0, 1), (1, 0)))

    term = (-0.25) * np.tensordot(t_1, u[o, o, o, o], axes=((1), (0)))  # blij
    term = np.tensordot(t_1, term, axes=((1), (1)))  # abij
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #Check
    term = (-0.25) * np.tensordot(t_1, u[v, v, v, v], axes=((0), (3)))  # iabc
    term = np.tensordot(t_1, term, axes=((0), (3)))  # jiab
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (1, 0, 2, 3))) #Check
   # term = (0.25) * np.tensordot(
   #     t_2, u[v, o, v, v], axes=((0, 1), (2, 3))
   # )  # ijbk
   # term = np.tensordot(t_1, term, axes=((1), (3)))  # aijb
   # energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (1, 2, 0, 3)))
   # term = (0.25) * np.tensordot(
   #     t_2, u[o, o, v, o], axes=((2, 3), (0, 1))
   # )  # abcj
   # term = np.tensordot(t_1, term, axes=((0), (2)))  # iabj
   # energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (0, 3, 1, 2)))

    term = (-1) * np.tensordot(t_1, u[o, o, v, v], axes=((0), (3)))  # ijkb
    term = np.tensordot(t_1, term, axes=((0, 1), (3, 1)))  # ik
    term = np.tensordot(t_1, term, axes=((1), (1)))  # ai
    energy += np.tensordot(l_1, term, axes=((0, 1), (1, 0))) #Check
    #term = (-1) * np.tensordot(
    #    t_2, u[o, o, v, v], axes=((1, 3), (3, 1))
    #)  # bjkc
    #term = np.tensordot(t_1, term, axes=((0), (3)))  # ibjk
    #term = np.tensordot(t_1, term, axes=((1), (3)))  # aibj
    #energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (1, 3, 0, 2)))

    term = (-0.5) * np.tensordot(t_1, u[v, o, v, v], axes=((0), (3)))  # ibkc
    term = np.tensordot(t_1, term, axes=((0), (3)))  # jibk
    term = np.tensordot(t_1, term, axes=((1), (3)))  # ajib
    energy += np.tensordot(l_2, term, axes=((0, 1, 2, 3), (2, 1, 0, 3))) #Check
    #term = (-0.5) * np.tensordot(
    #    t_1, u[o, o, v, v], axes=((0, 1), (2, 1))
    #)  # kd
    #term = np.tensordot(t_1, term, axes=((1), (0)))  # ad
    #term = np.tensordot(l_2, term, axes=((2), (0)))  # ijbd
    #energy += np.tensordot(term, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    term = (-0.5) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    term = np.tensordot(term, t_1, axes=((0), (1)))  # jlkc
    energy += np.tensordot(
        term, u[o, o, v, o], axes=((0, 1, 2, 3), (3, 1, 0, 2)) #Check
    )
    #term = (-0.5) * np.tensordot(l_2, t_1, axes=((0), (1)))  # jabc
    #term = np.tensordot(term, t_2, axes=((0, 1, 2), (2, 0, 1)))  # cl
    #term = np.tensordot(term, u[o, o, v, v], axes=((0, 1), (2, 1)))  # kd
    #energy += np.tensordot(t_1, term, axes=((0, 1), (1, 0)))

    #term = (-0.125) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbl
    #term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    #term = np.tensordot(term, t_2, axes=((0, 1), (2, 3)))  # lkcd
    #energy += np.tensordot(
    #    term, u[o, o, v, v], axes=((0, 1, 2, 3), (1, 0, 2, 3))
    #)
    #term = (-0.125) * np.tensordot(l_2, t_1, axes=((1), (1)))  # iabc
    #term = np.tensordot(term, t_1, axes=((0), (1)))  # abcd
    #term = np.tensordot(term, t_2, axes=((0, 1), (0, 1)))  # cdkl
    #energy += np.tensordot(
    #    term, u[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1))
    #)

    term = (0.25) * np.tensordot(l_2, t_1, axes=((2), (0)))  # ijbl
    term = np.tensordot(term, t_1, axes=((2), (0)))  # ijlk
    term = np.tensordot(term, t_1, axes=((1), (1)))  # ilkc
    term = np.tensordot(term, t_1, axes=((0), (1)))  # lkcd
    energy += np.tensordot(
        term, u[o, o, v, v], axes=((0, 1, 2, 3), (1, 0, 2, 3))#Check
    )

    ##############################################################################
    ##############################################################################
    ##########################         CCD        ################################
    ##############################################################################
    ##############################################################################

    energy += 0.25 * np.tensordot(
        l_2, u[v, v, o, o], axes=((0, 1, 2, 3), (2, 3, 0, 1)) #Check
    )

    energy += 0.25 * np.tensordot(
        t_2, u[o, o, v, v], axes=((0, 1, 2, 3), (2, 3, 0, 1)) #Check
    )

    #temp_ldem = np.tensordot(l_2, t_2, axes=((1, 3), (2, 1)))
    #energy += np.tensordot(
 #       temp_ldem, u[v, o, v, o], axes=((0, 1, 2, 3), (3, 0, 2, 1))
 #   )

    temp_dckl = 0.5 * np.tensordot(t_2, f[o, o], axes=((3), (0)))
    energy += np.tensordot(l_2, temp_dckl, axes=((0, 1, 2, 3), (3, 2, 0, 1))) #Check


    temp_dclk = 0.5 * np.tensordot(f[v, v], t_2, axes=((1), (0)))
    
    energy += np.tensordot(temp_dclk, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #Check
    energy += np.tensordot(temp_dclk, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))) #Check
    # np.testing.assert_allclose(result, energy)


    #temp_dclk = 0.125 * np.tensordot(u[v, v, v, v], t_2, axes=((2, 3), (0, 1)))
    #energy += np.tensordot(temp_dclk, l_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))

    #temp_clnf = 0.5 * np.tensordot(t_2, u[o, o, v, v], axes=((0, 3), (2, 0)))
    #temp_kdnf = np.tensordot(l_2, temp_clnf, axes=((0, 3), (1, 0)))
    #energy += np.tensordot(t_2, temp_kdnf, axes=((0, 1, 2, 3), (1, 3, 0, 2)))


    #temp_cf = -0.25 * np.tensordot(
    #    t_2, u[o, o, v, v], axes=((0, 2, 3), (2, 0, 1))
    #)
    #temp_lkdf = np.tensordot(l_2, temp_cf, axes=((3), (0)))
    #energy += np.tensordot(temp_lkdf, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    # np.testing.assert_allclose(result, energy)

    #temp_ln = -0.125 * np.tensordot(
    #    t_2, u[o, o, v, v], axes=((0, 1, 3), (2, 3, 0))
    #)
    #temp_dckl = np.tensordot(t_2, temp_ln, axes=((3), (1)))
    #energy += np.tensordot(l_2, temp_dckl, axes=((0, 1, 2, 3), (3, 2, 0, 1)))

#    temp_km = -0.125 * np.tensordot(l_2, t_2, axes=((0, 2, 3), (2, 0, 1)))
#    temp_knef = np.tensordot(temp_km, u[o, o, v, v], axes=((1), (0)))
#    energy += np.tensordot(temp_knef, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    # np.testing.assert_allclose(result, energy)

 #   if o.stop >= v.stop // 2:
 #       temp_dcef = 0.0625 * np.tensordot(
 #           t_2, u[o, o, v, v], axes=((2, 3), (0, 1))
 #       )
 #       temp_efdc = np.tensordot(t_2, l_2, axes=((2, 3), (0, 1)))
 #       energy += np.tensordot(
 #           temp_efdc, temp_dcef, axes=((0, 1, 2, 3), (2, 3, 0, 1))
 #       )
 #   else:
 #       temp_lkmn = 0.0625 * np.tensordot(l_2, t_2, axes=((2, 3), (0, 1)))
 #       temp_mnlk = np.tensordot(u[o, o, v, v], t_2, axes=((2, 3), (0, 1)))
 #       energy += np.tensordot(
 #           temp_lkmn, temp_mnlk, axes=((0, 1, 2, 3), (2, 3, 0, 1))
 #       )
    # np.testing.assert_allclose(result, energy)

    return energy
