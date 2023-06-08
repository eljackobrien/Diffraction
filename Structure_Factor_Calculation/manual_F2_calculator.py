# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:35:55 2023

@author: OBRIEJ25
"""
# =============================================================================
# A script to calculate the structure factors for any given lattice and reflection.
#
# The atomic form factors are calculated using the multi-Gaussian approximation, see
# ITC Vol C, section 6.1.1: Atomic Scattering Factors.
# Anomalous dispersion and nuclear scattering are neglected.
#
# The Structure Factors are calculated by summing over the atoms in the unit cell in the usual way:
# f(hkl) = sum_{i=1}^N  [ Occ_n * exp(-B_n |Q|**2 / 4) * f_n * exp( Q(hkl) * r_i ) ]
# The formula is confirmed from FullProf
# For now, I am keeping B constant ~ 1 and occupancy constant = 1, they can be added to the function easily.
# =============================================================================
import numpy as np
from os import path
path_prefix = path.dirname(path.abspath(__file__))

# Load and functionalise atomic form factors - Xrays
gauss_approx_coefs = np.genfromtxt( path.join(path_prefix, 'form_factors/Xray_form_factors.txt') , dtype=str)
gauss_elements, gauss_coefs = gauss_approx_coefs[:,0], gauss_approx_coefs[:,1:].astype(float)

# Equation valid for square lattice, all angles = 90
def get_Q_len(a,b,c, h,k,l):
    return (h**2/a**2 + k**2/b**2 + l**2/c**2)**0.5

# Calculate the form factor for a given element and scattering vector length (Q in 1/A)
def get_atomic_form_factor_low_Q(element, Q_len, temp=None):
    if np.mean(Q_len) > 1.5: print(temp, "Q_len is very large, are you sure you are using A and not nm?")
    Q_len = Q_len*10  # convert from A to nm, table in file is in nm but I use A
    if element not in gauss_elements: raise Exception(f"{element} is not in element list")
    a1,b1,a2,b2,a3,b3,a4,b4,c = gauss_coefs[(gauss_elements == element).argmax()]
    a, b = [a1,a2,a3,a4], [b1,b2,b3,b4]
    f = 0
    for i in range(4): f += a[i] * np.exp( -b[i] * (Q_len/(4*np.pi))**2 )
    return f + c

def get_F2_Xrays(a,b,c, h,k,l, atoms, positions):
    hkl = np.array([h,k,l])
    Q_len = get_Q_len(a,b,c, h,k,l)
    F = 0
    # Loop over atoms in the cell: get form factor and phase -> add to structure factor
    for atom,pos in zip(atoms,positions):
        O, B = 1, 0.0
        f = get_atomic_form_factor_low_Q(atom, Q_len, temp=hkl)
        # Terms in brackets are the occupancy and isotropic Deby-Waller factor, respectively
        # B in FullProf is in A^2 and I am using A (now...) so no scaling
        F += (O) * f * np.exp(-(B)*Q_len**2/4) * np.exp( -1j * 2 * np.pi * np.dot(pos, hkl) )
    # Return absolute squared F
    return abs( F * F.conj() )


# Load neutron isotopic form factors
neutron_fns = np.genfromtxt( path.join(path_prefix, 'form_factors/Neutron_form_factors.txt') , dtype=str)
isotope, occurence, b_coh, b_incoh, sig_coh, sig_incoh, sig_scatt, sig_absrb = neutron_fns.T
b_coh = b_coh.astype(complex)

# Store form factors (mostly constant with Q) in a dictionary
neutron_atomic_form_factor = {}
for iso, b in zip(isotope, b_coh):
    neutron_atomic_form_factor[iso] = b.real if b.imag == 0 else b

def get_F2_neutrons(h,k,l, atoms, positions):
    # Calculate squared structure factors
    hkl = np.array([h,k,l])
    F = 0
    # Loop over atoms in the cell: get form factor and phase -> add to structure factor
    for atom,pos in zip(atoms,positions):
        f = neutron_atomic_form_factor[atom]
        F += f * np.exp( -1j * 2 * np.pi * np.dot(pos, hkl) )
    # Return absolute squared F
    return abs( F * F.conj() )


def get_F2(a,b,c, h,k,l, atoms, positions, particle='photon'):
    if particle == 'photon': return get_F2_Xrays(a,b,c, h,k,l, atoms, positions)
    elif particle == 'neutron': return get_F2_neutrons(h,k,l, atoms, positions)
    else: raise Exception("Choose particle = 'photon' or 'neutron'")



if "__main__" in __name__:

    plot_tests = 1
    import matplotlib.pyplot as plt
    no_stem_plot_function = 0
    try: from mpl_stem_plot import plt_stem
    except: no_stem_plot_function = 1

    # Function to plot a unit cell in 3D given positions and colours of atoms
    def plot(lat_size, frac_pos, colours, save_path=None):
        cube_corners = lat_size*np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,1,1],[1,0,1],[1,1,0]])
        X, Y, Z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])

        fig = plt.figure(figsize=[5,5], constrained_layout=True)
        ax = fig.add_subplot(projection='3d')
        ax.view_init(7.5, 250)
        ax.set_xlim(0,lat_size.max()), ax.set_ylim(0,lat_size.max()), ax.set_zlim(0,lat_size.max())

        for pos,col in zip(frac_pos,colours): ax.scatter(*(pos*lat_size), c=col, s=200)

        for cor in cube_corners:
            if cor[0] == 0: plt.plot(*np.vstack((cor, cor+X*lat_size)).T, c='k', lw=1, alpha=0.75)
            if cor[1] == 0: plt.plot(*np.vstack((cor, cor+Y*lat_size)).T, c='k', lw=1, alpha=0.75)
            if cor[2] == 0: plt.plot(*np.vstack((cor, cor+Z*lat_size)).T, c='k', lw=1, alpha=0.75)

        plt.savefig(save_path) if save_path else plt.show()


    # Test "get_atomic_form_factor_low_Q" function
    if plot_tests:
        Q = np.linspace(0,1, 500)
        fn = get_atomic_form_factor_low_Q('Ru', Q)
        fig, ax = plt.subplots(1,1, figsize=[6,4], constrained_layout=True)
        ax.scatter(Q, fn, c='b', s=20)
        ax.grid(alpha=0.5), ax.set_xlim(0,1)
        ax.set_xlabel("$Q$ (nm$^{-1}$)"), ax.set_ylabel("$f_n(Q)$")
        plt.show()


    #%% Define structure
    # Mn2RuGa tetragonal unit cell (XA inverse Heusler structure - F-43m)
    a, b, c = 4.214, 4.214, 6.04  # in nm (because Q is in 1/nm)
    lat = np.array([a,b,c])
    atoms = 6*["Mn"] + 4*["Ru"] + 4*["Mn"] + 9*["Ga"]
    colours = 6*["r"] + 4*["k"] + 4*["b"] + 9*["g"]
    # atomic fractional coordinates in unit cell
    positions = [[0.5, 0.5, 0.0],  # Mn4a
                 [0.0, 0.0, 0.5],
                 [0.0, 1.0, 0.5],
                 [1.0, 0.0, 0.5],
                 [1.0, 1.0, 0.5],
                 [0.5, 0.5, 1.0],

                 [0.5, 0.0, 0.75],  # Mn4c
                 [0.5, 1.0, 0.75],
                 [0.0, 0.5, 0.25],
                 [1.0, 0.5, 0.25],

                 [0.5, 0.0, 0.25],  # Ru4d
                 [0.5, 1.0, 0.25],
                 [0.0, 0.5, 0.75],
                 [1.0, 0.5, 0.75],

                 [0.0, 0.0, 0.0],  # Ga4b
                 [0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 1.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.5, 0.5, 0.5]]

    if plot_tests: plot(lat, positions, colours)


    #%% Calculate Xray Structure Factors
    # Check for a few peaks + compare 002/004 F2 ratio for XA and L21 structure MRG
    print("MRG - XA structure - tetrgonal representation")
    for hkl in [[0,0,2],[0,0,3],[0,0,4],[1,1,4],[1,1,6]]:
        peak = ''.join( np.array(hkl).astype(str) )
        F2 = get_F2(a,b,c, *hkl, atoms, positions)
        print(f"F2({peak}) = {F2:.1f}")


    F2_002 = get_F2(a,b,c, 0,0,2, atoms, positions)
    F2_004 = get_F2(a,b,c, 0,0,4, atoms, positions)
    print(f"\nXA: S_002/004 = {F2_002/F2_004:.3f}")

    # Change from XA to L21 structure
    atoms2 = 6*["Ru"] + 4*["Mn"] + 4*["Mn"] + 9*["Ga"]
    colours2 = 6*["k"] + 4*["purple"] + 4*["purple"] + 9*["g"]

    if plot_tests: plot(lat, positions, colours2)

    F2_002 = get_F2(a,b,c, 0,0,2, atoms2, positions)
    F2_004 = get_F2(a,b,c, 0,0,4, atoms2, positions)
    print(f"L21: S_002/004 = {F2_002/F2_004:.3f}")



    #%% Calculate an entire corrected XRD spectrum
    # Note: not taking account of peak multiplicity which scales intensity in powder scan
    def rad(tt): return tt*np.pi/180

    tt_M = rad(45.296)  # Ge 220 monochromator angle
    # polarisation intentisy correction
    def P_corr(tt): return (1 + np.cos(tt_M)**2 * np.cos(tt)**2) / (1 + np.cos(tt_M)**2)
    # Lorentz intensity correction
    def L_corr(tt): return np.sin(tt)
    # Irradiated volume correction for collimated beam (UN-USED)
    def V_corr(w): return 1/np.sin(w)

    # A selection of peaks
    peaks = [[0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5], [0,0,6],
             [1,1,1], [1,1,2], [1,1,3], [1,1,4], [1,1,5], [1,1,6],
             [2,0,1], [2,0,2], [2,0,3], [2,0,4], [2,0,5], [2,0,6],
             [4,0,1], [1,2,3], [4,0,3], [4,0,4], [2,2,4], [2,2,2]]

    # Convert from reciprocal space units to 2*theta for Cu-Ka X-rays
    def q_to_tt(q):
        return 180/np.pi * 2*np.arcsin(1.54059*q/2)

    # Calculate 2*theta, structure factor and intensity (excluding volume correction)
    tt0s, F20s, I0s, names = [], [], [], []
    for peak in peaks:
        Q = get_Q_len(a,b,c, *peak)
        tt_hkl = q_to_tt( Q )
        if (20 > tt_hkl) | (tt_hkl > 130): continue
        F20 = get_F2(a,b,c, *peak, atoms, positions)
        I0 = F20 * P_corr(rad(tt_hkl))/L_corr(rad(tt_hkl))
        tt0s.append(tt_hkl), F20s.append(F20), I0s.append(I0)
        names.append('$('+''.join(np.array(peak).astype(str))+')$')

    # Normalise intensity to 100
    scale = 100/np.max(F20s)

    fig, ax = plt.subplots(1,1, figsize=[6,3], constrained_layout=True)

    ax.grid(alpha=0.5)
    for name,tt0,F20,I0 in zip(names,tt0s,F20s,I0s):
        z = (1 if name==names[-1] else 0)
        if no_stem_plot_function:
            plt.scatter(tt0, F20*scale, s=4, c='b')
            plt.scatter(tt0, I0*scale, s=4, c='r', alpha=0.5)
        else:
            plt_stem(tt0, F20*scale, 0)
            plt_stem(tt0, I0*scale, 0, c='r', alpha=0.5)
        ax.text(tt0, F20*scale+2, name, ha='center', va='bottom')
    ax.set_xlim(20,130), ax.set_ylim(-2.5,112.5)
    # Points for legend
    for clr,lab,alpha in zip(['b','r'],['$F^2$','$I$'],[1,0.5]):
        plt.scatter(None,None,s=30,c=clr,label=lab,alpha=alpha)
    ax.legend()
    ax.set_title("MRG - XA Structure - Simulated Powder Scan", fontsize=12)
    plt.show()



    #%% Neutron (nuclear only) structure factors
    # Test isotope with complex scattering length
    print("\n\n~~~~~ Neutrons ~~~~~")
    print(f"B (b=5.3-0.213j) -> F2(002) = {get_F2(1,1,1, 0,0,2, ['B'], [[0.5,0.5,0.5]], particle='neutron'):.1f}\n")

    # Test a few peaks for XA and L21 structure MRG Heusler alloy
    print("\tMRG\t\tXA\t\tL21")
    for peak in [[0,0,2],[0,0,4],[1,0,1],[1,0,3],[2,1,1],[2,0,2]]:
        pk = ''.join(np.array(peak).astype(str))
        F2_XA = get_F2(1,1,1, *peak, atoms, positions, particle='neutron')
        F2_L21 = get_F2(1,1,1, *peak, atoms2, positions, particle='neutron')
        print(f"{pk} peak:\t{F2_XA:4.0f}\t{F2_L21:.0f}")


