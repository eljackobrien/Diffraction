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
# The Structure Factors are calculated by summing over the atoms in the unit cell in the usual way,
# without Debye-Waller corrections:    f(hkl) = sum_{i=1}^N [ f_n * exp( Q(hkl) * r_i ) ]
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
plot_tests = 1

# Load and functionalise atomic form factors
gauss_approx_coefs = np.genfromtxt("Xray_atomic_form_factors_gaussian_approx.txt", dtype=str)
gauss_elements, gauss_coefs = gauss_approx_coefs[:,0], gauss_approx_coefs[:,1:].astype(float)

# Equation valid for square lattice, all angles = 90
def get_Q_len(a,b,c, h,k,l):
    return (h**2/a**2 + k**2/b**2 + l**2/c**2)**0.5

# Calculate the form factor for a given element and scattering vector length (Q in 1/nm)
def get_atomic_form_factor_low_Q(element, Q_len):
    a1,b1,a2,b2,a3,b3,a4,b4,c = gauss_coefs[(gauss_elements == element).argmax()]
    a, b = [a1,a2,a3,a4], [b1,b2,b3,b4]
    f = 0
    for i in range(4): f += a[i] * np.exp( -b[i] * (Q_len/(4*np.pi))**2 )
    return f + c

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
    Q = np.linspace(0,10, 500)
    fn = get_atomic_form_factor_low_Q('Ru', Q)
    fig, ax = plt.subplots(1,1, figsize=[6,4], constrained_layout=True)
    ax.scatter(Q, fn, c='b', s=20)
    ax.grid(alpha=0.5), ax.set_xlim(0,10)
    ax.set_xlabel("$Q$ (nm$^{-1}$)"), ax.set_ylabel("$f_n(Q)$")
    plt.show()


#%% Define structure
# Mn2RuGa tetragonal unit cell (XA inverse Heusler structure - F-43m)
a, b, c = 0.4214, 0.4214, 0.604  # in nm (because Q is in 1/nm)
lat = np.array([a,b,c])
atoms = 6*["Mn"] + 4*["Mn"] + 4*["Ru"] + 9*["Ga"]
colours = 6*["r"] + 4*["b"] + 4*["k"] + 9*["g"]
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


#%% Calculate Structure Factors
def get_F2(a,b,c, h,k,l, atoms, positions):
    hkl = np.array([h,k,l])
    Q_len = get_Q_len(a,b,c, h,k,l)
    F = 0
    # Loop over atoms in the cell: get form factor and phase -> add to structure factor
    for atom,pos in zip(atoms,positions):
        f = get_atomic_form_factor_low_Q(atom, Q_len)
        F += f * np.exp( -1j * 2 * np.pi * np.dot(pos, hkl) )
    # Return absolute squared F
    return abs( F * F.conj() )


# Check for a few peaks + compare 002/004 F2 ratio for XA and L21 structure MRG
F2_002 = get_F2(a,b,c, 0,0,2, atoms, positions)
F2_003 = get_F2(a,b,c, 0,0,3, atoms, positions)
F2_004 = get_F2(a,b,c, 0,0,4, atoms, positions)
F2_204 = get_F2(a,b,c, 1,1,4, atoms, positions)
F2_206 = get_F2(a,b,c, 1,1,6, atoms, positions)

print("MRG - XA structure - tetrgonal representation")
for peak,F2 in zip(['002','003','004','204','206'], [F2_002,F2_003,F2_004,F2_204,F2_206]):
    print(f"F2({peak}) = {F2:.1f}")
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
    return 180/np.pi * 2*np.arcsin(0.154059*q/2)

# Lorentzian (Cauchy) profile
def lorentzian(x, x0, gam, height):
    y = gam**2 / ((x-x0)**2 + gam**2)
    return height * y/y.max()

# Initialise spectrum
tt = np.linspace(10,120,1000)
I = 0*tt

# Add each peak via a narrow Lorentzian profile
for peak in peaks:
    Q = get_Q_len(a,b,c, *peak)
    tt_hkl = q_to_tt( Q )
    if (10 > tt_hkl) | (tt_hkl > 120): continue
    y = get_F2(a,b,c, *peak, atoms2, positions) * P_corr(rad(tt_hkl))/L_corr(rad(tt_hkl))
    contribution = lorentzian(tt, tt_hkl, 0.01, y)
    I += contribution

# Normalise intensity to 100
I = 100*I/I.max()

fig, ax = plt.subplots(1,1, figsize=[6,3], constrained_layout=True)
ax.plot(tt, I)
ax.grid(alpha=0.5)
ax.set_xlim(10,120)
plt.show()



#%% Neutron (nuclear only) structure factors
neutron_fns = np.genfromtxt("neutron_nuclear_atomic_form_factors.txt", dtype=str)
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

# Test isotope with complex scattering length
print(f"B (b=5.3-0.213j) -> F2(002) = {get_F2_neutrons(0,0,2, ['B'], [[0.5,0.5,0.5]]):.1f}\n")

# Test a few peaks for XA and L21 structure MRG Heusler alloy
print("\tMRG\t\tXA\t\tL21")
for peak in [[0,0,2],[0,0,4],[1,0,1],[1,0,3],[2,1,1],[2,0,2]]:
    pk = ''.join(np.array(peak).astype(str))
    F2_XA = get_F2_neutrons(*peak, atoms, positions)
    F2_L21 = get_F2_neutrons(*peak, atoms2, positions)
    print(f"{pk} peak:\t{F2_XA:4.0f}\t{F2_L21:.0f}")
