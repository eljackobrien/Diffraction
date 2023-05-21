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
# without Debye-Waller corrections:    f(hkl) = sum_{i=1}^N [ f_n * exp( hkl * r_i ) ]
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

plot_tests = 0

# Load and functionalise atomic form factors
gauss_approx_coefs = np.genfromtxt("atomic_form_factors_gaussian_approx.txt", dtype=str)
gauss_elements, gauss_coefs = gauss_approx_coefs[:,0], gauss_approx_coefs[:,1:].astype(float)

def get_atomic_form_factor_low_Q(element, G_len):
    a1,b1,a2,b2,a3,b3,a4,b4,c = gauss_coefs[(gauss_elements == element).argmax()]
    a, b = [a1,a2,a3,a4], [b1,b2,b3,b4]
    f = 0
    for i in range(4): f += a[i] * np.exp( -b[i] * (G_len/(4*np.pi))**2 )
    return f + c

if plot_tests:  # Plot approximation to atomic form factor
    element = 'Ru'
    G = np.linspace(0,10, 500)
    fig, ax = plt.subplots(1,1, figsize=[6,4], constrained_layout=True)
    I_low_Q = get_atomic_form_factor_low_Q(element, G)
    ax.scatter(G, I_low_Q, c='b', s=20)
    plt.show()


#%% Define structure
# Mn2RuGa tetragonal unit cell (XA inverse Heusler structure - F-43m)
a, b, c = 0.4214,0.4214, 0.604
lat = np.array([a,b,c])
atoms = 6*["Mn"] + 4*["Mn"] + 4*["Ru"] + 9*["Ga"]
colours = 6*["r"] + 4*["b"] + 4*["k"] + 9*["g"]
# XA F43m positions (Mn4a, Mn4c, Ru, Ga)
positions = [[0.5, 0.5, 0.0],
             [0.0, 0.0, 0.5],
             [0.0, 1.0, 0.5],
             [1.0, 0.0, 0.5],
             [1.0, 1.0, 0.5],
             [0.5, 0.5, 1.0],

             [0.5, 0.0, 0.75],
             [0.5, 1.0, 0.75],
             [0.0, 0.5, 0.25],
             [1.0, 0.5, 0.25],

             [0.5, 0.0, 0.25],
             [0.5, 1.0, 0.25],
             [0.0, 0.5, 0.75],
             [1.0, 0.5, 0.75],

             [0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 1.0, 1.0],
             [1.0, 0.0, 1.0],
             [1.0, 1.0, 1.0],
             [0.5, 0.5, 0.5]]

def plot(positions, colours):  # plot unit cell
    fig = plt.figure(figsize=[6,4], constrained_layout=True)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(7.5, 250)
    ax.set_xlim(0,0.6), ax.set_ylim(0,0.6), ax.set_zlim(0,0.6)
    [ax.scatter(*(positions[i]*lat), c=colours[i], s=200, marker="$\u263A$") for i in range(len(positions))]
    plt.show()

if plot_tests: plot(positions, colours)


#%% Calculate Structure Factors
def get_Q_len(a,b,c, h,k,l):  # Equation valid for square lattice, all angles = 90
    return (h**2/a**2 + k**2/b**2 + l**2/c**2)**0.5

def F2(a,b,c, h,k,l, atoms, positions):
    # Calculate squared structure factors
    hkl = np.array([h,k,l])
    G_len = get_Q_len(a,b,c, h,k,l)
    F = 0
    # Loop over atoms in the cell: get form factor and phase -> add to structure factor
    for atom,pos in zip(atoms,positions):
        f = get_atomic_form_factor_low_Q(atom, G_len)
        F += f * np.exp( -1j * 2 * np.pi * np.dot(pos, hkl) )
    # Return absolute squared F
    return abs( F * F.conj() )


# Check for a few peaks
# Using tetragonal unit cell -> to convert to full cell, rotate 45 + scale 2**0.2
# In other words, 204 full cell is 114 tetra cell, etc. 00L peaks unchanged
F2_002 = F2(a,b,c, 0,0,2, atoms, positions)
F2_003 = F2(a,b,c, 0,0,3, atoms, positions)
F2_004 = F2(a,b,c, 0,0,4, atoms, positions)
F2_204 = F2(a,b,c, 1,1,4, atoms, positions)
F2_206 = F2(a,b,c, 1,1,6, atoms, positions)


#print(f"{F2_002}\n{F2_003}\n{F2_004}\n{F2_204}\n{F2_206}")
print(f"XA: S = {F2_002/F2_004:.3f}")



# Change from XA to L21 structure
atoms2 = 6*["Ru"] + 4*["Mn"] + 4*["Mn"] + 9*["Ga"]
colours2 = 6*["k"] + 4*["purple"] + 4*["purple"] + 9*["g"]

if plot_tests: plot(positions, colours2)

F2_002 = F2(a,b,c, 0,0,2, atoms2, positions)
F2_004 = F2(a,b,c, 0,0,4, atoms2, positions)
print(f"L21: S = {F2_002/F2_004:.3f}")


#%% Calculate an entire corrected XRD spectrum
# Note: not taking account of peak multiplicity which scales intensity in powder scan

def rad(tt): return tt*np.pi/180
tt_M = rad(45.296)  # Ge 220 monochromator angle
def P_corr(tt):  # polarisation intentisy correction
    return (1 + np.cos(tt_M)**2 * np.cos(tt)**2) / (1 + np.cos(tt_M)**2)
def L_corr(tt):  # Lorentz intensity correction
    return np.sin(tt)

def V_corr(w):  # Irradiated volume correction for collimated beam (UN-USED)
    return 1/np.sin(w)

def LP_corr(tt): return P_corr(rad(tt)) / L_corr(rad(tt))

# A selection of peaks
peaks = [[0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5], [0,0,6],
         [1,1,1], [1,1,2], [1,1,3], [1,1,4], [1,1,5], [1,1,6],
         [2,0,1], [2,0,2], [2,0,3], [2,0,4], [2,0,5], [2,0,6],
         [4,0,1], [1,2,3], [4,0,3], [4,0,4], [2,2,4], [2,2,2]]


def q_to_tt(q): return 180/np.pi * 2*np.arcsin(0.154059*q/2)
x = [q_to_tt( get_Q_len(a,b,c, *peak) ) for peak in peaks]

# Initialise spectrum
tt = np.linspace(10,120,1000)
I = 0*tt

def lorentzian(x, x0, gam, height):
    y = gam**2 / ((x-x0)**2 + gam**2)
    return height * y/y.max()

# Add all the peaks with narrow Voigt profile
for peak in peaks:
    Q = get_Q_len(a,b,c, *peak)
    x = q_to_tt( Q )
    if (10 > x) | (x > 120): continue
    y = F2(a,b,c, *peak, atoms2, positions) * LP_corr(x)
    contribution = lorentzian(tt, x, 0.01, y)
    I += contribution
# Normalise intensity
I = 100*I/I.max()

fig = plt.figure(figsize=[8,4], constrained_layout=True)
ax = fig.add_subplot()
ax.plot(tt, I)
ax.set_xlim(10,120)
plt.show()

