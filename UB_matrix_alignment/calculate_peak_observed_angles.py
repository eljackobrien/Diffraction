# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:44:09 2023

@author: OBRIEJ25
"""
import numpy as np
from numpy.linalg import norm, inv
rad = np.pi/180
from scipy.optimize import curve_fit as fit
from rotate_vector import rotate, get_ang
from cif_reader import Material, get_tt
# =============================================================================
# A script which calculates and lists the predicted peak positions for a sample, given the position
# of a few substrate reflections and the lattice parameters of both materials. Allows knowledge of
# global alignment from the measurement of a few intense reflections, assuming perfect epitaxy.
# -----------------------------------------------------------------------------
#     Procedure:
# Load substrate and sample information from .cif files (B matrix).
# Measure at least 2, preferably 4+, substrate reflections which are far from parallel.
# Use the Busing and Levy technique to calculate the U matrix for the substrate from these:
#     [1] https://doi.org/10.1107/S0365110X67000970
# By the nature of epitaxy, the U matrix is the same for the sample
#
# We define 4 cooradinate systems:
#     crystal_miller "h", crystal_reciprocal "h_c"
#     diffractometer_cartesian "h_phi", diffractometer_angular "h_th"
# B converts hkl to h_c
# U converts h_c to h_phi
# R converts h_phi to h_th
# Mathematically, B, U and R are (3*3) matrices, while the h are column vectors (u is unit vector)
# The solution amounts to finding the angles of R such that h_th = R*U*B*h
# =============================================================================

run_tests = 0

#%% Load material information and calculate B-matrices
CuKa = 1.54059  # A
sample_file = "MRG_tetragonal_conventional.cif"
substrate_file = "MgO_conventional.cif"
sub_meas_refl = 'UB_mat_input.txt'

samp, sub = Material(sample_file), Material(substrate_file)

# The general expression for B (for non square angled lattices) can be found in [1]
def get_B(a,b,c): return np.array([[1/a, 0, 0], [0, 1/b, 0], [0, 0, 1/c]])

B_samp, B_sub = get_B(samp.a,samp.b,samp.c), get_B(sub.a,sub.b,sub.c)



#%% Define rotation matrices and other functions for converting between different spaces
def Phi(p):   return np.array([[np.cos(p), -np.sin(p), 0], [-np.sin(p), np.cos(p), 0],  [0, 0, 1]])
def Chi(x):   return np.array([[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0,  np.cos(x)]])
def Omega(w): return np.array([[np.cos(w), np.sin(w), 0], [-np.sin(w), np.cos(w), 0], [0, 0, 1]])

# Overall rotation matrix and function to get h_phi (diffractometer cartesian) coorads from angles
def R(w, x, p): return Omega(w) @ Chi(x) @ Phi(p)
def get_u_phi(w, x, p): return Phi(p).T @ Chi(x).T @ Omega(w).T @ np.array([1, 0, 0])[:,None]



#%% Get U:
#       from two reflections by defining orthonormal triplets of unit vectors (see [1])
#       by least-squares regression of all the measured substrate vectors - used here

# Load measured substrate reflections, hkl and angles
h, k, l, omega0, tts, chi0, phi0 = np.genfromtxt(sub_meas_refl).T
# IMPORTANT - transform the measured angles for your diffractometer into the Busing & Levy definitions
omega, chi, phi = omega0-tts/2, 90+chi0, -phi0
omega, tts, chi, phi = omega*rad, tts*rad, chi*rad, phi*rad

# Measured unit vectors for substrate: u_phi from angles and u_c from miller indices
u_phi_obs = np.squeeze([ get_u_phi(o,c,p) for o,c,p in zip(omega,chi,phi) ]).T
h_c_obs = B_sub @ np.stack((h,k,l))
u_c_obs = h_c_obs / norm(h_c_obs, axis=0)

# Regression is a lot easier on flat arrays (inputs) -> regression function:
def r_func(x, U0,U1,U2,U3,U4,U5,U6,U7,U8):
    x = x.reshape(3, x.size//3)
    return ( np.array([[U0,U1,U2],[U3,U4,U5],[U6,U7,U8]]) @ x ).flatten()

# Having guess parameters = 0 can mess with some regression algorithms -> add noise to guess
U_guess = np.identity(3) + np.random.normal(scale=0.1, size=(3,3))
U_opt, U_cov = fit(r_func, u_c_obs.flatten(), u_phi_obs.flatten(), p0=U_guess.flatten())
U_err = np.sqrt(np.diag(U_cov))
U_opt, U_err = U_opt.reshape(3,3), U_err.reshape(3,3)
# Over 1000 repeats, the maximum difference in refined parameter values for U_opt was = 0.002 %
# Showing that the addition of noise to the guess values is good for initialisation


# Constrainied_fitting to force tetragonality of U
tetragonal_constrained_fitting = 0

if tetragonal_constrained_fitting:
    scale = B_sub[0,0]**2 / B_sub[2,2]**2

    def get_U_tetra(U1,U2,U5):
        return np.array([[1, U1, U2], [-U1, 1, U5], [-scale*U2, -scale*U5, 1]])

    def r_func_tetra(x, U1,U2,U5):
        x = x.reshape(3, x.size//3)
        return ( get_U_tetra(U1,U2,U5) @ x ).flatten()

    U_guess = np.zeros(3) + np.random.normal(scale=0.1, size=(3))
    U_opt, U_cov = fit(r_func_tetra , u_c_obs.flatten(), u_phi_obs.flatten(), p0=U_guess)
    U_err = np.sqrt(np.diag(U_cov))
    U_opt, U_err = get_U_tetra(*U_opt), get_U_tetra(*U_err)

print(U_opt), print(U_err)


UB_sub_opt = U_opt @ B_sub
UB_samp_opt = U_opt @ B_samp


#%% Get R:
# R is obtained by definiing an orthogonal basis T_phi in the diffractometer cartesian space in
# terms of the desired reflection. The reflection we want has hkl and h_phi = U B_samp hkl.
# The columns of T_phi are t1, t2, and t3
# t1 = u_phi,    t2 is in the diffraction plane perp to u_phi and t3 is perp to those two.
# we use v1 = hkl and v2 = hk0 to obtain T_phi
def T_phi(UB, hkl):
    v1 = hkl
    # v2 is planar projection of v1, or [1,0,0] if v1 = [0,0,L]
    v2 = v1 * np.array([1,1,0]) if sum(abs(v1)>0)>1 else np.array([1,0,0])
    v1, v2 = UB@v1, UB@v2
    t1, t2 = 1*v1, np.cross(v1, np.cross(v1,v2))
    t3 = np.cross(t1, t2)
    test = np.stack((t1,t2,t3))
    return test.T/norm(test, axis=1)

# R rotates this orthogonal basis to be coincident with the theta axes, i.e. R T = 1
# So R = T^-1 and since T is orthogonal T^-1 = T.T  ->  R = T.T  (transpose)
# See [1] for detailed explanation

# Equations to get each of the angles omega, chi and phi from R
def get_angs(R):
    o = np.arctan2(-R[1,2], R[0,2])
    c = np.arctan2( np.sqrt(R[2,0]**2 + R[2,1]**2), R[2,2] )
    p = np.arctan2(-R[2,1], -R[2,0])
    return np.vstack((o, c, p))

# Test on measured substrate reflections
if run_tests:
    test_phi = U_opt @ B_sub @ np.stack((h,k,l))
    angs = []
    for x,tt in zip(test_phi.T, tts) :
        o,c,p = get_angs( T_phi(UB_sub_opt, x.T).T )
        o, c, p = o+tt/2, c-90*rad, -p
        angs.append(np.stack((o,c,p)))
    angs = np.squeeze(angs)/rad
    print(angs, end='\n\n')
    print(np.stack((omega0, chi0, phi0)).T)



#%% Decide Sample Peaks
# Get relevant peaks, sort by volume-corrected intensity, remove duplicates
samp.hkls = samp.hkls[(samp.omegas > 0.01) & (samp.tts > 20*rad)]
samp.calc_diffrac_vars(volume_corr=1)
inds = np.unique( np.stack((samp.tts, samp.F2s)), axis=1, return_index=True )[1]
samp.hkls = samp.hkls[inds]
samp.calc_diffrac_vars(volume_corr=1)

samp.hkls = samp.hkls[samp.F2s.argsort()[::-1]]
samp.calc_diffrac_vars(volume_corr=1)


# Decide what peaks to scan and for how long
# Loop over peak families, choose three random azimuths for each one and add to the list
# Stop once there are no available hours left
available_hours = 24+24+12
# Can raise counting_time_scale for very thin samples or lower for very thick
counting_time_scale = 1

write_to_file = 1

string = "# Peak        Omega     2*theta         Chi         Phi       hrs"
time_left, i = True, 0
while time_left:
    # Choose time to scan each peak from this family
    if samp.Is[i] > 50: time = 1
    elif samp.Is[i] > 15: time = 2
    elif samp.Is[i] > 5: time = 3
    else: time = 4
    if counting_time_scale > 1: time += 1
    elif counting_time_scale < 1: time = (time-1 if time>1 else 1)

    # Randomly choose three of the four possible azimuths (only two for 00L peaks)
    num_from_fam = (3 if sum(abs(samp.hkls[i])>0)>1 else 1)
    azimuths = np.random.choice([0,np.pi/2,np.pi,1.5*np.pi], num_from_fam, False)
    for azimuth in azimuths:
        hkl = rotate(samp.hkls[i], [0,0,1], azimuth)

        omega, chi, phi = get_angs( T_phi(UB_samp_opt, hkl).T ).flatten()
        omega, chi, phi = omega + samp.tts[i]/2, chi-90*rad, -phi
        hkl_str = ''.join((1.05*hkl).astype(int).astype(str))

        string += f"\n{'('+hkl_str+')':7.8s}{omega/rad:12.4f}{samp.tts[i]/rad:12.4f}{chi/rad:12.4f}{phi/rad:12.4f}{time:10.0f}"

        available_hours -= time
        if available_hours <= 0:
            time_left = False
            break

    if not time_left: break
    i += 1

print(string)
if write_to_file:
    with open("calculated_peak_positions.txt", "w+") as file:
        file.write(string)


# Can add in code to skip peaks with large chi angles (h,k > 0 but h!=k)
# (np.diff(samp.hkls)[:,0] == 0) | (samp.hkls[:,:2]==0).any(axis=1)
#
# And some of the high symmetry peaks such as 001 are overestimated by the cof reader,
#     so they could be skipped if the code is too keen to scan them.
#
# Also if peaks with low intensity are desired to "anchor" the values, can just loop over
#     a custom list of peaks






