# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 21:47:44 2021

@author: eljac
"""

import numpy as np

def get_ang(v1, v2, norm=None):
    """
    Return the angle between two vectors.
    If norm is specified, then have the angle be in the right hand rotated sense around said normal.
    The angle is rotation from v1 to v2 (if v2 is a matrix and v1 a vector, they will be swapped).
    """
    if type(norm) == list: norm = np.array(norm)
    if type(v1) == list: v1 = np.array(v1)
    if type(v2) == list: v2 = np.array(v2)
    if len(v2.shape) > len(v1.shape): v1, v2 = v2, v1
    # Use dot product definition to get angle between vectors
    ang_in_rad = np.arccos( np.dot(v1, v2) / ( np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2) + 1e-20 ) )
    # Use the cross product dotted with the normal vector to deduce the sign of the rotation, if necessary
    if type(norm) == np.ndarray:
        sign = np.sign( np.dot( np.cross(v1, v2), norm) )
        if type(sign) == np.ndarray: sign[sign==0] = 1
        elif sign == 0: sign = 1
        ang_in_rad = ang_in_rad*sign
    return ang_in_rad * 180 / np.pi

from sys import exit

# Polarisation, Lorentz and Irradiated Volume Corrections ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The Vesta Intensity values are already somewhat corrected
# These below can be applied to the raw structure factors

rd = np.pi/180
A_pol = np.cos(45 * rd)**2
def pol_cor(tth):  # Polarisation correction for X-ray intensity
    return ( 1 + A_pol*np.cos( tth * rd )**2 ) / (1 + A_pol)
def lorentz_cor(tth):  # Lorentz Correction for measured intensities
    return np.sin( tth * rd )

def vol_cor(w, a0=10e-3, beam_r=25e-6):
    samp_area = a0*5e-3   # 5e-3 is the width of the rectangular xray beam after the 2mm slit
    illuminated = 2*beam_r / ( np.sin( w * rd ) + 1e-30)   # Exposed length is beam diam over np.sin(w)
    if illuminated <= a0:  A = illuminated * 5e-3
    else:                  A = samp_area * (a0 / illuminated)
    return A/samp_area

def total_I_cor(tth, omega):
    ''' Enter angles in degrees '''
    return vol_cor(omega, a0=10e-3, beam_r=25e-6) * lorentz_cor(tth) / pol_cor(tth)

def qs_from_angles(w, tt):
    """ Given omega and tt (in deg or rad), returns qx and qz in nm^-1, assuming CuKa radiation"""
    if w.max()>np.pi or tt.max()>np.pi: w, tt = w*np.pi/180, tt*np.pi/180
    qx = 1 / 0.154059 * ( np.cos(w) - np.cos(tt - w) )
    qz = 1 / 0.154059 * ( np.sin(w) + np.sin(tt - w) )
    return qx, qz


def get_pk_names(pks):
    pk_names = np.empty(len(pks), dtype='<U12')
    for i in range(pk_names.shape[0]):
        pk_names[i] = '$(' + ''.join(pks.astype(str)[i]).replace('-','') + ')$'
    return pk_names


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_vesta_table(path_to_file, min_intensity=1e-4, max_omega=60, surface_normal=[0,0,1],
                     wavelength=1.54059e-10, return_lat_params=False, I_correct=False):
    """
    Parameters
    ----------
    path_to_file : Path to VESTA reflection table .txt file : string.
    min_intensity : Minimum intensity of peaks to include (normalised to 1) : float.
    max_omega : Maximum tube angle of peaks to include : float.
    surface_normal : Surface normal to calculate omega and phi values : list/array.
    return_lat_params : Return the lattice parameters as well as reflection angles : bool.
    I_correct : Correct the intensities for Irradiated Volume/Lorentz/Polarisation : bool.

    Returns
    -------
    Matrix with columns [2*theta, Omega, Phi, qx, qz, Intensity, [hkl], Peak_names].
    Optionally tuple with list of lattice parameters as second index.
    """
    data = np.genfromtxt(path_to_file, skip_header=1)
    #data = np.genfromtxt(file, skip_header=1)

    # Read in the structure factor squared instead of intensity
    tth, F2, pks = data[:,7], data[:,6]**2, data[:,:3]

    # Get the lattice parameters from the cif file
    with open(path_to_file.replace('txt','cif')) as cif_file:
        lines = np.array(cif_file.readlines())
        a = float( [line for line in lines if '_cell_length_a' in line][0].split()[1] )
        b = float( [line for line in lines if '_cell_length_b' in line][0].split()[1] )
        c = float( [line for line in lines if '_cell_length_c' in line][0].split()[1] )

    # Calculate tau for given surface normal and get omega values
    tau = get_ang(pks, surface_normal)
    om = tth/2 - tau  # Consider tube angles greater than 0 and less than max_omega
    inds_om = (om > 0) & (om < max_omega) if max_omega > 0 else (om > max_omega)
    tth, F2, pks, om = tth[inds_om], F2[inds_om], pks[inds_om].astype(int), om[inds_om]

    # Calculate the phi angles
    SN = surface_normal if type(surface_normal) == np.ndarray else np.array(surface_normal)
    IP_projection = pks - (np.dot(pks, SN) / np.dot(SN, SN))[:,None]*SN
    azi = [0,0,1] if (SN == np.array([1,1,0])).all() else [1,0,0]
    phis = np.around( get_ang(IP_projection, azi, SN), 4)

    # Apply the intensity correction if desired
    I = 1*F2
    if I_correct:
        for i in range(len(I)): I[i] = F2[i] * total_I_cor(tth[i], om[i])

    I = I / I.max()
    inds_I = I > min_intensity
    tth, I, pks, om, phis = tth[inds_I], I[inds_I], pks[inds_I], om[inds_I], phis[inds_I]

    # Get the qx and qz values for each peak
    qx, qz = qs_from_angles(om, tth)

    # Get strings of peak names for plot labelling
    pk_names = get_pk_names(pks)

    # Stack the relevant values in a matrix and return it
    M = [ tth, om, phis, qx, qz, I, pks, pk_names ]
    if return_lat_params: return M, [a, b, c]
    else: return M



