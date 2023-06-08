# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 21:47:44 2021

@author: eljac
"""
# =============================================================================
# This script provides a Material Class which, when given the path to a "conventional_standard" .cif file
# with P1/#1 space group symmetry (can be obtained from MaterialsProject.org), calculates the diffraction
# angles, structure factors and corrected intensities for reflections with a wide range of miller indices.
#
# Only unit cells with all angles = 90 degrees are likely to work correctly. Symmetry operations are
# not explicitly considered so only .cif files with P1 space group will result in a correct unit cell.
#
# The atomic form factors therein are calculated for each element using the standard multiple Gaussian
# function interpolation from the International Tables of Crystallography.
# The structure factor calculation uses the form and phase factors only. Occupancy and Debye-Waller
# factors are not used currently, but are denoted in that script and would be trivial to add.
#
# The Material class can plot the unit cell with given colours using matplotlib. The intensity
# calculation optionally includes an irradiated volume correction, relevant for single-crystal
# diffractometry with a collimated beam.
#
# A comparison of the data is made with similar data from the popular software VESTA to test functionality.
# The results are promising, bu this function tends to overestimate the intensity of weak peaks,
# especially those at small angles, likely due to the lack of explicit symmetry consideration.
#
# =============================================================================

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
plt.rc('font',  family='serif')
run_tests = 1
try: from mpl_stem_plot import plt_stem
except: run_tests = 0

import os
from manual_F2_calculator import get_F2

try: from print_clr import print_clr as prnt
except: raise Exception("print_clr module/function not found, get the module or remove the 'prnt()' call")

#%% Some initial functions and variables
hkls_list = []
for i in range(13):
    for j in range(13):
        for k in range(13):
            hkls_list.append([i,j,k])
hkls_arr = np.asarray(hkls_list)[1:]

# convert between radians and degrees
rad = np.pi/180

# Get inter-planar spacing
def get_d(a,b,c, hkls):
    ar = np.asarray(hkls)
    abc = np.array([a,b,c])
    return 1 / norm(ar/abc, axis=1)
# Get Bragg diffraction angle
def get_tt(a,b,c, hkls, wvln=1.54059):
    return 2 * np.arcsin( 0.5*wvln / get_d(a,b,c, hkls) )
def get_tt_from_d(d, wvln=1.54059):
    return 2 * np.arcsin( 0.5*wvln / d )
# Get inclination of hkl to the horizontal plane
def get_tau(a,b,c, hkls):
    hkl_ar = np.asarray(hkls)
    abc = np.array([a,b,c])
    return np.arctan2( c*norm( hkl_ar[:,:2]/abc[:2], axis=1 ), hkl_ar[:,-1] )

# Function to implement lorentz-polarisation correction (full FullProf)
def LP_corr(tt0, tt_mono=45*rad, K=0.26553):
    """
    K = {char_xray:0.5, synch_xray:~0.1, neutrons:0}
    Defualt values found from fitting powder reference sample measurement, with:
        CuKa radiation, Goebel mirror, Ge220 double-bounce mono
    """
    tt = np.asarray(tt0)*rad
    return (1 - K + (K * np.cos(tt_mono)**2 * np.cos(tt)**2) ) / np.sin(tt)


# Function to plot a unit cell
def plot_UC(lat_size, frac_pos, elements, colours0=None):
    try:
        check_colours_iterable = iter(colours0)
        clr_cycle = colours0
    except:
        clr_cycle = 2*['limegreen','purple','silver','goldenrod','r','b','pink']
    clr_dict = {}
    for ele,clr in zip(np.unique(elements),clr_cycle): clr_dict[ele] = clr
    colours = [clr_dict[ele] for ele in elements]
    print(clr_dict)

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

    plt.show()



#%% Define Material class: reads a .cif file and calculates d, angles & F2 for different [h,k,l]

class Material:
    def __init__(self, path_to_cif):

        self.F2_cutoff = 1

        if not os.path.exists(path_to_cif):
            raise Exception("Cannot find file at specified path")

        with open(path_to_cif, "r") as file:
            text = file.read()
            # Check if space group is P1 (so all atoms are listed)
            posb1, posb2 = "_symmetry_Int_Tables_number", "_space_group_IT_number"
            if posb1 in text: space_group = int(text.split(posb1)[-1].split("\n")[0])
            elif posb2 in text: space_group = int(text.split(posb2)[-1].split("\n")[0])
            else: raise Exception("Could not confirm space group from cif file")
            if space_group != 1: raise Exception("Please use a 'conventional standard' .cif with P1 (no) symmetry\nCan be obtained from e.g. https://materialsproject.org/materials")

            # Get the lattice parameters from the cif file
            self.name = text.split("data_")[-1].split("\n")[0]
            self.a = float(text.split("_cell_length_a")[-1].split("\n")[0])
            self.b = float(text.split("_cell_length_b")[-1].split("\n")[0])
            self.c = float(text.split("_cell_length_c")[-1].split("\n")[0])
            self.alpha = float(text.split("_cell_angle_alpha")[-1].split("\n")[0])
            self.beta = float(text.split("_cell_angle_beta")[-1].split("\n")[0])
            self.gamma = float(text.split("_cell_angle_gamma")[-1].split("\n")[0])

            # Get the atom list
            # Should be in format: [element, label, Mult, x, y, x, occupancy] -> test.shape[1]==7
            atoms = text.replace(',',' ').split("_atom_")[-1].split("\n")[1:]
            atoms_ar = np.array([atom.split() for atom in atoms if len(atom)>10])  # ignores empty lines

            # Even with P1 symmetry, cif files often only specify say (0,0,0) atom and not (1,0,0)
            #     (The "1 x,y,z" symmetry operation is always active)
            # Add additional atoms at all permutations of +1x,+1y,+1z
            pos = atoms_ar[:,3:-1].astype(float)
            pos_stack = 1*pos
            for shift in [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]:
                pos_stack = np.vstack((pos_stack, pos+np.array(shift)))
            # Ignore atoms outside cell and repeat atoms (shouldnt be any repeats)
            inside_cell_mask = (pos_stack<= 1).all(axis=1)
            self.positions = pos_stack[inside_cell_mask]
            # We have indices, now repeat the array similarly to get the MATCHING elements/occupancy
            atoms_ar = np.tile(atoms_ar.T, 8).T[inside_cell_mask]

            # Check whether labels or elements come first -> compare characters after string
            label_first = ( len(text.split('_site_label')[-1]) > len(text.split('_site_type_symbol')[-1]) )

            self.elements = atoms_ar[:,1] if label_first else atoms_ar[:,0]
            if sum(['_' in ele for ele in self.elements]):
                raise Exception("Underscore in element list: .cif file is likely wrong")
            self.hkls = hkls_arr
            self.calc_diffrac_vars()

    def plot_unit_cell(self, clrs=None):
        plot_UC(np.array([self.a,self.b,self.c]), self.positions, self.elements, clrs)

    def calc_diffrac_vars(self, wvln=1.54059, particle='photon', volume_corr=False):
        a,b,c = self.a, self.b, self.c
        # Calculate d, ignore too-small spacing
        d = get_d(a,b,c, self.hkls)
        d, self.hkls = d[wvln < 2*d], self.hkls[wvln < 2*d]
        # Calculate 2*theta, tau, omega and the structure factors
        tt = get_tt_from_d(d, wvln)
        tau = get_tau(a,b,c, self.hkls)
        omega = tt/2 - tau
        F2s = get_F2(a,b,c, *self.hkls.T, self.elements, self.positions, particle=particle)
        F2s = 100*F2s/F2s.max()
        # Ignore forbidden reflections
        inds2 = (F2s > self.F2_cutoff)
        self.hkls = self.hkls[inds2]
        self.d_spacings = d[inds2]
        self.tts = tt[inds2]
        self.taus = tau[inds2]
        self.omegas = omega[inds2]
        self.F2s = F2s[inds2]
        # Apply corrections to get intensity from structure factors
        self.Is = F2s[inds2] * LP_corr(self.tts)
        if volume_corr:
            self.Is = self.Is / np.sin(self.omegas)
            prnt("Only reflections with positive omega are accesible in geometries where volume correction is relevant.", fore_clr=(150,200,250))
        self.Is = np.around( 100*self.Is/self.Is.max(), 8 )
        # Names of peaks (Latex format) for labelling in matplotlib
        self.peak_names = np.array(['$('+''.join(np.array(pk).astype(str))+')$' for pk in self.hkls])



#%% Test functionality ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if "__main__" in __name__ and run_tests:
    samp_path = 'vesta_cif_reader_compare/Mn2RuGa_F-43m_conventional.cif'
    sub_path = 'vesta_cif_reader_compare/MgO_Fm-3m_conventional_standard.cif'
    sample, substrate = 'MRG', 'MgO'

    samp = Material(samp_path)
    samp.calc_diffrac_vars(volume_corr=True)  # Include the irradiated volume correction in intenisty calc
    samp.plot_unit_cell(clrs=['g','r','grey'])
    sub = Material(sub_path)
    sub.plot_unit_cell(clrs=['goldenrod','pink'])

    for key,val in zip(["major.pad","labelsize"],[2,8]):
        for ax in ['x','y']:
            plt.rcParams[f"{ax}tick.{key}"] = val


    fig, ax = plt.subplots(1,1, figsize=[5,3], constrained_layout=True)

    i = np.unique(samp.tts, return_index=True)[1]
    for tt0,F2,I,om,name in zip(samp.tts[i], samp.F2s[i], samp.Is[i], samp.omegas[i], samp.peak_names[i]):
        if om < 0: continue
        if tt0 > 140*rad: continue
        plt_stem(tt0/rad, F2, c='b', alpha=0.666)
        plt_stem(tt0/rad, I, c='cyan', alpha=0.333)
        ax.text(tt0/rad, F2 + 3, name, ha='center', fontsize=10, c='navy')

    i = np.unique(sub.tts, return_index=True)[1]
    bx = ax.twinx()
    for tt0,F2,om,name in zip(sub.tts[i], sub.F2s[i], sub.omegas[i], sub.peak_names[i]):
        if om < 0: continue
        if tt0 > 140*rad: continue
        plt_stem(tt0/rad, F2, c='r', ls='--', alpha=0.666)
        bx.text(tt0/rad, F2 + 3, name, ha='center', fontsize=10, c='maroon')
    bx.set_ylim(-5, bx.get_ylim()[1]*1.07), bx.tick_params(labelright=False)

    ax.scatter(None,None, c='b', label=sample+'$\:(F^2)$'), ax.scatter(None,None, c='r', label=substrate+'$\:(F^2)$')
    ax.scatter(None,None, c='cyan', label=sample+'$\:(I)$')
    ax.legend(handletextpad=0.0, ncol=2, columnspacing=0.3, borderpad=0.1, loc='upper right')

    ax.grid(alpha=0.5), ax.axhline(0, c='k', lw=0.75)
    ax.set_xlim(5, 140)
    ax.set_ylim(-5, ax.get_ylim()[1]*1.07)

    ax.set_xlabel("$2\\theta$ (deg)"), ax.set_ylabel("$F^2$ (arb)")
    plt.show()


#~~~~~ Compare to VESTA F2 and I calculations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    root = 'vesta_cif_reader_compare/'

    # Plot tt vs F2 and I for two tests, from my function and VESTA
    fig, [[ax00,ax01],[ax10,ax11]] = plt.subplots(2,2, figsize=[8,5], constrained_layout=True, sharex=True, sharey=True)

    path_to_cif1 = root + 'GaAs_mp-2534_conventional_standard.cif'
    path_to_cif2 = root + 'Mn2RuGa_F-43m_conventional.cif'

    for path, [ax,bx] in zip([path_to_cif1, path_to_cif2], [[ax00,ax10],[ax01,ax11]]):
        # Calculate parameters using my class
        samp = Material(path)
        tt, om, F2, I, peaks = samp.tts, samp.omegas, samp.F2s, samp.Is, samp.peak_names
        # Read the vesta version
        data_ves = np.genfromtxt(path.replace('.cif','.txt'), skip_header=1)
        h, k, l, d, Fr, Fi, F_ves, tt_ves, I_ves, _, _, _ = data_ves.T
        F2_ves, om_ves = F_ves**2, tt_ves/2 - get_tau(samp.a, samp.b, samp.c, np.stack((h,k,l)).T)
        inds = (om_ves>0)
        F2_ves, om_ves, tt_ves, I_ves = F2_ves[inds], om_ves[inds], tt_ves[inds], I_ves[inds]
        inds = np.unique( np.stack((F2_ves, tt_ves)), axis=1, return_index=True)[1]
        F2_ves, tt_ves, I_ves = F2_ves[inds], tt_ves[inds], I_ves[inds]

        # Make the plots
        tts_unique = []
        for x,y,s in zip(tt/rad, 100*F2/F2.max(), peaks):
            if (y < 5) or (x in tts_unique): continue
            ax.text(x, y, s), tts_unique.append(x)

        plt_stem(tt/rad, 100*F2/F2.max(), axis=ax, c='r')
        plt_stem(tt/rad, 100*I/I.max(), axis=ax, alpha=0.5, ls='--')
        plt_stem(tt_ves, 100*F2_ves/F2_ves.max(), axis=bx, c='r')
        plt_stem(tt_ves, 100*I_ves/I_ves.max(), axis=bx, alpha=0.5, ls='--')


    for ax in [ax00,ax01,ax10,ax11]:
        ax.grid(alpha=0.5)
        ax.scatter(None,None, c='r', label='F$^2$'), ax.scatter(None,None, c='b', label='I')
        ax.legend(handletextpad=0.2, borderpad=0.25, fontsize=12)

    ax.set_xlim(10, 170), ax.set_ylim(-5, 110)
    ax00.set_title("GaAs"), ax01.set_title("Mn2RuGa")
    ax00.set_ylabel("My Function", fontsize=12), ax10.set_ylabel("VESTA", fontsize=12)
    fig.supxlabel("$2\\theta$ (deg)")

    plt.show()


#~~~~~ Compare using tt vs omega plot for GaAs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    path_to_cif1 = root + 'GaAs_mp-2534_conventional_standard.cif'
    samp = Material(path_to_cif1)
    #samp.plot_unit_cell(clrs=['orange','limegreen'])

    fig, ax = plt.subplots(1,1, figsize=[6,5], constrained_layout=True)

    # Calculate parameters using my class
    tt, om, F2, I, peaks = samp.tts, samp.omegas, samp.F2s, samp.Is, samp.peak_names
    tt, om, F2, I, peaks = tt[om>0], om[om>0], F2[om>0], I[om>0], peaks[om>0]
    # Read the vesta version
    data_ves = np.genfromtxt(path_to_cif1.replace('.cif','.txt'), skip_header=1)
    h, k, l, d, Fr, Fi, F_ves, tt_ves, I_ves, _, _, _ = data_ves.T
    pks = np.array( ['$('+''.join(hkl)+')$' for hkl in np.squeeze((h,k,l)).T.astype(int).astype(str)] )
    F2_ves, om_ves = F_ves**2, tt_ves/2 - get_tau(samp.a, samp.b, samp.c, np.stack((h,k,l)).T)/rad
    inds = (om_ves > 0) & (h >= 0) & (k >= 0) & (l >= 0)
    F2_ves, om_ves, tt_ves, I_ves, pks = F2_ves[inds], om_ves[inds], tt_ves[inds], I_ves[inds], pks[inds]

    # Make the plots
    plt.scatter(tt/rad, om/rad, s=100*I/I.max(), c='limegreen')
    plt.scatter(tt_ves, om_ves, s=100*I_ves/I_ves.max(), c='b')
    plt.plot(np.arange(200), np.arange(200)/2, ls='--', c='k', zorder=-8, lw=1.0)


    # Label peaks (from My Function)
    tts_unique, oms_unique = [], []
    for x,y,z,s in zip(tt/rad, om/rad, 100*I/I.max(), peaks):
        if (z > 2.5) and ((x not in tts_unique) and (y not in oms_unique)):
            ax.text(x, y, s, c='darkgreen')
        tts_unique.append(x), oms_unique.append(y)

    # Label peaks (from VESTA)
    tts_unique, oms_unique = [], []
    for x,y,z,s in zip(tt_ves, om_ves, 100*I_ves/I_ves.max(), pks):
        if (z > 2.5) and ((x not in tts_unique) and (y not in oms_unique)):
            ax.text(x, y, s, c='navy')
        tts_unique.append(x), oms_unique.append(y)


    ax.grid(alpha=0.5)
    ax.scatter(None,None, s=0, label='GaAs Reflections:')
    ax.scatter(None,None, c='limegreen', label='My Function')
    ax.scatter(None,None, c='b', label='VESTA')
    ax.legend(handletextpad=0.2, borderpad=0.25, fontsize=12)

    ax.set_xlim(0, 170), ax.set_ylim(0, 70)
    ax.set_xlabel("$2\\theta$ (deg)")
    ax.set_ylabel("$\\omega$ (deg)", fontsize=12)

    plt.show()


