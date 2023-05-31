# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 20:51:01 2021

@author: eljac
"""
# =============================================================================
# Script to calculate and plot a diagram of various diffraction angles given:
#     - lattice parameters, miller indices of reflections and wavelength of probe
# Many common substrates and materials are listed
#
# Is not intended for hexagonal materials but some reflections can be calcuted
#
# The diagram shows a diffractometer in the reflection, grazing-incidence geometry.
# The script can always be used to calculate the bragg angle,
# for neutrons etc. the source and detector angle are less meaningful
# =============================================================================

import numpy as np
from numpy.linalg import norm

rd = np.pi/180

# Substrate lattice parameters (Angstrom) (Cubic/Tetragonal only)
# some common ones (uncomment one you want):
# Cubic
#sub, a, c, hexa =  'STO', 3.9051, 3.9051, 0
#sub, a, c, hexa =  'Si',  5.4307, 5.4307, 0
#sub, a, c, hexa =  'Ge',  5.658, 5.658, 0
#sub, a, c, hexa = 'GaAs', 5.7502, 5.7502, 0
#sub, a, c, hexa = 'Mn4NGa', 3.836, 3.903, 0
sub, a, c, hexa =  'MgO', 4.2112, 4.2112, 0
#sub, a, c, hexa =  'MRG', 5.954, 6.060, 0
#sub, a, c, hexa =  'MRG_tetra', 4.210, 6.060, 0
#sub, a, c, hexa =  'BTO', 3.9940, 4.0810, 0
#sub, a, c, hexa =  'CTS', 11.230, 11.230, 0
#sub, a, c, hexa =  'MVFA', 4.209, 6.087, 0
#sub, a, c, hexa = 'd-Mn', 3.0840, 3.0840, 0
#sub, a, c, hexa = 'V2FeAl', 5.960, 5.830, 0
#sub, a, c, hexa = 'Co2TiSi', 5.720, 5.720, 0
#sub, a, c, hexa = 'CrAl', 2.918, 2.918, 0

# Hexagonal
#sub, a, c, hexa = 'Al2O3', 4.805, 13.116, 1
#sub, a, c, hexa = 'MgCo2', 4.834, 7.855, 1

# Lattice vectors
av, cv = np.array([1, 0, 0]), np.array([0, 0, 1])

bv = np.array([-0.5, np.sqrt(3)/2, 0]) if hexa else np.array([0, 1, 0])

# Diffraction plane
H, K, L = 1,1,3

# X-Ray wavelength (also Angstrom) (uncomment one you want)
anode, w = 'Cu_{Ka}', 1.54059
#anode, w = 'Cu_{Kb}', 1.39223
#anode, w = 'Mo', 0.71070
#anode, w = 'W_{Lb}', 1.05
#anode, w = 'neutrons', 2.36

# Find d (plane spacing) and sides of triangle made by the plane/vertical/horizontal
def get_d(h, k, l):
    if not hexa: return 1/np.sqrt( (H**2 + K**2)/a**2 + (L**2)/c**2 )
    else: return 1 / np.sqrt( (4/3)*(h**2 + h*k + k**2)/(a**2) + l**2/c**2 )

# Get inclination of hkl to the horizontal plane
def get_tau(h, k, l):
    return np.arctan2( c*norm( h*av + k*bv ), a*l )

# Get Bragg diffraction angle
def get_tt(h, k, l):
    return 2 * np.arcsin( 0.5*w / get_d(h, k, l) )

# Find 2*theta, tau (plane tilt), Tube angle and Detector angle in grazing incidence
tt = get_tt(H,K,L)
tau = get_tau(H, K, L)
tub = tt/2 - tau
det = tt/2 + tau

textstr = '\n'.join((
    f'\N{Greek Small Letter Omega}\t=\t{tub/rd:.3f}\N{Degree sign}',
    f'2\N{Greek Small Letter theta}\t=\t{tt/rd:.3f}\N{Degree sign}',
    f'\N{Greek Small Letter tau}\t=\t{tau/rd:.3f}\N{Degree sign}',
    f'Det.=\t{det/rd:.3f}\N{Degree sign}'))
print(textstr)


#%% Make a diagram of a diffractometer in reflection geometry
import matplotlib.pyplot as plt
for key,value in zip(['font.family','font.weight','font.size'],['serif','normal',12.0]):
    plt.rcParams[key] = value

r = 10
x = np.linspace(-r, r, 500)
def circ(x, r): return np.sqrt(r**2 - x**2)
def arc(t, r): return (r*np.cos(t), r*np.sin(t))

fig, (ax, ay) = plt.subplots(1,2, figsize=[7,3], gridspec_kw={'width_ratios': [10,2.5]},
                             constrained_layout=True)

ax.plot(x, circ(x, r), lw=1, ls='--', c='k')                      # semi-circle
ax.plot([-1, 1], [-0.16, -0.16], lw=10, ls='-', c='g', zorder=5)  # sample
# Diffracting plane and scattering vector
ax.plot([-1.2*np.cos(tau), 1.2*np.cos(tau)], [-1.2*np.sin(tau), 1.2*np.sin(tau)], lw=2, ls='-', c='limegreen', zorder=5)
Qx,Qy = 1.5*np.cos(np.pi/2 + tau), 1.5*np.sin(np.pi/2 + tau)
ax.arrow(0, 0, Qx,Qy, lw=1.5, head_width=0.45, head_length=0.5, fc='limegreen', ec='limegreen')
ax.annotate('Q', (Qx,Qy), textcoords=('offset points'), xytext=(-20, 20))
# Grid lines through origin
ax.axhline(0, lw=1, ls='--', c='silver'), ax.axvline(0, lw=1, ls='--', c='silver')

# Plot Tube and Detector dots, labels, lines
s, nm = 1/rd, ['Tube', 'Detector']
x0, y0 = [-r*np.cos(tub), r*np.cos(det)], [r*np.sin(tub), r*np.sin(det)]
ax.scatter(x0, y0, c='r', s=15**2, zorder=10)
ax.annotate(nm[0], (x0[0], y0[0]), textcoords=('offset points'), xytext=(-40, 10))
ax.annotate(nm[1], (x0[1], y0[1]), textcoords=('offset points'), xytext=(6, 10))
ax.plot([x0[0], 0    ], [y0[0], 0.0  ], lw=1.5, c='r')
ax.plot([0,    -x0[0]], [0.0,  -y0[0]], lw=1.5, ls='--', c='r')
ax.plot([0,     x0[1]], [0.0,   y0[1]], lw=1.5, c='r')

# Plot the arcs for the angles
det_arc = arc(np.linspace(det,      0,    200), 3.5)
tt_arc  = arc(np.linspace(det,     -tub,  200), 6)
tub_arc = arc(np.linspace(np.pi - tub, np.pi,   100), 5)
tau_arc = arc(np.linspace(0,        tau,  100), 0.8)
ax.plot(tt_arc[0],  tt_arc[1],  'b', zorder=-5)
ax.plot(det_arc[0], det_arc[1], 'b', zorder=-5)
ax.plot(tub_arc[0], tub_arc[1], 'b', zorder=-5)
ax.plot(tau_arc[0], tau_arc[1], 'b', zorder=-5)

# Make and position the labels for the angles,
s2, nm2 = 1/rd, [r'$\omega$', r'$\tau$', r'Det', r'$2\theta$']
x1 = [tub_arc[0].mean(), tau_arc[0].mean(), det_arc[0].mean(), tt_arc[0].mean()]
y1 = [tub_arc[1].mean(), tau_arc[1].mean(), det_arc[1].mean(), tt_arc[1].mean()]
ax.scatter(x1, y1, c='r', s=0**2, zorder=10)
for nm,x,y, xyt in zip(nm2, x1, y1, [[-20,-3],[6,0],[3,3],[5,6]]):
    ax.annotate(nm, (x,y), textcoords=('offset points'), xytext=xyt)

# Get rid of axes numbers; grid, set limits so circle looks circular (depends on figsize)
ax.set_xlim([-12, 12])
ax.set_ylim([-2, 11.5])
ax.axis('off')

# Write info in the second panel (Latex math mode formatting)
textstr1 = '\n'.join((
    f'Substrate:    {sub}',
    f'$ \lambda \quad = \quad {w:.3f} \ \mathrm{{\AA}} \ \ ({anode})$',
    f'$ a \quad = \quad {a:.3f} \ \mathrm{{\AA}} $',
    f'$ c \quad = \quad {c:.3f} \ \mathrm{{\AA}} $',
    f'$ HKL \quad = \quad ({H}{K}{L}) $\n',
    f'$ \omega \qquad \ = \quad{tub/rd:.2f}^\circ $',
    f'$ 2\\theta \qquad = \quad{tt/rd:.2f}^\circ $',
    f'$ \\tau \qquad \ \ = \quad{tau/rd:.2f}^\circ $',
    f'$ Det \quad \ \: = \quad{det/rd:.2f}^\circ $'))

if hexa: textstr1 = '\n\n'.join((r'$ \quad --Hexagonal-- \quad $', textstr1))
props = dict(boxstyle='square', facecolor='red', alpha=0.1)
ay.text(0.0, 0.5, textstr1, fontsize=12, ha='left', va='center', bbox=props)
ay.axis('off')

plt.suptitle('Diffraction Geometry', fontsize=14)

save_name = sub + f'_{H}{K}{L}_plane_Bruker.pdf'

plt.show()

