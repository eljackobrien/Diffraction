# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:28:55 2020

@author: eljac

Various definitions of peak profile shapes, relevant to fitting xray diffraction data.
1D and 2D Gaussian, Lorentzian (Cauchy) and Voigtian profiles are defined.

The generalised 2D profiles which involve convolution have their horitzontal and vertical
    broadening parameters convoluted, so one should not draw quantitative conslusions
    for the separate variables from these parameters. Rather, vertical and horozontal
    slices through the 2D data should be used.

"""
import numpy as np
from scipy.special import voigt_profile
import scipy.signal as signal
import matplotlib.pyplot as plt

run_timing_tests = 0


#%% Classes for peak profiles, includes
class Gaussian_class:
    def  __init__(self, x, x0, sig, hieght=1):
        self.x = x
        self.x0 = x0
        self.sig = sig
        self.hieght = hieght
        self.re_calc()

    def re_calc(self):
        self.y = self.hieght * np.exp(-(self.x - self.x0)**2 / (2*self.sig**2))
        self.FWHM = 2.35482 * self.sig
        self.area = 2.50662827 * self.hieght * self.sig
        self.breadth = 2.50662827 * self.sig  # integral breadth

    def set(self, x=None, x0=None, sig=None, hieght=None):
        if type(x) != type(None): self.x = x
        if type(x0) != type(None): self.x0 = x0
        if type(sig) != type(None): self.sig = sig
        if type(hieght) != type(None): self.hieght = hieght
        self.re_calc()

    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=[6,3], constrained_layout=True)
        lab = f"Gaussian:\nx0 = {self.x0:.3g}\nFWHM = {self.FWHM:.3g}\nheight = {self.hieght:.3g}"
        ax.plot(self.x, self.y, lw=1.0, c='r', label=lab)
        ax.grid(), ax.legend()
        plt.show()


class Lorentzian_class:
    def  __init__(self, x, x0, gam, hieght=1):
        self.x = x
        self.x0 = x0
        self.gam = gam
        self.hieght = hieght
        self.re_calc()

    def re_calc(self):
        self.y = self.hieght * self.gam**2 / ( (self.x - self.x0)**2 + self.gam**2 )
        self.FWHM = 2 * self.gam
        self.area = np.pi * self.hieght * self.gam
        self.breadth = np.pi * self.gam

    def set(self, x=None, x0=None, gam=None, hieght=None):
        if type(x) != type(None): self.x = x
        if type(x0) != type(None): self.x0 = x0
        if type(gam) != type(None): self.gam = gam
        if type(hieght) != type(None): self.hieght = hieght
        self.re_calc()

    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=[6,3], constrained_layout=True)
        lab = f"Lorentzian:\nx0 = {self.x0:.3g}\nFWHM = {self.FWHM:.3g}\nheight = {self.hieght:.3g}"
        ax.plot(self.x, self.y, lw=1.0, c='r', label=lab)
        ax.grid(), ax.legend()
        plt.show()


class Voigt_class:
    def  __init__(self, x, x0, sig, gam, hieght=1):
        self.x = x
        self.x0 = x0
        self.sig = sig
        self.gam = gam
        self.hieght = hieght
        self.re_calc()

    def re_calc(self):
        g = np.exp(-(self.x - self.x.mean())**2 / (2*self.sig**2))  # gaussian, centred in x so convolution doesn't shift peak
        l = self.gam**2 / ( (self.x - self.x0)**2 + self.gam**2 )  # lorentzian
        self.y = np.convolve( g, l, mode='same' )  # do the convolution
        if self.y.max()!=0: self.y *= self.hieght/self.y.max()  # normalise the height

        self.FWHM = 1.0692*self.gam + (0.86639*self.gam**2 + 5.545177*self.sig**2)**0.5  # wikipedia

        k = self.gam/( (2**0.5) * self.sig )
        area, B, C, D, E = 0.9039645, 0.7699548, 1.364216, 1.136195, 0.9394372
        self.breadth = self.FWHM * (1 + C*k + D*k**2)/(E*(1 + area*k + B*k**2))  # birkholz 2006 book

        self.area = self.hieght * self.breadth

    def set(self, x=None, x0=None, sig=None, gam=None, hieght=None):
        if type(x) != type(None): self.x = x
        if type(x0) != type(None): self.x0 = x0
        if type(sig) != type(None): self.sig = sig
        if type(gam) != type(None): self.gam = gam
        if type(hieght) != type(None): self.hieght = hieght
        self.re_calc()

    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=[6,3], constrained_layout=True)
        lab = f"Lorentzian:\nx0 = {self.x0:.3g}\nFWHM = {self.FWHM:.3g}\nheight = {self.hieght:.3g}"
        ax.plot(self.x, self.y, lw=1.0, c='r', label=lab)
        ax.grid(), ax.legend()
        plt.show()


if __name__ in "__main__":
    def FWHM(x, y):
        temp = abs(y - y.max()/2)
        HM_left, HM_right = temp[:y.argmax()].argmin(), temp[y.argmax():].argmin()+y.argmax()
        return abs(x[HM_right] - x[HM_left])

    from scipy.integrate import simpson, trapezoid

    x = np.linspace(-90,90,11110)
    x0 = 0
    sig, gam, area = 2, 1, 1

    G = Gaussian_class(x, x0, sig, area)
    L = Lorentzian_class(x, x0, gam, area)
    V = Voigt_class(x, x0, sig, gam, area)

    print(f"{FWHM(G.x, G.y):.4f}, {G.FWHM:.4f}"), print(f"{trapezoid(G.y, G.x):.3f}, {simpson(G.y, G.x):.3f}, {G.area:.3f}")
    print(f"{FWHM(L.x, L.y):.4f}, {L.FWHM:.4f}"), print(f"{trapezoid(L.y, L.x):.3f}, {simpson(L.y, L.x):.3f}, {L.area:.3f}")
    print(f"{FWHM(V.x, V.y):.4f}, {V.FWHM:.4f}"), print(f"{trapezoid(V.y, V.x):.3f}, {simpson(V.y, V.x):.3f}, {V.area:.3f}")


#%% Standard normalised form Gaussian, Lorentzian, PV and Voigt Functions
def Gaussian(x, x0, sig, area=1):
    return area/(sig * np.sqrt(2*np.pi)) * np.exp( -(x-x0)**2 / (2*sig**2) )

def Lorentzian(x, x0, gam, area=1):
    return area/np.pi * gam / ( (x-x0)**2 + gam**2 )
def Lorentzian_centered(x, gam, area=1):
    return Lorentzian(x, x.mean(), gam, area=1)

def pseudo_Voigt(x, x0, sig, gam, area=1):  # Pseudo-Voigt from "Thompson, Cox & Hastings. (1987). J. areapp. Crys 20(2) 79-83"
    g, l = sig, gam
    F = ( g**5 + 2.69269*(g**4)*(l) + 2.42843*(g**3)*(l**2) + 4.47163*(g**2)*(l**3) + 0.07842*(g)*(l**4) + l**5 )**(1/5)
    n = 1.36603*(l/F) - 0.47719*((l/F)**2) + 0.11116*((l/F)**3)
    return area*( (1 - n)*Gaussian(x, x0, F) + n*Lorentzian(x, x0, F) )

def Voigt_scipy(x, x0, sig, gam, area=1):
    # Profile is normalised to area = 1, so can multiply by area
    return area * voigt_profile(x - x0, sig, gam)

# Do some test plots to make sure everything is working
if __name__ in "__main__":
    xt, yt, x0 = np.linspace(0, 20, 1000), np.linspace(0, 20, 1000), 9
    FWHM = 4.0
    fig, ax = plt.subplots(1,1, figsize=[10,7], tight_layout=True)
    [ax.axvline(q, ls='--', c='k', lw=1) for q in [x0+FWHM/2, x0-FWHM/2]]

    G = Gaussian(xt, x0, sig=FWHM/2.355, area=10)
    L = Lorentzian(xt, x0, gam=FWHM/2, area=10)
    PV = pseudo_Voigt(xt, x0, sig=0.5*FWHM/2.355, gam=0.5*FWHM/2, area=10)
    V = Voigt_scipy(xt, x0, sig=0.5*FWHM/2.355, gam=0.5*FWHM/2, area=10)

    for y,lab,clr in zip([G,L,PV,V],['Gaussian','Lorentzian','Pseudo-Voigt','Voigt'],['b','lime','goldenrod','r']):
        ax.plot(xt, y, lw=2.0, label=lab, c=clr), ax.axhline(y.max()/2, ls='--', lw=1, c=clr, alpha=0.5)

    ax.set_title("Comparison of the Peak Types - Similar Broadenings")
    ax.grid(alpha=0.5)
    ax.legend()
    plt.show()



#%% Timeit tests
if __name__ in "__main__" and run_timing_tests:
    from timeit import timeit
    num = 10000
    sG = """y=Gaussian(xt, x0, 1.53, 100)""" # 187.9ms -> #2
    Gaussian_time = timeit(stmt=sG, setup="from __main__ import Gaussian, xt, x0", number=num)

    sL = """y=Lorentzian(xt, x0, 1.8, 100)""" # 67.9ms -> #1
    Lorentz_time = timeit(stmt=sL, setup="from __main__ import Lorentzian, xt, x0", number=num)

    spV = """y=pseudo_Voigt(xt, x0, 1, 1, 100)""" # 336.9ms -> #3
    pseudo_voigt_time = timeit(stmt=spV, setup="from __main__ import pseudo_Voigt, xt, x0", number=num)

    sVs = """y=Voigt_scipy(xt, x0, 1, 1, 100)""" # 1505.8ms -> #4
    sp_voigt_time = timeit(stmt=sVs, setup="from __main__ import Voigt_scipy, xt, x0", number=num)

    names = ['Gaussian', 'Lorentzian', 'pseudo-Voigt', 'Scipy Voigt']
    times = [Gaussian_time, Lorentz_time, pseudo_voigt_time, sp_voigt_time]
    print(f"For {num} calculations of profiles with {len(xt)} points each:\n")
    for name,time in zip(names,times): print(f"{name} took: {1000*time:.1f}ms\n")


#%% 2-Dimensional Peak Functions
xt, yt, x0, y0 = np.linspace(0, 20, 1000), np.linspace(0, 16, 537), 12, 8
# 2D versions of NON-normalised Gaussian, Lorentzian and Pseudo-Voigt functions.
# pV is a bit wonky, see notes below

def Gaussian_2D(x, y, x0, y0, sigx, sigy, area=1):
    if np.ndim(x) == 1:
        return area * np.exp(-0.5*( ((x[None,:] - x0)/sigx)**2 + ((y[:,None] - y0)/sigy)**2 ))
    else:
        return area * np.exp(-0.5*( ((x - x0)/sigx)**2 + ((y - y0)/sigy)**2 ))

def Lorentzian_2D(x, y, x0, y0, gamx, gamy, area=1):
    if np.ndim(x) == 1:
        return area / ( 1 + ((x[None,:] - x0)/gamx)**2 + ((y[:,None] - y0)/gamy)**2 )
    else:
        return area / ( 1 + ((x - x0)/gamx)**2 + ((y - y0)/gamy)**2 )

def Lorentzian_2D_centered(x, y, gamx, gamy, area=1): # Centered Lorentzian for the convolution
    if np.ndim(x) == 1:
        return area / ( 1 + ((x[None,:] - x.mean())/gamx)**2 + ((y[:,None] - y.mean())/gamy)**2 )
    else:
        return area / ( 1 + ((x - x.mean())/gamx)**2 + ((y - y.mean())/gamy)**2 )

def pseudo_Voigt_2D(x, y, x0, y0, gx, gy, lx, ly, area=1):
    F = lambda g,l: ( g**5 + 2.69269*(g**4)*(l) + 2.42843*(g**3)*(l**2) + 4.47163*(g**2)*(l**3) + 0.07842*(g)*(l**4) + l**5 )**(1/5)
    n = lambda l,F: 1.36603*(l/F) - 0.47719*((l/F)**2) + 0.11116*((l/F)**3)
    Fx = F(gx, lx);  Fy = F(gy, ly);  nx = n(lx, Fx);  ny = n(ly, Fy)
    # I'm not sure there's any way to seperate the mixing ratio into x and y
    # The mathlab version used just 3 params, FWHMx, FWHMy and n. IP and OOP shared same n.
    n = 0.5*(nx + ny)  # This is incorrect and will correlate IP vs OOP
    # properties. i.e. there cannot be zero gaussian character IP and finite OOP
    return (1 - n)*Gaussian_2D(x, y, x0, y0, Fx, Fy, area) + n*Lorentzian_2D(x, y, x0, y0, Fx, Fy, area)

def Voigt_convolve_2D(x, y, x0, y0, gx, gy, lx, ly, area=1):
    Gaussian_peak = Gaussian_2D(x, y, x0, y0, gx, gy, area) # Store Gaussian peak to normalise intensity
    temp = signal.convolve(Gaussian_peak, Lorentzian_2D_centered(x, y, lx, ly, area), mode='same')
    return Gaussian_peak.max() * temp / temp.max()

# Do some test plots to make sure everything is working
if __name__ in "__main__":
    xt, yt, x0, y0 = np.linspace(0, 20, 1000), np.linspace(0, 16, 537), 12, 8
    FWHMx = 2.0; FWHMy = 3.5
    sigx=FWHMx/2.355; gamx=FWHMx/2; sigy=FWHMy/2.355; gamy=FWHMy/2
    fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=[16,5.5], constrained_layout=True)
    ax1.contourf(xt, yt, Gaussian_2D(xt, yt, x0, y0, sigx, sigy), cmap='plasma')
    ax2.contourf(xt, yt, pseudo_Voigt_2D(xt, yt, x0, y0, sigx, sigy, gamx, gamy), cmap='plasma')
    ax3.contourf(xt, yt, Voigt_convolve_2D(xt, yt, x0, y0, sigx, sigy, gamx, gamy), cmap='plasma')
    ax1.set_title("Gaussian"), ax2.set_title("pseudo-Voigt"), ax3.set_title("Voigt")
    ax.legend()
    plt.show()


#%% Timeit tests 2D
if __name__ in "__main__" and run_timing_tests:
    from timeit import timeit
    num = 25
    sG = """y=Gaussian_2D(xt, yt, x0, y0, 1.0, 1.0, 25)""" # 260.1ms -> #2
    Gaussian_time = timeit(stmt=sG, setup="from __main__ import Gaussian_2D, xt, yt, x0, y0", number=num)

    sL = """y=Lorentzian_2D(xt, yt, x0, y0, 1.0, 1.0, 25)""" # 94.5ms -> #1
    Lorentz_time = timeit(stmt=sL, setup="from __main__ import Lorentzian_2D, xt, yt, x0, y0", number=num)

    spV = """y=pseudo_Voigt_2D(xt, yt, x0, y0, 1.0, 1.0, 1.0, 1.0, 25)""" # 505.7ms -> #3
    pseudo_voigt_time = timeit(stmt=spV, setup="from __main__ import pseudo_Voigt_2D, xt, yt, x0, y0", number=num)

    sVs = """y=Voigt_convolve_2D(xt, yt, x0, y0, 1.0, 1.0, 1.0, 1.0, 25)""" # 5477.1ms -> #4
    voigt_convolve_time = timeit(stmt=sVs, setup="from __main__ import Voigt_convolve_2D, xt, yt, x0, y0", number=num)

    names = ['Gaussian', 'Lorentzian', 'pseudo-Voigt', 'Scipy FFT Convolve']
    times = [Gaussian_time, Lorentz_time, pseudo_voigt_time, voigt_convolve_time]
    print(f"For {num} calculations of peak profiles of size [{len(xt)} x {len(yt)}]:\n")
    for name,time in zip(names,times): print(f"{name} took: {1000*time:.1f}ms\n")


#%% Generalised 2D Peak Funtions
# Generalised 2D versions of non-normalised Gaussian, Lorentzian and Pseudo-Voigt functions.
# pV is a bit wonky, see notes below

def Gaussian_2D_general(x, y, th0, x0, y0, sigx, sigy, area):
    th = th0 * np.pi / 180
    a = 0.5 * ( np.cos(th)**2/sigx**2 + np.sin(th)**2/sigy**2 )
    b = 0.25 * ( -np.sin(2*th)/sigx**2 + np.sin(2*th)/sigy**2 )
    c = 0.5 * ( np.sin(th)**2/sigx**2 + np.cos(th)**2/sigy**2 )
    if np.ndim(x) == 1:
        return area * np.exp( -a*((x[None,:] - x0)**2) - 2*b*(x[None,:] - x0)*(y[:,None] - y0) - c*((y[:,None] - y0)**2))
    else:
        return area * np.exp( -a*((x - x0)**2) - 2*b*(x - x0)*(y - y0) - c*((y - y0)**2) )

def Lorentzian_2D_general(x, y, th0, x0, y0, gamx, gamy, area):
    th = th0 * np.pi / 180
    a = 0.5 * ( np.cos(th)**2/gamx**2 + np.sin(th)**2/gamy**2 )
    b = 0.25 * ( -np.sin(2*th)/gamx**2 + np.sin(2*th)/gamy**2 )
    c = 0.5 * ( np.sin(th)**2/gamx**2 + np.cos(th)**2/gamy**2 )
    if np.ndim(x) == 1:
        return area / (1 + a*((x[None,:] - x0)**2) + 2*b*(x[None,:] - x0)*(y[:,None] - y0) + c*((y[:,None] - y0)**2))
    else:
        return area / (1 + a*((x - x0)**2) + 2*b*(x - x0)*(y - y0) + c*((y - y0)**2))

def Lorentzian_2D_centered_general(x, y, th0, gamx, gamy, area):
    th = th0 * np.pi / 180
    a = 0.5 * ( np.cos(th)**2/gamx**2 + np.sin(th)**2/gamy**2 )
    b = 0.25 * ( -np.sin(2*th)/gamx**2 + np.sin(2*th)/gamy**2 )
    c = 0.5 * ( np.sin(th)**2/gamx**2 + np.cos(th)**2/gamy**2 )
    if np.ndim(x) == 1:
        return area / (1 + a*((x[None,:] - x.mean())**2) + 2*b*(x[None,:] - x.mean())*(y[:,None] - y.mean()) + c*((y[:,None] - y.mean())**2))
    else:
        return area / (1 + a*((x - x.mean())**2) + 2*b*(x - x.mean())*(y - y.mean()) + c*((y - y.mean())**2))


# This function calculates the peak at the relevant value and then shifts it to the center
#     rather than calculating directly at the center. Pretty sure this is unnecessary.
def Lorentzian_2D_shift_center_general(x, y, th0, x0, y0, gamx, gamy, area):
    # Number of indices to shift the peak in x and y so the peak is centered
    shift_x, shift_y = len(x)//2 - np.argmin(abs(x-x0)), len(y)//2 - np.argmin(abs(y-y0))
    sx, sy = abs(shift_x), abs(shift_y)
    # Step size to use for the extra x and y values
    x_step, y_step = np.diff(x).mean(), np.diff(y).mean()
    # Extra rows/columns to append to the profile so we can shift the peak by slicing
    x_extra, y_extra = np.arange(x_step, (sx + 1)*x_step, x_step), np.arange(y_step, (sy + 1)*x_step, y_step)
    # Depending on the sign of shift, add the extra rows/columns to the start or end
    if np.sign(shift_x)==1:
        if np.sign(shift_y)==1: x_new, y_new = np.hstack(( x[0]-x_extra, x )), np.hstack(( y[0]-y_extra, y ))
        else: x_new, y_new = np.hstack(( x[0]-x_extra, x )), np.hstack(( y[-1]+y_extra, y ))
    else:
        if np.sign(shift_y)==1: x_new, y_new = np.hstack(( x[-1]+x_extra, x )), np.hstack(( y[0]-y_extra, y ))
        else: x_new, y_new = np.hstack(( x[-1]+x_extra, x )), np.hstack(( y[-1]+y_extra, y ))
    x_new.sort(), y_new.sort()
    # Calculate temp for the whole expanded range
    temp = Lorentzian_2D_general(x_new, y_new, th0, x0, y0, gamx, gamy, area)
    # Cut to size, such that the shape of the peak is unchanged, but the max value is centered in the matrix
    if np.sign(shift_x)==1:
        if np.sign(shift_y)==1: x_new, y_new, temp = x_new[:-sx], y_new[:-sy], temp[:-sy, :-sx]
        else: x_new, y_new, temp = x_new[:-sx], y_new[sy:], temp[sy:, :-sx]
    else:
        if np.sign(shift_y)==1: x_new, y_new, temp = x_new[sx:], y_new[:-sy], temp[:-sy, sx:]
        else: x_new, y_new, temp = x_new[sx:], y_new[sy:], temp[sy:, sx:]
    return temp


def pseudo_Voigt_2D_general(x, y, th0, x0, y0, gx, gy, lx, ly, area):
    F = lambda g,l: ( g**5 + 2.69269*(g**4)*(l) + 2.42843*(g**3)*(l**2) + 4.47163*(g**2)*(l**3) + 0.07842*(g)*(l**4) + l**5 )**(1/5)
    n = lambda l,F: 1.36603*(l/F) - 0.47719*((l/F)**2) + 0.11116*((l/F)**3)
    Fx = F(gx, lx);  Fy = F(gy, ly);  nx = n(lx, Fx);  ny = n(ly, Fy)
    # I'm not sure there's any way to seperate the mixing ratio into x and y
    # The mathlab version used just 3 params, FWHMx, FWHMy and n. IP and OOP shared same n.
    n = 0.5*(nx + ny)  # This is incorrect and will correlate IP vs OOP
    # properties. i.e. there cannot be zero gaussian character IP and finite OOP
    return (1 - n)*Gaussian_2D_general(x, y, th0, x0, y0, Fx, Fy, area) + \
        n*Lorentzian_2D_general(x, y, th0, x0, y0, Fx, Fy, area)


def Voigt_2D_general(x, y, th0, x0, y0, gx, gy, lx, ly, area, x_st=0, x_en=0, y_st=0, y_en=0, plot=False):
    # Calculate the Voigt function from the Gaussian/Lorentzian over a much larger range than we are
    #     interested in, to avoid any edge effects from the convolution.
    x1, x2, y1, y2 = x[0], x[-1], y[0], y[-1]    # Get start and end for cutting down to size later
    if x_st == 0: x_st = x1
    if x_en == 0: x_en = x2
    if y_st == 0: y_st = y1
    if y_en == 0: y_en = y2

    x_len, y_len = len(x), len(y)
    # Get the step size/frequency for both matrices (ideally all steps would be the same size)
    # Take the most common step size as the one for the extra slices
    (stx, nx), (sty, ny) = np.unique(abs(np.diff(x)), return_counts=True), np.unique(abs(np.diff(y)), return_counts=True)
    xstep, ystep = stx[np.argmax(nx)], sty[np.argmax(ny)]

    # Calculate the extra rows and columns to add on to the start (00) and end(11)
    x_0_rows, x_1_rows = int((x_st - x[0])/-xstep), int((x[-1] - x_en)/-xstep)
    x00, x11 = np.arange(x[0] - x_0_rows*xstep, x[0], xstep), np.arange(x[-1], x[-1] + x_1_rows*xstep, xstep) + xstep    # New x range
    y_0_rows, y_1_rows = int((-y_st + y[0])/ystep), int((-y[-1] + y_en)/ystep)
    y00, y11 = np.arange(y[0] - y_0_rows*ystep, y[0], ystep), np.arange(y[-1], y[-1] + y_1_rows*ystep, ystep) + ystep    # New y range

    x, y = np.hstack((x00, x, x11)), np.hstack((y00, y, y11))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ The convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ttemp0 = signal.convolve(Gaussian_2D_general(x, y, th0, x0, y0, gx, gy, area),
                             Lorentzian_2D_centered_general(x, y, th0, lx, ly, area), mode='same')

    # Now cut the convolution back down to size, hoppefully having eliminated any edge effects
    if x_1_rows != 0: xt, temp0 = 1*x[x_0_rows:-x_1_rows], 1*ttemp0[:, x_0_rows:-x_1_rows]
    else: xt, temp0 = 1*x[x_0_rows:], 1*ttemp0[:, x_0_rows:]    # Cut the convolution back down to size np.since we know how many rows we added on

    if y_1_rows != 0: yt, temp0 = 1*y[y_0_rows:-y_1_rows], 1*temp0[y_0_rows:-y_1_rows, :]
    else: yt, temp0 = 1*y[y_0_rows:], 1*temp0[y_0_rows:, :]    # Cut the convolution back down to size np.since we know how many rows we added on

    # Fix matrix size discrepancy due to the instrument step size not being consistent
    x_dif, y_dif = temp0.shape[1] - x_len, temp0.shape[0] - y_len
    if temp0.shape[0] > y_len: temp0, yt = 1*temp0[:-y_dif, :], yt[:-y_dif]
    if temp0.shape[1] > x_len: temp0, xt = 1*temp0[:, :-x_dif], xt[:-x_dif]

    # Print message if there was an error in the convolution and 0 was returned
    if temp0.max() == 0:
        print("Error in the convolution! Return = 0")
        return temp0

    # Show the extended range and the cut to size range to make sure edge effects are satisfactory
    if plot:
        fig, (ap, av) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        av.contourf(xt, yt, temp0), av.set_title('cut to size', fontsize=26)
        ap.contourf(x, y, ttemp0), ap.set_title('full range -> adjust as necessary', fontsize=26)

        ap.axvline(x0, c='r', ls='--'), av.axvline(x0, c='r', ls='--')
        ap.axhline(y0, c='r', ls='--'), av.axhline(y0, c='r', ls='--')

    # Normalise the intensity and return
    return temp0 * (area / temp0.max())


# Make some test plots for the shited Lorentz peaks and the Voigt comparisons
if "__main__" in __name__:
    x, y = np.linspace(3, 6, 100), np.linspace(3, 6, 100)
    th0, x0, y0, sigx, sigy, gamx, gamy, area = -15, 4, 5, 40e-3, 69e-3, 60e-3, 100e-3, 100

    # Testing the precision of centered vs shifted Lorentzian peaks
    Lorentz_2D = Lorentzian_2D_general(x, y, th0, x0, y0, gamx, gamy, area)
    Lorentz_2D_cent = Lorentzian_2D_centered_general(x, y, th0, gamx, gamy, area)
    Lorentz_2D_shift = Lorentzian_2D_shift_center_general(x, y, th0, x0, y0, gamx, gamy, area)

    fig, axes = plt.subplots(2,2, figsize=[14,11], tight_layout=True); axes[-1,-1].set_axis_off()
    datas, names = [Lorentz_2D, Lorentz_2D_cent, Lorentz_2D_shift], ['unaltered', 'centered', 'shifted']
    for data,name,ax in zip(datas, names, axes.flatten()[:-1]):
        ax.contour(x, y, data), ax.set_title(f"{name} Lorentz peak - max = {data.max():.1f}")
        ax.set_xticks(np.linspace(3,6,7)), ax.set_yticks(np.linspace(3,6,7))
        ax.set_xlim(3.5,5.0), ax.set_ylim(4.0,5.5)
        ax.grid(alpha=0.75, lw=1.5)
    plt.show()


    # Comparing the different profiles
    Gaussian_2D = Gaussian_2D_general(x, y, th0, x0, y0, sigx, sigy, area)
    Lorentz_2D = Lorentzian_2D_general(x, y, th0, x0, y0, gamx, gamy, area)
    pseudo_Voigt_2D = pseudo_Voigt_2D_general(x, y, th0, x0, y0, sigx, sigy, gamx, gamy, area)
    Voigt_2D_gen = Voigt_2D_general(x, y, th0, x0, y0, sigx, sigy, gamx, gamy, area,
                                    x_st=0.95*x[0], x_en=1.05*x[-1], y_st=0.95*y[0], y_en=1.05*y[-1])

    fig, axes = plt.subplots(2,2, figsize=[14,13], tight_layout=True)
    datas = [Gaussian_2D, Lorentz_2D, pseudo_Voigt_2D, Voigt_2D_gen]
    names = ['Gaussian', 'Lorentz', 'pseudo-Voigt', 'Voigt1']
    for data,name,ax in zip(datas, names, axes.flatten()):
        ax.contour(x, y, data), ax.set_title(f"{name} 2D - max = {data.max():.1f}")
        ax.set_xlim(3.5,4.5), ax.set_ylim(4.5,5.5)
        ax.grid(alpha=0.75, lw=1.5)
    plt.show()


#%% Timing tests for generalised 2D profiles
if __name__ in "__main__" and run_timing_tests:
    num = 25
    G_2D = """y=Gaussian_2D_general(xt, yt, 45, 4.0, 5.0, 2.5, 1.53, 100)""" # 300.6ms -> #3
    time_G_2D = timeit(stmt=G_2D, setup="from __main__ import Gaussian_2D_general, xt, yt", number=num)

    L_2D = """y=Lorentzian_2D_general(xt, yt, 45, 4.0, 5.0, 3.2, 1.80, 100)""" # 162.2ms -> #1
    time_L_2D = timeit(stmt=L_2D, setup="from __main__ import Lorentzian_2D_general, xt, yt", number=num)

    L_shift_2D = """y=Lorentzian_2D_shift_center_general(xt, yt, 45, 4.0, 5.0, 3.2, 1.80, 100)""" # 227.7ms -> #2
    time_L_shift_2D = timeit(stmt=L_shift_2D, setup="from __main__ import Lorentzian_2D_shift_center_general, xt, yt", number=num)

    pV_2D = """y=pseudo_Voigt_2D_general(xt, yt, 45, 4.0, 5.0, 2.1, 1.3, 1.2, 0.5, 100)""" # 588.2ms -> #4
    time_pV_2D = timeit(stmt=pV_2D, setup="from __main__ import pseudo_Voigt_2D_general, xt, yt", number=num)

    V_2D = """y=Voigt_2D_general(xt, yt, 45, 4.0, 5.0, 2.1, 1.3, 1.2, 0.5, 100)""" # 5312.2ms -> #5
    time_V_2D = timeit(stmt=V_2D, setup="from __main__ import Voigt_2D_general, xt, yt", number=num)

    names = ['Gaussian_2D', 'Lorentzian_2D', 'Lorentzian_shifted_2D', 'pseudo-Voigt_2D', 'Voigt_2D']
    times = [time_G_2D, time_L_2D, time_L_shift_2D, time_pV_2D, time_V_2D]
    print(f"\nFor {num} calculations of peak profiles of size [{len(xt)} x {len(yt)}]:")
    for name,time in zip(names,times): print(f"{name} took: {1000*time:.1f}ms\n")
