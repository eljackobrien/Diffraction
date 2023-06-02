# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:24:25 2023

@author: OBRIEJ25
"""

import matplotlib.pyplot as plt

def plt_stem(x, y, y0=0,    c='b', alpha=1.0, label=None,   marker='o', ms=4,    ls='-', lw=1.0,   axis=None):
    """
    Shorthand function to make a stem plot in the active axis in matplotlib.
    Variables all have their usual meaning.
    Returns the stem plot components if further editing is desired.
    """
    if axis: mark, stem, _ = axis.stem(x, y, bottom=y0, linefmt=c, label=label)
    else: mark, stem, _ = plt.stem(x, y, bottom=y0, linefmt=c, label=label)
    plt.setp(stem,'ls',ls), plt.setp(stem,'lw',lw), plt.setp(stem,'alpha',alpha)
    plt.setp(mark,'marker',marker), plt.setp(mark,'ms',ms, c=c), plt.setp(mark,'alpha',alpha)
    return (mark, stem, _)

# Test
if "__main__" in __name__:
    fig, [bx, ax] = plt.subplots(1,2, figsize=[8,3], width_ratios=[1,4], constrained_layout=True)

    xs, ys = [1,2,3,4,5,6], [1,2,1,2,3,1.5]
    c, alpha = ['b','r','g','k','coral','cyan'], [1,0.9,0.8,0.7,0.6,0.5]
    m, ms = ['o','s','*','^','v','x'], [6,6,10,6,2,5]
    ls, lw = ['-','--','-.',':','--','-'], [1,1.5,2,3,2,1]
    for x,y,c0,a0,m0,ms0,ls0,lw0 in zip(xs, ys, c, alpha, m, ms, ls, lw):
        plt_stem(x, y, 0, c=c0, alpha=a0, marker=m0, ms=ms0, ls=ls0, lw=lw0)
    plt_stem(3.5, 1.8, -0.5, ms=15, lw=4, axis=bx)
    ax.axhline(0, c='k')
    ax.grid()
    plt.show()

