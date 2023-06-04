# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:30:53 2021

@author: OBRIEJ25
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from glob import glob

plt.rc('font',   family = 'Times New Roman')
plt.rc('axes',   titlesize = 28, linewidth = 1.5, labelsize = 26)
plt.rc('legend', fontsize = 26)
plt.rc('xtick',  labelsize = 24), plt.rc('ytick',  labelsize = 24)

# =============================================================================
# Interactive matplotlib script (using qt backend) for background subtraction.
# Oringally intended for use on XRD data.
#
# Works as-intended but is not a very efficient method. Suitable for cases where manual
# automatic routines do not work correctly.
#
# Press the "smooth" button to apply a Savitsky-Golay filter to the data (len=25, order=4).
# Double click points on the graph to place interpolation points.
# Press the "back" button to remove the most recently added point.
# Press the "spline" button to create a cubic beta spline through the selected points.
# Press the "clear" button to clear the points and spline
# Press the "subtract" button to subtract the interpolated spline from the data.
# Press the "reset" button to reset the data (smooth and subtraction)
# Press the "save" button to save the current data to a new file with "bg_corrected" appended.
# =============================================================================


import IPython
shell = IPython.get_ipython()
shell.enable_matplotlib(gui='qt')


path = glob('../../1_Ph.D._Stuff/XRay_Analysis/data/NIST_Calibration_Sample/*NIST*dat')[0]
data = np.genfromtxt(path)
tt, I = 1*data[:,0], 1*data[:,1]
ylims = [0.1, I.max()*1.05]
scale = 'linear'

fig, ax = plt.subplots(1,1, figsize=(13,7))
plt.subplots_adjust(right=0.85)

def on_close(event):
    global run
    run = False

coords = np.array([], dtype=object)
def on_click(event):
    if event.dblclick:
        ix, iy = event.xdata, event.ydata
        global coords
        if len(coords): coords = np.vstack((coords, [ix,iy]))
        else: coords = np.append(coords, [ix,iy])

    if len(coords) > 50: fig.canvas.mpl_disconnect(cid)

cid = fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('close_event', on_close)

x_fine, y_fine = None, None
class button:
    def plot_spline(self, event):
        if len(coords) >= 3:
            xy = coords[coords[:,0].argsort()]
            global x_fine, y_fine
            x, y = xy[:,0], xy[:,1]
            global spline_func
            spline_func = CubicSpline(x, y, bc_type='natural')
            x_fine = np.linspace(tt.min(), tt.max(), 250)
            y_fine = spline_func(x_fine)
        else: x_fine, y_fine = None, None

    def clear_previous(self, event):
        global coords
        coords = coords[:-1] if len(coords.shape) > 1 else coords[:0]
        #self.plot_spline(event)

    def clear_all(self, event):
        global coords
        coords = np.array([], object)
        self.plot_spline(event)

    def smooth_data(self, event):
        global I
        # increasing polyorder + decreasing window_length -> sharper variation in the filtered signal
        I = savgol_filter(I, window_length=25, polyorder=4, deriv=0, mode='nearest')

    def reset_data(self, event):
        global I
        I = 1*data[:,1]

    def subtract_background(self, event):
        global I
        I = I - spline_func(tt)

    def save_corrected_data(self, event):
        ext = path.split('.')[-1]
        new_path = path.replace('.'+ext, '_bg_corrected.'+ext)
        np.savetxt(new_path, np.hstack((tt[:,None], I[:,None])))

    def linear_log(self, event):
        global scale
        scale = "log" if scale=="linear" else "linear"

    def zoom(self, event):
        global ylims
        ylims = [0.001, 0.01*I.max()] if ylims==[0.1, I.max()*1.05] else [0.1, I.max()*1.05]

# Set up the buttons: positions; initialisation; linkage to functions in button class
y_pos  =  [0.80,  0.72,  0.64,  0.56,  0.51, 0.43, 0.35, 0.20, 0.12]
heights = [0.066,0.066,0.066,0.066,0.044,0.066,0.066,0.066,0.066]
back, clear_all, spline, smooth, reset, subtract, save, lin_log, zoom =\
    [plt.axes([0.875, i, 0.1, j]) for i,j in zip(y_pos,heights)]  # [x,y,width,height]

btns = button()
btn_back      = Button(back, 'Back')
btn_spline    = Button(spline, 'Spline')
btn_clear_all = Button(clear_all, 'Clear')
btn_smooth    = Button(smooth, 'Smooth')
btn_reset     = Button(reset, 'reset')
btn_subtract  = Button(subtract, 'Subtract')
btn_save      = Button(save, 'Save')
btn_scale     = Button(lin_log, 'Lin/Log')
btn_zoom      = Button(zoom, 'zoom')

btn_back.on_clicked(btns.clear_previous)
btn_spline.on_clicked(btns.plot_spline)
btn_clear_all.on_clicked(btns.clear_all)
btn_smooth.on_clicked(btns.smooth_data)
btn_reset.on_clicked(btns.reset_data)
btn_subtract.on_clicked(btns.subtract_background)
btn_save.on_clicked(btns.save_corrected_data)
btn_scale.on_clicked(btns.linear_log)
btn_zoom.on_clicked(btns.zoom)

for btn in [btn_back, btn_spline, btn_clear_all, btn_smooth, btn_reset, btn_subtract, btn_save, btn_scale, btn_zoom]:
    btn.label.set_fontsize(20)


def plot_data():
    # Make labels
    ax.set_xlabel(r'$ 2\theta $')
    ax.set_ylabel('Counts')
    ax.set_title('NIST Standard Sample')

    # Plot Xrays
    ax.plot(tt, I, lw=1.5, label='Data', zorder=5)
    ax.axhline(0, c='k', ls='--', lw=1.5, zorder=20)

    # Plot the chosen background points
    if len(coords.shape) > 1 and coords.shape[0]>0:
        coords_array = np.stack(coords)
        xp, yp = coords_array[:,0], coords_array[:,1]
        ax.scatter(xp, yp, s=8**2, c='k', label='Background Points', zorder=50)
    elif len(coords):
        xp, yp = coords[0], coords[1]
        ax.scatter(xp, yp, s=8**2, c='k', label='Background Points', zorder=50)

    # Plot the spline if button is pressed and enough points exist
    if x_fine is not None: ax.plot(x_fine, y_fine, c='limegreen', lw=2.0, zorder=100)

    ax.grid(True, zorder=0)
    ax.legend()
    ax.set_xlim(tt.min(), tt.max())
    ax.set_ylim(*ylims)
    ax.set_yscale(scale)


def update_plot():
    ax.cla()    # clear axes
    plot_data()
    plt.pause(0.1)

run = True
while run: update_plot()


