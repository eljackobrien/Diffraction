# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:42:04 2021

@author: OBRIEJ25
"""
#%% Try all the imports
import numpy as np
import sys
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

try: import VESTA_reflection_table_reader as vr
except ImportError: print("ImportError: Get the 'VESTA_reflection_table_reader' module off Jack"); quit()

try: import pygame as pg
except ImportError: print("ImportError: Script requires 'pygame' module"); quit()
try: import pygame_widgets
except ImportError: print("ImportError: Script requires 'pygame-widgets' module"); quit()
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.button import Button
#from pygame.locals import DOUBLEBUF

#%% Tabulate the file paths and material names
mat_paths = glob("./conventional_cifs_VESTA_data/*txt")

all_mats, all_mat_names = [], []
for i in range(len(mat_paths)):
    strings = list( mat_paths[i].split('\\')[-1].split('/')[-1].split('_')[0] )
    all_mats.append( ''.join(strings) )
    for j in range(len(strings)):
        for k in range(10):
            if str(k) in strings[j]:
                strings[j] = strings[j].replace(f"{k}","\\u208"+f"{k}")
                break
    all_mat_names.append( ''.join(strings) )

# get rid of unicode until I figure out how to work it
all_mat_names = all_mats[:]

path_name_dict = {}
for mat,name,path in zip(all_mats, all_mat_names, mat_paths):
    path_name_dict[f"{mat}"] = [f"{path}", f"{name}"]


#%% Pygame initialisations
pg.init()
pg.mixer.init()
LABEL_FONT = pg.font.SysFont('Times New Roman', 26)
MATH_FONT = pg.font.SysFont('Cambria Math', 26)
MATH_FONT_SMALL = pg.font.SysFont('Cambria Math', 22)
BTN_FONT = pg.font.SysFont('Times New Roman', 26, bold=True)

WIN_SIZE = (1200, 900)
PLOT_SIZE = (900, 850)
BCKGD, BLACK, WHITE = (230,242,255), (0,0,0), (255,255,255)
BTN_BCKGD, BTN_HOVER, BTN_PRESS = (20,30,40), (58,66,74), (89,202,145)
RED, GREEN = (150,0,0), (200,250,150)

#%% matplotlib in pygame

class Crystal_diffraction():
    def __init__(self):
        # Set default values for the various parameters
        self.substrate_dict = {'mat':'MgO', 'surface_normal':[0,0,1], 'c':'red', 'params':[4.2112,4.2112,4.2112]}
        self.sample_dict = {'mat':'Mn2RuGa', 'surface_normal':[0,0,1], 'c':'green', 'params':[5.960,5.960,6.040]}
        self.plot_units = 'Miller'
        self.mismatch_cube = -41.41
        self.mismatch_45 = 0.01

        self.recalculate()

    # Function to calculate cube-on-cube and 45 degree rotated growth mismatch, called by recalculate
    def get_mismatch(self):
        sub_a = self.substrate_dict['params'][0]
        samp_a = self.sample_dict['params'][0]
        self.mismatch_cube =  100 * (sub_a - samp_a) / sub_a

        if abs(self.mismatch_cube) > 10:
            if samp_a < sub_a: samp_a = samp_a * np.sqrt(2)
            else: sub_a = sub_a * np.sqrt(2)
        self.mismatch_45 = 100 * (sub_a - samp_a) / sub_a

    # Load all the data, create the plot object and update the RGB string, calculate mismatch
    def recalculate(self):
        fig = plt.figure(figsize=[PLOT_SIZE[0]/100, PLOT_SIZE[1]/100], dpi=100, tight_layout=True)

        x_max, y_max = 0, 0  # For setting axis limits
        dicts = [ self.substrate_dict, self.sample_dict ]
        for d in dicts:
            mat, norm, colour = d['mat'], d['surface_normal'], d['c']
            if mat==None: continue
            norm_render = '$' + str(d['surface_normal']).replace(', ','') + '$'

            # Load the data from the VESTA file for the correct material
            path, name = path_name_dict[mat]
            data, [a,b,c] = vr.read_vesta_table(path, surface_normal=norm, return_lat_params=1)
            # Set the lattice parameters
            d['params'] = [a,b,c]
            # Get the various x, y and z values for plotting
            tth, om, phis, qx, qz, I, peaks, pk_names = data
            inds = np.unique(I, return_index=True)[1]
            tth, om, qx, qz, I, pk_names = tth[inds], om[inds], qx[inds], qz[inds], I[inds], pk_names[inds]
            # Miller indices simply  L = qz*c  and  HK = qx*a (technically q_IP*sqrt(a^2+b^2))
            HK, L = 0.1*qx*self.substrate_dict['params'][0], 0.1*qz*self.substrate_dict['params'][2]

            # Adjust the limits of the plots depending on units
            if self.plot_units == 'Angles':
                if tth.max() > x_max: x_max = tth.max()
                if om.max() > y_max: y_max = om.max()
                plt.scatter(tth, om, c=colour, s=69*I, label=name+norm_render, zorder=5)
                # plot the labels for each peak that we kept
                for i in range(len(om)):
                    plt.text(tth[i]+1.2, om[i]-0.0, pk_names[i], fontsize=14, c=colour, zorder=10)

            elif self.plot_units == 'Reciprocal':
                if qx.max() > x_max: x_max = qx.max()
                if qz.max() > y_max: y_max = qz.max()
                plt.scatter(qx, qz, c=colour, s=69*I, label=name+norm_render, zorder=5)
                for i in range(len(om)):
                    plt.text(qx[i]+0.1, qz[i]-0.0, pk_names[i], fontsize=14, c=colour, zorder=10)

            elif self.plot_units == 'Miller':
                if HK.max() > x_max: x_max = HK.max()
                if L.max() > y_max: y_max = L.max()
                plt.scatter(HK, L, c=colour, s=69*I, label=name+norm_render, zorder=5)
                for i in range(len(om)):
                    plt.text(HK[i]+0.1, L[i]-0.0, pk_names[i], fontsize=14, c=colour, zorder=10)

        # Make the title for the plot
        sub = path_name_dict[ self.substrate_dict['mat'] ][1]
        samp = path_name_dict[ self.sample_dict['mat'] ][1]
        plt.title(f"Non-Forbidden Reflections:  {samp} on {sub}", fontsize=16)
        plt.grid(alpha=0.75, lw=1.0, ls='--', zorder=-5)

        # Make the labels, ticks, limits, legend, all the beels and whistles
        if self.plot_units == 'Angles':
            x_min = 1*plt.xlim()[0]  # get x-axis limits before drawing line
            plt.plot(np.linspace(10,160,100), np.linspace(10,160,100)/2, c='goldenrod', ls='--', zorder=-5)
            plt.xlabel(r'$2\theta$ (deg)', fontsize=14), plt.ylabel(r'$\omega$ (deg)', fontsize=14)
            plt.tick_params(labelsize=12)
            plt.xlim(x_min, x_max+10), plt.ylim(None, y_max+4)
            plt.legend(loc='upper left', fontsize=16)

        elif self.plot_units == 'Reciprocal':
            plt.xlabel(r'$q_x \: (nm^{-1})$', fontsize=16), plt.ylabel(r'$q_z \: (nm^{-1})$', fontsize=16)
            plt.tick_params(labelsize=12)
            plt.xlim(None, x_max+1.2), plt.ylim(None, y_max+0.5)
            plt.legend(loc='lower right', fontsize=16)

        elif self.plot_units == 'Miller':
            plt.xlabel(rf'$\sqrt{{H^2 + K^2}}$ ({sub})', fontsize=16), plt.ylabel(rf'$L$ ({sub})', fontsize=16)
            plt.tick_params(labelsize=12)
            plt.xlim(None, x_max+0.5), plt.ylim(None, y_max+0.1)
            plt.legend(loc='lower right', fontsize=16)

        # Convert the matplotlib figure into an RGB string, which pygame can read and render
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        raw_data = canvas.get_renderer().tostring_rgb()
        self.size = canvas.get_width_height()
        plt.close(fig)
        # Reconstruct the image in pygame and store in class
        self.surf = pg.image.fromstring(raw_data, self.size, "RGB")

        if (self.substrate_dict['mat'] != None) & (self.sample_dict['mat'] != None):
            self.get_mismatch()

    def draw(self, win):  # Centered in window, regardless of size
        win.blit(self.surf, (WIN_SIZE[0]/2 - self.size[0]/2, WIN_SIZE[1]/2 - self.size[1]/2))


#%% Pygame widgets help and main loop where all the buttons and logic are defined

# ~~~~~~~~~~~~~~~ Pygame Dropdown has the following parameters ~~~~~~~~~~~~~~~
# win, x, y, w, h, name=str, choices=[], direction=str, values=[], inactiveColour=RGB, pressedColour=RGB,
# hoverColour=RGB, onClick=func, onClickParams=(), onRelease=func, onReleaseParams=(),
# textColour=RGB_tuple, fontSize=int, font=pg.font.Font, textHAlign=str, borderColour=RGB,
# borderThickness=int, borderRadius=int

# Might replace them with ComboBox in futrue

# ~~~~~~~~~~~~~~~ Pygame Button has the following parameters ~~~~~~~~~~~~~~~
# win, x, y, w, h, text=str, inactiveColour=RGB, pressedColour=RGB, hoverColour=RGB, margin=int
# onClick=func, onClickParams=(), onRelease=func, onReleaseParams=(), textColour=RGB_tuple
# fontSize=int, font=pg.font.Font, textHAlign=str, textVAlign=str, radius=int


# Main loop
def main():
    window = pg.display.set_mode(WIN_SIZE)
    clock = pg.time.Clock()

    # Create crystal object where all the sub/samp info and plotting functions are stored
    crystal = Crystal_diffraction()

    # Right side (substrate) dropdown menus
    side, w, h = (WIN_SIZE[0]-PLOT_SIZE[0])/2, 120, 25
    x, y = WIN_SIZE[0]-(side+w)/2, 2*h
    sub_select = Dropdown(window, x, y, w, h, 'MgO', choices=all_mat_names, values=all_mats,
                          fontSize=24, inactiveColour=BTN_BCKGD, hoverColour=BTN_HOVER, textColour=WHITE,
                          pressedColour=BTN_PRESS)
    sub_select_label = [LABEL_FONT.render("Substrate", True, RED), (x, y-26)]

    sub_norm_select = Dropdown(window, x, y+3*h, w, 30, '[0,0,1]', choices=['[0,0,1]','[1,1,0]','[1,1,1]'],
                               values=[[0,0,1],[1,1,0],[1,1,1]], font=MATH_FONT, inactiveColour=BTN_BCKGD,
                               hoverColour=BTN_HOVER, textColour=WHITE, pressedColour=BTN_PRESS)
    sub_norm_label = [LABEL_FONT.render("Normal", True, RED), (x, y+3*h-26)]


    # Left side (sample) dropdown menus
    x, y = (side-w)/2, 2*h
    samp_select = Dropdown(window, x, y, w, h, 'Mn2RuGa', choices=all_mat_names, values=all_mats,
                          fontSize=24, inactiveColour=BTN_BCKGD, hoverColour=BTN_HOVER, textColour=WHITE,
                          pressedColour=BTN_PRESS)
    samp_select_label = [LABEL_FONT.render("Sample", True, RED), (x, y-26)]

    samp_norm_select = Dropdown(window, x, y+3*h, w, 30, '[0,0,1]', choices=['[0,0,1]','[1,1,0]','[1,1,1]'],
                               values=[[0,0,1],[1,1,0],[1,1,1]], font=MATH_FONT, inactiveColour=BTN_BCKGD,
                               hoverColour=BTN_HOVER, textColour=WHITE, pressedColour=BTN_PRESS)
    samp_norm_label = [LABEL_FONT.render("Normal", True, RED), (x, y+3*h-26)]

    unit_select = Dropdown(window, x, int(0.7*WIN_SIZE[1]), w, h, 'Miller', fontSize=24,
                               choices=['Reciprocal','Angles', 'Miller'], inactiveColour=BTN_BCKGD,
                               hoverColour=BTN_HOVER, textColour=WHITE, pressedColour=BTN_PRESS)
    unit_select_label = [LABEL_FONT.render("Units", True, RED), (x, int(0.7*WIN_SIZE[1])-26)]


    # List of texts to draw (each entry is list of the render itself and it's position)
    text_list = [sub_select_label, samp_select_label, sub_norm_label, samp_norm_label, unit_select_label]


    # Render the mismatch to the screen
    x, y = (WIN_SIZE[0]-(side+130)/2, int(0.7*WIN_SIZE[1]))
    mismatch = [LABEL_FONT.render("Mismatch", True, RED), [x,y]]
    mismatch_cube = [MATH_FONT_SMALL.render(f"a:a = {crystal.mismatch_cube:.2f}%", True, RED), [x-5,y+35]]
    mismatch_45 = [MATH_FONT_SMALL.render(f"a:\u221A2a = {crystal.mismatch_45:.2f}%", True, RED), [x-5,y+60]]
    text_list = text_list + [mismatch, mismatch_cube, mismatch_45]


    # define button, function to get all the selected values from the dropdown menus and re-do plot
    def btn1_func():
        # Get substrate info (if statements so we don't change anything back to None)
        if sub_select.getSelected(): crystal.substrate_dict['mat'] = sub_select.getSelected()
        if sub_norm_select.getSelected(): crystal.substrate_dict['surface_normal'] = sub_norm_select.getSelected()
        # Get sample info
        if samp_select.getSelected(): crystal.sample_dict['mat'] = samp_select.getSelected()
        if samp_norm_select.getSelected(): crystal.sample_dict['surface_normal'] = samp_norm_select.getSelected()
        # Get units info
        if unit_select.getSelected(): crystal.plot_units = unit_select.getSelected()
        # Recalculate everyhing
        crystal.recalculate()
        # Change Mismatch text
        text_list[-2] = [MATH_FONT_SMALL.render(f"a:a = {crystal.mismatch_cube:.2f}%", True, RED), [x-5,y+25]]
        text_list[-1] = [MATH_FONT_SMALL.render(f"a:\u221A2a = {crystal.mismatch_45:.2f}%", True, RED), [x-5,y+55]]

    Button(window, 15, WIN_SIZE[1]-25-40, 120, 40, text="Re-Calc", onRelease=btn1_func, font=BTN_FONT,
                  inactiveColour=GREEN, hoverColour=BTN_HOVER, textColour=RED)


    run = True
    while run:
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                run = False
                pg.quit()
                sys.exit()

        window.fill(BCKGD)

        for text in text_list: window.blit(*text)

        crystal.draw(window)
        pygame_widgets.update(events)
        pg.display.flip()  # update the entire display surface to the screen

        clock.tick(30)



#%%
if __name__ == '__main__':
    main()
    pg.quit()



