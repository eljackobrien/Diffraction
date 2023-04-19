This project uses matplotlib to plot the positions and intensities of reciprocal space peaks for two given materials.
Pygame is used for interactivity, with materials being chosen froma drop-down list, as well as out-of-plane orientation and the desried units of the graph.
The lattice mismatch for cube-on-cube and 45 degree rotated growth (with substrate the smaller lattice) is shown in the bottom right.

I made this code to aid in visualising and chosing suitable reflections to perform reciprocal space maps on.



.cif files for desired materials can be made by hand or using software such as VESTA or downloaded from databases such as MaterialsProject.
It is best to use "conventional standard" .cif files which don't manually account for the symmetry.

VESTA is then used to calculate the position and intensity for possible reflections from the .cif files, save these as .txt files with the same name in the same folder.

The VESTA_reflection_table_reader module reads these .txt files output by VESTA and parses the data.
