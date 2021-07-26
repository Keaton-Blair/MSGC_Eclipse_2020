## Gravity-Waves-2020


*Developed by Keaton Blair and Hannah Woody, MSGC*


This folder contains python code to detect and analyze gravity waves from radiosonde data, as well as data collected during the December 2020 field campaign in Chile.


### Descriptions of the python files are as follows:


 **DetectWaves.py:**
Run this file to analyze GRAWMET profile data. The file takes in user input, asking for the path to a directory containing the profile data, whether or not to display power surfaces, and whether or not to save the images and analysis files to a user provided directory. This code will detect and analyze waves using the wavelet method with stokes parameter analysis, then save the generated power surfaces and wave characteristics (as a JSON).


 **PlotWaves.py:**
Run this file to make specific plots of the output files from DetectWaves.py. This file takes in user input, asking for the path to a directory containing the output files, the units of time to plot on the x-axis, whether to make a 2D or 3D plot, and what the title of the plot should be. This code is experimental and constantly being changed for various purposes, so errors are common.


**Simulation.py:**
This file is highly experimental and currently does not work as intended.


**TorrenceCompoWavelets.py:**
See description contained in file. https://github.com/chris-torrence/wavelets/tree/master/wave_python  
Reference: Torrence, C. and G. P. Compo, 1998: A Practical Guide to
            Wavelet Analysis. <I>Bull. Amer. Meteor. Soc.</I>, 79, 61-78.


**WaveDetectionFunctions.py:**
This file contains all of the functions necessary to run DetectWaves.py. Keep this file in the same directory, or change the path in the import statement at the top of DetectWaves.py to match the location of this file.


### Descriptions of the folders are as follows:


**Data/Profiles:**
This folder contains all of the GRAWMET profiles collected during the December 2020 field campaign, organized by the site at which the data was collected.

**Data/Outputs:**
This folder contains the results of the gravity wave analysis code for every profile, organized by the site at which the data was collected.