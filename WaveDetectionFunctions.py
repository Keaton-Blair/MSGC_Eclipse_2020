########## IMPORT LIBRARIES AND FUNCTIONS ##########

import numpy as np  # Numbers (like pi) and math
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import matplotlib.pyplot as plt  # Easy plotting
import matplotlib.path as mpath  # Used for finding the peak region
import pandas as pd  # Convenient data formatting, and who doesn't want pandas
import os  # File reading and input
from io import StringIO  # Used to run strings through input/output functions
# Torrence & Compo (1998) wavelet analysis code, downloaded from https://github.com/chris-torrence/wavelets
from TorrenceCompoWavelets import wavelet as continuousWaveletTransform, wave_signif as significanceLevels
from skimage.feature import peak_local_max  # Find local maxima in power surface
import datetime  # Turning time into dates
from skimage.measure import find_contours  # Find contour levels around local max
from scipy.ndimage.morphology import binary_fill_holes  # Then fill in those contour levels
from scipy.signal import argrelextrema  # Find one-dimensional local min, for peak rectangle method
import json  # Used to save wave parameters to json file
from statsmodels.tsa.stattools import acf as autocorrelation  # Used to compute autocorrelation for significance test


########## USER INTERFACE ##########


def getUserInputFile(prompt):
    # FUNCTION PURPOSE: Get a valid path (relative or absolute) to a directory from user
    #
    # INPUTS:
    #   prompt: String that is printed to the console to prompt user input
    #
    # OUTPUTS:
    #   userInput: String containing path to an existing directory

    # Print the prompt to console
    print(prompt)

    # userInput starts as empty string
    userInput = ""

    # While userInput remains empty, get input
    while not userInput:
        userInput = input()

        # If input isn't a valid directory, set userInput to empty string
        if not os.path.isdir(userInput):
            # Console output to let user know requirements
            print("Please enter a valid directory:")
            userInput = ""

    # Now that the loop has finished, userInput must be valid, so return
    return userInput


def getUserInputTF(prompt):
    # FUNCTION PURPOSE: Get a valid boolean (True or False) from user
    #
    # INPUTS:
    #   prompt: String that is printed to the console to prompt user input
    #
    # OUTPUTS:
    #   userInput: Boolean containing the user's answer to 'prompt'

    # Print the prompt to console, followed by the user's input options ("Y" or "N")
    print(prompt+" (Y/N)")

    # userInput starts as empty string
    userInput = ""

    # While userInput remains empty, get input
    while not userInput:
        userInput = input()
        # If input isn't either "Y" or "N", set userInput to empty string
        if lower(userInput) != "y" and lower(userInput) != "n":
            print("Please enter a valid answer (Y/N):")
            # Console output to let user know requirements
            userInput = ""

    # Now that the loop has finished, return True for "Y" and False for "N"
    if lower(userInput) == "y":
        return True
    else:
        return False


def getAllUserInput():
    # FUNCTION PURPOSE: Get all required user input to begin running the program
    #
    # INPUTS: None
    #
    # OUTPUTS:
    #   results: Dictionary containing the user answers to the 3 or 4 questions below

    # Get the directory containing the data for analysis
    dataSource = getUserInputFile("Enter path to data input directory: ")

    # Get a boolean value for whether to display the generated plots
    showPlots = getUserInputTF("Do you want to display plots for analysis?")

    # Get a boolean value for whether to save calculated data
    saveData = getUserInputTF("Do you want to save the output data?")

    # If saving the data, get the directory in which to save it
    if saveData:
        savePath = getUserInputFile("Enter path to data output directory: ")
    else:
        savePath = "NA"

    # Print results to inform user and begin program
    # Could eventually add a "verbose" option into user input that regulates print() commands
    print("Running with the following parameters:")
    print("Path to input data: "+dataSource)
    print("Display plots: "+str(showPlots))
    print("Save data: "+str(saveData))
    if saveData:
        print("Path to output data: "+savePath+"\n")
    else:
        # Extra line for improved readability
        print()

    # Build a dictionary to return values
    results = {
        'dataSource': dataSource,
        'showPlots': showPlots,
        'saveData': saveData
    }
    if saveData:
        results.update( {'savePath': savePath })

    # Return the resulting dictionary
    return results


def displayProgress(peaks, length, data, dataList):
    # FUNCTION PURPOSE: Display console output detailing progress analyzing local maxima
    #
    # INPUTS:
    #   peaks: Numpy 2d array containing list of peaks yet to be analyzed
    #   length: Original number of peaks to be analyzed
    #   data: Pandas DataFrame containing the current section of the flight
    #   dataList: Full list of flight sections, each section a Pandas DataFrame
    #
    # OUTPUTS: None

    # Find the index of data in dataList
    currentSection = [i for i in range(len(dataList)) if dataList[i].equals(data)]

    # Print progress to the console, beginning with carriage return (\r) and ending without newline
    print("\rTracing and analyzing section " + str(currentSection[0] + 1) + "/" + str(len(dataList)) +
            " peak " + str(length - len(peaks) + 1) + "/" + str(length), end='')


def outputWaveParameters(userInput, waves, fileName):
    # FUNCTION PURPOSE: Save or print final wave parameters from finished analysis, based on user input
    #
    # INPUTS:
    #   userInput: Dictionary containing user input, especially data saving information
    #   waves: Dictionary containing final wave parameters from the completed analysis
    #   fileName: String with the name of the profile currently being analyzed
    #
    # OUTPUTS: None

    # If the user asked for data to be saved, do it
    if userInput.get('saveData'):

        # The following is code to sort the waves by alt, but it has trouble with the dictionary format and needs fixing
        # waves['waves'] = sorted(waves['waves'].items(), key=lambda x: x[1].get('Altitude [km]'))

        # Save waves data to a JSON file here
        with open(userInput.get('savePath') + "/" + fileName[0:-4] + '_wave_parameters.json', 'w') as writeFile:
            # Indent=4 sets human-readable whitespace, making the output viewable in a text editor
            json.dump(waves, writeFile, indent=4, default=str)

    # Otherwise, print the output to the console for user to see
    else:

        print("\nWave parameters found:")
        print(json.dumps(waves['waves'], indent=4, default=str), end='')


def drawPowerSurface(userInput, fileName, wavelets, altitudes, plotter, peaksToPlot, colorsToPlot):
    # FUNCTION PURPOSE: Create a power surface showing local maxima and their outlines
    #
    # INPUTS:
    #   userInput: Dictionary containing whether to save/show the plots, as well as a save path
    #   fileName: String, name of the profile file currently being analyzed
    #   wavelets: Dictionary containing power surface and corresponding wavelengths, significance and cone of influence
    #   altitudes: Pandas DataFrame column with altitudes (IN KM) corresponding to the power surface
    #   plotter: Boolean mask identifying traced regions on power surface
    #   peaksToPlot: Numpy 2d array containing peaks, e.g. [ [row1, col1], [row2, col2], ... [rowN, colN] ]
    #   colorsToPlot: Numpy array of strings corresponding to each peak, e.g. [ "color1", "color2", ... "colorN" ]
    #
    # OUTPUTS: Returns nothing, prints to console and saves files and/or shows images

    # If neither saving nor showing the plots, then don't bother making them
    if not userInput.get('saveData') and not userInput.get('showPlots'):
        return

    # Console output to keep the user up to date
    print("\r\nGenerating power surface plots", end='')

    # Get the vertical wavelengths in kilometers for the Y coordinates
    yScale = wavelets.get('wavelengths') / 1000

    # Define new array for power surface, then remove values < 0 for convenience (so that the scale is more manageable)
    power = np.log(wavelets.get('power').copy())
    power[power < 0] = 0

    # Contourf is a filled contour, which is the easiest tool to plot a colored surface
    # Levels is set to 50 to make it nearly continuous, which takes a while,
    # but looks good and handles the non-uniform yScale, which plt.imshow() does not
    plt.contourf(altitudes, yScale, power, levels=50, cmap=plt.get_cmap('turbo'))

    # Create a colorbar for the z scale
    cb = plt.colorbar()

    # Plot the outlines of the local maxima, contour is an easy way to outline a mask
    # The 'plotter' is a boolean mask, so levels is set to 0.5 to be between 1 and 0 (True and False)
    if plotter.any():
        plt.contour(altitudes, yScale, plotter, colors='red', levels=[0.5])

    # Plot 95% confidence level as a black line
    if wavelets.get('signif').any():
        plt.contour(altitudes, yScale, wavelets.get('signif'), colors='black', levels=[0.5])

    # Make a scatter plot of the identified peaks, coloring them according to which ones were confirmed as waves
    if len(peaksToPlot) > 0:
        plt.scatter(altitudes[peaksToPlot.T[1]], yScale[peaksToPlot.T[0]], c=colorsToPlot, marker='.')

    # Plot the cone of influence in black
    if wavelets.get('coi').any():
        plt.contour(altitudes, yScale, wavelets.get('coi'), colors='black', levels=[0.5])

    # Set the axis scales, labels, and titles
    plt.yscale("log")
    plt.xlabel(r"Altitude [$km$]")
    plt.ylabel(r"Vertical Wavelength [$km$]")
    plt.ylim(yScale[0], yScale[-1])
    plt.title("Power surface, including traced peaks")
    cb.set_label(r"LN( Power ) [$\frac{m^2}{s^2}$]")

    # Save and/or show the plot, according to user input.
    if userInput.get('saveData'):
        # Get current filenames in saving directory
        filenames = os.listdir(userInput.get('savePath'))
        # Filter for the current flight
        sections = [x for x in filenames if x.find(fileName[0:-4]+"_power_surface") >= 0]
        # Save file using the correct flight section number
        plt.savefig(userInput.get('savePath') + "/" + fileName[0:-4] +
                    "_power_surface_section_" + str(len(sections) + 1) + ".png")

    # Show plot if applicable, according to user input
    if userInput.get('showPlots'):
        plt.show()

    plt.close()  # Close plot when finished

    # Below is code to plot the power surface in 3D.
    # It's commented out because it doesn't look very good,
    # and it's confusing/not that useful.
    # However, with several innovations, it could be helpful,
    # so it's here for the future.

    """
    from matplotlib import cm
    # Make a meshgrid
    X, Y = np.meshgrid(altitudes, np.log10(yScale))
    fig = plt.figure()
    ax = fig.gca(projection='3d', proj_type = 'ortho')
    surf = ax.plot_surface(X, Y, wavelets.get('power'), cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    # Set top-down view, elevation and azimuth in degrees
    ax.view_init(elev=90, azim=270)
    # Turn off z-axis
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])
    # Set axis limits
    plt.xlim(altitudes[0], altitudes[len(altitudes)-1])
    plt.ylim(np.log10(yScale)[0], np.log10(yScale)[-1])
    #ax.set_zlabel('Power [m^2(s^-2)]')
    ax.set_ylabel('log10(vertical wavelength)')
    ax.set_xlabel('Altitude [km]')
    # Get the maximum value on the plot
    plotMax = np.max(wavelets.get('power'))
    # Plot the outlines of the local maxima, contour is an easy way to outline a mask
    # The 'plotter' is a boolean mask, so levels is set to 0.5 to be between 0 and 1
    ax.contour3D(X, Y, 2*plotMax*plotter, colors='red', levels=[plotMax])
    # Make a scatter plot of the identified peaks, coloring them according to which ones were confirmed as waves
    if len(peaksToPlot) > 0:
        ax.scatter3D(altitudes[peaksToPlot.T[1]], np.log10(yScale)[peaksToPlot.T[0]], [plotMax] * len(peaksToPlot), c=colorsToPlot, marker='.')
    # Plot the cone of influence in black
    ax.plot3D(altitudes, np.log10(wavelets.get('coi')), [plotMax] * len(altitudes), color='black')

    if userInput.get('saveData'):
        plt.savefig(userInput.get('savePath') + "/" + fileName[0:-4] + "_power_surface_3D.png")
    if userInput.get('showPlots'):
        plt.show()
    plt.close()
    """


def compareMethods(waveR, waveC, parametersR, parametersC, regionR, regionC):
    # FUNCTION PURPOSE: Get user input to compare results from two methods based on their hodographs.
    #                   This function is currently unused because we have settled on using the
    #                   rectangle method; however, this function could be useful for testing other
    #                   methods in the future, or improving on the contour tracing method.
    #
    # INPUTS:
    #   wave: Dictionary containing wavelet transformed surfaces, for rectangle (R) and contour (C) methods
    #   parameters: Dictionary containing wave parameters, for R and C methods
    #   region: Boolean mask tracing the wave on the power surface, for R and C methods
    #
    # OUTPUTS:
    #   parameters: Dictionary containing wave parameters, for the chosen method
    #   region: Boolean mask tracing the wave on the power surface, for the chosen method


    # First, filter based on half-max wind variance, from Murphy (2014)

    # Calculate the wind variance of the wave
    windVarianceR = np.abs(waveR.get('uTrim')) ** 2 + np.abs(waveR.get('vTrim')) ** 2
    windVarianceC = np.abs(waveC.get('uTrim')) ** 2 + np.abs(waveC.get('vTrim')) ** 2

    # Get rid of values below half-power, per Murphy (2014)
    uR = waveR.get('uTrim').copy()[windVarianceR >= 0.5 * np.max(windVarianceR)]
    vR = waveR.get('vTrim').copy()[windVarianceR >= 0.5 * np.max(windVarianceR)]
    uC = waveR.get('uTrim').copy()[windVarianceC >= 0.5 * np.max(windVarianceC)]
    vC = waveR.get('vTrim').copy()[windVarianceC >= 0.5 * np.max(windVarianceC)]

    # Discard imaginary components, which aren't needed for hodograph
    uR = uR.real
    vR = vR.real
    uC = uC.real
    vC = vC.real

    # Now, create hodograph subplots for easy comparison
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Which Hodograph Looks Better?')
    ax[0].plot(uR, vR)
    ax[0].set_title('Rectangle Peak Trace Method')
    ax[1].plot(uC, vC)
    ax[1].set_title('Contour Peak Trace Method')
    plt.show()

    # Get user input for selection
    print("\r\nPlease enter the name of the method that showed a more elliptical shape:")

    # userInput starts as empty string
    userInput = ""

    # While userInput remains empty, get input
    while not userInput:
        userInput = lower(input())

        # If input isn't either "rectangle" or "contour", set userInput to empty string
        if userInput != "rectangle" and userInput != "contour" and userInput != "r" and userInput != "c":
            # Console output to let user know requirements if they don't answer right
            print("Please enter either 'rectangle' or 'contour':")
            userInput = ""

    # Now that the loop has finished, return correct parameters and region
    if userInput == "rectangle" or userInput == "r":
        return parametersR, regionR
    else:
        return parametersC, regionC


########## DATA INPUT/MANAGEMENT ##########

def cleanData(file, path):
    # FUNCTION PURPOSE: Read a data file, and if the file contains GRAWMET profile data,
    #                   then clean the data and return the results
    #
    # INPUTS:
    #   file: The filename of the data file to read
    #   path: The path (absolute or relative) to the file
    #
    # OUTPUTS:
    #   data: Pandas DataFrame containing the time [s], altitude [m], temperature [deg C],
    #           pressure [hPa], wind speed [m/s], wind direction [deg], latitude [decimal deg],
    #           and longitude [decimal deg] of the radiosonde flight


    # If file is not a txt file, end now
    if not file.endswith(".txt"):
        return pd.DataFrame()  # Empty data frame means end analysis


    # Open and investigate the file
    contents = ""
    isProfile = False  # Check to see if this is a GRAWMET profile
    f = open(os.path.join(path, file), 'r', encoding='unicode_escape')  # Escape non-utf characters, i.e. \0xb0
    print("\nOpening file "+file+":")
    for line in f:  # Iterate through file, line by line
        if line.rstrip() == "Profile Data:":
            isProfile = True  # We found the start of the real data in GRAWMET profile format
            contents = f.read()  # Read in rest of file, discarding header
            print("File contains GRAWMET profile data")
            break
    f.close()  # Need to close opened file

    # If we checked the whole file and didn't find it, end analysis now.
    if not isProfile:
        print("File "+file+" is either not a GRAWMET profile, or is corrupted.")
        return pd.DataFrame()

    # Read in the data and perform cleaning

    # Need to remove space so Virt. Temp reads as one column, not two
    contents = contents.replace("Virt. Temp", "Virt.Temp")
    # Break file apart into separate lines
    contents = contents.split("\n")
    contents.pop(1)  # Remove units so that we can read table
    index = -1  # Used to look for footer
    for i in range(0, len(contents)):  # Iterate through lines
        if contents[i].strip() == "Tropopauses:":
            index = i  # Record start of footer
    if index >= 0:  # Remove footer, if found
        contents = contents[:index]
    contents = "\n".join(contents)  # Reassemble string

    # Read in the data
    data = pd.read_csv(StringIO(contents), delim_whitespace=True)
    del contents  # Free up a little memory

    # Get rid of extraneous columns that won't be used for further analysis
    essentialData = ['Time', 'Alt', 'T', 'P', 'Ws', 'Wd', 'Lat.', 'Long.', 'Rs']
    data = data[essentialData]

    # Coerce columns to numeric values to ensure that strings are interpreted as NA
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Find the end of usable (ascent) data
    badRows = []  # Index, soon to contain any rows to be removed
    for row in range(data.shape[0]):  # Iterate through rows of data
        # noinspection PyChainedComparisons
        if not str(data['Rs'].loc[row]).replace('.', '', 1).isdigit():  # Check for nonnumeric or negative rise rate
            badRows.append(row)
        # Check for stable or decreasing altitude (removes rise rate = 0)
        elif row > 0 and np.diff(data['Alt'])[row-1] <= 0:
            badRows.append(row)
        else:
            for col in range(data.shape[1]):  # Iterate through every cell in row
                # Check for values that (for GRAWMET) are equivalent to NA
                if data.iloc[row, col] == 999999.0 or data.iloc[row, col] == 1000275:
                    badRows.append(row)  # Remove row if NA value is found
                    break

    if len(badRows) > 0:
        print("Dropping "+str(len(badRows))+" rows containing unusable data")
        data = data.drop(data.index[badRows])  # Actually remove any necessary rows
    data.drop(columns='Rs')  # Remove rise rate from data as it is no longer needed
    data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]

    # If every row of data was removed, notify user
    if data.empty:
        print("No usable data left, quitting analysis")

    return data  # return cleaned pandas data frame


def readFromData(file, path):
    # FUNCTION PURPOSE: Find launch time and tropopause altitude in a given profile,
    #                   or return default values if not found.
    #
    # INPUTS:
    #   file: The filename of the data file to read
    #   path: The path (absolute or relative) to the file
    #
    # OUTPUTS:
    #   launchDateTime: datetime.datetime object containing the UTC date and time of launch
    #   tropopause: Number in meters representing PBL height



    # Establish default values, in case values are not contained in profile
    launchDateTime = datetime.datetime.now()
    tropopause = 12000

    # Open and investigate the file
    f = open(os.path.join(path, file), 'r', encoding='unicode_escape')  # Escape non unicode characters, i.e. '\0xb0'
    # Create flags to close loop once both values have been found
    timeFlag = False
    tropFlag = False
    for line in f:  # Iterate through file, line by line

        # If line has expected beginning, try to get datetime from file
        if line.rstrip() == "Flight Information:":
            # noinspection PyBroadException
            try:
                dateTimeInfo = f.readline().split()
                dateTimeInfo = ' '.join(dateTimeInfo[2:6] + [dateTimeInfo[8]])
                launchDateTime = datetime.datetime.strptime(dateTimeInfo, '%A, %d %B %Y %H:%M:%S')
                timeFlag = True
            except:
                # If an error is encountered, print a statement to the console and continue
                print("Error reading flight time info, defaulting to present")

        # If line has expected beginning, try to get tropopause info
        if line.split(' ')[0] == "Tropopauses:":
            # noinspection PyBroadException
            try:
                p = float(line.split(": ")[2].split(' ')[0])
                # Pressure to altitude conversion from https://www.weather.gov/media/epz/wxcalc/pressureAltitude.pdf
                tropopause = 44307.694 * (1 - (p/1013.25)**0.190284)
                tropFlag = True
            except:
                # If an error is encountered, print a statement to the console and continue
                print("Error reading flight tropopause info, defaulting to 12 kilometers")

        # Close loop early if both values have been found
        if tropFlag and timeFlag:
            print(f"Values from profile header: tropopause = {int(tropopause)/1000} km, launch = {launchDateTime} UTC")
            break

    f.close()  # Need to close opened file

    # Return values from profile, or default values if not found
    return launchDateTime, tropopause


def interpolateData(data, spatialResolution, tropopause, launchDateTime):
    # FUNCTION PURPOSE: Interpolate to create a Pandas DataFrame for the flight as a uniform
    #                   spatial grid, with datetime.datetime objects in the time column,
    #                   split into separate DataFrames for each section of the flight
    #
    # INPUTS:
    #   data: Pandas DataFrame containing flight information
    #   spatialResolution: Desired length (in meters) between rows of data, must be a positive integer
    #   tropopause: Tropopause altitude (in meters) as detected by GRAWMET software
    #   launchDateTime: A datetime.datetime object containing the launch date and time in UTC
    #
    # OUTPUTS:
    #   data: List of Pandas DataFrames containing the time [s], altitude [m], temperature [deg C],
    #           pressure [hPa], wind speed [m/s], wind direction [deg], latitude [decimal deg],
    #           and longitude [decimal deg] of the radiosonde flight


    # Console output to inform user
    print("Interpolating data to create a uniform spatial grid", end='')

    # First, filter data to remove sub-PBL data
    data = data[data['Alt'] >= tropopause]
    # If all data was removed, inform user
    if len(data) == 0:
        print("\nNo stratospheric data, quitting analysis")
        return []

    # Now, interpolate to create spatial grid, not temporal
    # Create index of heights with 1 meter spatial resolution
    heightIndex = pd.DataFrame({'Alt': np.arange(min(data['Alt']), max(data['Alt'])+1)})
    # Right merge data with index to keep all heights, allowing interpolation to every whole meter
    data = pd.merge(data, heightIndex, how="right", on="Alt")
    # Sort data by height for interpolation
    data = data.sort_values(by=['Alt'])
    # Reset data frame index to [0, 1, 2, ... nrow]
    data.reset_index(drop=True, inplace=True)

    # Use pandas built in interpolate function to fill in NAs

    missingDataLimit = 1000  # If 1 km --ARBITRARY-- or more missing data in a row, leave the NAs in place
    # Because pandas interpolate still fills in part of missing data, make a mask to refill NA's later
    # From https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe
    naMask = data.copy()
    grp = ((naMask.notnull() != naMask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for col in data.columns:
        naMask[col] = (grp.groupby(col)['ones'].transform('count') < missingDataLimit) | data[col].notnull()
    # Interpolate data, then replace the NA's that exceeded missingDataLimit
    data = data.interpolate(method="linear").bfill()[naMask]

    # Split the data along any null sections remaining

    # Get the list of rows with null data remaining
    nullIndices = [i for i in range(data.shape[0]) if data.isnull().values[i].any()]
    # Concatenate list with incremented one for edge parameters of numpy.split()
    nullIndices = nullIndices + [i+1 for i in nullIndices]
    nullIndices.sort()
    # Split data along sections with null values
    data = [x for x in np.split(data, nullIndices) if x.shape[0] > missingDataLimit and not x.isnull().values.any()]

    # For the section above, check whether it's faster to skip the two middle lines, then do np.split(data, nullIndices)
    # and remove all of the individual NA values via iteration or list comprehension?

    # If data was split into multiple sections, inform user
    if len(data) > 1:
        print("\nFound more than " + str(missingDataLimit) + " meters of consecutive missing data")
        print("Split data into " + str(len(data)) + " separate sections for analysis", end='')
    # If all data was removed, inform user that the analysis will end
    if len(data) == 0:
        print("\nNo salvageable data, quitting analysis")

    # For each section of split data, fix the index and time values
    for i in range(len(data)):
        data[i].reset_index(drop=True, inplace=True)  # Reset data frame index to [0,1,2,...,nrow]

        # Create index according to desired spatial resolution
        keepIndex = np.arange(0, len(data[i]['Alt']), spatialResolution)
        data[i] = data[i].iloc[keepIndex, :]  # Keep data according to index, lose the rest of the data

        data[i].reset_index(drop=True, inplace=True)  # Reset data frame index to [0,1,2,...,nrow]

        # Convert times from seconds since launch to UTC datetime.datetime objects
        data[i].loc[:, 'Time'] = [launchDateTime + datetime.timedelta(seconds=float(x)) for x in data[i].loc[:, 'Time']]


    return data  # Return list of pandas data frames


########## WAVELET TRANSFORMATION ##########

def waveletTransform(data, spatialResolution, waveletName):
    # FUNCTION PURPOSE: Perform the continuous wavelet transform on wind speed components and temperature
    #
    # INPUTS:
    #   data: Pandas DataFrame containing flight information
    #   spatialResolution: Length (in meters) between rows of data
    #   waveletName: String containing the name of the wavelet to use for the transformation. Based on
    #                Zink & Vincent (2001) and Murphy et. al (2014), this should be 'MORLET'
    #
    # OUTPUTS:
    #   results: Dictionary containing the power surface (|U|^2 + |V|^2), the wavelet transformed
    #               surfaces U, V, and T (zonal wind speed, meridional wind speed, and temperature
    #               in celsius), the wavelet scales and their corresponding fourier wavelengths,
    #               the cone of influence, significance levels and the reconstruction constant
    #               from Torrence & Compo (1998)


    # u and v (zonal & meridional) components of wind speed
    u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
    v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)
    t = data['T']

    # Remove background using a second order polynomial fit, from Moffat (2011)
    # Find polynomial fits
    uFit = np.polyfit(data['Alt'], u, 2)
    vFit = np.polyfit(data['Alt'], v, 2)
    tFit = np.polyfit(data['Alt'], t, 2)
    # Apply fits to remove background
    u = u - uFit[0] * data['Alt']**2 - uFit[1] * data['Alt'] - uFit[2] * np.ones(len(u))
    v = v - vFit[0] * data['Alt']**2 - vFit[1] * data['Alt'] - vFit[2] * np.ones(len(v))
    t = t - tFit[0] * data['Alt']**2 - tFit[1] * data['Alt'] - tFit[2] * np.ones(len(t))


    # In preparation for wavelet transformation, define variables
    # From Torrence & Compo (1998)
    padding = 1  # 'True', tells function to pad the data with zeros to allow convolution to edge of data
    scaleResolution = 0.125/8  # This controls the spacing in between scales
    smallestScale = 2 * spatialResolution  # This number is the smallest wavelet scale

    # Currently not used, pass this to parameter J1 in continuousWaveletTransform()
    # howManyScales = 10/scaleResolution  # This number is how many scales to compute
    # Check Zink & Vincent section 3.2 par. 1 to see their scales/wavelengths

    # Lay groundwork for inversions, outside of looping over local max. in power surface
    # Derived from Torrence & Compo (1998) Equation 11 and Table 2
    constant = scaleResolution * np.sqrt(spatialResolution) / (0.776 * np.pi**0.25)

    # Now, do the actual wavelet transform using library from Torrence & Compo (1998)
    print("\nPerforming wavelet transform on U... (1/3)", end='')  # Console output, to be updated
    coefU, periods, scales, coi = continuousWaveletTransform(u, spatialResolution, pad=padding, dj=scaleResolution,
                                                             s0=smallestScale, mother=waveletName)

    print("\rPerforming wavelet transform on V... (2/3)", end='')  # Update to keep user informed
    coefV, periods, scales, coi = continuousWaveletTransform(v, spatialResolution, pad=padding, dj=scaleResolution,
                                                             s0=smallestScale, mother=waveletName)

    print("\rPerforming wavelet transform on T... (3/3)", end='')  # Final console output for wavelet transform
    coefT, periods, scales, coi = continuousWaveletTransform(t, spatialResolution, pad=padding, dj=scaleResolution,
                                                             s0=smallestScale, mother=waveletName)


    # Power surface is sum of squares of u and v wavelet transformed surfaces
    power = abs(coefU) ** 2 + abs(coefV) ** 2  # abs() gets magnitude of complex number


    # Find the 95% confidence level, from Torrence & Compo (1998)
    signif = significanceLevels(u, spatialResolution, scales, lag1=autocorrelation(u, nlags=1, fft=False)[1]) + \
             significanceLevels(v, spatialResolution, scales, lag1=autocorrelation(v, nlags=1, fft=False)[1])
    # Turn 1D array into a 2D array for direct comparison with power surface
    signif = np.array([signif for _ in range(len(u))]).T
    # Create boolean mask that is True where power is significant and False otherwise
    signif = power > signif


    # Divide each column by sqrt of the scales so that it doesn't need to be done later to invert wavelet transform
    for col in range(coefU.shape[1]):
        coefU[:, col] = coefU[:, col] / np.sqrt(scales)
        coefV[:, col] = coefV[:, col] / np.sqrt(scales)
        coefT[:, col] = coefT[:, col] / np.sqrt(scales)

    # Create boolean mask from coi matching size and shape of power
    coiMask = np.array([np.array(periods) <= coi[i] for i in range(len(data['Alt']))]).T

    results = {
        'power': power,
        'coefU': coefU,
        'coefV': coefV,
        'coefT': coefT,
        'scales': scales,
        'wavelengths': periods,
        'signif': signif,
        'constant': constant,
        'coi': coiMask
    }

    return results  # Dictionary of wavelet-transformed surfaces


def invertWaveletTransform(region, wavelets):
    # FUNCTION PURPOSE: Invert the wavelet transformed U, V, and T in the traced region
    #
    # INPUTS:
    #   region: Boolean mask surrounding a local maximum in the power surface
    #   wavelets: Dictionary containing wavelet transformed surfaces of zonal & meridional wind and temperature
    #
    # OUTPUTS:
    #   results: Dictionary containing reconstructed time series for U, V, and T in 'region'


    # Perform the inversion, per Torrence & Compo (1998)
    uTrim = wavelets.get('coefU').copy()  # Create copy so that uTrim is not dependent on wavelets
    uTrim[np.invert(region)] = 0  # Trim U based on region
    # Sum across columns of U, then multiply by reconstruction constant
    uTrim = np.multiply(uTrim.sum(axis=0), wavelets.get('constant'))

    # Do the same with V
    vTrim = wavelets.get('coefV').copy()
    vTrim[np.invert(region)] = 0
    vTrim = np.multiply( vTrim.sum(axis=0), wavelets.get('constant') )

    # Again with T
    tTrim = wavelets.get('coefT').copy()
    tTrim[np.invert(region)] = 0
    tTrim = np.multiply( tTrim.sum(axis=0), wavelets.get('constant') )

    # Declare results in dictionary
    results = {
        'uTrim': uTrim,
        'vTrim': vTrim,
        'tTrim': tTrim
    }

    return results  # Dictionary of trimmed inverted U, V, and T


########## VARIABLE MANAGEMENT ##########

def removePeaks(region, peaks):
    # FUNCTION PURPOSE: Remove local maxima that are currently traced in 'region' from list of peaks
    #
    # INPUTS:
    #   region: Boolean mask surrounding a local maximum in the power surface
    #   peaks: List of local maxima in power surface
    #
    # OUTPUTS:
    #   peaks: Shortened list of local maxima, with local maxima in region removed

    # Remove local maxima that have already been traced from peaks list
    toRem = []  # Empty index of peaks to remove
    # Iterate through list of peaks
    for n in range(len(peaks)):
        if region[peaks[n][0], peaks[n][1]]:  # If peak in region,
            toRem.append(n)  # add peak to removal index
    # Then, remove those peaks from peaks list
    peaks = [ value for (i, value) in enumerate(peaks) if i not in set(toRem) ]

    return np.array(peaks)  # Return shortened list of peaks


def saveParametersInLoop(waves, plottingInfo, parameters, region, peaks):
    # FUNCTION PURPOSE: Set out-of-loop variables to save wave parameters and other variables local to the loop
    #
    # INPUTS:
    #   waves: Dictionary containing wave information for the current radiosonde flight
    #   plottingInfo: Dictionary keeping track of plotting information throughout successive loop iterations
    #   parameters: Dictionary of current wave parameters, could be empty if wave was determined non-physical
    #   region: Boolean mask showing the region surrounding the current local maximum
    #   peaks: List of local maxima, with peaks[0] being currently analyzed
    #
    # OUTPUTS:
    #   waves: Dictionary updated to contain the current wave's characteristics
    #   plottingInfo: Dictionary with updated mask of peak regions, color list, and wave number
    #   peaks: Shortened list of local maxima, with current peak(s) removed

    # If found, save parameters to dictionary of waves
    if parameters:

        # Copy the peak region estimate to a plotting map
        plottingInfo['regions'][region] = True

        # Set the name of the current wave
        if len(waves['waves']) == 0:
            # If there are no waves recorded, this is wave 1
            name = 'wave1'
        else:
            # Otherwise, get the maximum wave number recorded so far
            nums = [int(x[-1]) for x in waves['waves'].keys()]
            # Set the current wave number to the maximum plus one
            name = 'wave' + str(max(nums)+1)
        # Assign the parameters to that name in 'waves'
        waves['waves'][name] = parameters

        # Find similarities between the current peak and the list of peaks for plotting
        colorIndex = np.array(peaks[0] == plottingInfo.get('peaks')).sum(axis=1)
        # Where equal, set the color to red instead of blue for the output plot
        plottingInfo['colors'][np.where(colorIndex == 2)] = 'red'

    # Finally, update list of peaks that have yet to be analyzed by removing peaks defined in 'region'
    peaks = removePeaks(region, peaks)

    return waves, plottingInfo, peaks  # Return dictionaries and list of peaks


def defineWaves():
    # FUNCTION PURPOSE: Define the empty dictionary to keep track of wave parameters and flight path info
    #
    # INPUTS:
    #
    # OUTPUTS:
    #   waves: Dictionary that will keep track of all confirmed wave parameters and flight tracking information

    # Define dictionary to track waves and flight info
    waves = {
        'waves': {},  # Empty dictionary, to contain wave characteristics
        'flightPath': {  # Flight path for plotting results
            'time': [],
            'alt': []
        }
    }

    return waves


def setUpLoop(data, wavelets, peaks, waves):
    # FUNCTION PURPOSE: Define variables needed outside of the local maxima tracing & analysis loop
    #
    # INPUTS:
    #   data: Pandas DataFrame containing flight information
    #   wavelets: Dictionary containing wavelet transformed surfaces of zonal & meridional wind and temperature
    #   peaks: List of local maxima in power surface
    #   waves: Dictionary to contain wave parameters and flight path data (time and altitude)
    #
    # OUTPUTS:
    #   waves: Updated dictionary containing wave parameters and the increased flight path (for analysis plots)
    #   results: Dictionary containing a full list of local
    #               maxima, a corresponding list of colors, and a boolean mask of peak regions

    peaksToPlot = peaks.copy()  # Keep peaks for plot at end
    colorsToPlot = np.array(['blue'] * len(peaks))  # Keep track for plots at end

    # Numpy array for plotting purposes
    regionPlotter = np.zeros( wavelets.get('power').shape, dtype=bool )

    # Create index to only save 1/50 of the data for plotting, the detail isn't all needed
    step = int(len(data['Time'])/50)
    trimIndex = np.arange(0, len(data['Time']), step)

    # Update current flight info to include time and altitude of this flight section
    waves['flightPath']['time'].extend(np.array(data.iloc[trimIndex, data.columns.values == 'Time']).flatten())
    waves['flightPath']['alt'].extend( np.array(data.iloc[trimIndex, data.columns.values == 'Alt']).flatten() )

    # Define results dictionary
    results = {
        'peaks': peaksToPlot,
        'colors': colorsToPlot,
        'regions': regionPlotter
    }

    return waves, results


########## POWER SURFACE ANALYSIS ##########

def findPeaks(power, coi, waveSignif):
    # FUNCTION PURPOSE: Find the local maxima in the given power surface that are statistically significant
    #
    # INPUTS:
    #   power: Numpy 2d array containing sum of squares of wavelet transformed wind speeds
    #   coi: Boolean mask for power that is True inside the cone of influence, False outside
    #   waveSignif: Boolean mask for power that is True where power > 95% confidence interval, False elsewhere
    #
    # OUTPUTS:
    #   peaks: Numpy 2d array containing peak coordinates, e.g. [ [row1, col1], [row2, col2], ... [rowN, colN] ]


    # UI console output to keep user informed
    print("\nSearching for local maxima in power surface", end='')

    # Find and return coordinates of local maxima
    peaks = peak_local_max(power)

    # Define boolean mask based on 'waveSignif' and 'coi'
    mask = waveSignif.copy() & coi.copy()

    # Filter local maxima to significant peaks within cone of influence, per Torrence & Compo (1998)
    peaks = peaks[[mask[tuple(x)] for x in peaks]]

    print()  # Newline for next console output

    return np.array(peaks)  # List of peak coordinates


def findPeakRectangle(power, peak):
    # FUNCTION PURPOSE: Trace a rectangle around a local maximum in the power surface,
    #                   following the method from Zink & Vincent (2001), which iterates
    #                   in four directions until either 25% of peak power is reached,
    #                   or the power surface begins increasing.
    #
    # INPUTS:
    #   power: Numpy 2d array containing sum of squares of wavelet transformed wind speeds
    #   peak: Numpy array containing row and column coordinates of local maximum in power surface
    #
    # OUTPUTS:
    #   region: Boolean mask the size & shape of power that is True inside rectangle and false elsewhere

    # Create boolean mask, initialized as False
    region = np.zeros(power.shape, dtype=bool)

    # Per Zink & Vincent (2001), set the limit to 25% of peak power
    powerLimit = 0.25 * power[peak[0], peak[1]]

    # Get the row and column of the peak
    row = power[peak[0], :]
    col = power[:, peak[1]]

    # Create an array with coordinates of local minima on the row
    rowMins = np.array(argrelextrema(row, np.less))
    # Append all coordinates where the row is less than the power limit
    rowMins = np.append(rowMins, np.where(row <= powerLimit))
    # Add the peak itself, as well as the boundaries in case peak is near the edge
    rowMins = np.sort(np.append(rowMins, [0, peak[1], power.shape[1]-1]))
    # Get the two values on either side of the peak in the sorted array of indices
    cols = np.arange( rowMins[np.where(rowMins == peak[1])[0]-1],
                      rowMins[np.where(rowMins == peak[1])[0]+1] + 1).tolist()

    # Repeat for the column, to get the boundaries for the rows
    colMins = np.array(argrelextrema(col, np.less))
    colMins = np.append(colMins, np.where(col <= powerLimit))
    colMins = np.sort(np.append(colMins, [0, peak[0], power.shape[0]-1]))
    rows = np.arange(colMins[np.where(colMins == peak[0])[0] - 1][0],
                     colMins[np.where(colMins == peak[0])[0] + 1][0] + 1).tolist()

    # Set the boolean mask to true inside those boundaries
    region[np.ix_(rows, cols)] = True

    return region


def findPeakContour(power, peak):
    # FUNCTION PURPOSE: Trace a contour line around a local maximum in the power surface,
    #                   an alternative method to 'findPeakRectangle'. This method is currently
    #                   disused due to a lack of reliability, but a better method than the
    #                   rectangle algorithm should be found, as it leaves a lot to be desired.
    #
    # INPUTS:
    #   power: Numpy 2d array containing sum of squares of wavelet transformed wind speeds
    #   peak: Numpy array containing row and column coordinates of local maximum in power surface
    #
    # OUTPUTS:
    #   region: Boolean mask the size & shape of power that is True inside contour and false elsewhere

    # Create boolean mask, initialized as False
    region = np.zeros(power.shape, dtype=bool)

    # If for some reason this method can't isolate a region surrounding the peak,
    # set the peak itself to True so that it will be removed from list of peaks
    region[peak[0], peak[1]] = True

    # Find cut-off power level, based on height of peak
    # No one level works for all peaks, so iterate through different contours until one works
    relativePowerLevels = np.arange(0.55, 1.00, 0.05)  # Try levels 55%, 60%, 65%, ... 90%, 95%
    absolutePowerLevels = power[peak[0], peak[1]] * relativePowerLevels

    for level in absolutePowerLevels:

        # Find all the contours at cut-off level
        contours = find_contours(power, level)

        # Loop through contours to find the one surrounding the peak
        for contour in contours:

            # If the contour runs into multiple edges, skip as it's not worth trying
            if contour[0, 0] != contour[-1, 0] and contour[0, 1] != contour[-1, 1]:
                continue

            # Use matplotlib.path.Path to create a path
            p = mpath.Path(contour)

            # Check to see if the peak is inside the closed loop of the contour path
            if p.contains_point(peak):

                # If it is, set the boundary path to True
                region[contour[:, 0].astype(int), contour[:, 1].astype(int)] = True

                # Then fill in the contour to create mask surrounding peak
                region = binary_fill_holes(region)

                # The method is now done, so return region
                return region

    # If this method couldn't find a contour that surrounded the peak,
    # then return the boolean mask that is False except for the peak itself
    return region


########## STOKES PARAMETERS ANALYSIS ##########

def getParameters(data, wave, spatialResolution, waveAltIndex, wavelength):
    # FUNCTION PURPOSE: Get physical wave parameters based on the reconstructed time series of the potential wave
    #
    # INPUTS:
    #   data: Pandas DataFrame with time, altitude, temperature, pressure, latitude, and longitude of flight
    #   wave: Dictionary containing the reconstructed time series for zonal & meridional wind speeds and temperature
    #   spatialResolution: Height between rows in 'data', in meters
    #   waveAltIndex: Index of the altitude of the wave, taken to be the altitude at the local maximum power
    #   wavelength: Vertical wavelength, taken to be the equivalent fourier wavelength at the local maximum power
    #
    # OUTPUTS:
    #   waveProp: Dictionary of wave characteristics, to be written to JSON file for program output


    # Calculate the wind variance, defined as the sum of the power spectrums for the reconstructed U and V
    windVariance = np.abs(wave.get('uTrim')) ** 2 + np.abs(wave.get('vTrim')) ** 2

    # Filter U, V, T according to max half power - from Murphy (2014) & Zink & Vincent (2001) section 2.3 paragraph 3
    # This method is a more sophisticated algorithm to trim the wave to a single period according to max half power,
    # which avoids returning two disconnected sections and thus fulfills assumptions of Stokes parameters
    index = np.array([x[0] for x in enumerate(windVariance)])[windVariance <= 0.5 * np.max(windVariance)]
    index = np.append(index, argrelextrema(windVariance, np.less))  # Indices of all local minima & < half max power
    try:
        peakIndex = np.where(windVariance == np.max(windVariance))
        if np.array([peakIndex > index]).all() or np.array([peakIndex < index]).all():  # If the peak is on the edge,
            maxes = np.array(argrelextrema(windVariance, np.greater)).flatten()
            peakIndex = maxes[windVariance[maxes].argsort()[::-1]][1]  # then pick the second highest local maximum
        index = index - peakIndex  # Subtract peak index so that left is negative, right is positive
        # Find indices based on the closest value to zero that's either positive or negative
        leftIndex = index[index < 0]
        leftIndex = int(np.max(leftIndex) + peakIndex)
        rightIndex = index[index > 0]
        rightIndex = int(np.min(rightIndex) + peakIndex)
        # Trim U, V, T to a single period with amplitude > half max power
        uTrim = wave.get('uTrim').copy()[leftIndex:rightIndex]
        vTrim = wave.get('vTrim').copy()[leftIndex:rightIndex]
        tTrim = wave.get('tTrim').copy()[leftIndex:rightIndex]

    except (ValueError, IndexError):
        # Wave doesn't have a full period above half-max power, so it is rejected
        return {}


    # Separate imaginary/real components of U and V
    vHilbert = vTrim.copy().imag
    uvMatrix = [uTrim.copy(), vTrim.copy()]  # Combine into a matrix for easy rotation along propagation direction
    uTrim = uTrim.real
    vTrim = vTrim.real

    # Stokes parameters from Murphy (2014) appendix A and Eckerman (1996) equations 1-5
    I = np.mean(uTrim ** 2) + np.mean(vTrim ** 2)
    D = np.mean(uTrim ** 2) - np.mean(vTrim ** 2)
    P = np.mean(2 * uTrim * vTrim)
    Q = np.mean(2 * uTrim * vHilbert)
    degPolar = np.sqrt(D ** 2 + P ** 2 + Q ** 2) / I


    # First checks to rule out wave candidates that are non-physical, from Murphy (2014) section 2 paragraph 3
    if np.abs(P) < 0.05 or np.abs(Q) < 0.05 or degPolar < 0.5 or degPolar > 1.0:
        return {}


    # Find the angle of propagation (unit circle, not compass), from Vincent & Fritts (1987) Equation 15
    theta = 0.5 * np.arctan2(P, D)  # arctan2 has a range of [-pi, pi], as opposed to arctan's range of [-pi/2, pi/2]


    # Rotate by -theta so that u and v components of 'uvMatrix' are parallel/perpendicular to propogation direction
    rotate = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]  # Inverse of the rotation matrix
    uvMatrix = np.dot(rotate, uvMatrix)

    # From Murphy (2014) table 1, and Zink & Vincent (2001) equation A10
    axialRatio = np.linalg.norm(uvMatrix[0]) / np.linalg.norm(uvMatrix[1])
    # Alternative method that yields similar results (from Neelakantan et. al, 2019, Equation 8) is:
    # axialRatio = np.abs(1 / np.tan(0.5 * np.arcsin(Q / (degPolar * I))))

    # From Murphy (2014), citing Zink (2000) equation 3.17
    # This is the coherence function, measuring the coherence of U|| and T
    gamma = np.mean(uvMatrix[0] * np.conj(tTrim)) / \
            np.sqrt(np.mean(np.abs(uvMatrix[0]) ** 2) * np.mean(np.abs(tTrim) ** 2))
    # This comes from Marlton (2016) equation 2.5, but the coherence function performs better w/ our complex data
    # gamma = np.mean( uvMatrix[0].real * np.gradient(tTrim.real, spatialResolution) )

    # Check for the relative phase between U|| and T from Murphy (2014) Section 2 Paragraph 3
    # For a gravity wave, |angle| ~ 90 degrees, so remove waves that don't match
    if not 20 < np.abs(np.angle(gamma, deg=True)) < 160:
        # Need to determine exact parameters, Murphy uses 5 -- 175, cites Moffet (2011) which uses 20 -- 160 (par. 14)
        return {}

    # The phase of the coherence gives the phase shift between U|| and T
    # If the phase shift is negative, wave is propagating anti-parallel, so reverse the angle
    if np.angle(gamma) < 0:
        theta = theta + np.pi

    # Coriolis frequency, typically negative in the southern hemisphere (Murphy 2014 section 3.2 paragraph 1)
    # Equation from AMS (https://glossary.ametsoc.org/wiki/Coriolis_parameter)
    # However, we're taking the absolute value, yielding a positive number, which means that our intrinsic frequency
    # will be positive, regardless of hemisphere
    coriolisF = abs( 2 * 7.2921 * 10 ** (-5) * np.sin(data.iloc[waveAltIndex, data.columns.get_loc('Lat.')] * np.pi / 180) )

    # Intrinsic frequency, from Murphy (2014) table 1
    intrinsicF = coriolisF * axialRatio

    # Potential temperature, from AMS (https://glossary.ametsoc.org/wiki/Potential_temperature)
    potentialTemp = (1000.0 ** (2 / 7)) * (data['T'] + 273.15) / (data['P'] ** (2 / 7))  # kelvin

    # Brunt-Vaisala frequency squared, this equation assumes dry air, which is generally true @ altitude > ~16 km
    bvFreq2 = 9.81 / potentialTemp * np.gradient(potentialTemp, spatialResolution)  # Brunt-vaisala frequency squared
    # This is the most common equation w/ the most reasonable assumption, used by Wikipedia, MetPy, and other papers.
    # However, other equations that make slightly different assumptions and yield very different results are:
    # bvFreq2 = - 9.81 / data['D'] * np.gradient(data['D'], spatialResolution)
    # bvFreq2 = 9.81 / np.mean(pt) * np.gradient(pt, spatialResolution)
    # bvFreq2 = - 9.81 * data['T'] / (298.15) * np.gradient(np.log(pt), spatialResolution)
    # bvFreq2 = 9.81 / data['Virt.Temp'] * (np.gradient(data['Virt.Temp'], spatialResolution) - 9.8/1000)

    # Take the mean Brunt-Vaisala frequency across the filtered wave packet. Because of the massive fluctuations in
    # value, this yields more accurate results than taking the single value at the wave peak -- (find source for this)
    bvMean2 = np.mean(np.array(bvFreq2)[leftIndex:rightIndex])

    # Ensure that the intrinsic frequency of the wave is between the two theoretical bounds
    if not np.sqrt(bvMean2) > intrinsicF > coriolisF:
        return {}  # If not, reject the wave candidate as non-physical


    # Values that I should calculate and output are:
    # Intrinsic frequency
    # Ground based frequency
    # Periods for above frequencies
    # Propagation direction
    # Altitude
    # Horizontal phase speed
    # Vertical wavelength
    # (See Murphy (2014) table 3 for ground-based)


    # Wave parameters, see Murphy (2014) table 2 and Appendix B

    # Vertical wavenumber [1/m]
    m = 2 * np.pi / wavelength
    # Horizontal wavenumber [1/m]
    kh = np.sqrt((m ** 2 / bvMean2) * (intrinsicF ** 2 - coriolisF ** 2))  # Murphy (2014) Eqn B2
    # Intrinsic vertical wave velocity [m/s]
    intrinsicVerticalGroupVel = - (intrinsicF ** 2 - coriolisF ** 2) / (intrinsicF * m)  # Murphy (2014) Eqn B5

    #zonalWaveNumber = kh * np.sin(theta)  # [1/m]

    #meridionalWaveNumber = kh * np.cos(theta)  # [1/m]

    intrinsicVerticalPhaseSpeed = intrinsicF / m  # [m/s]

    intrinsicHorizPhaseSpeed = intrinsicF / kh  # [m/s]

    intrinsicZonalGroupVel = kh * np.sin(theta) * bvMean2 / (intrinsicF * m ** 2)  # [m/s]

    intrinsicMeridionalGroupVel = kh * np.cos(theta) * bvMean2 / (intrinsicF * m ** 2)  # [m/s]

    intrinsicHorizGroupVel = np.sqrt(intrinsicZonalGroupVel ** 2 + intrinsicMeridionalGroupVel ** 2)  # [m/s]
    # Horizontal wavelength [m]
    lambda_h = 2 * np.pi / kh
    # Altitude of wave peak
    altitudeOfDetection = data['Alt'][waveAltIndex]
    # Get latitude at index
    latitudeOfDetection = data['Lat.'][waveAltIndex]
    # Get longitude at index
    longitudeOfDetection = data['Long.'][waveAltIndex]
    # Get flight time at index
    timeOfDetection = data['Time'][waveAltIndex]

    # More tests to check that our basic assumptions are being satisfied, from Jie Gong at NASA
    # Make sure that the vertical wavelength < (delta z)/2, the theoretical maximum vertical wavelength for this method
    if m > (data['Alt'][data.shape[0]-1] - data['Alt'][0]):
        return {}  # If not, wave candidate is rejected

    # Make sure that horizontal wavelength >> balloon drift distance so that our assumption of a vertical profile holds
    # Unit conversion between lat-lon and km comes from:
    # https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance
    # Should find a peer-reviewed source eventually... check with Carl?
    # The methodology also is an overestimate, but because of the >> that works for our purposes
    if lambda_h / 1000 < np.sqrt(( (max(data['Lat.']) - min(data['Lat.'])) * 110.574) ** 2
                + ( (max(data['Long.']) - min(data['Long.'])) * 111.320*np.cos(latitudeOfDetection*np.pi/180) ) ** 2):
        return {}  # If horizontal wavelength is too short, wave candidate is rejected


    # Assemble all relevant wave properties into dictionary
    waveProp = {
        'Altitude [km]': altitudeOfDetection / 1000,
        'Latitude [deg]': latitudeOfDetection,
        'Longitude [deg]': longitudeOfDetection,
        'Date and Time [UTC]': timeOfDetection,
        'Vertical wavelength [km]': (2 * np.pi / m) / 1000,
        'Horizontal wavelength [km]': lambda_h / 1000,
        'Propagation direction [deg N from E]': theta * 180 / np.pi,
        'Axial ratio [no units]': axialRatio,
        'Intrinsic vertical group velocity [m/s]': intrinsicVerticalGroupVel,
        'Intrinsic horizontal group velocity [m/s]': intrinsicHorizGroupVel,
        'Intrinsic vertical phase speed [m/s]': intrinsicVerticalPhaseSpeed,
        'Intrinsic horizontal phase speed [m/s]': intrinsicHorizPhaseSpeed,
        'Degree of Polarization [no units]': degPolar
    }

    return waveProp  # Dictionary of wave characteristics
