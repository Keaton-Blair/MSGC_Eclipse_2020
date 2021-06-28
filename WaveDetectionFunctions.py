########## IMPORT LIBRARIES AND FUNCTIONS ##########

import numpy as np  # Numbers (like pi) and math
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import matplotlib.pyplot as plt  # Easy plotting
import matplotlib.path as mpath  # Used for finding the peak region
import pandas as pd  # Convenient data formatting, and who doesn't want pandas
import os  # File reading and input
from io import StringIO  # Used to run strings through input/output functions
from TorrenceCompoWavelets import wavelet as continuousWaveletTransform  # Torrence & Compo (1998) wavelet analysis code
from skimage.feature import peak_local_max  # Find local maxima in power surface
import datetime  # Turning time into dates
from skimage.measure import find_contours  # Find contour levels around local max
from scipy.ndimage.morphology import binary_fill_holes  # Then fill in those contour levels
from scipy.signal import argrelextrema  # Find one-dimensional local min, for peak rectangle method
import json  # Used to save wave parameters to json file
# from mpl_toolkits.basemap import Basemap  # For mapping with balloon flight


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
        print("Path to output data: "+savePath+"/\n")
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
    #
    # OUTPUTS: None

    # Find the index of data in dataList
    currentSection = [i for i in range(len(dataList)) if dataList[i].equals(data)]

    # Print progress to the console, beginning with carriage return (\r) and ending without newline
    print("\rTracing and analyzing section " + str(currentSection[0] + 1) + "/" + str(len(dataList)) +
            " peak " + str(length - len(peaks) + 1) + "/" + str(length), end='')


def outputWaveParameters(userInput, waves, fileName):
    # FUNCTION PURPOSE: Save or print final wave parameters from finished analysis
    #
    # INPUTS:
    #   userInput: Dictionary containing user input, especially data saving information
    #   waves: Dictionary containing final wave parameters from the completed analysis
    #   fileName: String with the name of the profile currently being analyzed
    #
    # OUTPUTS: Either save a file according to the user input save path, or print it to the console

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
        print(json.dumps(waves['waves'], indent=4, default=str))


def drawPowerSurface(userInput, fileName, wavelets, altitudes, plotter, peaksToPlot, colorsToPlot):
    # FUNCTION PURPOSE: Create a power surface showing local maxima and their outlines
    #
    # INPUTS:
    #   userInput: Dictionary containing whether to save/show the plots, as well as a save path
    #   fileName: String, name of the profile file currently being analyzed
    #   wavelets: Dictionary containing power surface and corresponding wavelengths
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

    # Get the vertical wavelengths for the Y coordinates
    yScale = wavelets.get('wavelengths')
    #yScaleShort = yScale[yScale <= max(wavelets.get('coi'))]  # EXPERIMENTAL!! -- DOESN'T WORK, need power surface to match scale...
    #powerSurf = wavelets.get('power')[yScaleShort, :]
    # Contourf is a filled contour, which is the easiest tool to plot a colored surface
    # Levels is set to 50 to make it nearly continuous, which takes a while,
    # but looks good and handles the non-uniform yScale, which plt.imshow() does not
    # plt.contourf(altitudes, yScale, wavelets.get('power'), levels=50)
    # NOTE -- this is experimental code for dealing with plotting issues, not permanent!
    temp = np.log(wavelets.get('power'))
    for i, _ in enumerate(altitudes):
        temp[yScale > wavelets.get('coi')[i], i] = np.NaN

    temp[temp < 0] = 0
    plt.contourf(altitudes, yScale, temp, levels=50, cmap=plt.get_cmap('turbo'))
    #x,y = np.meshgrid(altitudes, yScale)
    #plt.plot_surface(x, y, wavelets.get('power'), linewidth=0, antialiased=False)
    # Create a colorbar for the z scale
    cb = plt.colorbar()
    # Plot the outlines of the local maxima, contour is an easy way to outline a mask
    # The 'plotter' is a boolean mask, so levels is set to 0.5 to be between 0 and 1
    plt.contour(altitudes, yScale, plotter, colors='red', levels=[0.5])
    # Make a scatter plot of the identified peaks, coloring them according to which ones were confirmed as waves
    if len(peaksToPlot) > 0:
        plt.scatter(altitudes[peaksToPlot.T[1]], yScale[peaksToPlot.T[0]], c=colorsToPlot, marker='.')
    # Plot the cone of influence in black
    plt.plot(altitudes, wavelets.get('coi'), color='black')
    # Set the axis scales, labels, and titles
    plt.yscale("log")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Vertical Wavelength [m]")
    plt.ylim(yScale[0], yScale[-1])
    plt.title("Power surface, including traced peaks")
    cb.set_label("Power [m^2/s^2]")

    # Save and/or show the plot, according to user input.
    if userInput.get('saveData'):
        # Get current filenames in saving directory
        filenames = os.listdir(userInput.get('savePath'))
        # Filter for the current flight
        sections = [x for x in filenames if x.find(fileName[0:-4]+"_power_surface") >= 0]
        # Save file using the correct flight section number
        plt.savefig(userInput.get('savePath') + "/" + fileName[0:-4] +
                    "_power_surface_section_" + str(len(sections) + 1) + ".png")

    if userInput.get('showPlots'):
        plt.show()

    plt.close()

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
    # FUNCTION PURPOSE: Get user input to compare results from two methods based on their hodographs
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
    f = open(os.path.join(path, file), 'r')
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
    data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]

    # Get rid of extraneous columns that won't be used for further analysis
    essentialData = ['Time', 'Alt', 'T', 'P', 'Ws', 'Wd', 'Lat.', 'Long.']
    data = data[essentialData]

    return data  # return cleaned pandas data frame


def readFromData(file, path):
    # FUNCTION PURPOSE: Find launch time and pbl height in a profile file, or return default
    #                   values if not found. In particular, PBL height has to be written into
    #                   the profile by hand or by running companion software (CalculatePBL.py)
    #
    # INPUTS:
    #   file: The filename of the data file to read
    #   path: The path (absolute or relative) to the file
    #
    # OUTPUTS:
    #   launchDateTime: datetime.datetime object containing the UTC date and time of launch
    #   pblHeight: Number in meters representing PBL height



    # Establish default values, in case not contained in profile
    launchDateTime = datetime.datetime.now()
    pblHeight = 1500

    # Open and investigate the file
    f = open(os.path.join(path, file), 'r')
    for line in f:  # Iterate through file, line by line

        # If line has expected beginning, try to get datetime from file
        if line.rstrip() == "Flight Information:":
            # noinspection PyBroadException
            try:
                dateTimeInfo = f.readline().split()
                dateTimeInfo = ' '.join(dateTimeInfo[2:6] + [dateTimeInfo[8]])
                launchDateTime = datetime.datetime.strptime(dateTimeInfo, '%A, %d %B %Y %H:%M:%S')
            except:
                # If an error is encountered, print a statement to the console and continue
                print("Error reading flight time info, defaulting to present")

        # If line has expected beginning, try to get PBL info
        if line.rstrip() == "PBL Information:":
            # noinspection PyBroadException
            try:
                pblHeight = float(f.readline().split("\t")[3])
            except:
                # If an error is encountered, print a statement to the console and continue
                print("Error reading flight PBL info, defaulting to 1500 meters")

    f.close()  # Need to close opened file

    # Return values from profile, or default values if not found
    return launchDateTime, pblHeight


def interpolateData(data, spatialResolution, pblHeight, launchDateTime):
    # FUNCTION PURPOSE: Interpolate to create a Pandas DataFrame for the flight as a uniform
    #                   spatial grid, with datetime.datetime objects in the time column
    #
    # INPUTS:
    #   data: Pandas DataFrame containing flight information
    #   spatialResolution: Desired length (in meters) between rows of data, must be a positive integer
    #   pblHeight: The height above ground in meters of the PBL
    #   launchDateTime: A datetime.datetime object containing the launch date and time in UTC
    #
    # OUTPUTS:
    #   data: List of Pandas DataFrames containing the time [s], altitude [m], temperature [deg C],
    #           pressure [hPa], wind speed [m/s], wind direction [deg], latitude [decimal deg],
    #           and longitude [decimal deg] of the radiosonde flight


    # First, filter data to remove sub-PBL data
    data = data[ (data['Alt'] - data['Alt'][0]) >= pblHeight]

    # Now, interpolate to create spatial grid, not temporal

    # Create index of heights with 1 meter spatial resolution
    heightIndex = pd.DataFrame({'Alt': np.arange(min(data['Alt']), max(data['Alt']))})
    # Right merge data with index to keeping all heights
    data = pd.merge(data, heightIndex, how="right", on="Alt")
    # Sort data by height for interpolation
    data = data.sort_values(by=['Alt'])

    # Use pandas built in interpolate function to fill in NAs
    # Linear interpolation appears the most trustworthy, but more testing could be done
    missingDataLimit = 999  # If 1 km or more missing data in a row, leave the NAs in place
    data = data.interpolate(method="linear", limit=missingDataLimit)

    # If NA's remain, missingDataLimit was exceeded
    # Get the list of rows with null data remaining
    nullIndices = [i for i in range(data.shape[0]) if data.isnull().values[i].any()]
    # Concatenate list with incremented one for edge parameters of numpy.split()
    nullIndices = nullIndices + [i+1 for i in nullIndices]
    nullIndices.sort()
    # Split data along sections with null values
    data = [x for x in np.split(data, nullIndices) if x.shape[0] > missingDataLimit and not x.isnull().values.any()]

    # For the section above, check whether it's faster to skip the two middle lines, then do np.split(data, nullIndices)
    # and remove all of the individual NA values via iteration or list comprehension

    # If data was split into multiple sections, inform user
    if len(data) > 1:
        print("Found more than " + str(missingDataLimit) + " meters of consecutive missing data.")
        print("Split data into " + str(len(data)) + " separate sections for analysis.")

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

def waveletTransform(data, spatialResolution, wavelet):
    # FUNCTION PURPOSE: Perform the continuous wavelet transform on wind speed components and temperature
    #
    # INPUTS:
    #   data: Pandas DataFrame containing flight information
    #   spatialResolution: Length (in meters) between rows of data
    #   wavelet: String containing the name of the wavelet to use for the transformation. Based on
    #               Zink & Vincent (2001) and Murphy et. al (2014), this should be 'MORLET'
    #
    # OUTPUTS:
    #   results: Dictionary containing the power surface (|U|^2 + |V|^2), the wavelet transformed
    #               surfaces U, V, and T (zonal wind speed, meridional wind speed, and temperature
    #               in celsius), the wavelet scales and their corresponding fourier wavelengths,
    #               the cone of influence and the reconstruction constant from Torrence & Compo (1998)


    # u and v (zonal & meridional) components of wind speed
    u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
    v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)
    t = data['T']


    # In preparation for wavelet transformation, define variables
    # From Torrence & Compo (1998)
    padding = 1  # Pad the data with zeros to allow convolution to edge of data
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
                                                             s0=smallestScale, mother=wavelet)

    print("\rPerforming wavelet transform on V... (2/3)", end='')  # Update to keep user informed
    coefV, periods, scales, coi = continuousWaveletTransform(v, spatialResolution, pad=padding, dj=scaleResolution,
                                                             s0=smallestScale, mother=wavelet)

    print("\rPerforming wavelet transform on T... (3/3)", end='')  # Final console output for wavelet transform
    coefT, periods, scales, coi = continuousWaveletTransform(t, spatialResolution, pad=padding, dj=scaleResolution,
                                                             s0=smallestScale, mother=wavelet)


    # Power surface is sum of squares of u and v wavelet transformed surfaces
    power = abs(coefU) ** 2 + abs(coefV) ** 2  # abs() gets magnitude of complex number

    # Divide each column by sqrt of the scales so that it doesn't need to be done later to invert wavelet transform
    for col in range(coefU.shape[1]):
        coefU[:, col] = coefU[:, col] / np.sqrt(scales)
        coefV[:, col] = coefV[:, col] / np.sqrt(scales)
        coefT[:, col] = coefT[:, col] / np.sqrt(scales)

    results = {
        'power': power,
        'coefU': coefU,
        'coefV': coefV,
        'coefT': coefT,
        'scales': scales,
        'wavelengths': periods,
        'constant': constant,
        'coi': coi[0:len(data['Alt'])]  # Fix COI so that it has the same length as data
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

def filterPeaksCOI(wavelets, peaks):
    # FUNCTION PURPOSE: Remove local maxima that are outside the cone of influence
    #
    # INPUTS:
    #   wavelets: Dictionary containing wavelet transformation output, including COI
    #   peaks: List of local maxima in power surface
    #
    # OUTPUTS:
    #   peaks: Shortened list of local maxima, with local maxima outside COI removed

    # First, define an empty boolean mask
    peakRemovalMask = np.zeros(wavelets.get('power').shape, dtype=bool)

    # For each peak, if the peak is outside COI, set the mask to True
    for peak in peaks:
        if wavelets.get('wavelengths')[peak[0]] > wavelets.get('coi')[peak[1]]:
            peakRemovalMask[peak[0], peak[1]] = True

    # Then, pass the mask to the standard peak removal function
    peaks = removePeaks(peakRemovalMask, peaks)

    # Return shortened local maxima list
    return peaks


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


    # Temporary plotting changes for 6/25 meeting, delete later!!
    if parameters and 'check' in parameters.keys():
        parameters = {}
        # Find similarities between the current peak and the list of peaks for plotting
        colorIndex = np.array(peaks[0] == plottingInfo.get('peaks')).sum(axis=1)
        # Where equal, set the color to red instead of blue for the output plot
        plottingInfo['colors'][np.where(colorIndex == 2)] = '#2F2'

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
    # Create function header in comments eventually

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
    # FUNCTION PURPOSE: Define variables needed outside of the local maxima tracing/analysis loop
    #
    # INPUTS:
    #   data: Pandas DataFrame containing flight information
    #   wavelets: Dictionary containing wavelet transformed surfaces of zonal & meridional wind and temperature
    #   peaks: List of local maxima in power surface
    #   waves: Add description later...
    #
    # OUTPUTS:
    #   waves: Dictionary to contain wave parameters and the flight path (for analysis plots)
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


    results = {
        'peaks': peaksToPlot,
        'colors': colorsToPlot,
        'regions': regionPlotter
    }

    return waves, results


########## POWER SURFACE ANALYSIS ##########

def findPeaks(power):
    # FUNCTION PURPOSE: Find the local maxima in the give power surface
    #
    # INPUTS:
    #   power: Numpy 2d array containing sum of squares of wavelet transformed wind speeds
    #
    # OUTPUTS:
    #   peaks: Numpy 2d array containing peak coordinates, e.g. [ [row1, col1], [row2, col2], ... [rowN, colN] ]


    # UI console output to keep user informed
    print("\nSearching for local maxima in power surface", end='')

    # Find and return coordinates of local maximums
    cutOff = 300  # Disregard maximums less than this m^2/s^2, empirically determined via trial & error
    # Finds local maxima based on cutOff, margin
    peaks = peak_local_max(power, threshold_abs=cutOff)

    print()  # Newline for next console output

    return np.array(peaks)  # Array of coordinate arrays


def findPeakRectangle(power, peak):
    # FUNCTION PURPOSE: Trace a rectangle around a local maximum in the power surface,
    #                   following the method of Zink & Vincent (2001), which iterates
    #                   in four directions until either 25% of peak power is reached,
    #                   of the power surface begins increasing.
    #
    # INPUTS:
    #   power: Numpy 2d array containing sum of squares of wavelet transformed wind speeds
    #   peak: Numpy array containing row and column coordinates of local maximum in power surface
    #
    # OUTPUTS:
    #   region: Boolean mask the size & shape of power that is True inside rectangle and false elsewhere

    # Create boolean mask, initialized as False
    region = np.zeros(power.shape, dtype=bool)

    # Per Zink & Vincent (2001), the limit is 25% of peak power
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
    #                   possibly following Murphy (2014). The paper is unclear, and I still
    #                   need to investigate the IDL code to find the exact method.
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
    #   wave: Dictionary containing the reconstructed time series for zonal & meridional wind speed and temperature
    #   spatialResolution: Height between rows in 'data', in meters
    #   waveAltIndex: Index of the altitude of the wave, taken to be the altitude at the local maximum power
    #   wavelength: Vertical wavelength, taken to be the equivalent fourier wavelength at the local maximum power
    #
    # OUTPUTS:
    #   waveProp: Dictionary of wave characteristics,


    # Calculate the wind variance of the wave
    windVariance = np.abs(wave.get('uTrim')) ** 2 + np.abs(wave.get('vTrim')) ** 2

    # This code is for testing currently, see commented plotting below ... method to be approved during meeting on 25th!
    i = np.array([x[0] for x in enumerate(windVariance)])[windVariance <= 0.5 * np.max(windVariance)]
    i = np.append(i, argrelextrema(windVariance, np.less))
    i = np.append(i, [0, len(windVariance)-1])
    peakIndex = np.where(windVariance == np.max(windVariance))
    i = i - peakIndex
    if np.max(windVariance) == windVariance[-1]:
        i2 = len(windVariance) - 1
    else:
        i2 = i[i > 0]
        i2 = int(np.min(i2) + peakIndex)
    if np.max(windVariance) == windVariance[0]:
        i1 = 0
    else:
        i1 = i[i < 0]
        i1 = int(np.max(i1) + peakIndex)
    uTrim = wave.get('uTrim').copy()[i1:i2]
    vTrim = wave.get('vTrim').copy()[i1:i2]
    tTrim = wave.get('tTrim').copy()[i1:i2]
    """ Commented code to plot the new method, make sure we all agree on the method before I finalize this
        -- Almost done, just need to tweak the algorithm to avoid edge cases, how???
    """
    index = [x[0] for x in enumerate(windVariance)]
    plt.plot(index, windVariance)
    plt.plot(index, [0.5 * np.max(windVariance)] * len(index))
    plt.plot([i1] * len(windVariance), windVariance, 'green')
    plt.plot([i2] * len(windVariance), windVariance, 'green')
    # plt.show()
    plt.close()


    # Get rid of values below max half-power, per Zink & Vincent (2001) section 2.3 paragraph 3
    # uTrim = wave.get('uTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]
    # vTrim = wave.get('vTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]
    # tTrim = wave.get('tTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]

    # Separate imaginary/real parts
    vHilbert = vTrim.copy().imag
    uvComp = [uTrim.copy(), vTrim.copy()]  # Combine into a matrix for easy rotation along propagation direction
    uTrim = uTrim.real
    vTrim = vTrim.real

    # Potential temperature, needs to be appropriately sourced and verified
    pt = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286)  # kelvin

    # Stokes parameters from Murphy (2014) appendix A and Eckerman (1996) equations 1-5
    I = np.mean(uTrim ** 2) + np.mean(vTrim ** 2)
    D = np.mean(uTrim ** 2) - np.mean(vTrim ** 2)
    P = np.mean(2 * uTrim * vTrim)
    Q = np.mean(2 * uTrim * vHilbert)
    degPolar = np.sqrt(D ** 2 + P ** 2 + Q ** 2) / I
    # Check the variance to perform additional filtering

    # Tests to rule out waves that don't make sense. These restrictions seem fairly lax, so we should look at others
    # From Murphy (2014) section 2 paragraph 3
    if np.abs(P) < 0.05 or np.abs(Q) < 0.05 or degPolar < 0.5 or degPolar > 1.0:
        return {}

    # Find the angle of propagation, from Vincent & Fritts (1987) Equation 15
    theta = 0.5 * np.arctan2(P, D)  # arctan2 has a range of [-pi, pi], as opposed to arctan's range of [-pi/2, pi/2]


    # Rotate by theta so that u and v components of 'uvComp' are parallel/perpendicular to propogation direction
    rotate = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    uvComp = np.dot(rotate, uvComp)

    # From Murphy (2014) table 1, and Zink & Vincent (2001) equation A10
    axialRatio = np.linalg.norm(uvComp[0]) / np.linalg.norm(uvComp[1])
    # Alternative method that yields similar results (from Neelakantan et. al, 2019, Equation 8) is:
    # axialRatio = np.abs(1 / np.tan(0.5 * np.arcsin(Q / (degPolar * I))))

    # This equation is from Tom's code, but needs to be sourced because it gives different results than below
    # gamma = np.mean(uvComp[0] * np.conj(tTrim)) /
    #                           np.sqrt(np.mean(np.abs(uvComp[0]) ** 2) * np.mean(np.abs(tTrim) ** 2))

    # This comes from Marlton (2016) equation 2.5
    gamma = np.mean( uvComp[0] * np.gradient(tTrim, spatialResolution) )
    if np.angle(gamma) < 0:
        theta = theta + np.pi

    # Coriolis frequency, negative in the southern hemisphere (Murphy 2014 section 3.2 paragraph 1)
    # Equation given by wikipedia (https://en.wikipedia.org/wiki/Coriolis_frequency), but I should
    # get a peer reviewed source to verify the equation.
    # We're taking the absolute value, yielding a positive number, which shouldn't matter because
    # all the applications are coriolisF**2, but we decided as a team to do it this way.
    coriolisF = abs( 2 * 7.2921 * 10 ** (-5) * np.sin(data.iloc[waveAltIndex, data.columns.get_loc('Lat.')] * np.pi / 180) )

    # Intrinsic frequency, from Murphy (2014) table 1
    intrinsicF = coriolisF * axialRatio

    # From Zink & Vincent (2001), N^2 = - g/p0 * np.gradient(p0/Alt), where p0 is density
    # According to wikipedia (https://en.wikipedia.org/wiki/Brunt%E2%80%93V%C3%A4is%C3%A4l%C3%A4_frequency),
    # the above equation is for water, and the atmospheric equation below is correct. However, this needs to be
    # verified with peer reviewed sources to confirm
    # Fix to use eqn from https://glossary.ametsoc.org/wiki/Brunt-vaisala_frequency
    bvF2 = np.abs( 9.81 / pt * np.gradient(pt, spatialResolution) )  # Brunt-vaisala frequency squared

    # This code finds the mean across wave region: bvMean = np.mean(np.array(bvF2)[np.nonzero(region.sum(axis=0))])
    # However, my current code uses the Brunt-vaisala frequency squared at the wave altitude instead,
    # which is a departure from Murphy (2014), but which I defend by claiming that finding the BV frequency,
    # altitude, longitude, latitude, etc. at the strongest wave resemblance in our data (the power surface
    # maximum) is a better method for dealing with asymmetrical peaks, where the radiosonde was still in
    # contact with the wave for a while after capturing the best data, leading to a skewed hump shape in
    # the power surface. Finding the mean assumes that the data across the whole peak is all equally valid,
    # which I don't think is justified based on the appearance of many power surfaces.
    bvPeak = np.array(bvF2)[waveAltIndex]

    # Check that the axial ratio is positive, and that the intrinsic frequency is less than Brunt Vaisala
    if not np.sqrt(bvPeak) > intrinsicF > coriolisF:
        return {}


    # Values that I should calculate and output are:
    # Intrinsic frequency
    # Ground based frequency
    # Periods for above frequencies
    # Propagation direction
    # Altitude
    # Horizontal phase speed
    # Vertical wavelength
    # (See Murphy (2014) table 3 for ground-based)


    # Intrinsic values from Murphy (2014) table 2

    # Vertical wavenumber [1/m]
    m = 2 * np.pi / wavelength
    # Horizontal wavenumber [1/m]
    kh = np.sqrt(((coriolisF ** 2 * m ** 2) / bvPeak) * (intrinsicF ** 2 / coriolisF ** 2 - 1))  # Murphy (2014) Eqn B2
    # Intrinsic vertical wave velocity [m/s]
    intrinsicVerticalGroupVel = - (1 / (intrinsicF * m)) * (intrinsicF ** 2 - coriolisF ** 2)  # Murphy (2014) Eqn B5

    #zonalWaveNumber = kh * np.sin(theta)  # [1/m]

    #meridionalWaveNumber = kh * np.cos(theta)  # [1/m]

    intrinsicVerticalPhaseSpeed = intrinsicF / m  # [m/s]

    intrinsicHorizPhaseSpeed = intrinsicF / kh  # [m/s]

    intrinsicZonalGroupVel = kh * np.sin(theta) * bvPeak / (intrinsicF * m ** 2)  # [m/s]

    intrinsicMeridionalGroupVel = kh * np.cos(theta) * bvPeak / (intrinsicF * m ** 2)  # [m/s]

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
    # Look into this, find justification or refutation...
    if m > (data['Alt'][data.shape[0]-1] - data['Alt'][0]):  # Make sure that the vertical wavelength < (delta z)/2
        print("2nd check")
        return {'check':2}
    # Make sure that the horizontal wavelength >> balloon drift distance
    # Unit conversion comes from https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance
    # Should find a peer-edited source eventually... check with Carl
    # The methodology also isn't entirely accurate, but because of the >> we just need a rough estimate for now
    if lambda_h / 1000 < np.sqrt(( (max(data['Lat.']) - min(data['Lat.'])) * 110.574) ** 2
                + ( (max(data['Long.']) - min(data['Long.'])) * 111.320*np.cos(latitudeOfDetection*np.pi/180) ) ** 2):

        # Commented code to check errors within method, appears to be working fine!
        """
        print("3rd check")
        print(lambda_h / 1000)

        # Balloon trajectory plot code, here just in case I need it in the next week...
        m = Basemap(projection='lcc', resolution='h',
                    width=8E5, height=8E5,
                    lat_0=-40, lon_0=-70, )
        m.shadedrelief()
        m.drawcoastlines()
        m.drawcountries()

        # Map (long, lat) to (x, y) for plotting
        m.plot(data['Long.'], data['Lat.'], latlon=True, c='red')
        m.scatter(max(data['Long.']), max(data['Lat.']), latlon=True, c='red')
        m.scatter(min(data['Long.']), min(data['Lat.']), latlon=True, c='red')
        plt.show()
        """

        return {'check':3}


    # Assemble wave properties into dictionary
    waveProp = {
        'Altitude [km]': altitudeOfDetection / 1000,
        'Latitude [deg]': latitudeOfDetection,
        'Longitude [deg]': longitudeOfDetection,
        'Date and Time [UTC]': timeOfDetection,
        'Vertical wavelength [km]': (2 * np.pi / m) / 1000,
        'Horizontal wavelength [km]': lambda_h / 1000,
        'Propagation direction [deg]': theta * 180 / np.pi,
        'Axial ratio [no units]': axialRatio,
        'Intrinsic vertical group velocity [m/s]': intrinsicVerticalGroupVel,
        'Intrinsic horizontal group velocity [m/s]': intrinsicHorizGroupVel,
        'Intrinsic vertical phase speed [m/s]': intrinsicVerticalPhaseSpeed,
        'Intrinsic horizontal phase speed [m/s]': intrinsicHorizPhaseSpeed,
        'Degree of Polarization [no units]': degPolar
    }

    return waveProp  # Dictionary of wave characteristics
