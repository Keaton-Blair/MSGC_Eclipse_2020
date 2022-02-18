# Import all functions from 'WaveDetectionFunctions.py', the file which holds functions for this program
from WaveDetectionFunctions import *
# Import os to iterate through files in the given directory
import os


# First, get applicable user input.
userInput = getAllUserInput()

# Then, iterate over files in data directory, performing analysis on each one
for file in os.listdir( userInput.get('dataSource') ):

    # Import and clean the data, given the file path
    data = cleanData( file, userInput.get('dataSource') )

    # If nothing was returned, file is not recognized as a GRAWMET profile,
    # so skip ahead to the next loop iteration
    if data.empty:
        continue

    # Get the launch time and tropopause altitude from the profile header
    launchDateTime, tropopause = readFromData( file, userInput.get('dataSource') )

    # Set the height in between data points, currently 5m because it's the nominal data acquisition rate
    spatialResolution = 5  # meters, must be pos integer

    # Interpolate to create a uniform spatial grid of data, rather than temporal
    dataList = interpolateData( data, spatialResolution, tropopause, launchDateTime )

    # If nothing was returned, file was missing too much data,
    # so skip ahead to the next loop iteration
    if len(dataList) == 0:
        continue

    # Otherwise, loop through each separate section of data to perform analysis
    waves = defineWaves()  # Initialize wave characteristic dictionary outside of loop
    for data in dataList:

        # Perform the continuous wavelet transform to get the power surface
        wavelets = waveletTransform( data, spatialResolution, 'MORLET')  # Use the morlet wavelet

        # Find local maxima in power surface, based on cone of influence and significance levels
        peaks = findPeaks( wavelets.get('power'), wavelets.get('coi'), wavelets.get('signif') )

        # Define the needed variables, outside of the local max. loop
        waves, plottingInfo = setUpLoop(data, wavelets, peaks, waves)

        # Iterate over local maxima to identify wave characteristics
        while len(peaks) > 0:

            # Output progress to console, keeping user informed
            displayProgress( peaks, len(plottingInfo.get('peaks')), data, dataList )

            # Identify the region surrounding the peak using the rectangle method from Zink & Vincent (2001)
            region = findPeakRectangle( wavelets.get('power'), peaks[0] )

            # Get reconstructed time series by inverting the wavelet transform inside the region
            wave = invertWaveletTransform( region, wavelets )

            # Perform stokes parameters analysis to find wave information and confirm/deny wave candidate
            parameters = getParameters(data, wave, spatialResolution, peaks[0, 1],
                                       wavelets.get('wavelengths')[peaks[0, 0]], 
                                       wavelets.get('power')[peaks[0,0], peaks[0,1]])

            # Update pertinent variables to save current wave parameters, increment counters, and shorten peaks list
            waves, plottingInfo, peaks = saveParametersInLoop(waves, plottingInfo, parameters, region, peaks)

        # Build and show/save power surface plot for current section of the current profile
        drawPowerSurface(userInput, file, wavelets, data['Alt'] / 1000, plottingInfo.get('regions'),
                            plottingInfo.get('peaks'), plottingInfo.get('colors'))

    # Finished profile, now save or print wave parameters, depending on user input
    outputWaveParameters(userInput, waves, file)

    # Done with this radiosonde flight, print 'finished' and continue to next file
    print("\nFinished file analysis")

# Finished with every file in folder, now done running program
print("\n\nAnalyzed all files in folder "+userInput.get('dataSource'))
