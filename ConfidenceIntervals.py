from WaveDetectionFunctions import getAllUserInput, cleanData, readFromData, interpolateData
import os
import numpy as np
import matplotlib.pyplot as plt


# First, get applicable user input.
userInput = getAllUserInput()

# Then, iterate over files in data directory
for file in os.listdir( userInput.get('dataSource') ):

    # Import and clean the data, given the file path
    data = cleanData( file, userInput.get('dataSource') )

    # If nothing was returned, file is not recognized as a GRAWMET profile,
    # so skip ahead to the next loop iteration
    if data.empty:
        continue

    # Get the launch time and pbl height from profile header
    launchDateTime, pblHeight = readFromData( file, userInput.get('dataSource'))
    # Set the height in between data points, currently 5 because it's the nominal data acquisition rate
    spatialResolution = 10  # meters, must be pos integer
    # Interpolate to create a uniform spatial grid of data, rather than temporal
    dataList = interpolateData( data, spatialResolution, pblHeight, launchDateTime )

    # If nothing was returned, file was missing too much data,
    # so skip ahead to the next loop iteration
    if len(dataList) == 0:
        continue

    for data in dataList:

        print("Starting...")

        # u and v (zonal & meridional) components of wind speed
        u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
        v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)

        print("About to do the stuff")

        N = len(u)
        var = np.sum([(x - np.mean(u))**2 for x in u])/N
        u = [1/N * np.sum([u[n]*np.exp(-2*np.pi*1j*n*k/N) for n in range(N)]) for k in range(N)]
        u = [N * np.abs(x)**2 / (2 * var) for x in u]

        print("Finished! Plotting...")

        plt.scatter(np.arange(N)[100:3000], u[100:3000])
        plt.ylim(0, 0.0025)
        plt.show()

        print("Totally done! Next try...")