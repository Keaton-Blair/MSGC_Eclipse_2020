import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

riseRate = 5  # m/s balloon speed
u = np.zeros(6000)
v = u.copy()
t = u.copy()
time = np.array([x[0] for x in enumerate(u)])
altitude = time*riseRate

azimuth = 20  # degrees from vertical
wavelength = 10000  # meters
theta = 30  # degrees north from east
amplitude = 10  # m/s wind speed
velocity = 3  # m/s intrinsic group velocity
waveAlt = 20000  # m above sea level

vertWavelength = wavelength * np.cos(azimuth*np.pi/180)
vertAmplitude = amplitude * np.cos(azimuth*np.pi/180)
vertVelocity = velocity * np.cos(azimuth*np.pi/180)
vertWaveNumber = 2*np.pi/vertWavelength
vertAngularFrequency = 2*np.pi*vertVelocity/vertWavelength

axialRatio = (velocity / wavelength) / (-0.915 * 10**(-4))
phaseShift = np.arcsin(1/axialRatio)

wave1 = np.exp(- .5 * ((altitude - waveAlt) / (.25 * vertWavelength) )**2) \
        * vertAmplitude * np.cos(vertAngularFrequency * (time - waveAlt/riseRate) + vertWaveNumber * (altitude - waveAlt))

wave2 = np.exp(- .5 * ((altitude - waveAlt) / (.25 * vertWavelength) )**2) \
        * vertAmplitude * np.cos(vertAngularFrequency * (time - waveAlt/riseRate) + vertWaveNumber * (altitude - waveAlt) + phaseShift)


u = u + wave1 * np.cos(theta*np.pi/180)
v = v + wave2 * np.sin(theta*np.pi/180)

tAmplitude = 2  # K
t = t + np.exp(- .5 * ((altitude - waveAlt) / (.25 * vertWavelength) )**2) \
        * tAmplitude * np.sin(vertAngularFrequency * (time - waveAlt/riseRate) + vertWaveNumber * (altitude - waveAlt))


# NOW WE ADD THE NOISE!!!

t = t + np.polyval([0.1,-3,-33],altitude/1000)  #(0.003 * altitude - 0.033 * altitude**2)
for waveL in np.arange(100,1000,10):
    t = t + 0.1 * np.random.random() * np.sin(2*np.pi/waveL * altitude + np.random.random()*2*np.pi)
    u = u + .5 * np.random.random() * np.sin(2*np.pi/waveL * altitude + np.random.random()*2*np.pi)
    v = v + .5 * np.random.random() * np.sin(2*np.pi/waveL * altitude + np.random.random()*2*np.pi)

p = 1000 * np.exp(-altitude / 8400)
lat = [-39] * len(altitude)
lon = [-72] * len(altitude)
rs = [5] * len(altitude)


data = pd.DataFrame({
    'Alt': altitude,
    'Time': time,
    'P': p,
    'T': t,
    'Ws': np.sqrt(u**2 + v**2),
    'Wd': np.arctan2(v,u),
    'Lat.': lat,
    'Long.': lon,
    'Rs': rs
})

data.to_csv('fakeprofile.txt', header=True, index=None, sep='\t')

"""
plt.subplot(1,3,1)
plt.plot(u,altitude)
plt.title('U')
plt.subplot(1,3,2)
plt.plot(v, altitude)
plt.title('V')
plt.subplot(1,3,3)
plt.plot(t, altitude)
plt.title('T')
plt.show()
plt.close()

plt.plot(u[3500:4500],v[3500:4500])
plt.show()
plt.close()"""
