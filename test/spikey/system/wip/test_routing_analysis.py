import numpy as np
import sys

limitSilence = 1.0  # max allowed spontaneous rate
limitActive = 2.0  # min rate if stimulated

filename = "routing_data.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]
print 'loading filename', filename

data = np.loadtxt(filename, skiprows=1)

synDrv = []
neuronID = []
rate = []
rateMatrix = np.ones((256, 192)) * -100
for i in range(2, 5):  # this is active neuron 1-3
    mask = data[:, i] < limitSilence
    synDrv = np.concatenate((synDrv, data[mask][:, 0]))
    neuronID = np.concatenate((neuronID, data[mask][:, 1]))

    rate = np.concatenate((rate, data[:, i]))

for i in range(len(data)):
    for j in range(2, 5):
        rateMatrix[data[i][0]][data[i][1] + (j - 2) * 64] = data[i][j]

import matplotlib.pyplot as plt

# fails
plt.figure()
plt.hist(synDrv, range=(-0.5, 255.5), bins=256)
plt.xlabel('syndrv ID')
plt.ylabel('# fails')
plt.xlim(-0.5, 255.5)

plt.figure()
plt.hist(neuronID, range=(-0.5, 63.5), bins=64)
plt.xlabel('neuron ID')
plt.ylabel('# fails')
plt.xlim(-0.5, 63.5)

# rate
plt.figure()
plt.hist(rate)
plt.xlabel('rate [Hz]')
plt.ylabel('# measurements')

plt.figure()
plt.imshow(rateMatrix)
plt.xlabel('neuron ID')
plt.ylabel('syndrv ID')
cb = plt.colorbar()
cb.set_label('rate [Hz]')

plt.show()
