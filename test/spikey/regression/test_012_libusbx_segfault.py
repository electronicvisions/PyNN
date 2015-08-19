import pyNN.hardware.spikey as pynn
import numpy as np
import time


def test_runs():
    '''
    after approximately 1000 runs the system fails with segfault
    this effect accounts for version 5, but not version 4 of the chip
    '''
    runtime = 10000.0
    i = 0

    timeStart = time.time()

    while i < 2000:
        pynn.setup(loglevel=0)

        #neurons = pynn.Population(192, pynn.IF_facets_hardware1, {'v_rest': -40.0})
        #pynn.Projection(neurons, neurons, pynn.FixedNumberPreConnector(15, weights=2*pynn.minExcWeight()), target='inhibitory')
        # neurons.record()

        # while True:
        i += 1
        # pynn.run(runtime)

        #spikes = neurons.getSpikes()
        #noSpikes = len(spikes)
        #uniqueNeurons = len(np.unique(spikes[:,0]))
        #rate = float(noSpikes) / uniqueNeurons / runtime * 1e3
        # print 'run', i, 'after', round((time.time() - timeStart) / 3600.0,
        # 4), 'hours runtime:', noSpikes, 'spikes from', uniqueNeurons, 'unique
        # neurons with av. rate', rate
        print 'run', i, 'after', round((time.time() - timeStart) / 3600.0, 4), 'hours runtime'

        pynn.end()

# inner loop, no recording of spikes: run 1104907 after 21.2418 hours runtime
# outer loop, no recording of spikes: run 1013 after 0.1643 hours runtime
# outer loop, no network: run 1013 after 0.0719 hours runtime
