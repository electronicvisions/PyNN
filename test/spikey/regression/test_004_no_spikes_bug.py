def test_noSpikesBug():
    import numpy as np
    import pyNN.hardware.spikey as pynn

    duration = 10 * 1000.0  # ms
    neuronParams = {
        'v_reset': -80.0,  # mV
        'e_rev_I': -80.0,  # mV
        'v_rest': -45.0,  # mV / rest above threshold
        'v_thresh': -55.0,  # mV
        'g_leak':  20.0,  # nS / without Scherzer calib approx. tau_m = 2ms
    }
    noTrials = 100
    failCount = 0
    for i in range(noTrials):
        pynn.setup()
        neuron = pynn.Population(1, pynn.IF_facets_hardware1, neuronParams)
        neuron.record()
        pynn.run(duration)
        spikes = neuron.getSpikes()[:, 1]
        pynn.end()  # comment this out and everything is fine

        if len(spikes) == 0:
            failCount += 1

    assert failCount == 0, str(
        float(failCount) / noTrials * 1e2) + ' % of runs did not have spikes'
