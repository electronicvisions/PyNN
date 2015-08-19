def testRegularMaxPacked():
    '''Maximum rate with packing:

    Each clock cycle a full (filled with 3 spikes) spike packet.'''

    import numpy as np
    import pyNN.hardware.spikey as pynn

    duration = 10000.0  # ms
    h = 1e3 / 5000.0 / 2.0  # 10kHz for each of 3 sources = 30kHz
    spikeTimes = np.arange(0, duration + h / 2.0, h)

    pynn.setup()

    stim = pynn.Population(256, pynn.SpikeSourceArray)
    stim[0].set_parameters(spike_times=spikeTimes)
    stim[63].set_parameters(spike_times=spikeTimes)
    stim[127].set_parameters(spike_times=spikeTimes)
    neuron = pynn.Population(1, pynn.IF_facets_hardware1)
    pynn.Projection(stim, neuron, method=pynn.AllToAllConnector(
        weights=0), target='inhibitory')
    neuron.record()

    pynn.run(duration)
    print 'no out spikes:', len(neuron.getSpikes())
    lost, sent = pynn.getInputSpikes()
    print 'no in spikes (lost, sent)', lost, sent
    assert lost == 0, 'there should not be any spikes lost!'

    pynn.end()
