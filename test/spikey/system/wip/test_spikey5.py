import pyNN.hardware.spikey as pynn


def test_spikey5_allneurons():
    '''
    Tests mapping and firing of all 384 neurons.
    '''
    runtime = 1000.0
    stimRate = 10.0
    weight = 7

    pynn.setup()

    neurons = pynn.Population(384, pynn.IF_facets_hardware1)
    stim = pynn.Population(10, pynn.SpikeSourcePoisson, {
                           'start': 0, 'duration': runtime, 'rate': stimRate})

    prj = pynn.Projection(stim, neurons, method=pynn.AllToAllConnector(
        weights=pynn.minExcWeight() * weight), target='excitatory')

    pynn.run(runtime)

    spikes = neurons.getSpikes()
    print 'spikes from', len(np.unique(spikes)), 'different neurons'
    # TODO: check for spikes from all neurons

    pynn.end()

if __name__ == '__main__':
    test_spikey5_allneurons()
