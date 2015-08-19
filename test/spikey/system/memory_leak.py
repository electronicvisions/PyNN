import pyNN.hardware.spikey as pynn
loops = 100
runtime = 10 * 1000.0
rate = 5.0
weight = 7.0


def emulate():
    # pynn.setup(useUsbAdc=True)
    pynn.setup()
    stimI = pynn.Population(40, pynn.SpikeSourcePoisson, {
                            'start': 0, 'duration': runtime, 'rate': rate})
    stimE = pynn.Population(20, pynn.SpikeSourcePoisson, {
                            'start': 0, 'duration': runtime, 'rate': rate})
    neuron = pynn.Population(192, pynn.IF_facets_hardware1)
    prjI = pynn.Projection(stimI, neuron, pynn.AllToAllConnector(
        weights=weight * pynn.minInhWeight()), target='inhibitory')
    prjE = pynn.Projection(stimE, neuron, pynn.AllToAllConnector(
        weights=weight * pynn.minExcWeight()), target='excitatory')
    stimI.record()
    stimE.record()
    neuron.record()
    pynn.record_v(neuron[0], '')

    pynn.run(runtime)
    spikesInI = stimI.getSpikes()
    spikesInE = stimE.getSpikes()
    spikes = neuron.getSpikes()
    mem = pynn.membraneOutput

    print 'spikes out', len(spikes)
    print 'spikes in', len(spikesInI), len(spikesInE)
    print 'mem data points', len(mem)

    pynn.end()


for i in range(loops):
    emulate()
