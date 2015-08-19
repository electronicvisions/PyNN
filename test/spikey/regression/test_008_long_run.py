import pyNN.hardware.spikey as pynn


def maxSpikesIn(runtime):
    '''Maximum number of spikes that can be sent:
    Should be limited by memory size on FPGA board (256MB => 256x1024x1024x8/32x3 approx. 200e6 spikes).'''

    rate = 10.0
    weight = 1.0

    poissonParam = {'start': 0, 'duration': runtime, 'rate': rate}

    pynn.setup()

    stim = pynn.Population(256, pynn.SpikeSourcePoisson, poissonParam)
    neuron = pynn.Population(192, pynn.IF_facets_hardware1)
    prj = pynn.Projection(stim, neuron, pynn.AllToAllConnector(
        weights=pynn.minInhWeight() * weight), target='inhibitory')

    neuron.record()

    pynn.run(runtime)
    spikes = neuron.getSpikes()
    lost, sent = pynn.getInputSpikes()

    print 'spikes in / out', sent, len(spikes)

    pynn.end()


def maxSpikesOut(runtime):
    '''Maximum number of spikes that can be received:
    Should be limited by memory size on FPGA board (128MB approx. 100e6 spikes, other half for ADC).'''

    neuronParams = {
        'v_reset': -80.0,  # mV
        'e_rev_I': -80.0,  # mV
        'v_rest': -45.0,  # mV / rest above threshold
        'v_thresh': -55.0,  # mV
        'g_leak':  20.0,  # nS / without Scherzer calib approx. tau_m = 2ms
    }

    pynn.setup()

    neuron = pynn.Population(192, pynn.IF_facets_hardware1, neuronParams)
    neuron.record()

    pynn.run(runtime)

    spikes = neuron.getSpikes()[:, 1]
    lost, sent = pynn.getInputSpikes()

    print 'spikes in / out', sent, len(spikes)

    pynn.end()


def maxRuntime(runtime):
    '''Maximum runtime:
    Limited by wrap around of counter (after approx. 6600s).
    Can be extended to infinitly long runtimes by considering wrap around.
    Subtract/Add offset to in/out spike times for each wrap around.'''

    rate = 1.0
    weight = 1.0

    poissonParam = {'start': 0, 'duration': runtime, 'rate': rate}

    pynn.setup()

    stim = pynn.Population(1, pynn.SpikeSourcePoisson, poissonParam)
    neuron = pynn.Population(1, pynn.IF_facets_hardware1)
    prj = pynn.Projection(stim, neuron, pynn.AllToAllConnector(
        weights=pynn.minInhWeight() * weight), target='inhibitory')

    neuron.record()

    pynn.run(runtime)
    spikes = neuron.getSpikes()
    lost, sent = pynn.getInputSpikes()

    print 'spikes in / out', sent, len(spikes)

    pynn.end()


def test():
    maxSpikesIn(1000 * 1000.0)  # 2.56e6 spikes
    maxSpikesOut(1000 * 1000.0)  # ~3.2e6
    maxRuntime(1000 * 1000.0)
    maxRuntime(3000 * 1000.0)
