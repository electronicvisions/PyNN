#!/usr/bin/env python

# TODO: increase runtime if FPGA fixed, see regression test 008

"""
Tests encoding of spikes sent to the system.
Max input rate is half of system clock (100MHz/2 in hardware = 5kHz bio),
because each not full spike packet needs command packet.
If packing possible max input rate should be three times the system clock (300MHz in hardware = 30kHz bio).
"""


def test_poisson():
    """Test with Poisson source."""

    import pyNN.hardware.spikey as pynn

    duration = 1000.0  # ms
    rate = 5000.0  # 1/s
    poissonParam = {'start': 0, 'duration': duration, 'rate': rate}
    limLost = 1.0  # %

    pynn.setup()

    stim = pynn.Population(1, pynn.SpikeSourcePoisson, poissonParam)
    neuron = pynn.Population(1, pynn.IF_facets_hardware1)
    pynn.Projection(stim, neuron, method=pynn.AllToAllConnector(
        weights=0), target='inhibitory')
    neuron.record()

    pynn.run(duration)
    lost, sent = pynn.getInputSpikes()
    print 'Number of input spikes (lost, sent, %lost)', lost, sent, float(lost) / sent * 1e2

    assert float(lost) / sent * 1e2 < limLost, 'Too many spikes lost!'

    pynn.end()


def test_regular():
    """Maximum rate without packing:
    Each second clock cycle a minimal loaded (filled with 1 spikes) spike packet."""

    import numpy as np
    import pyNN.hardware.spikey as pynn
    import time

    np.random.seed(int(time.time()))
    lineDriverNo = np.random.random_integers(0, 255)
    print 'Using line driver number', lineDriverNo

    duration = 1000.0  # ms
    h = 1e3 / 5000.0  # 0.2 ms

    pynn.setup()

    stim = pynn.Population(256, pynn.SpikeSourceArray)
    stim[lineDriverNo].set_parameters(
        spike_times=np.arange(0, duration + h / 2.0, h))
    neuron = pynn.Population(1, pynn.IF_facets_hardware1)
    pynn.Projection(stim, neuron, method=pynn.AllToAllConnector(
        weights=0), target='inhibitory')
    neuron.record()

    pynn.run(duration)
    lost, sent = pynn.getInputSpikes()
    print 'Number of input spikes (lost, sent)', lost, sent
    assert lost == 0, 'There should not be any spikes lost!'

    pynn.end()


def test_regular_packed():
    """Maximum rate with packing:
    Each clock cycle a full (filled with 3 spikes) spike packet."""

    import numpy as np
    import pyNN.hardware.spikey as pynn

    duration = 1000.0  # ms
    h = 1e3 / 5000.0 / 2.0  # 0.1 ms
    spikeTimes = np.arange(0, duration + h / 2.0, h)

    pynn.setup()

    stim = pynn.Population(256, pynn.SpikeSourceArray)
    # spikes have to be distributed over blocks of line drivers for efficient
    # packing
    stim[0].set_parameters(spike_times=spikeTimes)
    stim[64].set_parameters(spike_times=spikeTimes)
    stim[128].set_parameters(spike_times=spikeTimes)
    neuron = pynn.Population(1, pynn.IF_facets_hardware1)
    pynn.Projection(stim, neuron, method=pynn.AllToAllConnector(
        weights=0), target='inhibitory')
    neuron.record()

    pynn.run(duration)
    lost, sent = pynn.getInputSpikes()
    print 'Number of input spikes (lost, sent)', lost, sent
    assert lost == 0, 'There should not be any spikes lost!'

    pynn.end()

# last seen on 2015-06-10 by TP
