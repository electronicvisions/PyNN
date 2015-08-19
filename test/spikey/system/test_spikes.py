#!/usr/bin/env python

# last seen on 2015-06-10 by TP

import time
import numpy as np


def plot(memTime, mem, spikes, spikesMem, deriv, thresh):
    import matplotlib.pyplot as plt
    plt.figure()

    for spike in spikes:
        plt.axvline(spike, c='r')

    for spike in spikesMem:
        plt.axvline(spike, c='b')

    plt.twinx()
    plt.plot(memTime[:-1], deriv, c='gray')
    plt.axhline(thresh, c='g')

    plt.plot(memTime, mem, c='k')

    plt.savefig('result_spikes.png')


def test_compareSpikesToMembrane_restOverThresh():
    """
    Tests the precise timing of digital spikes and spikes extracted from the membrane potential.
    The resting potential is set over the firing threshold, which results in regular firing.
    """

    import pyNN.hardware.spikey as pynn
    np.random.seed(int(time.time()))

    trials = 3
    duration = 5 * 1000.0  # ms
    freqLimit = 10.0  # 1/s
    limitSpikesMissing = 2

    for trial in range(trials):
        neuronNo = np.random.random_integers(0, 191)
        print 'Using neuron number', neuronNo

        pynn.setup(mappingOffset=neuronNo)

        neuron = pynn.Population(1, pynn.IF_facets_hardware1, {
                                 'v_rest': pynn.IF_facets_hardware1.default_parameters['v_thresh'] + 10.0})
        neuron.record()
        pynn.record_v(neuron[0], '')

        pynn.run(duration)

        mem = pynn.membraneOutput
        memTime = pynn.timeMembraneOutput
        spikes = neuron.getSpikes()[:, 1]
        pynn.end()

        print 'Mean membrane:', np.mean(mem)
        noSpikes = len(spikes)
        print 'Number of spikes:', noSpikes
        assert noSpikes > freqLimit * \
            (duration / 1e3), 'Too less spikes: ' + str(noSpikes)

        spikesMem, deriv, thresh = spikesFromMem(memTime, mem)

        #plot(memTime, mem, spikes, spikesMem, deriv, thresh)

        # calculate ISIs
        spikes = np.array(spikes)
        isiDigital = spikes[1:] - spikes[:-1]
        isiDigitalMean = isiDigital.mean()

        spikesMem = np.array(spikesMem)
        isiAnalog = spikesMem[1:] - spikesMem[:-1]
        isiAnalogMean = isiAnalog.mean()

        # any digital or analog spikes missing?
        missingDigital = 0
        missingDigital += len(isiDigital[isiDigital > isiDigitalMean * 1.5])
        missingDigital += len(isiDigital[isiDigital < isiDigitalMean * 0.5])

        missingAnalog = 0
        missingAnalog += len(isiAnalog[isiAnalog > isiAnalogMean * 1.5])
        missingAnalog += len(isiAnalog[isiAnalog < isiAnalogMean * 0.5])

        print 'Number of spikes (digital, analog):', len(spikes), len(spikesMem)
        print 'Spikes missing (digital, analog):', missingDigital, missingAnalog
        print 'Frequency (digital, analog) in 1/s:', 1e3 / isiDigitalMean, 1e3 / isiAnalogMean
        ratioDigAna = isiDigitalMean / isiAnalogMean
        print 'Frequency digital to analog (abs, %):', ratioDigAna, (ratioDigAna - 1.0) * 1e3

        assert abs(len(spikes) - len(spikesMem)
                   ) <= limitSpikesMissing, 'Numbers of digital and analog spikes differ by more than ' + str(limitSpikesMissing) + '.'
        assert missingDigital == 0, 'Digital spikes are missing.'
        assert missingAnalog == 0, 'Analog spikes are missing.'
        assert (ratioDigAna - 1) < 2e-4, 'Time axes differ more than 0.2% between digital spikes and membrane (is ' + \
            str((ratioDigAna - 1.0) * 1e3) + '%).'


def spikesFromMem(memTime, mem):
    """Extract spikes from membrane."""

    maxSpikesLength = 0.5  # ms
    thresh = 0.6  # ratio between min and max derivation of membrane

    # derive membrane
    derivedMem = mem[1:] - mem[:-1]
    threshSpike = (np.max(derivedMem) - np.min(derivedMem)) * \
        thresh + np.min(derivedMem)
    derivedMemTh = memTime[derivedMem < threshSpike]
    derivedMemTh = np.concatenate([[0], derivedMemTh])
    spikesMem = []

    # only one spike if derivative is above threshold over many consecutive
    # samples
    for i in range(1, len(derivedMemTh), 1):
        if not derivedMemTh[i] - derivedMemTh[i - 1] < maxSpikesLength:
            spikesMem.append(derivedMemTh[i])
    return spikesMem, derivedMem, threshSpike


def compareSpikesToMembrane(duration):
    """
    Tests the precise timing of digital spikes and spikes extracted from the membrane potential.
    The neuron is stimulated with Poisson spike sources.
    """
    np.random.seed(int(time.time()))
    neuronNo = np.random.random_integers(0, 191)
    print 'Using neuron number', neuronNo

    poissonParams = {'start': 100.0, 'duration': duration -
                     100.0, 'rate': 30.0}  # offset of 100 ms to get all spikes
    weightExc = 4  # digital hardware value
    weightInh = 15  # digital hardware value
    freqLimit = 1.0  # 1/s
    meanLimit = 0.2  # ms
    stdLimit = 0.2  # ms

    import pyNN.hardware.spikey as pynn

    pynn.setup(mappingOffset=neuronNo)

    stimExc = pynn.Population(64, pynn.SpikeSourcePoisson, poissonParams)
    stimInh = pynn.Population(192, pynn.SpikeSourcePoisson, poissonParams)
    neuron = pynn.Population(1, pynn.IF_facets_hardware1)
    prj = pynn.Projection(stimExc, neuron, pynn.AllToAllConnector(
        weights=weightExc * pynn.minExcWeight()), target='excitatory')
    prj = pynn.Projection(stimInh, neuron, pynn.AllToAllConnector(
        weights=weightInh * pynn.minInhWeight()), target='inhibitory')

    neuron.record()
    pynn.record_v(neuron[0], '')

    pynn.run(duration)

    spikes = neuron.getSpikes()[:, 1]
    membrane = pynn.membraneOutput
    memTime = pynn.timeMembraneOutput
    spikesMem, deriv, thresh = spikesFromMem(memTime, membrane)

    pynn.end()

    #plot(memTime, membrane, spikes, spikesMem, deriv, thresh)

    print 'Spikes and spikes on membrane:', len(spikes), '/', len(spikesMem)
    assert len(spikes) / duration * 1e3 >= freqLimit, 'Too less spikes.'
    assert len(spikes) == len(spikesMem), 'Spikes do not match membrane.'
    spikesDiff = spikesMem - spikes
    spikesDiffMean = np.mean(spikesDiff)
    spikesDiffStd = np.std(spikesDiff)
    print 'Offset between spikes and membrane:', spikesDiffMean, '+-', spikesDiffStd
    assert spikesDiffMean < meanLimit, 'Spike and membrane have too large offset.'
    assert spikesDiffStd < stdLimit, 'Time axes of spikes and membrane are different.'


def test_compareSpikesToMembrane_1s():
    compareSpikesToMembrane(1000.0)


def test_compareSpikesToMembrane_10s():
    compareSpikesToMembrane(10 * 1000.0)


def test_compareSpikesToMembrane_100s():
    compareSpikesToMembrane(100 * 1000.0)
