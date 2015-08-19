#!/usr/bin/env python

import pyNN.hardware.spikey as pynn
import numpy as np


def test_change_input():
    """
    Increase input rate, record output rate.
    Compare output rates between experiments with new pynn.setup() before each pynn.run()
    and one pynn.setup() for all pynn.run().
    """

    runtime = 50 * 1000.0  # in ms
    numNeurons = 192
    rateRange = np.arange(0.25, 3.01, 0.25)
    maxRelErr = 10.0  # %

    stim = None
    neurons = None

    def build(rateIn):
        global neurons, stim
        poissonParams = {'start': 0, 'duration': runtime, 'rate': rateIn}
        stim = pynn.Population(32, pynn.SpikeSourcePoisson, poissonParams)
        neurons = pynn.Population(numNeurons, pynn.IF_facets_hardware1)
        pynn.Projection(stim, neurons, pynn.AllToAllConnector(
            weights=pynn.minExcWeight() * 5.0), target='excitatory')
        neurons.record()

    def runClosed():
        global neurons
        rateOutList = []
        for rate in rateRange:
            pynn.setup()
            build(rate)
            pynn.run(runtime)
            rateOutList.append(len(neurons.getSpikes()) /
                               runtime * 1e3 / numNeurons)
            pynn.end()
        return rateOutList

    def runLoop():
        global neurons, stim
        rateOutList = []
        pynn.setup()
        build(0)
        for rate in rateRange:
            poissonParams = {'start': 0, 'duration': runtime, 'rate': rate}
            stim.set(poissonParams)
            pynn.run(runtime)
            rateOutList.append(len(neurons.getSpikes()) /
                               runtime * 1e3 / numNeurons)
        pynn.end()
        return rateOutList

    rateListClosed = runClosed()
    rateListLoop = runLoop()

    #import matplotlib.pyplot as plt
    # plt.figure()
    #plt.plot(rateRange, rateListClosed, 'b')
    #plt.plot(rateRange, rateListLoop, 'r')
    # plt.savefig('result_change_input.png')

    polyClosed = np.polyfit(rateRange, rateListClosed, 1)
    polyLoop = np.polyfit(rateRange, rateListLoop, 1)
    print 'min rate closed / loop', np.min(rateListClosed), np.min(rateListLoop)
    print 'max rate closed / loop', np.max(rateListClosed), np.max(rateListLoop)

    print 'polynom closed', polyClosed
    print 'polynom loop', polyLoop
    relErrSlope = abs(polyClosed[0] / polyLoop[0] - 1.0) * 100.0
    relErrOffset = abs(polyClosed[1] / polyLoop[1] - 1.0) * 100.0

    print 'error slope', relErrSlope
    print 'error offset', relErrOffset
    assert relErrSlope < maxRelErr
    assert relErrOffset < maxRelErr

# last seen on 2015-06-09 by TP
