#!/usr/bin/env python

import pyNN.hardware.spikey as pynn
import hwconfig_default_s1v2 as default
import numpy as np

runtime = 10000.0  # in ms
gLeakRange = np.arange(0.2, 1.2, 0.1)
slopeMin = 20.0
trials = 10


def run(mappingOffset):
    """
    Measures firing rate of one neuron (determined by mappingOffset) in dependence on value of g_leak.
    If linear fit to these firing rates does not show a significantly large slope,
    g_leak is assumed to be not set correctly.
    """
    pynn.setup(mappingOffset=mappingOffset, calibTauMem=False,
               calibSynDrivers=False, calibVthresh=False)
    # set v_rest over v_reset to get neuron firing
    neuron = pynn.Population(1, pynn.IF_facets_hardware1, {
                             'v_rest': pynn.IF_facets_hardware1.default_parameters['v_thresh'] + 10.0})
    neuron.record()

    rateList = []
    for gLeak in gLeakRange:
        neuron.set({'g_leak': gLeak / default.iLeak_base})
        pynn.hardware.hwa._neuronsChanged = True
        pynn.run(runtime)
        rateList.append(
            [gLeak, float(len(neuron.getSpikes())) / runtime * 1e3])

    pynn.end()

    rateList = np.array(rateList)
    pol = np.polyfit(rateList[:, 0], rateList[:, 1], 1)  # linear fit
    print 'fitted polynom:', pol
    assert pol[0] > slopeMin, 'rate does not change with g_leak'

    #import matplotlib.pyplot as plt
    # plt.figure()
    #plt.plot(rateList[:,0], rateList[:,1])
    #index = np.linspace(np.min(gLeakRange), np.max(gLeakRange), 1000)
    #plt.plot(index, np.polyval(pol, index))
    #plt.savefig('result_change_ileak_' + str(mappingOffset).zfill(3) + '.png')


def test_change_ileak():
    mappingOffsetList = np.random.random_integers(0, 191, trials)
    for mappingOffset in mappingOffsetList:
        print 'mappingOffset', mappingOffset
        run(mappingOffset)

# last seen on 2015-06-09 by TP
