#!/usr/bin/env python

# last seen on 2015-06-08 by TP

# TODO: detect EPSPs with Fourier analysis, see Johann's Bachelor thesis

import time
import numpy as np
import copy
import pyNN.hardware.spikey as pynn

# initialize random number generator
seed = int(time.time())
print 'seed', seed
np.random.seed(seed)

runtime = 2000.0  # ms
limitSilence = 1.0  # max allowed spontaneous rate
limitActive = 2.0  # min rate if stimulated
stimRate = 100.0
weight = 15.0  # in digital values

noConnections = 10
connectionList = zip(np.random.random_integers(
    0, 255, noConnections), np.random.random_integers(0, 63, noConnections))

filename = 'routing_data.txt'
with open(filename, 'w') as myfile:
    myfile.write('syndrv neuron\n')

failList = []


def route(connection):
    """
    Connect synapse driver to each neuron individually and stimulate.
    Check other neurons for spontaneous activity.
    To take care of "ghost spikes", one neuron in each 64-block of neurons is stimulated.
    """

    synDriverIndex, neuronIndexBlock = connection
    print 'testing route:', synDriverIndex, '->', neuronIndexBlock

    neuronParam = copy.copy(pynn.IF_facets_hardware1.default_parameters)
    neuronParam['v_thresh'] = neuronParam['v_rest'] + 10.0
    neuronParam['g_leak'] = 40.0  # TODO: to avoid warnings of tau_mem calib

    # one neuron in each block of 64 neurons is stimulated
    pynn.setup()
    chipVersion = pynn.getChipVersion()
    noNeuronBlocks = 6
    if chipVersion == 4:
        noNeuronBlocks = 3  # use only one half of chip

    # create stimulus
    if synDriverIndex > 0:
        stimDummy = pynn.Population(synDriverIndex, pynn.SpikeSourceArray)
    stim = pynn.Population(1, pynn.SpikeSourcePoisson, {
                           'start': 0, 'duration': runtime, 'rate': stimRate})

    # create neurons
    neuronList = []
    dummyNeuronsList = []

    if neuronIndexBlock > 0:
        dummyNeurons = pynn.Population(
            neuronIndexBlock, pynn.IF_facets_hardware1, neuronParam)
        dummyNeurons.record()
        dummyNeuronsList.append(dummyNeurons)
    for neuronBlock in range(noNeuronBlocks):
        neuron = pynn.Population(1, pynn.IF_facets_hardware1, neuronParam)
        neuron.record()
        neuronList.append(neuron)
        if neuronBlock < noNeuronBlocks - 1:
            dummyNeurons = pynn.Population(
                63, pynn.IF_facets_hardware1, neuronParam)
            dummyNeurons.record()
            dummyNeuronsList.append(dummyNeurons)
    if neuronIndexBlock < 63:
        dummyNeurons = pynn.Population(
            63 - neuronIndexBlock, pynn.IF_facets_hardware1, neuronParam)
        dummyNeurons.record()
        dummyNeuronsList.append(dummyNeurons)

    # connect stimulus to neurons
    for neuron in neuronList:
        pynn.Projection(stim, neuron, method=pynn.AllToAllConnector(
            weights=weight * pynn.minExcWeight()), target='excitatory')

    pynn.run(runtime)

    def getRate(spikes):
        return len(spikes) / runtime * 1e3

    rateList = []
    rateDummyList = []
    for neuronIndex, neuron in enumerate(neuronList):
        neuronIndexGlobal = neuronIndex * 64 + neuronIndexBlock
        spikes = neuron.getSpikes()
        if len(spikes) > 0:
            assert (neuronIndexGlobal == np.squeeze(
                np.unique(spikes[:, 0]))).all()
        rate = getRate(spikes)
        rateList.append([neuronIndexGlobal, rate])

    for dummyNeurons in dummyNeuronsList:
        rateDummyList.append(getRate(dummyNeurons.getSpikes()))

    print 'rate neurons:', rateList
    print 'rate dummy neurons', rateDummyList

    pynn.end()

    # evaluate firing rates
    def addFail():
        if not connection in failList:
            failList.append(connection)

    def didFire(rate):
        addFail()
        return 'Neurons did fire with rate ' + str(round(rate, 2)) + ' although not connected and stimulated.'

    def didNotFire(neuronIndexGlobal, rate):
        addFail()
        return 'Neuron ' + str(neuronIndexGlobal) + ' did fire with too low rate ' + str(round(rate, 2)) + ' although connected and stimulated.'

    for neuronIndexGlobal, rate in rateList:
        if rate < limitActive:
            print didNotFire(neuronIndexGlobal, rate)

    for rate in rateDummyList:
        if rate > limitSilence:
            print didFire(rate)

    # save data for collecting statistics
    with open(filename, 'a') as myfile:
        myfile.write('%i %i %i\n' % (synDriverIndex,
                                     neuronIndexBlock, int(connection in failList)))


def test_routing():
    for connection in connectionList:
        route(connection)
    assert len(failList) == 0, "List of failed connections: " + str(failList)
