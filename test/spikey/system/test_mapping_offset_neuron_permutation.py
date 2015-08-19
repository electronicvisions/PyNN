#!/usr/bin/env python

import pyNN.hardware.spikey as pynn
import numpy as np
import copy


def run_network(mappingOffset=0, neuronPermutation=[], noNeurons=-1):
    pynn.setup(mappingOffset=mappingOffset,
               neuronPermutation=neuronPermutation)
    if noNeurons == -1:
        noNeurons = 384
        if pynn.getChipVersion() == 4:
            noNeurons = 192
    a = pynn.Population(noNeurons, pynn.IF_facets_hardware1)
    b = pynn.Population(10, pynn.SpikeSourcePoisson)
    prj = pynn.Projection(b, a, method=pynn.AllToAllConnector())
    pynn.run(1.0)


def test_mappingOffset_0():
    # example without mapping offset and without neuron permutator
    run_network(mappingOffset=0)

    should = np.arange(384)
    if pynn.getChipVersion() == 4:
        should = np.concatenate((np.ones(192, int) * -1, np.arange(192)))
    assert np.array_equal(pynn.hardware.hwa.neuronIndexMap,
                          should), 'error in mapping'
    should = np.arange(384)
    if pynn.getChipVersion() == 4:
        should = np.concatenate((np.arange(192, 384), np.ones(192, int) * -1))
    assert np.array_equal(pynn.hardware.hwa.hardwareIndexMap,
                          should), 'error in mapping'

    pynn.end()


def test_mappingOffset_10():
    # example with mapping offset
    mappingOffset = 10
    run_network(mappingOffset=mappingOffset)

    should = np.concatenate(
        (np.arange(384 - mappingOffset, 384), np.arange(384 - mappingOffset)))
    if pynn.getChipVersion() == 4:
        should = np.concatenate((np.ones(
            192, int) * -1, np.arange(192 - mappingOffset, 192), np.arange(192 - mappingOffset)))
    assert np.array_equal(pynn.hardware.hwa.neuronIndexMap,
                          should), 'error in mapping'
    should = np.concatenate(
        (np.arange(mappingOffset, 384), np.arange(0, mappingOffset)))
    if pynn.getChipVersion() == 4:
        should = np.concatenate((np.arange(192 + mappingOffset, 384),
                                 np.arange(192, 192 + mappingOffset), np.ones(192, int) * -1))
    assert np.array_equal(pynn.hardware.hwa.hardwareIndexMap,
                          should), 'error in mapping'

    pynn.end()


def test_mappingOffset_and_Permutation():
    # example with neuron permutator and mapping offset for less than 192
    # neurons
    mappingOffset = 2
    noNeurons = 6
    permutator = [192, 195, 193, 196, 194, 197] + range(198, 384) + range(192)

    run_network(mappingOffset=mappingOffset,
                neuronPermutation=permutator, noNeurons=noNeurons)

    permutator = np.array(permutator)
    should = np.concatenate((np.ones(permutator[mappingOffset], int) * -1, [
                            0, 2, -1, 1, 3, 4, 5], np.ones(384 - permutator[mappingOffset] - noNeurons - 1, int) * -1))
    assert np.array_equal(pynn.hardware.hwa.neuronIndexMap,
                          should), 'error in mapping'
    should = np.concatenate((permutator[
                            mappingOffset:mappingOffset + noNeurons], np.ones(384 - noNeurons, int) * -1))
    assert np.array_equal(pynn.hardware.hwa.hardwareIndexMap,
                          should), 'error in mapping'

    pynn.end()


def test_mappingOffset_and_Permutation_random():
    # example with random neuron permutator
    trials = 3

    import time
    seed = int(time.time())
    print 'seed', seed
    np.random.seed(seed)

    pynn.setup()
    chipVersion = pynn.getChipVersion()
    pynn.end()

    permutatorWorking = range(384)
    if chipVersion == 4:
        permutatorWorking = range(192, 384)

    for i in range(trials):
        np.random.shuffle(permutatorWorking)
        mappingOffset = np.random.random_integers(0, 383)
        permutator = copy.copy(permutatorWorking)
        if chipVersion == 4:
            mappingOffset = np.random.random_integers(0, 191)
            permutator = copy.copy(permutatorWorking) + range(192)
        run_network(mappingOffset=mappingOffset, neuronPermutation=permutator)

        permutator = np.array(permutator)
        should = np.concatenate(
            (permutator[mappingOffset:384], permutator[0:mappingOffset]))
        if chipVersion == 4:
            should = np.concatenate((permutator[mappingOffset:192], permutator[
                                    0:mappingOffset], np.ones(192, int) * -1))
        assert np.array_equal(pynn.hardware.hwa.hardwareIndexMap, should)
        neuronIndexMap = pynn.hardware.hwa.neuronIndexMap
        noNeurons = 384
        if chipVersion == 4:
            noNeurons = 192
        assert len(neuronIndexMap[
                   neuronIndexMap >= 0]) == noNeurons, 'number of hardware neuron IDs does not match'
        assert len(neuronIndexMap[
                   neuronIndexMap >= 0]) == noNeurons, 'hardware neuron IDs not adjacent in map'
        if chipVersion == 4:
            noNeurons += 1  # +1 for "-1" entries
        assert len(np.unique(neuronIndexMap)
                   ) == noNeurons, 'not all hardware neuron IDs in map'

        pynn.end()


def test_Permutation_NumpyArray():
    neuronPermutation = np.concatenate(
        (np.random.permutation(range(192, 384)), range(192)))
    # example with no list, but numpy array (test for termination)
    run_network(neuronPermutation=neuronPermutation)

    pynn.end()

# last seen on 2015-06-09 by TP
