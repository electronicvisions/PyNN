import pyNN.hardware.spikey as pynn
import numpy as np


mappingOffset = np.random.random_integers(0, 192)
print 'mapping offset', mappingOffset
noNeurons = 0
if mappingOffset < 192:
    noNeurons = np.random.random_integers(0, 192 - mappingOffset) / 2
print 'number alternating neuron pairs:', noNeurons
shouldPatternWeights = mappingOffset * [0] + noNeurons * [1, 0]


def test():
    '''mapping of bio index to hardware index should work for networks
    where not all neurons are recorded'''
    pynn.setup()

    if mappingOffset > 0:
        dummy = pynn.Population(mappingOffset, pynn.IF_facets_hardware1)

    neuronList = []
    for i in range(noNeurons):
        neuronList.append(pynn.Population(1, pynn.IF_facets_hardware1))
        neuronList[-1].record()
        dummy = pynn.Population(1, pynn.IF_facets_hardware1)

    stim = pynn.Population(1, pynn.SpikeSourcePoisson)
    for neuron in neuronList:
        pynn.Projection(stim, neuron, pynn.AllToAllConnector(
            weights=pynn.minExcWeight()))

    pynn.run(1000.0)
    pynn.end()

    f = open('spikeyconfig.out')
    for line in f:
        for i in range(mappingOffset + 2 * noNeurons):
            if line.find('w ' + str(192 + i)) >= 0:
                weight = int(line.split(' ')[256 + 2 - 1])
                print 192 + i, weight
                assert(weight == shouldPatternWeights[
                       i]), 'results do not fit expectation'
    f.close()
