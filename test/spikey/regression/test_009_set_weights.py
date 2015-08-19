import pyNN.hardware.spikey as pynn
import numpy as np


def emulation(doesWork):
    numberSynapses = 10
    runtime = 1000.0
    weights = range(0, numberSynapses * numberSynapses)

    pynn.setup()
    pre = pynn.Population(numberSynapses, pynn.SpikeSourcePoisson)
    post = pynn.Population(numberSynapses, pynn.IF_facets_hardware1)
    if doesWork:
        conn = pynn.Projection(pre, post, method=pynn.AllToAllConnector())
        conn.setWeights(weights)
    else:
        conn = pynn.Projection(
            pre, post, method=pynn.AllToAllConnector(weights=weights))

    pynn.run(runtime)

    pynn.end()


def test_set_weight():
    '''
    Setting the synaptic weight via an array does work with Projection.setWeights(),
    but not in constructor of Connector().
    '''
    emulation(True)
    emulation(False)
