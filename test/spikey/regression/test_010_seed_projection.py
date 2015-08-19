import pyNN.hardware.spikey as pynn
import numpy as np
from multiprocessing import Process, Manager


def emulation(seed, connType=0, returnValue=None):
    numberNeurons = 192
    noInputs = 15

    pynn.setup()

    rngPrj = pynn.random.NumpyRNG(
        seed=seed, parallel_safe=True)  # this may not work?!
    neurons = pynn.Population(numberNeurons, pynn.IF_facets_hardware1)
    connector = None
    if connType == 0:
        connector = pynn.FixedNumberPreConnector(
            noInputs, weights=pynn.minExcWeight())
    elif connType == 1:
        connector = pynn.FixedNumberPostConnector(
            noInputs, weights=pynn.minExcWeight())
    elif connType == 2:
        connector = pynn.FixedProbabilityConnector(
            float(noInputs) / numberNeurons, weights=pynn.minExcWeight())
    else:
        assert False, 'invalid connector type'

    prj = pynn.Projection(neurons, neurons, method=connector,
                          target='inhibitory', rng=rngPrj)

    connList = []
    for conn in prj.connections():
        connList.append(conn)

    assert len(connList) > 0, 'no connections'
    assert len(connList) < numberNeurons * \
        (numberNeurons - 1), 'all-to-all connection'

    pynn.run(1.0)

    pynn.end()

    if returnValue != None:
        returnValue = connList
    else:
        return connList


def checkConnLists(connListA, connListB):
    assert len(connListA) == len(
        connListB), 'number of connections does not fit'
    assertString = 'connections differ although identical seed'
    for i in range(len(connListA)):
        assert connListA[i][0] == connListB[i][0], assertString
        assert connListA[i][1] == connListB[i][1], assertString


def test_seed_projection():
    '''Two runs with same seed handed over to Projection() should result in same connectivity matrix'''
    for connType in range(3):
        connListA = emulation(0, connType=connType)
        connListB = emulation(0, connType=connType)
        checkConnLists(connListA, connListB)

    # with multiprocessing
    connListColl = []
    for i in range(2):
        manager = Manager()
        results = manager.list()
        proc = Process(target=emulation, args=(0, results))
        proc.start()
        proc.join()
        connList = results
        connListColl.append(connList)
        assert proc.exitcode == 0, 'emulation crashed'
    checkConnLists(connListColl[0], connListColl[1])
