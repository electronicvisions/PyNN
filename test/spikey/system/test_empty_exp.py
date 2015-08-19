#!/usr/bin/env python

import pyNN.hardware.spikey as pynn


def test_empty_exp():
    """
    Initialize hardware and create one neuron.
    """

    pynn.setup()
    pynn.Population(1, pynn.IF_facets_hardware1)
    pynn.run(1000.0)
    pynn.end()

# last seen on 2015-06-09 by TP
