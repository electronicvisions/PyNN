import pyNN.hardware.spikey as pynn


def run(withSTDP):
    runtime = 1000.0

    pynn.setup()
    stim = pynn.Population(1, pynn.SpikeSourcePoisson, {
                           'start': 0, 'duration': runtime, 'rate': 100.0})
    neuron = pynn.Population(1, pynn.IF_facets_hardware1)

    if withSTDP:
        stdp_model = pynn.STDPMechanism(timing_dependence=pynn.SpikePairRule(),
                                        weight_dependence=pynn.AdditiveWeightDependence())
        pynn.Projection(stim, neuron,
                        method=pynn.AllToAllConnector(
                            weights=pynn.maxExcWeight()),
                        target='excitatory',
                        synapse_dynamics=pynn.SynapseDynamics(slow=stdp_model))
    else:
        pynn.Projection(stim, neuron,
                        method=pynn.AllToAllConnector(
                            weights=pynn.maxExcWeight()),
                        target='excitatory')

    pynn.run(runtime)
    pynn.end()


def test():
    """
    Experiments without STDP after experiments with STDP fail.
    """
    run(True)  # with STDP
    run(False)  # without STDP
