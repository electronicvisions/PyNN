import pyNN.hardware.spikey as pynn
import hwconfig_default_s1v2 as default


def run(lowThreshold):
    runtime = 1000.0
    pynn.setup()

    # set STDP params for low threshold  -> fails when vcthigh-vctlow < 0.04
    if lowThreshold:
        pynn.hardware.hwa.setSTDPParams(
            0.0, default.tpcsec, default.tpcorperiod, 1.0, 1.0, 1.0, 0.98, 2.5)
    else:
        pynn.hardware.hwa.setSTDPParams(
            0.0, default.tpcsec, default.tpcorperiod, 1.0, 1.0, 1.0, 0.85, 2.5)

    neuron = pynn.Population(1, pynn.IF_facets_hardware1)
    spikeArray = pynn.Population(1, pynn.SpikeSourceArray)

    stdp_model = pynn.STDPMechanism(timing_dependence=pynn.SpikePairRule(),
                                    weight_dependence=pynn.AdditiveWeightDependence())

    prj = pynn.Projection(spikeArray, neuron,
                          method=pynn.AllToAllConnector(
                              weights=pynn.minExcWeight() * 0),
                          target='excitatory',
                          synapse_dynamics=pynn.SynapseDynamics(slow=stdp_model))

    pynn.run(runtime)
    pynn.end()


def test():
    run(False)
    run(True)
