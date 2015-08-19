import unittest


class test_stdp_basic(unittest.TestCase):
    """
    Test one synapse to show weight update on causal and anti-causal branch.
    """

    # TODO: extend to randomly chosen synapse and more than one synapse
    def runTest(self):
        import numpy as np

        column = 4
        row = 4
        n = 20  # number of spike pairs
        deltaTList = [-1.0, 1.0]  # ms
        deltaTLimit = 0.3  # allowed deviation
        delay = 2.9  # ms (between stimulus and post)
        # at beginning in ms (should be larger than max deltaT)
        experimentOffset = 100.0
        deltaTPairs = 100.0  # time between pre-post-pairs in ms

        noStimulators = 3
        weightStimulator = 15  # weight for stimulator neurons
        weightMeasure = 0  # weight for measured neuron
        procCorrOffset = 100.0  # time after experiment until correlations are processed in ms

        for deltaT in deltaTList:
            stimulus = np.arange(experimentOffset, (n - 0.5)
                                 * deltaTPairs + experimentOffset, deltaTPairs)
            self.assertTrue(len(stimulus) == n)
            stimulusMeasure = stimulus + delay - deltaT

            import pyNN.hardware.spikey as pynn
            import hwconfig_default_s1v2 as default

            pynn.setup()

            if column > 0:
                pynn.Population(column, pynn.IF_facets_hardware1)
            # stimulated neuron
            neuron = pynn.Population(1, pynn.IF_facets_hardware1)

            spikeSourceStim = None
            spikeSourceMeasure = None
            # stimulators above measured synapse
            if row < noStimulators:
                if row > 0:
                    dummy = pynn.Population(row, pynn.SpikeSourceArray)
                spikeSourceMeasure = pynn.Population(1, pynn.SpikeSourceArray, {
                                                     'spike_times': stimulusMeasure})

            spikeSourceStim = pynn.Population(
                noStimulators, pynn.SpikeSourceArray, {'spike_times': stimulus})

            # stimulators below measured synapse
            if row >= noStimulators:
                if row > noStimulators:
                    dummy = pynn.Population(
                        row - noStimulators, pynn.SpikeSourceArray)
                spikeSourceMeasure = pynn.Population(1, pynn.SpikeSourceArray, {
                                                     'spike_times': stimulusMeasure})

            # connect and record
            stdp_model = pynn.STDPMechanism(timing_dependence=pynn.SpikePairRule(),
                                            weight_dependence=pynn.AdditiveWeightDependence())
            pynn.Projection(spikeSourceStim, neuron,
                            method=pynn.AllToAllConnector(
                                weights=pynn.minExcWeight() * weightStimulator),
                            target='excitatory')
            prj = pynn.Projection(spikeSourceMeasure, neuron,
                                  method=pynn.AllToAllConnector(
                                      weights=pynn.minExcWeight() * weightMeasure),
                                  target='excitatory',
                                  synapse_dynamics=pynn.SynapseDynamics(slow=stdp_model))
            neuron.record()

            #######
            # RUN #
            #######
            # correlation flags:
            # 0: no weight change
            # 1: causal weight change
            # 2: anti-causal weight change
            pynn.hardware.hwa.setLUT([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            lastInputSpike = np.max(
                np.concatenate((stimulus, stimulusMeasure)))
            runtime = lastInputSpike + procCorrOffset
            pynn.hardware.hwa.autoSTDPFrequency = runtime
            print 'runtime: ' + str(runtime) + '; last input spike: ' + str(lastInputSpike) + '; STDP readout: ' + str(runtime)
            pynn.run(runtime)

            # get flag and spikes
            corrFlag = (np.array(prj.getWeightsHW(
                readHW=True, format='list')) / pynn.minExcWeight())[0]
            spikes = neuron.getSpikes()[:, 1]
            print 'stimulus:', stimulus
            print 'measure:', stimulusMeasure
            print 'post:', spikes
            self.assertTrue(len(stimulusMeasure) == len(
                spikes), 'No proper spiking!')
            print 'correlation flag: ' + str(corrFlag)
            print 'deltaT (is / should / limit):', np.mean(spikes - stimulusMeasure), '/', deltaT, '/', deltaTLimit
            self.assertTrue(abs(np.mean(spikes - stimulusMeasure) -
                                deltaT) <= deltaTLimit, 'No precise spiking!')
            if deltaT > 0:  # causal
                self.assertTrue(corrFlag == 1, 'Wrong correlation flag!')
            else:  # anti-causal
                self.assertTrue(corrFlag == 2, 'Wrong correlation flag!')

            pynn.end()


if __name__ == "__main__":
    unittest.main()
