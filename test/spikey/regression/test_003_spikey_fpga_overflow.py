import unittest


class test_003_spikey_fpga_overflow(unittest.TestCase):
    '''
    Tests for Spikey->FPGA memory overflow.
    '''

    def test(self):
        def run(noNeurons):
            runtime = 1000.0

            import numpy as np
            import pyNN.hardware.spikey as pynn
            pynn.setup()

            neurons = pynn.Population(noNeurons, pynn.IF_facets_hardware1)
            neurons.record()
            stim = pynn.Population(10, pynn.SpikeSourcePoisson, {
                                   'rate': 20.0, 'duration': runtime})
            prj = pynn.Projection(stim, neurons, pynn.AllToAllConnector())
            prj.setWeights(pynn.maxExcWeight())

            pynn.run(runtime)
            spikes = neurons.getSpikes([])
            # for neuron in np.unique(spikes[:,0]):
            # print 'neuron', int(neuron), 'has', len(spikes[spikes[:,0] ==
            # neuron]), 'spikes'
            noSpikes = len(spikes)
            lost, sent = pynn.getInputSpikes()
            pynn.end()

            print 'no neurons / spikes in / lost / out:', noNeurons + 1, sent, lost, noSpikes

            return noSpikes

        noSpikesList = []
        for noNeurons in range(1, 192):
            noSpikes = run(noNeurons)
            noSpikesList.append([noNeurons, noSpikes])

if __name__ == "__main__":
    unittest.main()
