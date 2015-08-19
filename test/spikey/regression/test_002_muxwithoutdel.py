import unittest
import shutil
import difflib


class test_002_muxwithoutdel(unittest.TestCase):
    '''
    Test proper functionality of mux in consecutive pynn experiments.
    '''

    def test(self):
        limFreq = 0.5
        runtime = 2000.0
        memLim = 1.0  # mV

        vRestEven = -70.0
        vRestOdd = -80.0

        spikeyconfigFilename = 'spikeyconfig.out'
        noNeurons = 4
        extListWith = ['withend' + str(i) for i in range(noNeurons)]
        extListNo = ['noend' + str(i) for i in range(noNeurons)]

        import pyNN.hardware.spikey as pynn

        def getMemNoLoop():
            result = []
            for i in range(noNeurons):
                pynn.setup(useUsbAdc=True)
                neuron = pynn.Population(noNeurons, pynn.IF_facets_hardware1)

                for j in range(noNeurons):
                    if j % 2 == 0:
                        neuron[j].set_parameters(v_rest=vRestEven)
                    else:
                        neuron[j].set_parameters(v_rest=vRestOdd)

                neuron.record()
                pynn.record_v(neuron[i], '')

                pynn.run(runtime)

                mem = pynn.membraneOutput
                spikes = neuron.getSpikes()

                pynn.end()

                shutil.copy(spikeyconfigFilename,
                            spikeyconfigFilename + extListWith[i])
                self.assertTrue((float(len(spikes)) / runtime * 1e3) <=
                                limFreq, 'there should not be any (too much) spikes')
                result.append([mem.mean(), mem.std()])

            return result

        def getMemLoop():
            result = []

            pynn.setup(useUsbAdc=True)
            neuron = pynn.Population(noNeurons, pynn.IF_facets_hardware1)

            for j in range(noNeurons):
                if j % 2 == 0:
                    neuron[j].set_parameters(v_rest=vRestEven)
                else:
                    neuron[j].set_parameters(v_rest=vRestOdd)

            neuron.record()

            for i in range(noNeurons):
                pynn.record_v(neuron[i], '')

                pynn.run(runtime)

                mem = pynn.membraneOutput
                spikes = neuron.getSpikes()

                shutil.copy(spikeyconfigFilename,
                            spikeyconfigFilename + extListNo[i])
                self.assertTrue((float(len(spikes)) / runtime * 1e3) <=
                                limFreq, 'there should not be any (too much) spikes')
                result.append([mem.mean(), mem.std()])

            pynn.end()

            return result

        def emuWithMappingOffset():
            pynn.setup()
            pynn.Population(1, pynn.IF_facets_hardware1)
            pynn.run(runtime)
            pynn.end()

        emuWithMappingOffset()  # this caused failure
        memShouldList = getMemNoLoop()
        memIsList = getMemLoop()
        print 'should', memShouldList
        print 'is', memIsList

        for extID in range(len(extListWith)):
            spikeyconfigFileWith = open(
                spikeyconfigFilename + extListWith[extID])
            spikeyconfigFileNo = open(spikeyconfigFilename + extListNo[extID])
            spikeyconfigWith = spikeyconfigFileWith.read()
            spikeyconfigNo = spikeyconfigFileNo.read()
            diffResult = difflib.unified_diff(spikeyconfigWith, spikeyconfigNo)
            for line in diffResult:
                self.assertTrue(
                    True, 'there should not be any differences in spikeyconfigs')

        for i in range(len(memShouldList)):
            self.assertTrue(abs(memShouldList[i][
                            0] - memIsList[i][0]) < memLim, 'membrane mux not working properly')
            self.assertTrue(abs(memShouldList[i][
                            0] - memIsList[i][0]) < memLim, 'membrane mux not working properly')

if __name__ == "__main__":
    unittest.main()
