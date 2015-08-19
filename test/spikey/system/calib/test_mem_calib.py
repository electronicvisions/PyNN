import unittest
import numpy as np


class test_mem_calib(unittest.TestCase):
    '''
    Tests output pin and neuron membrane calibration (and hence also adc and vout calibration)
    '''

    def setUp(self):
        self.doPlot = False
        self.voltageRange = np.arange(-80.0, -44.5, 3.5)
        self.pins = 4
        self.mappingOffsetRange = range(192)
        self.runtime = 2000.0
        self.limFreq = 1.0

        self.neuronParams = {
            'v_reset': -80.0,  # mV
            'e_rev_I': -80.0,  # mV
            'v_rest': -75.0,  # mV
            'v_thresh': -55.0,  # mV
            'g_leak':  250.0  # nS large leakage conductance to pull strongly to rest potential
        }

    def getMem(self, voltageRest, mappingOffset, calibOutputPins, calibNeuronMems):
        import pyNN.hardware.spikey as pynn
        pynn.setup(useUsbAdc=True, avoidSpikes=True, mappingOffset=mappingOffset,
                   calibTauMem=False, calibOutputPins=calibOutputPins, calibNeuronMems=calibNeuronMems)

        neuron = pynn.Population(
            1, pynn.IF_facets_hardware1, self.neuronParams)
        #neuronDummy = pynn.Population(1, pynn.IF_facets_hardware1, self.neuronParams)
        neuron.set({'v_rest': voltageRest})
        #neuronDummy.set({'v_rest': self.voltageRange[0]})
        neuron.record()
        pynn.record_v(neuron[0], '')

        pynn.run(self.runtime)

        mem = pynn.membraneOutput
        spikes = neuron.getSpikes()

        pynn.end()

        self.assertTrue((float(len(spikes)) / self.runtime * 1e3) <=
                        self.limFreq, 'there should not be any (too much) spikes')
        return mem.mean()

    def getData(self, calibOutputPins, calibNeuronMems):
        results = []
        for mappingOffset in self.mappingOffsetRange:
            for voltageValue in self.voltageRange:
                results.append([mappingOffset, voltageValue, self.getMem(
                    voltageValue, mappingOffset, calibOutputPins, calibNeuronMems)])
        return np.array(results)

    def fitData(self, data):
        fit, res, rank, singular_values, rcond = np.polyfit(
            data[:, 1], data[:, 2], 1, full=True)
        return fit[0], fit[1], res[0]

    def test_calibCombination(self):
        self.assertRaises(Exception, self.getData, (True, True))
        self.assertRaises(Exception, self.getData, (False, False))

    def test_calibOutputPins(self):
        results = self.getData(True, False)

        for pin in range(self.pins):
            resultsOnePin = results[results[:, 0] % self.pins == pin]

            slope, offset, res = self.fitData(resultsOnePin)

            print 'pin', pin, 'fit slope, offset, residual:', slope, offset, res
            self.assertTrue(abs(1 - slope) < 0.1,
                            'output pin fit slope is very different from 1')
            self.assertTrue(abs(offset) < 3.0,
                            'output pin fit offset is very different from 0')

        if self.doPlot:
            import matplotlib.pyplot as plt
            colormap = ['r', 'g', 'b', 'k']
            plt.figure()
            for i in range(self.pins):
                resultsOnePin = results[results[:, 0] % self.pins == i]
                if len(resultsOnePin) == 0:
                    continue
                plt.plot(resultsOnePin[:, 1], resultsOnePin[
                         :, 2], 'x-', c=colormap[i])
            plt.savefig('calibOutputPins.png')

    def test_calibNeuronMems(self):
        results = self.getData(False, True)
        for mappingOffset in self.mappingOffsetRange:
            resultsOneNeuron = results[results[:, 0] == mappingOffset]

            slope, offset, res = self.fitData(resultsOneNeuron)

            print 'neuron', mappingOffset, 'fit slope, offset, residual:', slope, offset, res
            self.assertTrue(abs(1 - slope) < 1.0,
                            'output pin fit slope is very different from 1')
            self.assertTrue(abs(offset) < 20.0,
                            'output pin fit offset is very different from 0')

        if self.doPlot:
            import matplotlib.pyplot as plt
            plt.figure()
            index = np.array([np.min(self.voltageRange),
                              np.max(self.voltageRange)])
            for mappingOffset in self.mappingOffsetRange:
                resultsOneNeuron = results[results[:, 0] == mappingOffset]
                plt.plot(resultsOneNeuron[:, 1], resultsOneNeuron[:, 2], 'bx')

                slope, offset, res = self.fitData(resultsOneNeuron)
                plt.plot(index, index * slope + offset, 'r')
            plt.plot(index, index, 'k:', lw=2)
            plt.savefig('calibNeuronMems.png')
            #np.savetxt('neuronMemRaw.dat', results)

if __name__ == "__main__":
    unittest.main()
    # plt.show()
