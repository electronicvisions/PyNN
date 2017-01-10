import unittest
# TODO
# - Test for correlation in spiking. Right now just the number of spikes are
#       compared
# - Parameter and code documentation


class test_adjacent_block_connection(unittest.TestCase):
    """
    Test synapse connection between the different neuron blocks
    """

    def runTest(self):
        with_figure = False
        import numpy
        import pyNN.hardware.spikey as pynn
        if with_figure:
            import pylab

        # some test parameters
        weight = 12.0  # in units of pyn.minExcWeight

        neuron_params = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -100.0,         # mV
            'e_rev_I': -100.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh': -63.0          # mV
        }

        stim_offset = 100.0  # ms
        stim_isi = 500.0  # ms
        stim_num = 10       # Number of external input spikes
        stim_weight = 8.0  # in units of pyn.minExcWeight
        stim_pop_size = 10     # size of stimulating population

        # set up the test network where pre_neuron and pst_neuron are on different
        # blocks
        #======================================================================
        # stim ==> pre_neuron ==> pst_neuron
        #
        #                    sil_neuron
        #======================================================================
        # there shouldn't be a connection between sil_neuron and pst_neuron
        neuron_order = [0, 1, 192] + range(2, 192, 1) + range(193, 384)
        if with_figure:
            pynn.setup(neuronPermutation=neuron_order, useUsbAdc=True)
        else:
            pynn.setup(neuronPermutation=neuron_order)

        # create the populations
        pre_neuron = pynn.Population(
            1, pynn.IF_facets_hardware1, neuron_params)
        sil_neuron = pynn.Population(
            1, pynn.IF_facets_hardware1, neuron_params)
        pst_neuron = pynn.Population(
            1, pynn.IF_facets_hardware1, neuron_params)
        if with_figure:
            pynn.record_v(pst_neuron[0], '')

        # create the connection
        pre_pst_con = pynn.AllToAllConnector(
            weights=weight * pynn.minExcWeight())
        pre_pst_prj = pynn.Projection(pre_neuron, pst_neuron, pre_pst_con)

        # create the external stimulus
        stim_times = numpy.arange(stim_offset, stim_num * stim_isi, stim_isi)
        stim_pop = pynn.Population(
            stim_pop_size,
            pynn.SpikeSourceArray,
            {'spike_times': stim_times}
        )

        # connect the external simulus
        stim_con = pynn.AllToAllConnector(
            weights=stim_weight * pynn.minExcWeight())
        stim_prj = pynn.Projection(stim_pop, pre_neuron, stim_con)

        # record spikes of all involved neurons
        pre_neuron.record()
        pst_neuron.record()
        sil_neuron.record()

        # run the emulation
        pynn.run(stim_offset + ((stim_num + 1) * stim_isi))

        if with_figure:
            pylab.figure()
            pylab.plot(pynn.timeMembraneOutput, pynn.membraneOutput)
            pylab.show()

        # let's see what we've got, accept up to 1 ghost spike
        assert(len(pre_neuron.getSpikes()) >= 10)
        assert(len(pre_neuron.getSpikes()) < 12)
        assert(len(pst_neuron.getSpikes()) >= 10)
        assert(len(pst_neuron.getSpikes()) < 12)
        assert(len(sil_neuron.getSpikes()) < 2)


class test_adjacent_block_crosstalk(unittest.TestCase):
    """
    Test for synapse routing permutation error between the different neuron blocks
    """

    def runTest(self):
        with_figure = False
        import numpy
        import pyNN.hardware.spikey as pynn
        if with_figure:
            import pylab

        # some test parameters
        weight = 12.0  # in units of pyn.minExcWeight

        neuron_params = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -100.0,         # mV
            'e_rev_I': -100.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh': -63.0          # mV
        }

        stim_offset = 100.0  # ms
        stim_isi = 500.0  # ms
        stim_num = 10       # Number of external input spikes
        stim_weight = 8.0  # in units of pyn.minExcWeight
        stim_pop_size = 10     # size of stimulating population

        # set up the test network where pre_neuron and pst_neuron are on different
        # blocks
        #======================================================================
        #                    pre_neuron ==> pst_neuron
        #
        # stim ==> sil_neuron
        #======================================================================
        # there shouldn't be a connection between sil_neuron and pst_neuron
        neuron_order = [0, 1, 192] + range(2, 192, 1) + range(193, 384)
        if with_figure:
            pynn.setup(neuronPermutation=neuron_order, useUsbAdc=True)
        else:
            pynn.setup(neuronPermutation=neuron_order)

        # create the populations
        pre_neuron = pynn.Population(1, pynn.IF_facets_hardware1)
        sil_neuron = pynn.Population(1, pynn.IF_facets_hardware1)
        pst_neuron = pynn.Population(1, pynn.IF_facets_hardware1)
        if with_figure:
            pynn.record_v(pst_neuron[0], '')

        # create the connection
        pre_pst_con = pynn.AllToAllConnector(
            weights=weight * pynn.minExcWeight())
        pre_pst_prj = pynn.Projection(pre_neuron, pst_neuron, pre_pst_con)

        # create the external stimulus
        stim_times = numpy.arange(stim_offset, stim_num * stim_isi, stim_isi)
        stim_pop = pynn.Population(
            stim_pop_size,
            pynn.SpikeSourceArray,
            {'spike_times': stim_times}
        )

        # connect the external simulus
        stim_con = pynn.AllToAllConnector(
            weights=stim_weight * pynn.minExcWeight())
        stim_prj = pynn.Projection(stim_pop, sil_neuron, stim_con)

        # record spikes of all involved neurons
        pre_neuron.record()
        pst_neuron.record()
        sil_neuron.record()

        # run the emulation
        pynn.run(stim_offset + ((stim_num + 1) * stim_isi))

        if with_figure:
            pylab.figure()
            pylab.plot(pynn.timeMembraneOutput, pynn.membraneOutput)
            pylab.show()

        # let's see what we've got, accept up to 1 ghost spike
        assert(len(pre_neuron.getSpikes()) < 2)
        assert(len(pst_neuron.getSpikes()) < 2)
        assert(len(sil_neuron.getSpikes()) >= 10)
        assert(len(sil_neuron.getSpikes()) < 12)


class test_same_block_connection(unittest.TestCase):
    """
    Test synapse connection between the same neuron blocks
    """

    def runTest(self):
        with_figure = False
        import numpy
        import pyNN.hardware.spikey as pynn
        if with_figure:
            import pylab

        # some test parameters
        weight = 12.0  # in units of pyn.minExcWeight

        neuron_params = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -100.0,         # mV
            'e_rev_I': -100.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh': -63.0          # mV
        }

        stim_offset = 100.0  # ms
        stim_isi = 500.0  # ms
        stim_num = 10       # Number of external input spikes
        stim_weight = 8.0  # in units of pyn.minExcWeight
        stim_pop_size = 10     # size of stimulating population

        # set up the test network where pre_neuron and pst_neuron are on different
        # blocks
        #======================================================================
        # stim ==> pre_neuron ==> pst_neuron
        #
        #                    sil_neuron
        #======================================================================
        # there shouldn't be a connection between sil_neuron and pst_neuron
        neuron_order = range(0, 384, 1)
        if with_figure:
            pynn.setup(neuronPermutation=neuron_order, useUsbAdc=True)
        else:
            pynn.setup(neuronPermutation=neuron_order)

        # create the populations
        pre_neuron = pynn.Population(
            1, pynn.IF_facets_hardware1, neuron_params)
        sil_neuron = pynn.Population(
            1, pynn.IF_facets_hardware1, neuron_params)
        pst_neuron = pynn.Population(
            1, pynn.IF_facets_hardware1, neuron_params)
        if with_figure:
            pynn.record_v(pst_neuron[0], '')

        # create the connection
        pre_pst_con = pynn.AllToAllConnector(
            weights=weight * pynn.minExcWeight())
        pre_pst_prj = pynn.Projection(pre_neuron, pst_neuron, pre_pst_con)

        # create the external stimulus
        stim_times = numpy.arange(stim_offset, stim_num * stim_isi, stim_isi)
        stim_pop = pynn.Population(
            stim_pop_size,
            pynn.SpikeSourceArray,
            {'spike_times': stim_times}
        )

        # connect the external simulus
        stim_con = pynn.AllToAllConnector(
            weights=stim_weight * pynn.minExcWeight())
        stim_prj = pynn.Projection(stim_pop, pre_neuron, stim_con)

        # record spikes of all involved neurons
        pre_neuron.record()
        pst_neuron.record()
        sil_neuron.record()

        # run the emulation
        pynn.run(stim_offset + ((stim_num + 1) * stim_isi))

        if with_figure:
            pylab.figure()
            pylab.plot(pynn.timeMembraneOutput, pynn.membraneOutput)
            pylab.show()

        # let's see what we've got, accept up to 1 ghost spike
        assert(len(pre_neuron.getSpikes()) >= 10)
        assert(len(pre_neuron.getSpikes()) < 12)
        assert(len(pst_neuron.getSpikes()) >= 10)
        assert(len(pst_neuron.getSpikes()) < 12)
        assert(len(sil_neuron.getSpikes()) < 2)


class test_same_block_crosstalk(unittest.TestCase):
    """
    Test for synapse routing permutation error between the same neuron blocks
    """

    def runTest(self):
        with_figure = False
        import numpy
        import pyNN.hardware.spikey as pynn
        if with_figure:
            import pylab

        # some test parameters
        weight = 12.0  # in units of pyn.minExcWeight

        neuron_params = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -100.0,         # mV
            'e_rev_I': -100.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh': -63.0          # mV
        }

        stim_offset = 100.0  # ms
        stim_isi = 500.0  # ms
        stim_num = 10       # Number of external input spikes
        stim_weight = 8.0  # in units of pyn.minExcWeight
        stim_pop_size = 10     # size of stimulating population

        # set up the test network where pre_neuron and pst_neuron are on different
        # blocks
        #======================================================================
        #                    pre_neuron ==> pst_neuron
        #
        # stim ==> sil_neuron
        #======================================================================
        # there shouldn't be a connection between sil_neuron and pst_neuron
        neuron_order = range(0, 384, 1)
        if with_figure:
            pynn.setup(neuronPermutation=neuron_order, useUsbAdc=True)
        else:
            pynn.setup(neuronPermutation=neuron_order)

        # create the populations
        pre_neuron = pynn.Population(1, pynn.IF_facets_hardware1)
        sil_neuron = pynn.Population(1, pynn.IF_facets_hardware1)
        pst_neuron = pynn.Population(1, pynn.IF_facets_hardware1)
        if with_figure:
            pynn.record_v(pst_neuron[0], '')

        # create the connection
        pre_pst_con = pynn.AllToAllConnector(
            weights=weight * pynn.minExcWeight())
        pre_pst_prj = pynn.Projection(pre_neuron, pst_neuron, pre_pst_con)

        # create the external stimulus
        stim_times = numpy.arange(stim_offset, stim_num * stim_isi, stim_isi)
        stim_pop = pynn.Population(
            stim_pop_size,
            pynn.SpikeSourceArray,
            {'spike_times': stim_times}
        )

        # connect the external simulus
        stim_con = pynn.AllToAllConnector(
            weights=stim_weight * pynn.minExcWeight())
        stim_prj = pynn.Projection(stim_pop, sil_neuron, stim_con)

        # record spikes of all involved neurons
        pre_neuron.record()
        pst_neuron.record()
        sil_neuron.record()

        # run the emulation
        pynn.run(stim_offset + ((stim_num + 1) * stim_isi))

        if with_figure:
            pylab.figure()
            pylab.plot(pynn.timeMembraneOutput, pynn.membraneOutput)
            pylab.show()

        # let's see what we've got, accept up to 1 ghost spike
        assert(len(pre_neuron.getSpikes()) < 2)
        assert(len(pst_neuron.getSpikes()) < 2)
        assert(len(sil_neuron.getSpikes()) >= 10)
        assert(len(sil_neuron.getSpikes()) < 12)


class TestZeroWeightConnection(unittest.TestCase):
    """
    Test for correct configuration of synapse line driver for connections
    weight 0
    """
    with_figures = True

    def test_zero_projection(self):
        #======================================================================
        # set up the following network and record the membrane potential of neuron0
        # for three different combination of weights:
        # run 0: w0 = 0 | w1 = 0
        # run 1: w0 = w | w1 = w
        # run 2: w0 = w | w1 = 0
        #======================================================================
        #           inh weight w0
        #            --> neuron 0
        #           /
        # stim
        #           \
        #            --> neuron 1
        #           inh weight w1
        #======================================================================
        # TODO: write test with call of connect() instead of projection as well
        import numpy
        import pyNN.hardware.spikey as pynn
        if self.with_figures:
            import pylab

        duration = 1000.0  # ms
        w = 10.0  # in units of pynn.minInhWeight
        stim_rate = 40.0  # Hz

        neuron_params = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -80.0,         # mV
            'e_rev_I': -80.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh':       0.0          # mV // no spiking
        }

        # call setup with enabled oscilloscope
        pynn.setup(useUsbAdc=True)
        # create the populations
        nrn0 = pynn.Population(1, pynn.IF_facets_hardware1)
        nrn1 = pynn.Population(1, pynn.IF_facets_hardware1)
        stim = pynn.Population(1, pynn.SpikeSourceArray, dict(spike_times=np.arange(100., duration, 100.)))
        # record the membrane potential of neuron0
        pynn.record_v(nrn0[0], '')
        # create the connectors
        w_con = pynn.AllToAllConnector(weights=w * pynn.minInhWeight())
        zero_con = pynn.AllToAllConnector(weights=0.0 * pynn.minInhWeight())

        # run the first time
        nrn0_prj = pynn.Projection(stim, nrn0, zero_con, target='inhibitory')
        nrn1_prj = pynn.Projection(stim, nrn1, zero_con, target='inhibitory')
        pynn.run(duration)
        u_run0 = numpy.mean(pynn.membraneOutput)
        du_run0 = numpy.std(pynn.membraneOutput)
        if self.with_figures:
            pylab.figure()
            pylab.plot(pynn.timeMembraneOutput, pynn.membraneOutput, label="Conn = (0,0)")

        # second time
        nrn0_prj = pynn.Projection(stim, nrn0, w_con, target='inhibitory')
        nrn1_prj = pynn.Projection(stim, nrn1, w_con, target='inhibitory')
        pynn.run(duration)
        u_run1 = numpy.mean(pynn.membraneOutput)
        du_run1 = numpy.std(pynn.membraneOutput)
        if self.with_figures:
            pylab.plot(pynn.timeMembraneOutput, pynn.membraneOutput, label="Conn = (w,w)")

        # third time
        nrn0_prj = pynn.Projection(stim, nrn0, w_con, target='inhibitory')
        nrn1_prj = pynn.Projection(stim, nrn1, zero_con, target='inhibitory')
        pynn.run(duration)
        u_run2 = numpy.mean(pynn.membraneOutput)
        du_run2 = numpy.std(pynn.membraneOutput)
        if self.with_figures:
            pylab.plot(pynn.timeMembraneOutput, pynn.membraneOutput, label="Conn = (w,0)")
            pylab.legend()
            pylab.show()

        # plot and evaluate
        if self.with_figures:
            x_val = numpy.arange(1, 4, 1)
            y_val = [u_run0, u_run1, u_run2]
            y_err = [du_run0, du_run1, du_run2]
            pylab.figure()
            pylab.xlim([0, 4])
            pylab.xlabel("Run")
            pylab.ylabel("Mean membrane voltage [mV]")
            pylab.errorbar(x_val, y_val, yerr=y_err, fmt='o')
            pylab.show()
        assert(((u_run0 - u_run1) / du_run0) > 0.2)
        assert(((u_run0 - u_run2) / du_run0) > 0.2)
        assert(abs((u_run2 - u_run1) / du_run0) < 0.2)

if __name__ == "__main__":
    unittest.main()
