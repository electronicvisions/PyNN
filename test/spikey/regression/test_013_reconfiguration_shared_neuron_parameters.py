import unittest


def plotVoltageAndSpikes(pylab, t_mem, v_mem, *args, **kwargs):
    """Plot the voltage and the spike data.

    Arguments:
        pylab -- pylab module
        t_mem -- time array for membrane voltage
        v_mem -- membrane voltage array
        *args -- arbitrary number of spike data matrices
    Keyword arguments:
        nrn_idx_lim -- limit for neuron indices range in plot, e.g. [0,192]
    """
    if 'nrn_idx_lim' in kwargs.keys():
        nrn_idx_lim = kwargs['nrn_idx_lim']
    else:
        nrn_idx_lim = [0, 192]
    pylab.figure()
    pylab.subplot(211)
    pylab.xlabel("time [ms]")
    pylab.xlim([min(t_mem), max(t_mem)])
    pylab.ylabel("pynn neuron id")
    pylab.ylim([0, 192])
    for spike_data in args:
        pylab.plot(spike_data[:, 1], spike_data[:, 0], marker='o', ls='')
    pylab.subplot(212)
    pylab.xlabel("time [ms]")
    pylab.xlim([min(t_mem), max(t_mem)])
    pylab.ylabel("$U_{mem}$ of 1st neuron [mV]")
    pylab.plot(t_mem, v_mem)
    pylab.show()


class test_valid_param_config_change(unittest.TestCase):
    """
    Test a valid parameter configuration change for a population
    """

    def runTest(self):
        with_figure = False
        import numpy
        import pyNN.hardware.spikey as pynn
        if with_figure:
            import pylab

        # some test parameters
        neuron_param_even = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -100.0,         # mV
            'e_rev_I': -100.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh': -62.0          # mV
        }
        neuron_param_uneven = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -100.0,         # mV
            'e_rev_I': -100.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh':       0.0          # mV
        }
        stim_offset = 100.0  # ms
        stim_isi = 500.0  # ms
        stim_num = 10   # Number of external input spikes
        stim_weight = 8.0  # in units of pyn.minExcWeight
        stim_pop_size = 10   # size of stimulating population
        duration = stim_offset + ((stim_num + 1) * stim_isi)

        # neuron order: {0, 2, ..., 190, 1, 3, ..., 191, 192, 193, ... 343}
        neuron_order = range(0, 191, 2) + range(1, 192, 2) + range(192, 384, 1)
        if with_figure:
            pynn.setup(neuronPermutation=neuron_order, useUsbAdc=True)
        else:
            pynn.setup(neuronPermutation=neuron_order)

        # create the population with an even hardware neuron index
        even_population = pynn.Population(
            96, pynn.IF_facets_hardware1, neuron_param_even
        )
        # create the population with an uneven hardware neuron index
        uneven_population = pynn.Population(
            96, pynn.IF_facets_hardware1, neuron_param_uneven
        )
        if with_figure:
            pynn.record_v(even_population[0], '')

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
        stim_prj_even = pynn.Projection(stim_pop, even_population, stim_con)
        stim_prj_uneven = pynn.Projection(
            stim_pop, uneven_population, stim_con)

        # record spikes of all involved neurons
        even_population.record()
        uneven_population.record()

        # run the emulation
        pynn.run(duration)

        # get the spike data
        pre_swap_spikes_even = even_population.getSpikes()
        pre_swap_spikes_uneven = uneven_population.getSpikes()
        if with_figure:
            plotVoltageAndSpikes(
                pylab,
                pynn.timeMembraneOutput, pynn.membraneOutput,
                pre_swap_spikes_even, pre_swap_spikes_uneven
            )

        # swap the configurations
        pynn.set(even_population[0],
                 pynn.IF_facets_hardware1, {'v_thresh': 0.0})
        pynn.set(uneven_population[0], pynn.IF_facets_hardware1, {
                 'v_thresh': -62.0})

        # run the emulation
        pynn.run(duration)

        # get the spike data
        pst_swap_spikes_even = even_population.getSpikes()
        pst_swap_spikes_uneven = uneven_population.getSpikes()
        if with_figure:
            plotVoltageAndSpikes(
                pylab,
                pynn.timeMembraneOutput, pynn.membraneOutput,
                pst_swap_spikes_even, pst_swap_spikes_uneven
            )

        pre_spikes_count_even = float(len(pre_swap_spikes_even[:, 0]))
        pre_spikes_count_uneven = float(len(pre_swap_spikes_uneven[:, 0]))
        pst_spikes_count_even = float(len(pst_swap_spikes_even[:, 0]))
        pst_spikes_count_uneven = float(len(pst_swap_spikes_uneven[:, 0]))
        # let's see what we've got
        assert(pre_spikes_count_even > 0)
        assert(pst_spikes_count_uneven > 0)
        assert(pre_spikes_count_uneven / pre_spikes_count_even < 0.01)
        assert(pst_spikes_count_even / pst_spikes_count_uneven < 0.01)
        assert(pre_spikes_count_uneven / pst_spikes_count_uneven < 0.01)
        assert(pst_spikes_count_even / pre_spikes_count_even < 0.01)


class test_invalid_param_config_change_hidx(unittest.TestCase):
    """
    Test invalid parameter configuration change for a population. This
    test changes the neurons with the higher even hardware index.

    The call of "pynn.set" should be stronger than the previous neuron
    parameter setting. It should overwrite the previous configuration
    and even change the other population sharing the same parameter.
    """

    def runTest(self):
        with_figure = False
        import numpy
        import pyNN.hardware.spikey as pynn
        if with_figure:
            import pylab

        # some test parameters
        neuron_param = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -100.0,         # mV
            'e_rev_I': -100.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh': -62.0          # mV
        }
        stim_offset = 100.0  # ms
        stim_isi = 500.0  # ms
        stim_num = 10   # Number of external input spikes
        stim_weight = 8.0  # in units of pyn.minExcWeight
        stim_pop_size = 10   # size of stimulating population
        duration = stim_offset + ((stim_num + 1) * stim_isi)

        # neuron order: {0, 2, ..., 190, 1, 3, ..., 191, 192, 193, .., 383}
        neuron_order = range(0, 191, 2) + range(1, 192, 2) + range(192, 384, 1)
        if with_figure:
            pynn.setup(neuronPermutation=neuron_order, useUsbAdc=True)
        else:
            pynn.setup(neuronPermutation=neuron_order)

        # create first population with an even hardware neuron index
        fst_population = pynn.Population(
            48, pynn.IF_facets_hardware1, neuron_param
        )
        # create second population with an even hardware neuron index
        snd_population = pynn.Population(
            48, pynn.IF_facets_hardware1, neuron_param
        )
        if with_figure:
            pynn.record_v(fst_population[0], '')

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
        stim_prj_fst = pynn.Projection(stim_pop, fst_population, stim_con)
        stim_prj_snd = pynn.Projection(stim_pop, snd_population, stim_con)

        # record spikes of all involved neurons
        fst_population.record()
        snd_population.record()

        # run the emulation
        pynn.run(duration)

        # get the spike data
        pre_change_spikes_fst = fst_population.getSpikes()
        pre_change_spikes_snd = snd_population.getSpikes()
        if with_figure:
            plotVoltageAndSpikes(
                pylab,
                pynn.timeMembraneOutput, pynn.membraneOutput,
                pre_change_spikes_fst, pre_change_spikes_snd,
                nrn_idx_lim=[0, 96]
            )

        # change the configuration for the second group
        # desired behaviour: change the configuration for all even neurons
        # "set" should be stronger than "previous setting in even/uneven group"
        pynn.set(snd_population[0],
                 pynn.IF_facets_hardware1, {'v_thresh': 0.0})

        # run the emulation
        pynn.run(duration)

        # get the spike data
        hidx_change_spikes_fst = fst_population.getSpikes()
        hidx_change_spikes_snd = snd_population.getSpikes()
        if with_figure:
            plotVoltageAndSpikes(
                pylab,
                pynn.timeMembraneOutput, pynn.membraneOutput,
                hidx_change_spikes_fst, hidx_change_spikes_snd,
                nrn_idx_lim=[0, 96]
            )

        pre_spikes_count_fst = float(len(pre_change_spikes_fst[:, 0]))
        pre_spikes_count_snd = float(len(pre_change_spikes_snd[:, 0]))
        hidx_spikes_count_fst = float(len(hidx_change_spikes_fst[:, 0]))
        hidx_spikes_count_snd = float(len(hidx_change_spikes_snd[:, 0]))
        # let's see what we've got
        assert(pre_spikes_count_fst > 0)
        assert(pre_spikes_count_snd > 0)
        assert(hidx_spikes_count_fst / pre_spikes_count_fst < 0.01)
        assert(hidx_spikes_count_snd / pre_spikes_count_snd < 0.01)


class test_invalid_param_config_change_lidx(unittest.TestCase):
    """
    Test invalid parameter configuration change for a population. This
    test changes the neurons with the lower even hardware index.

    The call of "pynn.set" should be stronger than the previous neuron
    parameter setting. It should overwrite the previous configuration
    and even change the other population sharing the same parameter.
    """

    def runTest(self):
        with_figure = False
        import numpy
        import pyNN.hardware.spikey as pynn
        if with_figure:
            import pylab

        # some test parameters
        neuron_param = {
            'g_leak':       1.0,         # nS
            'tau_syn_E':       5.0,         # ms
            'tau_syn_I':       5.0,         # ms
            'v_reset': -100.0,         # mV
            'e_rev_I': -100.0,         # mV,
            'v_rest': -65.0,         # mV
            'v_thresh': -62.0          # mV
        }
        stim_offset = 100.0  # ms
        stim_isi = 500.0  # ms
        stim_num = 10   # Number of external input spikes
        stim_weight = 8.0  # in units of pyn.minExcWeight
        stim_pop_size = 10   # size of stimulating population
        duration = stim_offset + ((stim_num + 1) * stim_isi)

        # neuron order: {0, 2, ..., 190, 1, 3, ..., 191, 192, 193, .., 383}
        neuron_order = range(0, 191, 2) + range(1, 192, 2) + range(192, 384, 1)
        if with_figure:
            pynn.setup(neuronPermutation=neuron_order, useUsbAdc=True)
        else:
            pynn.setup(neuronPermutation=neuron_order)

        # create first population with an even hardware neuron index
        fst_population = pynn.Population(
            48, pynn.IF_facets_hardware1, neuron_param
        )
        # create second population with an even hardware neuron index
        snd_population = pynn.Population(
            48, pynn.IF_facets_hardware1, neuron_param
        )
        if with_figure:
            pynn.record_v(fst_population[0], '')

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
        stim_prj_fst = pynn.Projection(stim_pop, fst_population, stim_con)
        stim_prj_snd = pynn.Projection(stim_pop, snd_population, stim_con)

        # record spikes of all involved neurons
        fst_population.record()
        snd_population.record()

        # run the emulation
        pynn.run(duration)

        # get the spike data
        pre_change_spikes_fst = fst_population.getSpikes()
        pre_change_spikes_snd = snd_population.getSpikes()
        if with_figure:
            plotVoltageAndSpikes(
                pylab,
                pynn.timeMembraneOutput, pynn.membraneOutput,
                pre_change_spikes_fst, pre_change_spikes_snd,
                nrn_idx_lim=[0, 96]
            )

        # change the configuration for the first group
        # desired behaviour: change the configuration for all even neurons
        # "set" should be stronger than "previous setting in even/uneven group"
        pynn.set(fst_population[0],
                 pynn.IF_facets_hardware1, {'v_thresh': 0.0})

        # run the emulation
        pynn.run(duration)

        # get the spike data
        lidx_change_spikes_fst = fst_population.getSpikes()
        lidx_change_spikes_snd = snd_population.getSpikes()
        if with_figure:
            plotVoltageAndSpikes(
                pylab,
                pynn.timeMembraneOutput, pynn.membraneOutput,
                lidx_change_spikes_fst, lidx_change_spikes_snd,
                nrn_idx_lim=[0, 96]
            )

        pre_spikes_count_fst = float(len(pre_change_spikes_fst[:, 0]))
        pre_spikes_count_snd = float(len(pre_change_spikes_snd[:, 0]))
        lidx_spikes_count_fst = float(len(lidx_change_spikes_fst[:, 0]))
        lidx_spikes_count_snd = float(len(lidx_change_spikes_snd[:, 0]))
        # let's see what we've got
        assert(pre_spikes_count_fst > 0)
        assert(pre_spikes_count_snd > 0)
        assert(lidx_spikes_count_fst / pre_spikes_count_fst < 0.01)
        assert(lidx_spikes_count_snd / pre_spikes_count_snd < 0.01)

if __name__ == "__main__":
    unittest.main()
