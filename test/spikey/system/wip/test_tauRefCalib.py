""" tests if calibrated refractory periods are set correctly"""

import numpy as np
import matplotlib.pyplot as plt
import pyNN.hardware.spikey as p
import utilities_calib_tauMem as utils
import hwconfig_default_s1v2 as default
import time
import pickle
import time
import random
import os

######################### neurons and tau_ref values to be measured ######
neuronIDs = range(192, 384)

# tau_ref value (in ms) at which to compare calibrated and uncalibrated
# tau_ref of individual neurons
tau_ref_target = 1.0
# file for this data
file_comparison = 'calib_vs_uncalib.txt'

# tau_ref values (in ms) at which to record mean calibrated tau_ref of all
# given neurons
tau_ref_range = np.arange(0.5, 2.0, 0.2)

# file for this data
file_range = 'calib_range.txt'

# plot only from given data?
plotOnly = False

# save recorded data?
saveData = True

# use seeded neuron parameters?
seededParams = False

# report problems?
reportFile = None

# neurons for which tau_ref calibration fails. These neurons will be
# marked in compare_neurons() and omitted in test_calib_range()
faultyNeurons = []

# debug plots for fitting tau_ref?
debugPlot = False

################################# testing routine ########################

# measure tau_ref for a given neuron and target tau_ref value


def testNeuron(neuronNr, tau_ref_target, calibIcb=True, tries=5, workstation=None, seededParams=False):
    # measured tau_ref values
    tau = []
    # failed fits
    failed_tries = 0
    # short membrane time constant for good results
    tau_mem = 1.0   # ms
    # time interval between two subsequently stimulated spikes
    meanISI = 300.  # ms
    # number of test runs for each neuron and icb value
    runs = 30

    # seeded neuron parameters; neurons were calibrated with v_rest = -60.0 mV
    if seededParams:
        v_rest = random.randint(-70, -60)
        v_thresh = random.randint(-55, -50)
        v_reset = random.randint(-90, -85)
    else:
        v_reset = -90.0
        v_rest = -65.0
        v_thresh = -55.0

    neuronParams = {
        'v_reset': v_reset,             # mV
        'e_rev_I': -90.0,               # mV
        'v_rest': v_rest,              # mV
        'v_thresh': v_thresh,            # mV
        'g_leak':  0.2 / (tau_mem / 1000.),  # nS
        'tau_syn_E':  5.0,                # ms
        'tau_syn_I':  10.0,               # ms
        'tau_refrac': tau_ref_target,      # ms
    }

    # experiment duration
    duration = (runs + 1) * meanISI

    # stimulating population
    inputParameters = {
        'numInputs': 10,           # number of spike sources connected to observed neuron
        'weight': 0.012,        # uS
        'weightIncrement': 0.003,        # uS
        'duration': duration,     # ms
        # ms
        'inputSpikes': {'spike_times': np.arange(10, duration, meanISI)},
    }

    # hardware setup
    p.setup(useUsbAdc=True, calibTauMem=True, calibVthresh=False, calibSynDrivers=False,
            calibIcb=calibIcb, mappingOffset=neuronNr - 192, workStationName=workstation)
    # observed neuron
    neuron = p.Population(1, p.IF_facets_hardware1, neuronParams)
    # stimulating population
    input = p.Population(inputParameters[
                         'numInputs'], p.SpikeSourceArray, inputParameters['inputSpikes'])
    # connect input and neuron
    conn = p.AllToAllConnector(
        allow_self_connections=False, weights=inputParameters['weight'])
    proj = p.Projection(input, neuron, conn,
                        synapse_dynamics=None, target='excitatory')

    # record spikes and membrane potential
    neuron.record()
    p.record_v(neuron[0], '')

    # run experiment
    p.run(duration)

    # evaluate results
    spikesDig = neuron.getSpikes()[:, 1]

    # change weight if too few or too many spikes occured
    tries = 0
    while tries < 5 and (len(spikesDig) < runs - 1 or len(spikesDig) > runs + 1):
        if len(spikesDig) < runs - 1:
            inputParameters['weight'] += inputParameters['weightIncrement']
            print 'increasing weight to {0}, try {1}'.format(inputParameters['weight'], tries)
        else:
            inputParameters['weight'] -= inputParameters['weightIncrement']
            print 'decreasing weight to {0}, try {1}'.format(inputParameters['weight'], tries)
        conn = p.AllToAllConnector(
            allow_self_connections=False, weights=inputParameters['weight'])
        proj = p.Projection(input, neuron, conn,
                            synapse_dynamics=None, target='excitatory')
        p.run(duration)
        spikesDig = neuron.getSpikes()[:, 1]
        tries += 1

    membrane = p.membraneOutput
    time = p.timeMembraneOutput
    # clean up
    p.end()

    # determine sampling bins
    timestep = time[1] - time[0]

    # detect analog spikes
    spikesAna, isiAna = utils.find_spikes(
        membrane, time, spikesDig, reportFile=reportFile)

    # determine refractory period from measurement of analog spikes
    tau_ref, tau_ref_err, doubles_spikes = utils.fit_tau_refrac(
        membrane, timestep, spikesAna, isiAna, noDigSpikes=len(spikesDig), debugPlot=debugPlot)

    return tau_ref, tau_ref_err


# compare mean tau_ref of calibrated neurons to target values in tau_ref_values
def test_calib_range(neuronIDs, tau_ref_values, filename='calib_range.txt', workstation=None, plotOnly=False, faultyNeurons=[], fig=None, saveData=False):
    meanTau_all = []
    tries = 5
    if not plotOnly:
        for tau_ref in tau_ref_values:
            meanTau_calib = []
            for neuron in neuronIDs:
                if neuron in faultyNeurons:
                    continue
                tau, tau_err = testNeuron(
                    neuron, tau_ref, calibIcb=True, tries=tries, workstation=workstation)
                meanTau_calib.append([neuron, tau])
            meanTau_calib = np.array(meanTau_calib)
            # exclude neurons which have been declared faulty
            faulty = [x for x in neuronIDs if x in faultyNeurons]
            missing_neurons = len(neuronIDs) - len(faulty) - len(meanTau_calib)
            # calculate mean tau_ref value for measured neurons (this excludes
            # neurons in faultyNeurons)
            meanTau_all.append([tau_ref, np.mean(meanTau_calib[:, 1]), np.std(
                meanTau_calib[:, 1]), len(neuronIDs) - len(faulty), missing_neurons])
        # save data
        meanTau_all = np.array(meanTau_all)
        if saveData:
            with file(filename, 'w') as outfile:
                pickle.dump(meanTau_all, outfile)
                pickle.dump(faultyNeurons, outfile)
                outfile.close()

    if plotOnly and not os.path.isfile(filename):
        print 'Cannot plot data, no file {0} exists'.format(filename)
        return
    # plot results
    if fig == None:
        fig = plt.figure()
    data = open(filename)
    meanTau_all = pickle.load(data)
    faultyNeurons_recorded = pickle.load(data)
    data.close()
    if len(meanTau_all) == 0:
        print 'No data was recorded for given neurons'
        return fig
    print 'Set tau_ref / measured tau_ref / error of measured tau_ref / measured neurons / neurons for which measuring tau_ref failed'
    print meanTau_all
    print 'Faulty neurons are:', faultyNeurons_recorded
    ax = fig.add_subplot(111)
    ax.errorbar(meanTau_all[:, 0], meanTau_all[:, 1], yerr=meanTau_all[
                :, 2], fmt='bo', label='all neurons')
    ax2 = fig.add_subplot(111)
    ax2.plot(meanTau_all[:, 0], meanTau_all[:, 0], color='k', label='set')
    plt.xlabel('Set tau_ref [ms]')
    plt.ylabel('Mean measured tau_ref [ms]')
    plt.title('Measurements for {0} neurons'.format(int(meanTau_all[0, 3])))
    if saveData:
        figname = file_range[:len(file_range) - 4]
        plt.savefig(figname)
    return fig


def compare_neurons(neuronIDs, tau_ref, filename='calib_vs_uncalib.txt', workstation=None, faultyNeurons=[], seed=None, plotOnly=False, fig=None, saveData=False, seededParams=True):
    if not plotOnly:
        neuronIDs = sorted(neuronIDs)
        tau_all_calib = []
        tau_all_uncalib = []
        failed_fits = []
        tries = 5

        # if no seed given, use time as seed
        if seed == None:
            random.seed(time.time())

        for neuron in neuronIDs:
            # record calibrated tau
            tau_calib, tau_calib_err = testNeuron(
                neuronNr=neuron, tau_ref_target=tau_ref, calibIcb=True, tries=tries, workstation=workstation, seededParams=seededParams)
            tau_all_calib.append([neuron, tau_calib, tau_calib_err])

            # record uncalibrated tau
            tau_uncalib, tau_uncalib_err = testNeuron(
                neuronNr=neuron, tau_ref_target=tau_ref, calibIcb=False, tries=tries, workstation=workstation, seededParams=seededParams)
            tau_all_uncalib.append([neuron, tau_uncalib, tau_uncalib_err])
            if tau_calib == -1 or tau_uncalib == -1:
                faultyNeurons.append(neuron)

            print '###################### result for neuron {0} #################'.format(neuron)
            print 'set refractory period is {0} ms'.format(tau_ref)
            print 'mean tau_ref after {0} measurements:'.format(tries)
            print 'calibrated: {0} +/- {1} ms'.format(tau_calib, tau_calib_err)
            print 'uncalibrated: {0} +/- {1} ms'.format(tau_uncalib, tau_uncalib_err)
            print '##############################################################'

        tau_all_calib = np.array(tau_all_calib)
        tau_all_uncalib = np.array(tau_all_uncalib)

        if saveData:
            with file(filename, 'w') as outfile:
                pickle.dump(tau_ref, outfile)
                pickle.dump(seed, outfile)
                pickle.dump(seededParams, outfile)
                pickle.dump(tau_all_calib, outfile)
                pickle.dump(tau_all_uncalib, outfile)
                pickle.dump(faultyNeurons, outfile)
                outfile.close()

    # plot recorded data
    if plotOnly and not os.path.isfile(filename):
        print 'cannot plot data, no file {0} exists'.format(filename)
        return
    data = open(filename)
    tau_ref = pickle.load(data)
    seed = pickle.load(data)
    seededParams = pickle.load(data)
    tau_all_calib = pickle.load(data)
    tau_all_uncalib = pickle.load(data)
    faultyNeurons_recorded = pickle.load(data)
    data.close()

    # mark neurons where fit failed
    faulty = np.array([x for x in tau_all_calib if x[0]
                       in faultyNeurons or x[0] in faultyNeurons_recorded])
    toPlot_calib = np.array([x for x in tau_all_calib if x[
                            0] in neuronIDs and x[1] >= 0])
    toPlot_uncalib = np.array(
        [x for x in tau_all_uncalib if x[0] in neuronIDs and x[1] >= 0])
    if fig == None:
        fig = plt.figure()
    if len(toPlot_calib) == 0:
        print 'No data for neurons {0} in file {1}'.format(neuronIDs, filename)
        return fig
    ax = fig.add_subplot(111)
    ax.errorbar(toPlot_calib[:, 0], toPlot_calib[:, 1],
                yerr=toPlot_calib[:, 2], fmt='bo', label='calib')
    ax2 = fig.add_subplot(111)
    ax2.errorbar(toPlot_uncalib[:, 0], toPlot_uncalib[
                 :, 1], yerr=toPlot_uncalib[:, 2], fmt='ro', label='uncalib')

    ax.set_xlabel('Neuron ID')
    ax.set_ylabel('Mean tau_ref [ms]')
    ax3 = fig.add_subplot(111)
    ax3.axhline(y=tau_ref, label='set tau_ref', color='k')
    if len(faulty > 0):
        # mark neurons for which calibration has failed
        ax4 = fig.add_subplot(111)
        ax4.plot(faulty[:, 0], faulty[:, 1], 'ko', label='faulty')

    plt.xlim(min(neuronIDs) - 2, max(neuronIDs) + 2)
    if seededParams:
        plt.title('Calibrated vs uncalibrated (seeded params)')
    else:
        plt.title('Calibrated vs uncalibrated (no seeds)')
    plt.legend()
    if saveData:
        figname = file_comparison[:len(file_comparison) - 4] + '.png'
        plt.savefig(figname)
    plt.ylim(tau_ref - 3, tau_ref + 3)

    return fig


##################### run test #####################
start = time.time()

compare_neurons(neuronIDs, tau_ref=tau_ref_target, plotOnly=plotOnly, faultyNeurons=faultyNeurons,
                filename=file_comparison, saveData=saveData, seededParams=seededParams)

test_calib_range(neuronIDs, tau_ref_values=tau_ref_range, plotOnly=plotOnly,
                 faultyNeurons=faultyNeurons, filename=file_range, saveData=saveData)

print 'testing neurons {0} to {1} took {2} minutes'.format(min(neuronIDs), max(neuronIDs), (time.time() - start) / 60.)
plt.show()
