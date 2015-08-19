""" tests if calibrated membrane time constants are set correctly"""

# TODO: tidy up preamble
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
from copy import deepcopy

######################### neurons and tau_mem values to be measured ######
neuronIDs = range(192, 195)

# tau_mem value (in ms) at which to compare calibrated and uncalibrated
# tau_mem of individual neurons
tau_mem_target = 5.0
# file for this data
file_comparison = 'calib_vs_uncalib.txt'

# tau_mem values (in ms) at which to record mean calibrated tau_mem of all
# given neurons
tau_mem_range = np.arange(2.0, 13.0, 2.0)

# file for this data
file_range = 'tau_calib.txt'
file_plot = 'tau_calib.png'

# workstation number
workstation = ''  # tests get workstation name from file

# plot only from given data?
plotOnly = False

# save recorded data?
saveData = True

# use seeded neuron parameters?
randomNeuronParams = False

# report problems?
reportFile = None

# neurons for which tau_mem calibration fails. These neurons will be
# marked in compare_neurons() and omitted in test_calib_range()
faultyNeurons = []

# debug plots for fitting tau_mem?
debugPlot = False

trialsAverage = 5


################################# testing routine ########################

# measure tau_mem for a given neuron and target tau_mem value
def measure_tau_mem(neuronNr, tau_mem_target, calibTauMem=True):
    tauMeasured = []  # measured tau_mem values
    failed_trials = 0  # failed fits
    duration = 5000.  # duration of each measurement in ms
    targetSpikes = 20  # minimum number of spikes (target rate)
    vRestStep = 1.0   # increase of resting potential if too less spikes

    v_rest = p.IF_facets_hardware1.default_parameters['v_thresh'] + 3.0
    if randomNeuronParams:
        v_rest = random.randint(-65, -55)
    neuronParams = {'v_thresh':   p.IF_facets_hardware1.default_parameters['v_thresh'],
                    'v_rest':   v_rest,
                    'v_reset':   p.IF_facets_hardware1.default_parameters['v_reset'],
                    'g_leak':   0.2 / tau_mem_target * 1e3}  # nS
    maxVRest = neuronParams['v_rest'] + 10.0

    # measure tau_mem
    for i in range(trialsAverage):
        p.setup(useUsbAdc=True, calibTauMem=calibTauMem, calibVthresh=False, calibSynDrivers=False,
                calibIcb=False, mappingOffset=neuronNr - 192, workStationName=workstation)
        neuron = p.Population(1, p.IF_facets_hardware1, neuronParams)
        neuron.record()
        p.record_v(neuron[0], '')

        params = deepcopy(neuronParams)
        trials = 0
        crossedTargetRate = False
        while params['v_rest'] < maxVRest:
            print 'now at trial', trials, '/ v_rest =', params['v_rest']
            p.run(duration)
            trace = p.membraneOutput
            dig_spikes = neuron.getSpikes()[:, 1]
            memtime = p.timeMembraneOutput
            timestep = memtime[1] - memtime[0]

            # if neuron spikes with too low rate, try again with higher resting
            # potential
            if len(dig_spikes) < targetSpikes:
                params['v_rest'] = params['v_rest'] + vRestStep
                neuron.set(params)
                print 'Neuron spiked with too low rate, trying again with parameters', params
                trials += 1
            else:  # proper spiking
                crossedTargetRate = True
                break

        p.end()

        # determine tau_mem from measurements
        result = utils.fit_tau_mem(
            trace, memtime, dig_spikes, timestep=timestep, reportFile=reportFile)
        if result == None:  # fit failed
            failed_trials += 1
            continue
        tauMeasured.append(result[0])

    return tauMeasured, failed_trials


# compare mean tau_mem of calibrated neurons to target values in tau_mem_range
def test_calib_range(saveData=True):
    tauMeasuredList = []
    for tau_mem in tau_mem_range:
        for neuron in neuronIDs:
            if neuron in faultyNeurons:
                continue
            tau, failed_trials = measure_tau_mem(
                neuron, tau_mem, calibTauMem=True)
            # if tau cannot be measured, proceed to next neuron
            tau = np.array(tau)
            tau = tau[np.where(np.isfinite(tau))]
            assert len(tau) > 0, 'measurement of tau_mem failed!'
            tauMeasuredList.append([neuron, tau_mem, np.mean(tau)])

    if saveData:
        np.savetxt(file_range, tauMeasuredList)

    plot()

    # TODO: add uncalib and add asserts


def plot(saveData=True):
    assert os.path.isfile(file_range), 'file missing'
    data = np.loadtxt(file_range)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    resultList = []
    for tau_mem in tau_mem_range:
        measuredTau = data[data[:, 1] == tau_mem][:, 2]
        resultList.append([tau_mem, np.mean(measuredTau), np.std(measuredTau)])
    resultList = np.array(resultList)

    ax.errorbar(resultList[:, 0], resultList[:, 1],
                yerr=resultList[:, 2], fmt='bo')
    ax.plot(resultList[:, 0], resultList[:, 0], ls='--', c='k')  # expectation

    plt.xlabel(r'Set $\tau_m$ (ms)')
    plt.ylabel(r'Measured $\tau_m$ (ms)')
    plt.title('Measurements for {0} neurons'.format(
        len(np.unique(data[:, 0]))))

    if saveData:
        plt.savefig(file_plot)


#compare_neurons(neuronIDs, tau_mem=tau_mem_target, plotOnly=plotOnly, workstation=workstation, faultyNeurons=faultyNeurons, filename=file_comparison, saveData=saveData)
# TODO: remove?
def compare_neurons(neuronIDs, tau_mem, filename='calib_vs_uncalib.txt', workstation=None, faultyNeurons=[], seed=None, plotOnly=False, fig=None, saveData=False):
    if not plotOnly:
        neuronIDs = sorted(neuronIDs)
        tau_all_calib = []
        tau_all_uncalib = []
        failed_fits = []

        # if no seed given, use time as seed
        if seed == None:
            random.seed(time.time())

        for neuron in neuronIDs:
            # record calibrated tau
            tau_calib, failed_trials = measure_tau_mem(
                neuron, tau_mem, calibTauMem=True)
            tau_all_calib.append(
                [neuron, np.mean(tau_calib), np.std(tau_calib), failed_trials])
            # record uncalibrated tau
            tau_uncalib, failed_trials = measure_tau_mem(
                neuron, tau_mem, calibTauMem=False)
            tau_all_uncalib.append(
                [neuron, np.mean(tau_uncalib), np.std(tau_uncalib), failed_trials])
            if len(tau_calib) == 0 or len(tau_uncalib) == 0:
                faultyNeurons.append(neuron)

            print '###################### result for neuron {0} #################'.format(neuron)
            print 'set membrane time constant is {0} ms'.format(tau_mem)
            print 'mean tau_mem after {0} measurements:'.format(trialsAverage)
            print 'calibrated: {0} +/- {1} ms'.format(np.mean(tau_calib), np.std(tau_calib))
            print 'uncalibrated: {0} +/- {1} ms'.format(np.mean(tau_uncalib), np.std(tau_uncalib))
            print '##############################################################'

        tau_all_calib = np.array(tau_all_calib)
        tau_all_uncalib = np.array(tau_all_uncalib)

        if saveData:
            with file(filename, 'w') as outfile:
                pickle.dump(tau_mem, outfile)
                pickle.dump(seed, outfile)
                pickle.dump(randomNeuronParams, outfile)
                pickle.dump(tau_all_calib, outfile)
                pickle.dump(tau_all_uncalib, outfile)
                pickle.dump(faultyNeurons, outfile)
                outfile.close()

    # plot recorded data
    if plotOnly and not os.path.isfile(filename):
        print 'cannot plot data, no file {0} exists'.format(filename)
        return
    data = open(filename)
    tau_mem = pickle.load(data)
    seed = pickle.load(data)
    randomNeuronParams = pickle.load(data)
    tau_all_calib = pickle.load(data)
    tau_all_uncalib = pickle.load(data)
    faultyNeurons_recorded = pickle.load(data)
    data.close()

    # mark neurons where fit failed
    faulty = np.array([x for x in tau_all_calib if x[0]
                       in faultyNeurons or x[0] in faultyNeurons_recorded])
    toPlot_calib = np.array([x for x in tau_all_calib if x[0] in neuronIDs])
    toPlot_uncalib = np.array(
        [x for x in tau_all_uncalib if x[0] in neuronIDs])
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
    ax2.set_xlabel('Neuron ID')
    ax2.set_ylabel('Mean tau_mem [ms]')

    ax3 = fig.add_subplot(111)
    ax3.axhline(y=tau_mem, label='set tau_mem', color='k')

    if len(faulty > 0):
        # mark neurons for which calibration has failed
        ax4 = fig.add_subplot(111)
        ax4.plot(faulty[:, 0], faulty[:, 1], 'ko', label='faulty')

    plt.xlim(min(neuronIDs) - 2, max(neuronIDs) + 2)
    if randomNeuronParams:
        plt.title('Calibrated vs uncalibrated (seeded params)')
    else:
        plt.title('Calibrated vs uncalibrated (no seeds)')
    plt.legend()
    if saveData:
        figname = file_comparison[:len(file_comparison) - 4] + '.png'
        plt.savefig(figname)
    return fig
