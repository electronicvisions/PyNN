import pyNN.hardware.spikey as p
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import os
import sys

# save string to report file TODO: use docstrings


def report(string, filename, append=True):
    if filename == None:
        return
    if append:
        option = 'a'
    else:
        option = 'w'
    outfile = file(filename, option)
    outfile.write(string + '\n')
    outfile.close()

# save 1D array entries in separate columns


def save_to_columns(array, filename, delimiter='\t', append=True):
    array = np.array(array)
    assert len(array.shape) == 1, 'save_to_columns only takes 1D arrays'
    if append:
        outfile = file(filename, 'a')
    else:
        outfile = file(filename, 'w')
    np.savetxt(outfile, np.atleast_2d(array), delimiter=delimiter)
    outfile.close()

# smoothen by means of simple moving average


def smoothen(array, n):
    smooth_values = []
    for j in range(len(array) - n + 2):
        mean = 1. / n * sum(array[j:j + n - 1])
        smooth_values.append(mean)
    return np.array(smooth_values)

# dependency of membrane potential on tau_mem


def v(t, tau_mem, v_rest, v_reset):
    return v_rest - (v_rest - v_reset) * np.exp(-t / tau_mem)

# find analog spikes in given membrane trace


def find_spikes(mem, time, spikesDig=[], returnDigitalSpikes=False, reportFile=None):
    # detect spikes in membrane
    maxSpikesLength = 0.5  # ms
    # derive membrane
    derivedMem = mem[1:] - mem[:-1]
    threshSpike = (np.max(derivedMem) - np.min(derivedMem)) / \
        2.0 + np.min(derivedMem)
    derivedMemTh = time[derivedMem < threshSpike]
    spikesMem = []
    # only one spike if derivative is above threshold over many consecutive
    # samples
    for i in range(0, len(derivedMemTh) - 1, 1):
        if not derivedMemTh[i + 1] - derivedMemTh[i] < maxSpikesLength:
            spikesMem.append(derivedMemTh[i])

    # calculate ISIs
    spikesDig = np.array(spikesDig)
    isiDigital = spikesDig[1:] - spikesDig[:-1]
    isiDigitalMean = isiDigital.mean()

    spikesMem = np.array(spikesMem)
    isiAnalog = spikesMem[1:] - spikesMem[:-1]
    isiAnalogMean = isiAnalog.mean()

    print 'number of spikes (digital, analog):', len(spikesDig), len(spikesMem)
    print 'frequency (digital, analog) [Hz]:', 1e3 / isiDigitalMean, 1e3 / isiAnalogMean
    ratioDigAna = isiDigitalMean / isiAnalogMean
    print 'frequency digital to analog (abs, %):', ratioDigAna, (ratioDigAna - 1) * 1e2
    if (ratioDigAna - 1) > 5e-3:
        print 'digital and analog time domain differ more than 0.5 %'
        spikesMem = None
        isiAnalogMean = isiDigitalMean

    if returnDigitalSpikes:
        return spikesDig, isiDigitalMean
    return spikesMem, isiAnalogMean


### get tau_ref from given trace ###
# in order to fit tau_ref correctly, the membrane trace should comprise
# several spikes with long temporal distances between them where the
# membrane potential returns to resting potential!
def fit_tau_refrac(trace, timestep, spikes, meanISI, noDigSpikes, reportFile=None, debugPlot=False):

    # if no or too few spikes occured, tau_ref cannot be determined
    if noDigSpikes <= 3:
        return -1, 0, False

    # check if any bursts occured and rule out double spikes
    spikes = np.array(spikes)
    diffs = spikes[1:] - spikes[:-1]
    spikes_corrected = []
    double_spikes = False
    if not (diffs > 0.5 * meanISI).all():
        double_spikes = True
        j = 0
        while j < len(diffs):
            spikes_corrected.append(spikes[j])
            if j == len(diffs) - 1:
                if spikes[j + 1] - spikes[j] > 0.5 * meanISI:
                    spikes_corrected.append(spikes[j + 1])
                break
            if diffs[j] > 0.5 * meanISI:
                j += 1
            else:
                j += 2
        spikes = np.array(spikes_corrected)
    meanISI = np.mean(spikes[1:] - spikes[:-1])

    tau_refrac = []
    for i in range(1, len(spikes) - 1):
        # the part of the trace which is examined comprises spike i and stops
        # well before the next spike to avoid measurement failures
        stop = int(spikes[i + 1] / timestep - 0.3 * meanISI / timestep)
        voltage_values = trace[int(spikes[i] / timestep):stop + 1]
        start = np.where(voltage_values[:20] == min(voltage_values[:20]))
        voltage_values = voltage_values[min(start[0]):]

        if len(voltage_values) == 0:
            return -1, 0, double_spikes

        # determine tau_refrac
        x = 0
        n = 5
        # choose this part of the trace as reference for the trace's values at
        # resting potential
        rest_start = int(0.2 * meanISI / timestep)
        rest_stop = int(0.4 * meanISI / timestep)

        assert rest_stop - rest_start > 0.1 * meanISI / \
            timestep, 'rest_start/rest_stop: {0}, {1}'.format(
                rest_start, rest_stop)

        deriv = voltage_values[1:] - voltage_values[:-1]
        deriv_smooth = smoothen(deriv, n)
        resting_state = deriv_smooth[rest_start:rest_stop]
        if len(resting_state) == 0:
            return -1, 0, double_spikes
        limit = 1.1 * max(deriv_smooth[rest_start:rest_stop])
        for k in range(len(deriv_smooth) - n):
            if (deriv_smooth[k:k + n] > limit).all():
                x = k + 1
                break

        # tau_ref value determined from measurement
        tau_ref = (x + (n - 1) / 2) * timestep

        # check if tau_ref is plausible
        voltage_smooth = smoothen(voltage_values, n)
        # normalize voltage values
        diffs = voltage_smooth - np.mean(voltage_smooth[rest_start:rest_stop])
        # first and second derivative
        diffs_der1 = diffs[1:] - diffs[:-1]
        reference = smoothen(diffs_der1, n)
        index = int(tau_ref / timestep) - 2
        #diffs_der2 = diffs_der1[1:] - diffs_der1[:-1]
        #index = min(np.where(diffs_der2 <= max(diffs_der2[rest_start:rest_stop])[0]))

        vals = []
        if (abs(reference[index:index + 10]) > 1.5 * max(reference[rest_start:rest_stop])).any():
            if x > 3:
                tau_refrac.append(tau_ref)
            # treat very small values of tau_ref differently. Better results
            # since sampling is very coarse compared to measured tau_ref
            else:
                vals = deriv_smooth[:10]
                vals = deriv_smooth[
                    np.where(vals >= 5 * min(deriv_smooth[rest_start:rest_stop]))]
                vals = deriv_smooth[
                    np.where(vals <= 5 * max(deriv_smooth[rest_start:rest_stop]))]
                tau_ref = len(vals) * timestep
                tau_refrac.append(tau_ref)
        else:
            tau_ref = 0.0
            tau_refrac.append(tau_ref)

        if debugPlot:
            import matplotlib.pyplot as plt
            if i == 5:
                print 'measured tau:', tau_ref
                print 'measured x:', x * timestep
                time = timestep * np.arange(len(voltage_values))
                plt.figure()
                plt.subplot(311)
                plt.plot(time[:len(voltage_smooth)], voltage_smooth)
                plt.axhline(y=min(voltage_smooth))
                plt.axvline(x=tau_ref, color='r')
                plt.subplot(312)
                plt.plot(time[:len(deriv_smooth)], deriv_smooth)
                plt.axhline(y=limit, color='g')
                plt.axvline(x=x * timestep, color='r')
                plt.subplot(313)
                plt.plot(time, voltage_values)
                plt.axvline(x=tau_ref, color='r')
                plt.xlabel('time [ms]')
                plt.show()

    if len(tau_refrac) < 0.5 * len(spikes):
        report('Could not determine tau_refrac because {0} % of the fits failed'.format(
            (1 - len(tau_refrac) / float(len(spikes))) * 100), reportFile)
        return -1, 0, double_spikes

    # return mean tau_refrac and corresponding error for this neuron
    tau_refrac = np.array(tau_refrac)
    tau = np.mean(tau_refrac)
    err = np.std(tau_refrac)
    print 'refractory period is {0} +/- {1} ms'.format(tau, err)
    return tau, err, double_spikes


### get tau_mem from given trace ###
def fit_tau_mem(trace, memtime, dig_spikes, timestep, reportFile=None, debugPlot=False):
    #debugPlot = True
    if debugPlot:
        plt.figure()

    # read out spikes in trace
    spikes, meanISI = find_spikes(trace, memtime, dig_spikes, reportFile)
    if spikes == None:
        report('Digital and analog spikes differ too much in find_spikes()', reportFile)
        assert False, 'bad membrane recording'

    memList = []
    shortest = np.inf
    # skip first inter-spike interval, could be incomplete
    for i in range(1, len(spikes) - 1):
        interspikeMem = trace[
            int(round(spikes[i] / timestep)):int(round(spikes[i + 1] / timestep))]
        shortest = np.min([shortest, len(interspikeMem)])
        memList.append(interspikeMem)
    # cut to identical length
    memListAligned = []
    for interspikeMem in memList:
        memListAligned.append(interspikeMem[:shortest])
        if debugPlot:
            plt.plot(np.arange(len(interspikeMem)) *
                     timestep, interspikeMem, 'gray')
    mem = np.mean(memListAligned, axis=0)
    memStd = np.std(memListAligned, axis=0)
    if debugPlot:
        plt.axvline(shortest * timestep, c='b')
        plt.errorbar(np.arange(len(mem)) * timestep,
                     mem, yerr=memStd, c='k', lw=2)

    # cut refractory period
    timeStart = 2.0  # TODO: from ankis talk #0.2 * meanISI #TODO: detect?, see tau_refrac calib
    timeStartSamples = int(round(timeStart / timestep))
    # and approx. 0.5 ms before next spike
    timeStop = 0.5
    timeStopSamples = int(round(timeStop / timestep))
    mem = mem[timeStartSamples:-timeStopSamples]
    memStd = memStd[timeStartSamples:-timeStopSamples]
    time = (np.arange(len(mem)) + timeStartSamples) * timestep

    # too short membrane potential trace
    if len(mem) == 0:
        report('Too short membrane potential trace', reportFile)
        return None

    try:  # TODO: one day curve_fit has return value for fit result status like leastsq
        params, cov = curve_fit(v, time, mem, sigma=memStd / mem)
    except:
        report('Fit of membrane time constant failed', reportFile)
        return None
    if type(cov) == float:
        report('Fit of membrane time constant failed', reportFile)
        return None
    if params[0] > 500.0:
        report('Bad fit of membrane time constant', reportFile)
        return None

    tau_mem = params[0]
    v_rest = params[1]
    v_reset = params[2]

    # diagonal elements for variance
    tau_mem_error = cov[0, 0]
    v_rest_error = cov[1, 1]
    v_reset_error = cov[2, 2]

    if debugPlot:
        plt.axvline(timeStart, c='r')
        plt.axvline(shortest * timestep - timeStop, c='r')
        plt.plot(time, v(time, params[0], params[1], params[2]), 'm')
        plt.show()

    print 'tau_mem:', tau_mem, '+/-', tau_mem_error
    print 'v_rest:', v_rest, '+/-', v_rest_error
    print 'v_reset:', v_reset, '+/-', v_reset_error

    return tau_mem, tau_mem_error, v_rest, v_rest_error, v_reset, v_reset_error


################################# fit and plot dependency of tau_mem on iL

### fit tau_mem vs. iLeak dependency ###
def fit_dependency(neuronIDs, prefix_rawData, filename_result, overwrite_data=True, reportFile='./report.dat', fitMaxRes=0.5):

    report('FITTING DEPENDENCY OF TAU MEM ON ILEAK FOR NEURONS {0} TO {1}'.format(
        min(neuronIDs), max(neuronIDs)), reportFile)
    # save zeros for missing neurons or failed fits

    def save_ones(neuron):
        to_save = np.concatenate(
            (np.array([neuron]), np.ones(5), [sys.maxint]))
        save_to_columns(to_save, filename_result)

    failed_fits = []
    missing_data = []

    # if file for results already exists, overwrite it or return
    if os.path.isfile(filename_result):
        if overwrite_data:
            os.remove(filename_result)
        else:
            print 'File {0}.data already exists, using existing fit'.format(filename_result)
            report('Cannot fit tau_mem dependency because file {0} already exists'.format(
                filename_result), reportFile)
            return

    # fill lines for left block of chip with zeros
    for neuron in range(384):
        if neuron not in neuronIDs:
            save_ones(neuron)
            continue

        # if no data for given neuron exists, fill fit column with zeros
        filename_data = prefix_rawData + '_neuron' + \
            str(neuron).zfill(3)  # TODO: one file
        if not os.path.isfile(filename_data + '.dat'):
            missing_data.append(neuron)
            save_ones(neuron)
            report('no data file exists for neuron {0}'.format(
                neuron), reportFile)
            continue
        else:
            data = np.atleast_2d(np.loadtxt(
                filename_data + '.dat', delimiter='\t'))

            # get recorded tau_mem and iLeak values
            assert data.shape[1] == 14, 'data in file {0} must have 14 columns'.format(
                filename_data + '.dat')
            tau_mem = data[:, 1]
            tau_mem_error = data[:, 7]
            iLeak_inv = 1. / data[:, 0]

            # exclude possible NaNs and infs
            mask = np.where(np.isfinite(tau_mem))
            tau_mem = tau_mem[mask]
            tau_mem_error = tau_mem_error[mask]
            iLeak_inv = iLeak_inv[mask]

            # don't fit if too few data points for fit have been recorded
            if len(tau_mem) < 2:
                report('Too few data points ({0}) for a quadratic fit for neuron {1}. Will proceed to next neuron'.format(
                    len(tau_mem), neuron), reportFile)
                failed_fits.append(neuron)
                save_ones(neuron)
                continue

            # fit according to given dependency
            # TP (07.07.2015): TODO: include weights (errors), then allow bad
            # fits of membrane time constant
            result = np.polyfit(tau_mem, iLeak_inv, 2, full=True)
            pol = result[0]
            res = result[1][0] / len(tau_mem)

            if res > fitMaxRes:
                report('Bad fit (residual = {0}) for neuron {1}. Will proceed to next neuron'.format(
                    res, neuron), reportFile)
                failed_fits.append(neuron)

            # save neuron number, fitted parameters
            to_save = np.concatenate(
                ([neuron], pol, [min(1. / iLeak_inv), max(1. / iLeak_inv)], [res]))
            save_to_columns(to_save, filename_result)

    print 'successful fits: {0}/{1}'.format(len(neuronIDs) - len(missing_data) - len(failed_fits), len(neuronIDs) - len(missing_data))
    if len(missing_data) > 0:
        print 'Missing data for neurons {0}'.format(missing_data)

    if len(failed_fits) > 0:
        report('Bad fit or too few data points for neurons {0}'.format(
            failed_fits), reportFile)


### plot dependency and fit if it exists ###
def plot_dependency(ax, neuron, prefix_rawData, filename_result):

    # choose random but same color for data and fit
    color = np.random.rand(3,)

    # choose neuron to plot
    filename_data = prefix_rawData + '_neuron' + str(neuron).zfill(3)

    # does raw data for neuron exist?
    if not os.path.isfile(filename_data + '.dat'):
        print 'No data was recorded for neuron {0}'.format(neuron)
        return

    # load data
    data = np.atleast_2d(np.loadtxt(filename_data + '.dat', delimiter='\t'))
    tau_mem = data[:, 1]
    tau_mem_error = data[:, 7]
    iLeak_inv = 1. / data[:, 0]
    # exclude possible NaNs and infs
    tau_mem = tau_mem[np.where(np.isfinite(tau_mem))]
    tau_mem_error = tau_mem_error[np.where(np.isfinite(tau_mem_error))]
    iLeak_inv = iLeak_inv[np.where(np.isfinite(tau_mem))]

    # plot recorded tau_mem against iLeak
    #ax.errorbar(iLeak_inv, tau_mem, yerr=tau_mem_error, marker='o', ls='', color=color, label='neuron {0}'.format(neuron))
    ax.plot(iLeak_inv, tau_mem, marker='o', ls='',
            color=color, label='neuron {0}'.format(neuron))

    # load fit
    if not os.path.isfile(filename_result):
        print 'no fit file with filename {0}'.format(filename_result)
        return
    fit = np.atleast_2d(np.loadtxt(filename_result, delimiter='\t'))
    assert fit.shape[
        1] == 7, 'entries in fit file must include neuron, polynom coefficient 1, 2, 3, min iLeak, max iLeak, residual'

    mask = fit[:, 0] == neuron
    if (mask == False).all():
        # if neuron not in given file, return
        print 'no fit for neuron {0} exists in file {1}'.format(neuron, filename_result)
        return

    # if line for this neuron just comprises zeros, return
    if (fit[mask][0][1:] == 0).all():
        return

    # plot fit
    pol = fit[mask][0][1:4]
    tauMem_range = np.linspace(
        np.min(tau_mem) * 0.8, np.max(tau_mem) * 1.2, 1000)
    ax.plot(np.polyval(pol, tauMem_range), tauMem_range,
            color=color, label='neuron {0}'.format(neuron))
    ax.set_xlabel('1/iLeak')
    ax.set_ylabel('tau_mem [ms]')
