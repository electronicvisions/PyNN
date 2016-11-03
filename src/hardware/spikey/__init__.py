# *******************************************************************************
# To help you understand the flow of parameter setting within the architecture
# of pyNN and its hardware.spikey module, here is what happens after calling a
# simple command like
#     >> neuron.v_thresh = -50.0
# where type(neuron) == ID and neuron.cellclass == IF_facets_hardware1 .
# The class ID is derived from the class common.IDMixin .
#
#
# (1) class common.IDMixin.__setattr__ converts the command to       <--+
#                                                                       |
# (2) common.IDMixin.set_parameters(**{'v_thresh': -50.0}) which     <--+
#                                                                       |
# (3) "translates" the standard parameter 'v_thresh' to the    +--------+--------+
#     hardware-specific parameter 'v_thresh' via               |methods of neuron|
#     common.StandardModelType.translate() using the           +--------+--------+
#     translations built in class cells.IF_facets_hardware1             |
#     and calls                                                         |
#                                                                       |
# (4) ID.set_native_parameters({'v_thresh': -50.0}). This function   <--+
#     calls the pyNN-low-level function
#     set(neuron, cells.IF_facets_hardware1, {'v_thresh': -50.0})
#
# (5) Now, information splits up. set() stores it in two
#     different neuron types:
#     (a) neuron.cell.parameters.update({'v_thresh': -50.0})
#         Here the pyNN-command-flow ends. And:
#     (b) hardware.net.neuron[int(neuron)].parameters.update({'v_thresh': -50.0})
#         Notice: hardware.net.neuron[] holds
#             pyhal.pyhal_buildingblocks_s1v2.Neuron's, a hardware specific type.
#
# All lower hardware modules will use the data stored in (b)!
# *******************************************************************************

import sys
import os
if os.environ.has_key('PYNN_HW_PATH'):
    pynn_hw_path = os.environ['PYNN_HW_PATH']
else:
    raise EnvironmentError(
        'ERROR: The environment variable PYNN_HW_PATH is not defined!')

sys.path.insert(0, pynn_hw_path)
sys.path.insert(0, os.path.join(pynn_hw_path, "pyhal"))
sys.path.insert(0, os.path.join(pynn_hw_path, "tools"))
sys.path.insert(0, os.path.join(pynn_hw_path, "config"))

import logger
import pylogging as pylog
myLogger = pylog.get("PyN.ini")

from pyhal_c_interface_s1v2 import hardwareAvailable

import numpy
import types
import time
import fcntl
import sets

from pyNN import common
from pyNN import utility
import workstation_control
import pyhal_s1v2 as hardware
import pyhal_neurotypes as neurotypes
import hwconfig_default_s1v2 as hwconfig_default

import simulator
common.simulator = simulator

from population import *
from projection import *
from cells import *
from synapses import *
from simulator import *


# dummy function that always returns False
def alwaysReturnFalse():
    return False

############################
##  VARIABLES DEFINITION  ##
############################

# variables intended for public access
# reference to the container holding the network's spike output of the last run
spikeOutput = None
# reference to the container holding the recorded neuron's membrane trace
# of the last run
membraneOutput = numpy.array([])
# reference to the container holding the time stamps for membrane trace
timeMembraneOutput = numpy.array([])
# number of input spikes lost during the last run
numLostInputSpikes = 0
# total number of input spikes (lost ones included) during the last run
numInputSpikes = 0


# variables NOT intended for public access
# this module's random number generator
_globalRNG = common.random.NumpyRNG().rng
# flag to check if the setup() method has been called already
_calledSetup = False
# flag that determines if debug output is generated or not
_debug = False
# flag for signalling a neuron parameter change, helps to reduce PC <->
# chip traffic
_neuronsChanged = True
# flag for signalling a synapse parameter change, helps to reduce PC <->
# chip traffic
_synapsesChanged = True
# flag for signalling a connectivity parameter change, helps to reduce PC
# <-> chip traffic
_connectivityChanged = True
# flag for signalling an input parameter change, helps to reduce PC <->
# chip traffic
_inputChanged = True
_iteration = 0                             # counts the number of experiment runs
# container for external input objects (not IDs)
_externalInputs = []
# container for the indices of those neurons of which the output spikes
# will be recorded
_spikes_recordedNeurons = []
_spikes_recordedInputs = []
_spikes_recordInputFilenames = []
# container for the neuron output spike recordings' target filenames
_spikes_recordFilenames = []
# the minimum inter-spike interval that is allowed; spikes which violate
# this will be dropped.
_minimumISI = 0.0
# container for the indices of those neurons of which the membrane
# potential will be recorded
_memPot_recordedNeurons = []
# container for the neuron membrane potential recordings' target filenames
_memPot_recordFilenames = []
# save spikepatterns and membrane potentials to file 'filename.#.ext'
# instead of 'filename.ext' where # is pyNN._iteration
_incrementFilename = False
# dictionary for hardware specific parameters, to be filled from a file
# called 'workstations.xml'
_hardwareParameters = {}
# container for the random number generation seeds
_seeds = [0]
# temporal resolution of the acquisition of the analog data (currently via
# oscilloscope), in msec
_dt = 0.0
# station params from workstations.xml - written by setup()
_stationDict = {}
# decides if a warning is printed in case a neuron is not recordable
_suppressRecWarnings = True
# this flag tells if the setup() method has been called already - if not,
# check if the user owns the workstation
_isFirstSetup = True
# the path to look for calibration files
_basePath = workstation_control.basePath
# flag for triggering read-back of weights
_retrieveWeights = False
# flag for triggering read-back of correlation flags
_retrieveCorrelationFlags = False
_appliedInputs = None
_useUsbAdc = False
# write spiketrain.in and spikeyconfig.out to file
_writeConfigToFile = True
# for the runWithDeveloperInterrupt keyword
_calledRunSinceLastExperimentExecution = False

# soft profiling
_timeSetupPyNN = 0
_timeRunPyNN = 0
_timeConfigPyNN = 0
_timeEncodePyNN = 0
_timeRunPyHAL = 0
_timeMemPyNN = 0
_timeEndPyNN = 0


###########################
##  PyNN Procedural API  ##
###########################

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def reset():
    """This function does currently nothing."""
    pass


def setup(timestep=0.1, min_delay=0.1, max_delay=0.1, debug=False,
          useUsbAdc=False, workStationName='', mappingOffset=0, neuronPermutation=[],
          calibOutputPins=True, calibNeuronMems=False, calibTauMem=True, calibSynDrivers=True, calibVthresh=True, calibBioDynrange=False,
          **extra_params):
    """
    Sets up all necessary data structures. Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given simulator but not by others.
        hw-specific extra_params:
        * boolean incrementFilename (default: False):
            save spikepatterns and membrane potentials to file 'filename.#.ext' instead of 'filename.ext' where # is pyNN._iteration
        * boolean outAmpZero (default: False):
            set membrane output amplifier biases to zero
        * boolean writeConfigToFile (default: True):
            write spiketrain.in and spikeyconfig.out to file
        * boolean dryRun (default: False):
            run without using hardware
    """

    startTime = time.time()

    # instantiate logger
    if 'loglevel' in extra_params.keys():
        loglevel = extra_params['loglevel']
    else:
        loglevel = 2  # INFO

    if 'logfile' in extra_params.keys():
        logfile = extra_params['logfile']
    else:
        logfile = 'logfile.txt'

    logger.loggerAppendFile(logfile)
    logger.loggerSetLevel(loglevel)

    if extra_params.has_key('dryRun') and extra_params['dryRun']:
        _dryRun = True

    # print the current spikey chip version
    if not hardwareAvailable() and not _dryRun:
        raise Exception(
            'ERROR: The neuromorphic hardware system is not available!')
    else:
        myLogger.debug('Loaded hardware module from ' + pynn_hw_path)

    # get write access for global parameters
    global _neuronsChanged
    global _synapsesChanged
    global _connectivityChanged
    global _inputChanged
    global _chipParamsChanged
    global _hardwareParameters
    global _externalInputs
    global _spikes_recordedNeurons
    global _spikes_recordedInputs
    global _spikes_recordFilenames
    global _spikes_recordInputFilenames
    global _minimumISI
    global _memPot_recordedNeurons
    global _memPot_recordFilenames
    global _incrementFilename
    global _calledSetup
    global _dt
    global acquisitionTrigger
    global _stationDict
    global _isFirstSetup
    global _seeds
    global _globalRNG
    global _iteration
    global _debug
    global _useUsbAdc
    global _writeConfigToFile
    global _timeSetupPyNN

    # check if minimum ISI is given
    if 'minimumISI' in extra_params:
        assert isinstance(extra_params[
                          'minimumISI'], (float)), 'ERROR: pyNN.setup: minimumISI must be of type float!'
        _minimumISI = extra_params['minimumISI']
        myLogger.warn('All spikes with an ISI smaller than ' +
                      str(_minimumISI) + 'ms will be ignored!')

    # check for random number generator seed
    if 'rng_seeds' in extra_params:
        _seeds = extra_params['rng_seeds']
        _globalRNG.seed(extra_params['rng_seeds'][0])

    # check if full reset shall be performed
    if 'fullReset' in extra_params:
        fullReset = extra_params['fullReset']
    else:
        fullReset = False

    # check if spike-silence shall be guaranteed between experiments
    if 'assertSilence' in extra_params:
        assertSilence = extra_params['assertSilence']
    else:
        assertSilence = False

    # check if minimum parameter value shall be zero or LSB
    if 'defaultValue' in extra_params:
        defaultValue = extra_params['defaultValue']
    else:
        defaultValue = 0.05
    myLogger.debug(
        'The default value for undefined parameters is ' + str(defaultValue) + '.')

    # check if the ratio SuperThreshold-to-SubThreshold was defined explicitly
    if 'ratioSuperthreshSubthresh' in extra_params:
        ratioSuperthreshSubthresh = extra_params['ratioSuperthreshSubthresh']
    else:
        ratioSuperthreshSubthresh = 0.8
    myLogger.debug('The ratio SuperThreshold-to-SubThreshold is ' +
                   str(ratioSuperthreshSubthresh) + '.')

    # check if spiketrain.in and spikeyconfig.out should be written to file
    if 'writeConfigToFile' in extra_params:
        _writeConfigToFile = extra_params['writeConfigToFile']
    else:
        _writeConfigToFile = True

    # fill global parameter values
    _dt = timestep
    _hardwareParameters = {
        'spikeyNr':   0,
        'spikeyVersion':   3,
        'spikeyClk':   10.0e-9,
        'mappingOffset':   mappingOffset,
        'speedup':   1.0e5,
        'calibfileOutputPins':   '/config/calibration/calibOutputPins.pkl',
        'calibfileIcb':   '/config/calibration/calibIcb.dat',
        'calibfileTauMem':   '/config/calibration/calibTauMem.dat',
        'calibfileSynDriverExc':   '/config/calibration/calibSynDrivers.dat',
        'calibfileSynDriverInh':   '/config/calibration/calibSynDrivers.dat',
        'calibfileVthresh':   '/config/calibration/calibVthresh.dat',
        # Additional calibration files
        'calibBioDynrange':   calibBioDynrange,
        'calibfileBioDynrange':   '/config/calibration/calibfileBioDynrange.dat',
        # Acquisition device
        'acquisitionDevice':   'facetsscope1',
        'acquisitionTriggerOffset':   0.0,
        'usbadcTimeFactor':   1.0,
        'acquisitionDeviceInputs':   {'channel1': -1,
                                      'channel2': -1,
                                      'channel3': -1,
                                      'channel4': -1},
        'avoidSpikes':   False,
        'recNeurons':   '',
        #        'voutLower'                      :   0.8,
        'voutLower':   0.5,
        'voutUpper':   1.4,
        #        'voutMins'                        :   numpy.ones(50) * 0.8,
        'voutMins':   numpy.ones(50) * 0.5,
        'voutMaxs':   numpy.ones(50) * 1.4,
        'adcCalibFactor': -2.8,
        'adcCalibOffset':   3235.0
    }
    _debug = debug

    # check if avoidSpikes
    if 'avoidSpikes' in extra_params:
        _hardwareParameters['avoidSpikes'] = extra_params['avoidSpikes']

    # this calib flag is kept for backwards compatibility, but will probably
    # be never needed again
    if 'calibIcb' in extra_params:
        calibIcb = extra_params['calibIcb']
    else:
        calibIcb = False

    # set all changed-flags to true
    _neuronsChanged = True
    _synapsesChanged = True
    _connectivityChanged = True
    _inputChanged = True
    _chipParamsChanged = True
    if fullReset:
        _iteration = 0

    # initialize hardware access
    hardware.initialize(defaultValue=defaultValue,
                        fullReset=fullReset, debug=_debug, **extra_params)
    workStationName = hardware.getBus(workStationName)

    # get workstation of individual user, fill _hardwareParameters and check
    # if the user owns the workstation
    _stationDict = workstation_control.getWorkstation(
        workStationName, _isFirstSetup)
    _isFirstSetup = False
    for k in _stationDict.keys():
        if _hardwareParameters.has_key(k):
            if (len(k) > 3) and (k[:4] == 'vout'):
                _hardwareParameters[k] = _stationDict[k]
            else:
                _hardwareParameters[k] = type(
                    _hardwareParameters[k])(_stationDict[k])

    # bio time <-> chip clock conversion
    # TP: for Spikey v4 and v5 clock for events runs with 200MHz, "Spikey
    # clock" runs with 100MHz
    # event generation clock is half of spikey clock!
    eventGenerationClk = _hardwareParameters['spikeyClk'] / 2.0
    hardwareBinsPerBioSec = 16.0 / \
        (eventGenerationClk * _hardwareParameters['speedup'])
    _hardwareParameters['hardwareBinsPerBioSec'] = hardwareBinsPerBioSec
    _hardwareParameters[
        'hardwareBinsPerBioMilliSec'] = hardwareBinsPerBioSec / 1000.0

    # check if programmable output pin is connected to scope
    acquisitionChannel = -1
    aquisitionTrigger = -1
    for channel in _hardwareParameters['acquisitionDeviceInputs'].keys():
        if int(_hardwareParameters['acquisitionDeviceInputs'][channel]) == 8:
            acquisitionChannel = int(channel[-1])
            # break
        elif int(_hardwareParameters['acquisitionDeviceInputs'][channel]) == 9:
            aquisitionTrigger = int(channel[-1])

    # prepare hardware
    hardware.getHardware(workStationName,
                         _hardwareParameters['spikeyNr'],
                         _hardwareParameters['spikeyClk'] * 1.0e9,
                         _hardwareParameters['voutMins'],
                         _hardwareParameters['voutMaxs'],
                         calibOutputPins,
                         calibNeuronMems,
                         calibIcb,
                         calibTauMem,
                         calibSynDrivers,
                         calibVthresh,
                         # New calibration options
                         calibBioDynrange,
                         #                         calibWeightsExc,
                         #                         calibWeightsInh,
                         #
                         os.path.join(_basePath, _hardwareParameters[
                                      'calibfileOutputPins']),
                         os.path.join(_basePath, _hardwareParameters[
                                      'calibfileIcb']),
                         os.path.join(_basePath, _hardwareParameters[
                                      'calibfileTauMem']),
                         os.path.join(_basePath, _hardwareParameters[
                                      'calibfileSynDriverExc']),
                         os.path.join(_basePath, _hardwareParameters[
                                      'calibfileSynDriverInh']),
                         os.path.join(_basePath, _hardwareParameters[
                                      'calibfileVthresh']),
                         # New calibration options
                         os.path.join(_basePath, _hardwareParameters[
                                      'calibfileBioDynrange']),
                         #                         os.path.join(_basePath, _hardwareParameters['calibfileWeightsExc']),
                         #                         os.path.join(_basePath, _hardwareParameters['calibfileWeightsInh']),
                         #
                         neuronPermutation,
                         _hardwareParameters['mappingOffset'],
                         ratioSuperthreshSubthresh)
    hardware.disableSTDP()

    if assertSilence:
        hardware.interruptActivity()

    # create containers for input sources and observable recording
    _externalInputs = []
    _spikes_recordedNeurons = []
    _spikes_recordedInputs = []
    _spikes_recordFilenames = []
    _spikes_recordInputFilenames = []
    _memPot_recordedNeurons = []
    _memPot_recordFilenames = []

    _calledSetup = True

    _timeSetupPyNN += time.time() - startTime

    return 0


def end():
    '''Do any necessary cleaning up before exiting.'''

    global _timeEndPyNN
    startTime = time.time()
    hardware.delHardware()
    _timeEndPyNN += time.time() - startTime


def run(simtime=0, **extra_params):
    """
    Run the simulation for simtime ms.
    extra_params contains any keyword arguments that are required by a given simulator but not by others.
     - ['replay'] if true, the spiketrain will be replayed (faster than retransmitting)
     - ['voltageOnTestPin'] in the range [0,numBlocks*numVoutsPerBlock]:
           - if int: the corresponding vout value of block 0 will be connected to the test pin.
           - if list or numpy.array: multiple vout values of both blocks will be connected to the IBTest-Jumper. The
                                     test pin will be connected to the MUX - so vouts are only accessable by the jumper.
                                     Allows writing vouts instead of reading them.
           Otherwise, the neuron membrane which was flagged to be recorded last will be connected to the test pin.
     - ['retrieveWeights'] retrieves the weight matrix (for all synapses)
     - ['retrieveCorrelationFlags'] retrieves the correlation flag matrix (for all synapses)
     - ['translateToBioVoltage'] determines if recorded voltages will be translated into their biological interpretation (default: True)
    """

    startTimeRun = time.time()

    replay = False
    voltageOnTestPin = None
    if extra_params.has_key('replay'):
        replay = extra_params['replay']
    if extra_params.has_key('voltageOnTestPin'):
        voltageOnTestPin = extra_params['voltageOnTestPin']

    global _retrieveWeights
    global _retrieveCorrelationFlags
    _retrieveWeights = None
    _retrieveCorrelationFlags = None
    if extra_params.has_key('retrieveWeights'):
        retrieveWeights = extra_params['retrieveWeights']
    if extra_params.has_key('retrieveCorrelationFlags'):
        retrieveCorrelationFlags = extra_params['retrieveCorrelationFlags']
    if extra_params.has_key('translateToBioVoltage'):
        translateToBioVoltage = extra_params['translateToBioVoltage']
    else:
        translateToBioVoltage = True

    global _neuronsChanged
    global _synapsesChanged
    global _connectivityChanged
    global _inputChanged
    global _chipParamsChanged
    global _iteration
    global _minimumISI
    global spikeOutput
    global membraneOutput
    global timeMembraneOutput
    global numLostInputSpikes
    global numInputSpikes
    global _appliedInputs
    global _useUsbAdc
    global _writeConfigToFile
    global _timeRunPyNN
    global _timeConfigPyNN
    global _timeEncodePyNN
    global _timeRunPyHAL
    global _timeMemPyNN
    global _dt
    global _calledRunSinceLastExperimentExecution

    # check basic preconditions
    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")
    if hardware.net.size() == 0:
        raise Exception("ERROR: Network size is zero!")
    if hardware.net.size() > hardware.numNeurons():
        raise Exception("ERROR: Network is larger than chip size!")

    # check for changes in setup and, if necessary, reconfigure hardware
    if (not _calledRunSinceLastExperimentExecution) and (_neuronsChanged or _synapsesChanged or _connectivityChanged or _chipParamsChanged):
        startTime = time.time()

        #######                                          ##
        #      ##                                         ####                     #                 H
        #      ##               ##################################                #     xxxxxx      H
        #      ##               #####################################            #     xxxxxx      H
        # xxxxxx      H
       # H
       # HHHHHHHHHHHHHHHHHH

        # this parameter transfer has to happen first, otherwise the other
        # programming will not work
        if _chipParamsChanged:
            hardware.applyChipParamConfig()

        if (_neuronsChanged or _synapsesChanged) and isinstance(voltageOnTestPin, (list, numpy.ndarray)):
            # for writing vouts with external source: clear vouts and
            # voutbiases before spikeycfg is transmitted!!!
            hardware.setVoutsToMin(
                voltageOnTestPin, leftBlock=True, rightBlock=True)

        # prepare hardware for operation, i.e. transfer configuration and input
        # data
        hardware.mapNetworkToHardware(_hardwareParameters['hardwareBinsPerBioSec'], doFlush=(not _neuronsChanged),
                                      neuronsChanged=_neuronsChanged, synapsesChanged=_synapsesChanged, connectivityChanged=_connectivityChanged, avoidSpikes=_hardwareParameters['avoidSpikes'])

        pin = 0  # default membrane pin to multiplex if multiple vouts are assigned to ibtest
        if _neuronsChanged:
            # enable membrane recordings
            membraneCopyToTestPin = not(voltageOnTestPin in range(
                hardware.hwa.numBlocks() * hardware.hwa.numVoutsPerBlock()))
            pin = hardware.monitorMembrane(
                _memPot_recordedNeurons, True, doFlush=False, copyToTestPin=membraneCopyToTestPin)
            # although not clear why: here ALL parameters have to be
            # transferred to the chip again
            hardware.applyConfig()

        if (_neuronsChanged or _synapsesChanged) and isinstance(voltageOnTestPin, (list, numpy.ndarray)):
            # for writing vouts with external source: after clearing voutbiases the source can be connected.
            # testpin is assigned to the MUX and the correct membrane pin.
            hardware.assignMultipleVoltages2IBTest(
                voltageOnTestPin, leftBlock=True, rightBlock=True, pin4Mux=pin)

        if (_neuronsChanged or _synapsesChanged) and (voltageOnTestPin in range(hardware.hwa.numBlocks() * hardware.hwa.numVoutsPerBlock())):
            hardware.assignVoltage2TestPin(voltageOnTestPin % hardware.hwa.numVoutsPerBlock(
            ), int(voltageOnTestPin / hardware.hwa.numVoutsPerBlock()))

        _neuronsChanged = False
        _synapsesChanged = False
        _connectivityChanged = False
        _chipParamsChanged = False

        _timeConfigPyNN += time.time() - startTime

        # Interrupt experiment execution to permit low level hardware parameter manipulation.
        if ("interruptRunAfterMapping" in extra_params) and (extra_params["interruptRunAfterMapping"] is True):
          _calledRunSinceLastExperimentExecution = True
          if _writeConfigToFile:
              hardware.writeConfigFile('spikeyconfig_afterFirstMapping.out')
              hardware.writeInSpikeTrain('spiketrain_afterFirstMapping.in')
          myLogger.info("* * * INTERRUPT pynn.run() since 'interruptRunAfterMapping=True' * * *")
          return hardware.hwa.cfg

    # Rentry point after hardware manipulation
    if _calledRunSinceLastExperimentExecution:
        myLogger.info("* * * CONTINUE pynn.run() after 'interruptRunAfterMapping=True' * * *")
        hardware.hwa.applyConfig()


    if _inputChanged:
        startTime = time.time()

        netsize = hardware.net.size()
        # determine if mirroring of external input sources to both Spikey
        # blocks is necessary
        mirroringNecessary = (hardware.hardwareIndexMax()
                              >= hardware.numNeuronsPerBlock())
        if mirroringNecessary:
            myLogger.debug('Mirroring input to both halves of chip.')
        # generate the spike train container to be received by the hardware
        (hw_st, orig_st) = generateHardwareSpikeTrain(
            mirrorInputs=mirroringNecessary)
        _appliedInputs = orig_st
        # set this container to be the input for this run
        hardware.setInput(hw_st, _hardwareParameters[
                          'hardwareBinsPerBioMilliSec'], duration=simtime * _hardwareParameters['hardwareBinsPerBioMilliSec'])
        # clear playback memory
        hardware.clearPlaybackMem()
        _inputChanged = False

        _timeEncodePyNN += time.time() - startTime

    if _writeConfigToFile:
        hardware.writeConfigFile('spikeyconfig.out')
        hardware.writeInSpikeTrain('spiketrain.in')

    # if not _iteration % 10:
    myLogger.info('### Hardware run ' + str(_iteration).zfill(8) +
                  ' - Tsim = ' + str(simtime) + ' ###')

    # ms in HTD; delay between ADC trigger and replay of playback memory
    offsetTriggerExp = _hardwareParameters['acquisitionTriggerOffset'] * 1e3
    samplingDuration = simtime / \
        _hardwareParameters['speedup'] - offsetTriggerExp  # ms in HTD
    # instantiate the USB ADC device
    if _useUsbAdc:
        startTime = time.time()
        myLogger.debug('setup USB ADC')
        hardware.setupUsbAdc(samplingDuration)
        _timeMemPyNN += time.time() - startTime

    ##########################################################################
    startTimeRunPyhal = time.time()
    (numLostInputSpikes, numInputSpikes) = hardware.run(replay=replay)
    _timeRunPyHAL += time.time() - startTimeRunPyhal
    _iteration += 1
    ##########################################################################

    _calledRunSinceLastExperimentExecution = False

    if _useUsbAdc:
        startTime = time.time()
        myLogger.info('Receive membrane data.')
        _dt = 1 / 96.0 / 1000.0  # ADC has 96MHz clock
        membraneOutput = hardware.readUsbAdc(
            _hardwareParameters['adcCalibFactor'], _hardwareParameters['adcCalibOffset'])
        lowerIndex = int(round(-offsetTriggerExp / _dt))
        higherIndex = int(round(samplingDuration / _dt))
        membraneOutput = membraneOutput[lowerIndex:higherIndex + 1]
        myLogger.debug('acquired %i samples which corresponds to %.2fms' % (len(
            membraneOutput), (len(membraneOutput) - 1) * _dt * _hardwareParameters['speedup']))
        timeMembraneOutput = numpy.arange(
            0, samplingDuration + _dt, _dt) * _hardwareParameters['speedup']
        timeMembraneOutput = timeMembraneOutput[0:len(membraneOutput)]
        _timeMemPyNN += time.time() - startTime

    # hardware.writeOutSpikeTrain('spiketrain.out')

    def writeSpikeData(recCells, recFilenames, spikeData):

        filenameToNeuronList = {}
        for filename in sets.Set(recFilenames):
            filenameToNeuronList[filename] = []
        for neuron, filename in zip(recCells, recFilenames):
            filenameToNeuronList[filename].append(neuron)
        for filename, neuronlist in filenameToNeuronList.items():
            if filename in ['', None]:
                continue
            # Even if there is no spike data, we don't want to throw
            # exceptions, see software simulators:
            myLogger.warn(
                'this may be very slow: using old syntax (pynn.record(neuronID)) to get spikes')
            spikeDataTemp = numpy.transpose(
                numpy.array([spikeData[1], spikeData[0]]))
            try:
                cleanedOutput = [spikeDataTemp[
                    spikeDataTemp[:, 1] == r] for r in neuronlist]
                cleanedOutput = numpy.concatenate(tuple(cleanedOutput), axis=0)
                sortidx = numpy.argsort(cleanedOutput[:, 0])
                cleanedOutput = cleanedOutput[sortidx, :]
                numpy.savetxt(filename, cleanedOutput, delimiter='\t')
            except IndexError:
                numpy.savetxt(filename, [])

    # get digital output spike data
    numRecs = len(_spikes_recordFilenames)

    if numRecs > 0:
        out = hardware.getOutput(
            runtime=simtime, numNeurons=hardware.net.size(), minimumISI=_minimumISI)
        # TODO: maybe obsolete, but needed for old pyNN syntax, used e.g. in
        # tau_mem calib
        spikeOutput = out
        writeSpikeData(_spikes_recordedNeurons, _spikes_recordFilenames, out)

    # get digital input data
    numRecs = len(_spikes_recordInputFilenames)

    if numRecs > 0:
        out = _appliedInputs
        writeSpikeData(_spikes_recordedInputs,
                       _spikes_recordInputFilenames, out)

    if _useUsbAdc and len(_memPot_recordFilenames) > 0:
        startTime = time.time()

        # transform scope output with respect to impedance distortion
        if not _hardwareParameters['calibBioDynrange']:
            myLogger.info('Transforming membrane data.')

            # manipulate scope output according to output pin calibfile entry
            # because there are different resistances on nathan boards
            recordedNeuron = hardware.hwa.hardwareIndex(
                _memPot_recordedNeurons[0])
            # For spikey version 4 hardwareIndex returns 192..383 (right block only),
            # consequently only the rightmost pins must be used:
            neuronsPerBlock = hardware.hwa.numNeuronsPerBlock()
            recordPin = recordedNeuron % 4 + \
                int(recordedNeuron / neuronsPerBlock) * 4

            if hardware.hwa.calibOutputPins:
                membraneOutput = numpy.polyval(hardware.hwa.outputPinsFit[
                                               recordPin], membraneOutput * hardware.hwa.voltageDivider)
            elif hardware.hwa.calibNeuronMems:
                membraneOutput = numpy.polyval(hardware.hwa.neuronMemsFit[
                                               recordedNeuron], membraneOutput * hardware.hwa.voltageDivider)
            else:
                membraneOutput *= hardware.hwa.voltageDivider

        # translate hardware voltage to biological interpretation
        if translateToBioVoltage:
            # Optionally translate to the bio voltage domain
            if _hardwareParameters['calibBioDynrange']:
                # translate hardware voltage to biological interpretation
                try:
                    recordedNeuron = hardware.hwa.hardwareIndex(
                        _memPot_recordedNeurons[0])
                    slope, offset = hardware.hwa.bioDynrange[recordedNeuron]
                    membraneOutput = membraneOutput * slope + offset
                except TypeError:
                    myLogger.error(
                        'The parameter file calibBioDynrange is probably not installed!')
            # Default method
            else:
                membraneOutput = hardware.translateToBioVoltage(membraneOutput)

        recordFilename = _memPot_recordFilenames[0]
        if recordFilename != '' and recordFilename != None:
            # generate correct filename
            if _incrementFilename:
                lastDotIdx = recordFilename.find('.')
                if lastDotIdx == -1:  # no . in filename >> append _iteration
                    recordFilename += '.' + str(_iteration - 1)
                else:
                    while not recordFilename[lastDotIdx + 1:].find('.') == -1:
                        lastDotIdx += recordFilename[
                            lastDotIdx + 1:].find('.') + 1
                    recordFilename = recordFilename[
                        0:lastDotIdx] + '.' + str(_iteration - 1) + recordFilename[lastDotIdx:]
            # create file
            f = open(recordFilename, 'w')
            myLogger.info('Recording ' + str(len(membraneOutput)
                                             ) + ' samples to ' + recordFilename)
            for t in xrange(len(membraneOutput)):  # TODO: use numpy savetxt!
                f.write(str(membraneOutput[t]) + '\t' +
                        str(_memPot_recordedNeurons[0]) + '\n')
            f.close()
            del f

        _timeMemPyNN += time.time() - startTime

    myLogger.info('Network emulation finished.')

    _timeRunPyNN += time.time() - startTimeRun

    #######                           ##
    #      ##                        ####                                        #                 H
    #      ##                    ##################################             #     xxxxxx      H
    #      ##                 #####################################            #     xxxxxx      H
    # xxxxxx      H
    # H
    # HHHHHHHHHHHHHHHHHH

##########################################################################


def setRNGseeds(seedList):
    '''Globally set rng seeds. Not supported by PyNN 0.4.0 anymore!'''

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")

    myLogger.warn(
        'YOU USE THE FUNCTION setRNGseeds(...). BETTER DONT DO THIS! IT IS NOT SUPPORTED BY PYNN 0.4.0 ANYMORE. USE FUNCTION ARGUMENT setup(..., rng_seeds=[x,y,z]) INSTEAD.')

    global _seeds
    global _inputChanged
    _seeds = seedList
    _globalRNG.seed(_seeds[0])

    _inputChanged = True


def generateHardwareSpikeTrain(mirrorInputs=True):
    """
    Prepare the spike train to be sent to the hardware.
    A maximum of 256 external input sources to one chip is possible.
    If mirrorInputs is True, every input to a synapse driver on the left chip block will be mirrored to the
    corresponding synapse driver on the right chip block.
    This should be done if neurons on both chip blocks are used, i.e. if the network size plus the chosen
    mapping offset is larger than 192!
    """

    global _globalRNG

    # spikey4 workaround (left block can not be used)
    if hardware.chipVersion() == 4:
        avoidLeftBlock = True
        myLogger.info('Using right block only (Spikey v4 feature)')
    else:
        avoidLeftBlock = False
    blockOffset = hardware.numInputsPerNeuron()

    spikeTrainInHardwareIDs = numpy.array([], int)
    # still in bio time, gets translated in hardware.setInput()
    spikeTrainInHardwareTimes = numpy.array([], float)
    spikeTrainPyNNIDs = numpy.array([], int)
    spikeTrainPyNNTimes = numpy.array([], float)

    # prepare list of spike trains, for each external input one train
    for inputIndex, extInput in enumerate(_externalInputs):
        spikeTrainExtInput = []
        if extInput.__class__.__name__ == 'SpikeSourcePoisson':
            spikeTrainExtInput = poisson(extInput.parameters['start'], extInput.parameters[
                                         'duration'], extInput.parameters['rate'], _globalRNG, sorted=False)
        elif extInput.__class__.__name__ == 'SpikeSourceArray':
            spikeTrainExtInput = extInput.parameters['spike_times']

        spikeTrainInHardwareTimesTemp = numpy.array(spikeTrainExtInput)
        spikeTrainPyNNIDsTemp = numpy.ones(
            len(spikeTrainExtInput), int) * (-inputIndex - 1)
        spikeTrainPyNNTimesTemp = numpy.array(spikeTrainExtInput)
        if hardware.chipVersion() != 4:
            spikeTrainInHardwareIDsTemp = numpy.ones(
                len(spikeTrainExtInput), int) * inputIndex

            spikeTrainInHardwareIDs = numpy.concatenate(
                (spikeTrainInHardwareIDs, spikeTrainInHardwareIDsTemp))
            spikeTrainInHardwareTimes = numpy.concatenate(
                (spikeTrainInHardwareTimes, spikeTrainInHardwareTimesTemp))
            spikeTrainPyNNIDs = numpy.concatenate(
                (spikeTrainPyNNIDs, spikeTrainPyNNIDsTemp))
            spikeTrainPyNNTimes = numpy.concatenate(
                (spikeTrainPyNNTimes, spikeTrainPyNNTimesTemp))
        else:
            assert mirrorInputs

        if mirrorInputs:  # this is always true for chipVersion() == 4
            spikeTrainInHardwareIDsTemp = numpy.ones(
                len(spikeTrainExtInput), int) * (inputIndex + blockOffset)

            spikeTrainInHardwareIDs = numpy.concatenate(
                (spikeTrainInHardwareIDs, spikeTrainInHardwareIDsTemp))
            spikeTrainInHardwareTimes = numpy.concatenate(
                (spikeTrainInHardwareTimes, spikeTrainInHardwareTimesTemp))

            if hardware.chipVersion() == 4:
                spikeTrainPyNNIDs = numpy.concatenate(
                    (spikeTrainPyNNIDs, spikeTrainPyNNIDsTemp))
                spikeTrainPyNNTimes = numpy.concatenate(
                    (spikeTrainPyNNTimes, spikeTrainPyNNTimesTemp))

    # sort (input, time)-pairs for spike times
    sortIdx = numpy.argsort(spikeTrainInHardwareTimes)
    spikeTrainInHardwareIDs = spikeTrainInHardwareIDs[sortIdx]
    spikeTrainInHardwareTimes = spikeTrainInHardwareTimes[sortIdx]

    sortIdx = numpy.argsort(spikeTrainPyNNTimes)
    spikeTrainPyNNIDs = spikeTrainPyNNIDs[sortIdx]
    spikeTrainPyNNTimes = spikeTrainPyNNTimes[sortIdx]

    spikeTrainPyNN = (spikeTrainPyNNIDs, spikeTrainPyNNTimes)
    spikeTrainHardware = (spikeTrainInHardwareIDs, spikeTrainInHardwareTimes)

    return (spikeTrainHardware, spikeTrainPyNN)


# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass, cellparams=None, n=1, **extra_params):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    extra_params contains any keyword arguments that are required by a given simulator but not by others.
    """

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")
    if extra_params.has_key('param_dict'):
        myLogger.warn(
            "The argument 'param_dict' soon will not be supported anymore, use 'cellparams' instead!")
        cellparams = extra_params['param_dict']
    if extra_params.has_key('paramDict'):
        myLogger.warn(
            "The argument 'paramDict' soon will not be supported anymore, use 'cellparams' instead!")
        cellparams = extra_params['paramDict']
    assert n > 0, 'n must be a positive integer'

    global _neuronsChanged
    _neuronsChanged = True

    if cellclass == IF_facets_hardware1:
        oldsize = hardware.net.size()
        returnList = []
        indexCount = oldsize
        for i in xrange(n):
            # map from bio to hardware index
            hardware.hwa.hardwareIndex(oldsize + i)
            hardneuron = cellclass(cellparams)
            returnList.append(ID(indexCount, hardneuron))
            indexCount += 1

        hardware.create(hardneuron.parameters, n)
        if n == 1:
            returnList = returnList[0]
        return returnList

    elif cellclass in [SpikeSourcePoisson, SpikeSourceArray]:
        returnList = []
        # print "creating ", cellclass, "with", cellparams
        for i in xrange(n):
            ss = cellclass(cellparams)
            index = -1 - numpy.size(_externalInputs)
            _externalInputs.append(ss)
            returnList.append(ID(index, ss))
        if n == 1:
            returnList = returnList[0]
        # range(-1-numpy.size(_externalInputs)+n,-1-numpy.size(_externalInputs),-1)
        return returnList

    else:
        exceptionString = "Has to be cell type " + IF_facets_hardware1.__name__ + \
            " or " + SpikeSourcePoisson.__name__ + " or " + SpikeSourceArray.__name__
        raise Exception("ERROR: " + exceptionString)


def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None, **extra_params):
    """
    Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    extra_params contains any keyword arguments that are required by a given simulator but not by others.
    Spikey neuromorphic system: STDP & shortTermPlasticity parameters are passed as dicts named STDP and STP. If not
    provided, the coresponding mechanism is not used.
    STP-dict holds keys: 'cap'(= 1,3,5,7) and 'mode'(= 'fac', 'dep')
    """

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")

    if (type(source) != type([])) and (not isinstance(source, numpy.ndarray)):
        source = [source]
    if (type(target) != type([])) and (not isinstance(source, numpy.ndarray)):
        target = [target]

    if "synapse_dynamics" in extra_params.keys() and extra_params["synapse_dynamics"]:
        if extra_params["synapse_dynamics"].fast:
            if type(extra_params["synapse_dynamics"].fast) != TsodyksMarkramMechanism:
                raise Exception(
                    "ERROR: The only short-term synaptic plasticity type supported by the Spikey neuromorphic system is TsodyksMarkram!")
            # print "STP parameters: ",
            # extra_params["synapse_dynamics_fast"].parameters
            STP = extra_params["synapse_dynamics"].fast.parameters
        else:
            STP = None
        if extra_params["synapse_dynamics"].slow:
            STDP = extra_params["synapse_dynamics"].slow.parameters
        else:
            STDP = None
    else:
        STP = None
        STDP = None

    global _synapsesChanged
    _synapsesChanged = True  # comment here for Schmuker paper 2014 profiling
    global _connectivityChanged
    _connectivityChanged = True
    global _inputChanged
    for src in source:
        if type(src.cell) != IF_facets_hardware1:
            _inputChanged = True

    global _dt
    if weight is None:
        weight = 0.0
    if synapse_type == 'inhibitory' and weight > 0.0:
        weight *= -1.

    numConns = 0
    try:
        for src in source:
            rarr = None
            if type(src.cell) is SpikeSourcePoisson or type(src.cell) is SpikeSourceArray:  # external input
                connection_type = neurotypes.connectionType["ext"]
                srcIndex = -1 - src
            else:  # type(src.cell) == IF_facets_hardware1
                connection_type = neurotypes.connectionType["int"]
                srcIndex = src
            if p < 1:
                if rng:  # use the supplied RNG
                    rarr = rng.uniform(0., 1., len(target))
                else:   # use the default RNG
                    rarr = _globalRNG.uniform(0., 1., len(target))
                if type(rarr) is types.FloatType:
                    rarr = [rarr]
            for j, tgt in enumerate(target):
                if p >= 1 or rarr[j] < p:
                    # print 'connecting',srcIndex,'with',tgt.index,' and
                    # weight',weight
                    hardware.connect(connection_type, srcIndex,
                                     tgt, weight, STDP=STDP, STP=STP)
                    numConns += 1

    except Exception, e:
        raise common.ConnectionError(e)
    return numConns


def set(cells, cellclass, param, val=None):
    """
    Set one or more parameters of an individual cell or list of cells (provided as IDs).
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names.
    """

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")

    global _neuronsChanged
    global _inputChanged

    if val is not None:
        param = {param: val}
    if type(cells) != types.ListType:
        cells = [cells]
    for c in cells:
        for key in param:
            if not c.cell.parameters.has_key(key):
                raise Exception('ERROR: Cell ' + str(c) +
                                ' has no parameter ' + str(key) + '!')
        c.cell.parameters.update(param)
        if isinstance(c.cell, IF_facets_hardware1):
            _neuronsChanged = True
            hardware.net.neuron[c].parameters.update(
                param)     # write to pyhal-neuron
        elif isinstance(c.cell, SpikeSourcePoisson) or isinstance(c.cell, SpikeSourceArray):
            # print "input changed!"
            _inputChanged = True


def record(source, filename, **extra_params):
    """
    Record spikes to a file. source can be an individual cell or a list of cells (provided as IDs).
    Prints a warning if cells are not labeled recordable in workstations.xml.
    extra_params contains any keyword arguments that are required by a given simulator but not by others.
    """

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")
    import re
    global _suppressRecWarnings
    try:
        if _hardwareParameters['recNeurons'] != '':
            recordable = [int(x) for x in re.findall(
                '\d{1,3}', _hardwareParameters['recNeurons'])]
        else:
            recordable = hwconfig_default.recordableNeurons
    except:
        if not _suppressRecWarnings:
            myLogger.warn(
                "Can't find list of recordable neurons! Further warnings will be suppressed.")
        _suppressRecWarnings = True
        recordable = hwconfig_default.recordableNeurons

    global _neuronsChanged
    _neuronsChanged = True

    global _spikes_recordedNeurons
    global _spikes_recordFilenames

    if type(source) != types.ListType:
        source = [source]

    for src in source:
        if type(src.cell) == IF_facets_hardware1:
            if src >= hardware.net.size:
                raise Exception(
                    "ERROR: Invalid neuron index for spike recording!")
            if (not hardware.hwa.hardwareIndex(src) in recordable) and (not _suppressRecWarnings):
                myLogger.warn('Spikes from neuron ' + str(src).zfill(3) + ' (mapped to ' + str(
                    hardware.hwa.hardwareIndex(src)).zfill(3) + ' in hardware) can only be recorded separately!')
            _spikes_recordedNeurons.append(src)
            # print 'recording spikes of neuron',src
            _spikes_recordFilenames.append(filename)
            hardware.net.neuron[src].recordSpikes = True
        if type(src.cell) in [SpikeSourcePoisson, SpikeSourceArray]:
            insrc = -src - 1
            if insrc >= len(_externalInputs):
                raise Exception(
                    "ERROR: Invalid input index for spike recording!")
            _spikes_recordedInputs.append(src)
            # print 'recording spikes of input',src
            _spikes_recordInputFilenames.append(filename)


def stopRecording(source):
    '''Stop recording spikes to a file. Source can be an individual cell or a list of cells (provided as IDs).'''

    global _neuronsChanged
    _neuronsChanged = True

    global _spikes_recordedNeurons
    global _spikes_recordFilenames

    if type(source) != types.ListType:
        source = [source]

    for src in source:
        if type(src.cell) != IF_facets_hardware1:
            raise Exception(
                "ERROR: Can only record neurons, not external spike sources!")
        if src >= hardware.net.size:
            raise Exception("ERROR: Invalid neuron index for spike recording!")

        for i in xrange(len(_spikes_recordedNeurons)):
            if _spikes_recordedNeurons[i] == src:
                _spikes_recordedNeurons.remove(src)
                _spikes_recordFilenames.remove(_spikes_recordFilenames[i])
                break
        hardware.net.neuron[src].recordSpikes = False


def record_v(source, filename, **extra_params):
    """
    Record membrane potential to a file.
    source can be an individual cell or a list of cells (provided as IDs).
    extra_params contains any keyword arguments that are required by a given simulator but not by others.
    """

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")

    global _neuronsChanged
    global _memPot_recordedNeurons
    global _memPot_recordFilenames
    global _useUsbAdc

    _useUsbAdc = True

    if type(source) != type([]):
        source = [source]
    for src in source:
        # check if maximum number of recorded neurons is already reached
        if len(_memPot_recordedNeurons) == 1:
            myLogger.warn(
                "Currently the hardware supports only the recording of one membrane potential at a time! Replacing old record_v entry...")
        # only IF_facets_hardware1 neurons are recordable
        if not src.cell.__class__ == IF_facets_hardware1:
            raise Exception(
                'ERROR: Can only record membrane potential of neurons!')
        # check if neuron is already marked for being recorded
        else:
            if not (src in _memPot_recordedNeurons):
                _neuronsChanged = True
            _memPot_recordedNeurons = [src]
            _memPot_recordFilenames = [filename]


# pyNN.hardware implements some additional functions here (future pyNN?)
def getWeight(source, target):
    """Weight of connection between source and target is returned."""
    return hardware.hwa.sp.getSynapseWeight(source, target)


def setWeight(source, target, w):
    """
    Weight of connection between source and target is set to w. Weights should
    be in nA for current-based and uS for conductance-based synapses.
    """
    # this is a hardware specific thing... ;) (ECM)
    _synapsesChanged = True
    return connect(source, target, w)


def setSynapseDynamics(source, target, param, value):
    """
    Set parameters of the synapse dynamics linked with the synapse.
    """
    # TODO: ECM :), just interfacing... (see pyNN.projection)
    pass


class ID(int, common.IDMixin):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    Hardware specific: the additional member .cell is of type:
        IF_facets_hardware1, SpikeSourcePoisson or SpikeSourceArray
        It is generated by spikey.create() automatically.
    """
    non_parameter_attributes = ('parent', '_cellclass', 'cellclass',
                                '_position', 'position', 'hocname', '_cell',
                                'inject', '_v_init', 'local', 'node',
                                'nodeParams', 'tag', 'local')

    def __init__(self, index, cell):
        """
        target is an object of type: IF_facets_hardware1, SpikeSourcePoisson or
        SpikeSourceArray
        """
        int.__init__(index)
        common.IDMixin.__init__(self)
        object.__setattr__(self, 'cell', cell)
        self.cellclass = cell.__class__

    def __new__(cls, index, cell):
        inst = super(ID, cls).__new__(cls, index)
        return inst

    def __getattr__(self, name):
        if self.cell.parameters.has_key(name):
            return self.cell.parameters[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in ID.non_parameter_attributes:
            object.__setattr__(self, name, value)
        elif name == 'parameters':
            return set(self, self.cellclass, value)
        else:
            return self.set_parameters(**{name: value})

    def set_native_parameters(self, parameters):
        """Set cell parameters, given as a sequence of parameter=value arguments."""
        return set(self, self.cellclass, parameters)

    def get_native_parameters(self):
        """Return a dict of all cell parameters."""
        return self.cell.parameters


#################
##  UTILITIES  ##
#################

def findRecordableNeurons(neuronList):
    '''
    Finds out which neurons in neuronList are recordable and returns their IDs.
    Use this function after creating all neurons, but before recording them...
    '''

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")

    try:
        import re
        tmp = [int(x)
               for x in re.findall('\d{1,3}', _stationDict['recNeurons'])]
    except:
        myLogger.warn(
            "Can't retrieve recordable neurons. Assuming all as recordable.")
        tmp = range(hardware.numNeurons())
    recNeurons = []
    for n in neuronList:
        if hardware.hwa.hardwareIndex(n) in tmp:
            recNeurons.append(n)
    return recNeurons

Timer = utility.Timer


def poisson(start, duration, freq, prng, sorted=True):
    '''Receives a start time and a duration in msec, a frequency in Hz and a random number generator - returns a corresponding poisson spike train in msec.'''

    # determine number of spikes
    N = prng.poisson(duration * freq / 1000.0)
    p = prng.uniform(start, start + duration, N)
    p = p.tolist()
    if sorted:
        p.sort()
    return p


def spiketrainHeapsort(st):
    '''Sorts a given spike train by applying a heapsort implementation. Works merely on the given allocated memory.'''
    buf0 = buf1 = i = j = 0
    n = len(st[0, :]) - 1
    l = (n >> 1) + 1
    ir = n

    while (1):
        if (l > 1):
            l -= 1
            buf0 = st[0, l]
            buf1 = st[1, l]
        else:
            buf0 = st[0, ir]
            buf1 = st[1, ir]
            st[0, ir] = st[0, 1]
            st[1, ir] = st[1, 1]
            ir -= 1
            if (ir == 1):
                st[0, 1] = buf0
                st[1, 1] = buf1
                return
        i = l
        j = l << 1
        while (j <= ir):
            if ((j < ir) and (st[1, j] < st[1, j + 1])):
                j += 1
            if (buf1 < st[1, j]):
                st[0, i] = st[0, j]
                st[1, i] = st[1, j]
                i = j
                j += i
            else:
                j = ir + 1
        st[0, i] = buf0
        st[1, i] = buf1


def minExcWeight():
    '''Returns the minimum excitatory weight larger than zero, i.e. the weight given in uS that corresponds to the discrete hardware weight value 1.'''

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")
    return hardware.minExcWeight()


def minInhWeight():
    '''Returns the minimum inhibitory weight larger than zero, i.e. the weight given in uS that corresponds to the discrete hardware weight value 1.'''

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")
    return hardware.minInhWeight()


def maxExcWeight(neuronIndex=0):
    '''Returns the maximum excitatory weight, i.e. the weight given in uS that corresponds to the discrete hardware weight value 15.'''

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")
    return hardware.maxExcWeight(neuronIndex)


def maxInhWeight(neuronIndex=0):
    '''Returns the maximum inhibitory weight, i.e. the weight given in uS that corresponds to the discrete hardware weight value 15.'''

    if not _calledSetup:
        raise Exception("ERROR: Call function 'setup(...)' first!")
    return hardware.maxInhWeight(neuronIndex)


def systemInfo():
    #if not _calledSetup: raise Exception("ERROR: Call function 'setup(...)' first!")

    def svnrev(path, repo):  # returns svn revision of repository at 'path'
        try:
            if repo == 'svn':
                fs = os.popen('svnversion -n ' + path)
                res = fs.readlines()[0]
            elif repo == 'git':
                fs = os.popen('git rev-parse HEAD ' + path)
                res = fs.readlines()[0]
            fs.close()
        except:
            myLogger.warn("Couldn't retrieve current revision of: " + path)
            return 'unknown'

        if res == 'exported':
            myLogger.warn("Couldn't retrieve current revision of: " + path)
            return 'unknown'
        return res

    pyhalver = svnrev('$PYNN_HW_PATH', 'git')
    spikeyver = svnrev('$SPIKEYHALPATH', 'git')

    result = 'Host: ' + str(os.uname()) + '\n'
    result += 'Python: ' + sys.version + '\n'
    result += 'Time: ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n'
    result += 'pynnhw svn revision: ' + pyhalver + '\n'
    result += 'spikeyhal git revision: ' + spikeyver + '\n'
    result += '_hardwareParameters: ' + str(_hardwareParameters) + '\n'
    return result

get_current_time = common.get_current_time


def getInputSpikes():
    global numLostInputSpikes, numInputSpikes
    return numLostInputSpikes, numInputSpikes


def getSoftProfiling():
    '''See also https://gitviz.kip.uni-heidelberg.de/projects/symap2ic/wiki/Profiling'''
    myDict = hardware.getSoftProfiling()
    myDict['setupPyNN'] = _timeSetupPyNN - myDict['initCpp']

    myDict['configPyNN'] = _timeConfigPyNN - \
        myDict['mapNetworkPyHAL'] - myDict['configCpp']
    myDict['encodePyNN'] = _timeEncodePyNN
    myDict['runPyHAL'] = _timeRunPyHAL - myDict['encodeCpp'] - \
        myDict['sendCpp'] - myDict['runCpp'] - myDict['receiveCpp']
    myDict['memPyNN'] = _timeMemPyNN - myDict['adcCpp']
    myDict['runPyNN'] = _timeRunPyNN - _timeConfigPyNN - \
        _timeEncodePyNN - _timeRunPyHAL - _timeMemPyNN - myDict['decodePyHAL']

    myDict['endPyNN'] = _timeEndPyNN
    return myDict


def getChipVersion():
    '''Get version of Spikey chip from connected hardware.'''

    if not _calledSetup:
        raise Exception('ERROR: Call function setup() first!')
    return hardware.chipVersion()

# TP (10.09.2014): TODO: remove acquisition*, check _hardwareParameters
