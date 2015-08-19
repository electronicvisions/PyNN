# This file contains the most convenient access functions for the Spikey neuromorphic system.
# The functions contained here are intended to be an
# intermediate interface for wrapping hardware access with PyNN.
# The functions try to hide hardware specific details as far as possible.

import pylogging as pylog
myLogger = pylog.get("PyN.hal")

# these modules will be needed
import pyhal_config_s1v2 as conf
import pyhal_buildingblocks_s1v2 as bb
from pyhal_c_interface_s1v2 import vectorInt, vectorVectorInt
import numpy
import os
import sys
import pyhal_neurotypes as neurotypes
import time

if os.environ.has_key('PYNN_HW_PATH'):
    basePath = os.path.join(os.environ['PYNN_HW_PATH'], 'config')
else:
    raise EnvironmentError(
        'ERROR: The environment variable PYNN_HW_PATH is not defined!')


class HardwareError(Exception):
    '''Exception caused by communication with the Spikey neuromorphic system.'''


class HardwareAccessError(HardwareError):
    '''Exception caused by trying to access the Spikey neuromorphic system.'''


hwa = None
net = None
dictNeuronType = None
dictConnectionType = None

# profiling
_timeDecodePyHAL = 0


############################################
# functions that can be interfaced by PyNN #
############################################


def flush():
    '''Flushes the playback memory'''

    global hwa
    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.sp.Flush()


def initialize(defaultValue=0.0, debug=False, fullReset=False, **extra_params):
    '''This function has to be called once before all others.'''

    global hwa
    global net
    global dictNeuronType
    global dictConnectionType

    # print all of numpy arrays, no shortening of output
    numpy.set_printoptions(threshold=sys.maxint)

    # if the flag 'fullReset' is set, destroy hardware access and network
    # objects
    if fullReset:
        hwa = None
        net = None

    # create instances of both hardware access and network classes
    if hwa == None:
        hwa = conf.HWAccess(
            debug=debug, defaultValue=defaultValue, **extra_params)
    if net == None:
        net = bb.Network(maxSize=conf.numNeurons, maxExternalInputs=conf.numExternalInputs,
                         maxExternalInputsPerNeuron=conf.numPresyns, debug=debug)
    net.clear()

    # get pointers to the type dictionaries
    dictNeuronType = neurotypes.neuronType
    dictConnectionType = neurotypes.connectionType

    return True


def create(paramDict=None, number=1):
    '''Create a network, possibly with information about neuron type, size and parameters (given in a dictionary).'''

    net.create(paramDict, number)


def clearNetwork():
    '''Clears the network.'''

    net.clear()


def setSize(number):
    '''Set the size of the network to \'number\'.'''

    net.setSize(number)


def connect(sourceType, source, target, weight, **extra_params):
    '''Connects object of type \'sourceType\' (external or internal signal) and with index \'source\' to network neuron with index \'target\' using \'weight\'.'''

    net.connect(sourceType, source, target, weight, delay=0, **extra_params)


def printConnectivity():
    '''Prints the network connectivity to the console.'''

    net.printConnectivity()


def generateNetlist(filename):
    '''Prints the network connectivity to the console.'''

    net.generateNetlist(filename)


def mapNetworkToHardware(hardwareBinsPerBioSec=1, doFlush=True,
                         synapsesChanged=True, neuronsChanged=True, connectivityChanged=True, avoidSpikes=False):
    '''Tries to map the network \'net\' to the hardware.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    if hwa.haveHardware:
        # updateChip and updateDAC can be always false, since the corresponding
        # values are set in function getHardware() already
        hwa.mapNetworkToHardware(net=net, hardwareBinsPerBioSec=hardwareBinsPerBioSec,
                                 doFlush=doFlush, updateChip=False, updateDAC=False, updateParam=(synapsesChanged or neuronsChanged), updateRowConf=synapsesChanged,
                                 updateColConf=neuronsChanged, updateWeight=connectivityChanged, avoidSpikes=avoidSpikes)
    else:
        raise Exception('pyhal_s1v2: Call function getHardware() first!')


def getBus(workStationName):
    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.getBus(workStationName)


def getHardware(workStationName='',
                spikeyNr=0,
                spikeyClk=10.0,
                voutMin=0.0,
                voutMax=2.0,
                calibOutputPins=False,
                calibNeuronMems=True,
                calibIcb=True,
                calibTauMem=True,
                calibSynDrivers=True,
                calibVthresh=True,
                # New calibration options
                calibBioDynrange=True,
                #                         calibWeightsExc=False,
                #                         calibWeightsInh=False,
                calibfileOutputPins=basePath + '/config/calibration/calibOutputPins.pkl',
                calibfileIcb=basePath + '/config/calibration/calibIcb.dat',
                calibfileTauMem=basePath + '/config/calibration/calibTauMem.dat',
                calibfileSynDriverExc=basePath + '/config/calibration/calibSynDriver.dat',
                calibfileSynDriverInh=basePath + '/config/calibration/calibSynDriver.dat',
                calibfileVthresh=basePath + '/config/calibration/calibVthresh.dat',
                # New calibration options
                calibfileBioDynrange=basePath + '/config/calibration/calibBioDynrange.dat',
                #                calibfileWeightsExc=basePath+'/config/calibration/calibWeightsExc.dat',
                #                calibfileWeightsInh=basePath+'/config/calibration/calibWeightsInh.dat',
                neuronPermutation=[],
                mappingOffset=0,
                ratioSuperthreshSubthresh=0.8):
    '''Tries to get the Spikey neuromorphic system with number \'spikeyNr\', initializes it with a clock period of \'spikeyClk\'.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.getHardware(workStationName, spikeyNr, spikeyClk, voutMin, voutMax,
                    calibOutputPins, calibNeuronMems, calibIcb, calibTauMem, calibSynDrivers, calibVthresh, \
                    # New calibration options
                    calibBioDynrange,
                    #                    calibWeightsExc,
                    #                    calibWeightsInh,
                    calibfileOutputPins, calibfileIcb, calibfileTauMem, calibfileSynDriverExc, calibfileSynDriverInh, \
                    calibfileVthresh,
                    # New calibration options
                    calibfileBioDynrange,
                    #                    calibfileWeightsExc,
                    #                    calibfileWeightsInh,
                    neuronPermutation, mappingOffset, ratioSuperthreshSubthresh)


def monitorMembrane(neuronIndex, yesno, doFlush=True, copyToTestPin=False):
    '''Enables or disables membrane potential monitoring for neuron with index \'neuronIndex\'.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    if hwa.haveHardware:
        if type(neuronIndex) != type([]):
            neuronIndex = [neuronIndex]
        pin = 0
        for i in neuronIndex:
            tmpPin = hwa.monitorMembrane(
                i, yesno, doFlush, (copyToTestPin and i == neuronIndex[-1]))
            if (copyToTestPin and i == neuronIndex[-1]):
                pin = tmpPin
        return pin
    else:
        raise HardwareAccessError("Have no hardware access!")


def applyNeuronRecordConfig():
    '''Sends neuron configuration data to the chip.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.applyConfig(updateChip=False, updateDAC=False, updateParam=False,
                    updateRowConf=False, updateColConf=True, updateWeight=False)


def applyChipParamConfig():
    '''Sends hardware specific configuration data to the chip.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.applyConfig(updateChip=True, updateDAC=True, updateParam=False,
                    updateRowConf=False, updateColConf=False, updateWeight=False)


def applyConfig():
    '''Sends complete configuration data to the chip.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.applyConfig(updateChip=True, updateDAC=True, updateParam=True,
                    updateRowConf=True, updateColConf=True, updateWeight=True)


def assignMembrane2TestPin(membrane):
    '''Assigns the membrane potential output of recha pin \'membrane\' to the test pin (ibtest).'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    if hwa.haveHardware:
        hwa.assignMembrane2TestPin(membrane)
        return True
    else:
        raise HardwareAccessError("Have no hardware access!")


def assignVoltage2TestPin(signal, block):
    '''Assigns the spikey voltage \'signal\' to the test pin (ibTest).'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    if hwa.haveHardware:
        hwa.assignVoltage2TestPin(signal, block)
        return True
    else:
        raise HardwareAccessError("Have no hardware access!")


def setVoutsToMin(vouts, leftBlock=True, rightBlock=True):
    """
    Set vouts and voutbiases to zero. Call before assignMultipleVoltages2IBTest
    in combination with an external source to save spikey's life!
    """
    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    if hwa.haveHardware:
        blocks = []
        if leftBlock:
            blocks.append(0)
        if rightBlock:
            blocks.append(1)
        for i in blocks:
            for j in vouts:
                if not j in range(21):
                    raise Exception(
                        'pyhal_s1v2: vouts for test pin must be in range [0 , 21]!')
                myLogger.info("Set to zero: " + str(i) + ' ' + str(j))
                # best is 0.0; if oszillating a small value like 0.0025 or
                # 0.005 is necessary.
                hwa.voutbiases[i, j] = 0.0
                # ask ANDI: "clear" vouts, too? A: yes.
                hwa.vouts[i, j] = 0.0
    myLogger.info("Vouts " + str(vouts) + " ( leftBlock: " + str(leftBlock) + " , rightBlock: " + str(rightBlock) + " ) \
and corresponding biases set to zero. Make sure to flush config before activating external source!")


def assignMultipleVoltages2IBTest(vouts, leftBlock=True, rightBlock=True, pin4Mux=1):
    """
    Assigns the vouts listed to the test pin (ibTest) for one block or both blocks.
    The coaxial test pin is assigned to the membrane MUX and membrane pin pin4Mux. So this function allows to
    write spikey's vouts with an external source.
    """
    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    if hwa.haveHardware:
        hwa.assignMultipleVoltages2IBTest(
            vouts, leftBlock, rightBlock, pin4Mux)
        return True
    else:
        raise HardwareAccessError("Have no hardware access!")


def setInput(data, hardwareBinsPerBioMilliSec, duration=0):
    """
    Define input spike train. This function automatically shifts all events by a necessary hardware offset and
    adds the also necessary dummy spike.
    """

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')

    if len(data) != 2:
        myLogger.warn(
            "Wrong spike train format, size of first dimension must be 2!")
        return numpy.zeros((2, 0), int)

    spikeTrainIDs = data[0]
    spikeTrainTimes = data[1]

    # scale and convert to int (round to floor only)
    spikeTrainTimes = spikeTrainTimes * hardwareBinsPerBioMilliSec
    spikeTrainTimesInt = spikeTrainTimes.astype(int)
    # shifting in time
    spikeTrainTimesInt += startOffset()

    if len(spikeTrainTimesInt) != 0 and spikeTrainTimesInt.max() > pow(2, 31):
        # not 32 bit because of unsigned int
        raise Exception(
            'pyhal_s1v2: maximum runtime (due to 32-bit time stamps) is approx. 6600s. Work in progress!')

    # from software to hardware indices
    numPresyns = int(conf.numPresyns)
    numExternalInputs = int(conf.numExternalInputs)
    lut = numpy.concatenate((numpy.arange(
        numPresyns, 0, -1) - 1, numpy.arange(numExternalInputs, numPresyns, -1) - 1))
    spikeTrainIDs = numpy.take(lut, spikeTrainIDs)

    # prepare SpikeTrain object format
    intrain = vectorVectorInt()
    intrain.append(vectorInt())  # first  dimension
    intrain.append(vectorInt())  # second dimension
    intrain[0].extend(spikeTrainTimesInt.tolist())
    intrain[1].extend(spikeTrainIDs.tolist())

    # add a dummy spike behind the last real spike (needed by Spikey
    # controller)
    intrain[1].append(conf.numPresyns - 1)
    intrain[0].append(int(duration) + hwa.dummyWaitTime + startOffset())
    myLogger.trace('Appended extra spike to experiment (for technical reasons ' + str(
        hwa.dummyWaitTime) + ' after last input spike) with time stamp: ' + str(intrain[0][-1]))

    hwa.stin.data = intrain
    return intrain[0][-1]


def setNeuronParameters(neuronIndex, paramDict):
    '''Set parameters of a neuron.'''

    net.setNeuronParameters(neuronIndex, paramDict)


def startOffset():
    '''Get hardware starting time bin.'''

    # TP: increased startOffset() to 2^21, which corresponds to 2^17 clock cycles and 6.5536s in biological time domain (speed-up 10^4)
    # this was necessary to enable clearance of STDP capacitances before
    # experiment start that requires 15ms for each row of synapses (300 clock
    # cycles)

    # return hwa.bus.minTimebin()
    return 1 << 21


def clearPlaybackMem():
    '''Clear the playback memory (all data stored in pb mem should be flushed before calling this function, otherwise it will be lost).'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.clearPlaybackMem()


def run(replay=False):
    '''Runs the hardware once.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    (numLostInputSpikes, numInputSpikes) = hwa.run(
        replay, expOffset=startOffset())
    return (numLostInputSpikes, numInputSpikes)


def getOutput(runtime=numpy.infty, numNeurons=numpy.infty, minimumISI=0.0):
    '''
    Get back network output spikes. If runtime in ms is given, spikes generated after runtime are ignored.
    The return value is a 2-column numpy array, 1st column holds sorted spike times, 2nd column holds firing neuron index.
    '''

    global _timeDecodePyHAL
    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')

    startTime = time.time()
    dataTimes = numpy.array(hwa.stout.data[0])
    dataIDs = numpy.array(hwa.stout.data[1])
    myLogger.debug('number of spikes (raw) ' + str(len(dataIDs)))

    # remove temporal offset
    dataTimes -= startOffset()

    # re-scale time to biological interpretation
    factor = hwa.hardwareBinsPerBioSec / 1000.0
    dataTimes /= factor

    # re-establish biological indexing
    deltaN = conf.numPresyns - conf.neuronsPerBlock
    dataIDs[dataIDs > conf.neuronsPerBlock] -= deltaN
    dataIDs = numpy.take(hwa.neuronIndexMap, dataIDs)

    # remove those entries with negative times and with times beyond
    # experiment duration and invalid neuron indices
    mask = (dataTimes > 0) * (dataTimes <= runtime) * \
        (dataIDs > -1) * (dataIDs < numNeurons)
    dataIDsMasked = dataIDs[mask]
    dataTimesMasked = dataTimes[mask]
    myLogger.debug(
        'number of spikes (filtered for runtime and ID) ' + str(len(dataTimesMasked)))

    # optionally: remove too small inter-spike intervals
    # TODO: TP: obsolete?
    if minimumISI > 0.00001:    # to avoid floating point precision problems
        myLogger.info('removing spikes with too small ISIs from spike train')
        myLogger.warn(
            'this may be very slow: removing spikes with too small ISIs from spike train')
        lastSpikeDict = {}
        doubleCleanedTimes = []
        doubleCleanedNeurons = []
        for i in xrange(len(dataIDsMasked)):
            skipThisSpike = False
            if lastSpikeDict.has_key(dataIDsMasked[i]):
                if (dataTimesMasked[i] - lastSpikeDict[dataIDsMasked[i]]) < minimumISI:
                    skipThisSpike = True
            if not skipThisSpike:
                doubleCleanedTimes.append(dataTimesMasked[i])
                doubleCleanedNeurons.append(dataIDsMasked[i])
            lastSpikeDict[dataIDsMasked[i]] = dataTimesMasked[i]
        result = (doubleCleanedNeurons, doubleCleanedTimes)
    else:
        # here cast of int to float again
        result = (dataIDsMasked, dataTimesMasked)

    _timeDecodePyHAL += time.time() - startTime

    return result


def getOutputOfNeuron(neuronIndex, runtime=numpy.infty, minimumISI=0.0):
    '''
    Get back a single neuron\'s output spikes. If runtime in ms is given, spikes generated after runtime are ignored.
    Rarely used, and hence, not very efficiently implemented
    '''

    global _timeDecodePyHAL
    startTime = time.time()

    myLogger.warn('this may be very slow: getting spikes of single neurons')
    dataSpikes = getOutput(runtime, conf.numNeurons, minimumISI)
    dataSpikes = dataSpikes[dataSpikes[1] == neuronIndex]

    _timeDecodePyHAL += time.time() - startTime

    return dataSpikes


def writeConfigFile(filename):
    '''Writes the SpikeyConfig object to a file.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.writeConfigFile(filename)


def writeInSpikeTrain(filename):
    '''Writes the input spike train to file \'filename\'.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.writeInSpikeTrain(filename)


def writeOutSpikeTrain(filename):
    '''Writes the output spike train to file \'filename\'.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    hwa.writeOutSpikeTrain(filename)


def numInputsPerNeuron():
    '''Returns the number of presynaptic inputs per neuron.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.numInputsPerNeuron()


def numNeurons():
    '''Returns the number of neurons per chip.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.numNeurons()


def numNeuronsPerBlock():
    '''Returns the number of neurons per block.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.numNeuronsPerBlock()


def numBlocks():
    '''Returns the number of neuron blocks per chip.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.numBlocks()


def minExcWeight():
    '''Returns the minimum excitatory weight larger than zero, i.e. the weight given in uS that corresponds to the discrete hardware weight value 1.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.minExcWeight()


def minInhWeight():
    '''Returns the minimum inhibitory weight larger than zero, i.e. the weight given in uS that corresponds to the discrete hardware weight value 1.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.minInhWeight()


def maxExcWeight(neuronIndex=0):
    '''Returns the maximum excitatory weight, i.e. the weight given in uS that corresponds to the discrete hardware weight value 15.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.maxExcWeight(neuronIndex)


def maxInhWeight(neuronIndex=0):
    '''Returns the maximum inhibitory weight, i.e. the weight given in uS that corresponds to the discrete hardware weight value 15.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.maxInhWeight(neuronIndex)


def chipVersion():
    '''Returns the Spikey version as an integer.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.chipVersion()


def interruptActivity():
    '''Interrupts possible remaining activity from previous experiments by temporariliy setting all synaptic weights to zero.'''

    if hwa == None:
        raise Exception('pyhal_s1v2: Call function initialize() first!')
    return hwa.interruptActivity()


def translateToBioVoltage(hardwareVoltage):
    '''Translates the given hardware voltage into a biological interpretation.'''

    return hwa.translateToBioVoltage(hardwareVoltage)


def getWeightsHW(connList, synapseType, format, readHW):
    """Get hardware weights (in bits) before (readHW=False) and after (readHW=True) the experiment run"""

    return hwa.getWeightsHW(connList, synapseType, format, readHW)


def setSTDPRowsCont(presynaptic_neurons):
    """Set range of synapse rows enabled for STDP"""

    hwa.setSTDPRowsCont(presynaptic_neurons)


def initSTDP():
    """Initialize STDP"""
    hwa.initSTDP()


def disableSTDP():
    """turn off STDP"""
    hwa.disableSTDP()


def setupUsbAdc(simtime):
    """Setup fast USB ADC"""
    hwa.setupUsbAdc(simtime)


def readUsbAdc(slope, offset):
    """Read voltage data from fast USB ADC"""
    return hwa.readUsbAdc(slope, offset)


def delHardware():
    global hwa, net
    hwa.delHardware()
    hwa = None
    net = None


def setLUT(causalLUT, acausalLUT, first=False):
    """Set STDP look-up table entries"""

    hwa.setLUT(causalLUT, acausalLUT, first)


def getSoftProfiling():
    myDict = hwa.getSoftProfiling()
    myDict['decodePyHAL'] = _timeDecodePyHAL
    return myDict


def hardwareIndexMax():
    '''Get highest hardware index of allocated neurons.'''
    return hwa.hardwareIndexMax()
