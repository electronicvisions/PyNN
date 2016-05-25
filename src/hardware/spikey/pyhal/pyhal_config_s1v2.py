# This file containts information, classes and routines for accessing the
# Spikey neuromorphic system.

import pylogging as pylog
myLogger = pylog.get("PyN.cfg")

# these modules will be needed
import pyhal_buildingblocks_s1v2 as bb
import pyhal_c_interface_s1v2 as pyhalc
import pyhal_neurotypes as neurotypes
import hwconfig_default_s1v2 as default

import pylab
import numpy
import time
import random
import os
import sys
import pickle


# constants describing the Spikey chip
numPresyns = default.numPresyns
numBlocks = default.numBlocks
neuronsPerBlock = default.neuronsPerBlock
numExternalInputs = numPresyns * numBlocks
numNeurons = neuronsPerBlock * numBlocks
numVout = default.numVout

if os.environ.has_key('PYNN_HW_PATH'):
    basePath = os.path.join(os.environ['PYNN_HW_PATH'], 'config')
else:
    raise EnvironmentError(
        'ERROR: The environment variable PYNN_HW_PATH is not defined!')

sys.path.append(os.path.join(basePath, "calibration"))
import calibSTPglobals


class HWAccess:
    '''This class has to be instanciated once in order to access the Spikey neuromorphic system.'''

    def __init__(self, debug=False, defaultValue=0.0, **extra_params):
        """
        Constructor of class HWAccess. Instanciates a SpikeyConfig object, one SpikeTrain container for input and output each,
        plus arrays for the storage of calibration data.
        """

        # hardware access model pointers
        self.bus = None
        self.sp = None

        self.dryRun = False
        if extra_params.has_key('dryRun') and extra_params['dryRun']:
            self.dryRun = True
        self.debug = debug

        self.cfg = pyhalc.SpikeyConfig()
        self.stin = pyhalc.SpikeTrain()
        self.stout = pyhalc.SpikeTrain()

        # chip version
        self._chipVersion = None
        # instantiated bus model already?
        self.haveBus = False
        # instantiated hardware access already?
        self.haveHardware = False
        # determines at which hardware neuron index mapping starts
        self.mappingOffset = 0
        # self explaining
        self.hardwareBinsPerBioSec = 0
        # range for stdp parameter-readback
        self.stdp = {'enable': False,
                     'minC': 192,  # column range (=> neuron indexes)
                     'maxC': 383,
                     'minR':   0,  # row range (=> synapse drivers)
                     'maxR': 255}
        # use continuous STDP
        self.contSTDP = True
        # order in which hardware neurons are allocated
        self.neuronPermutation = numpy.array([])
        # counter of hardware neurons needed to build network
        self.hwNeuronCounter = 0

        #############################
        # fixed hardware parameters #
        #############################

        # spikey internal timing (given in external chip clock periods): after
        # this period the output of synapse ram bitline reading is valid
        self.tsense = default.tsense
        # pre-charge time for secondary read when processing correlations
        # (given in external chip clock periods)
        self.tpcsec = default.tpcsec
        # minimum time used for the correlation processing of a single row
        # (given in external chip clock periods)
        self.tpcorperiod = default.tpcorperiod
        # DAC reference current determining possible hardware currents:
        # minimal programmable current (!=0) possible = irefdac / 10. * (1./1024.)
        # maximal programmable current possible = irefdac / 10. * (1023./1024.)
        # irefdac given in uA
        self.irefdac = default.irefdac
        # decide if minimum parameter allowed is zero or one MSB
        self.defaultParamValue = defaultValue
        # cascode DAC voltage (given in V): Never touch this value!
        self.vcasdac = default.vcasdac
        # A reference voltage for STDP measurement (given in V): The larger vm, the smaller is the charge
        # which is stored per measured pre-/post-synaptic correlation -> STDP
        # curve amplitude gets smaller
        self.vm = default.vm
        # start value of rising voltage ramp (given in V), influence on
        # integral over PSP is weak
        self.vstart = default.vstart
        # baseline of voltage ramp (given in V), stronly influences efficacy of ramp, i.e. huge impact on PSP!
        # set very low if you want weak PSPs
        self.vrest = default.vrest
        # "adjust delay": this value controls the delay between a spike running into a synapse driver
        # and the triggered voltage up/down ramp
        self.adjdel = default.adjdel
        # threshold comparator bias current: the larger this value, the faster
        # is the comparator
        self.icb_base = default.icb_base
        # biasb  0.. 3: Vdtc0=0,Vdtc1=1,Vdtc2=2,Vdtc3=3,       short term plasticity time constant for spike history, higher current->shorter averaging windowtime
        # biasb  4.. 7: Vcb0=4,Vcb1=5,Vcb2=6,Vcb3=7 ,          spike driver comparator bias
        # biasb  8..11: Vplb0=8,Vplb1=9,Vplb2=10,Vplb3=11,     spike driver pulse length bias, higher current->shorter internal pulse,important for short term plasticity
        # biasb 12..13: Ibnoutampba=12,Ibnoutampbb=13,         both add together to the neuronoutampbias
        # biasb     14: Ibcorrreadb=14,                        correlation read
        # out bias

        # outamp 0..7: bias current for 50 ohm membrane voltage monitors,
        # outamp 8: current memory for ibtest_pin, should be 0.0!
        self.outamp = default.outamp

        # zero outamp biases for ADC calibration
        Ibnoutampba = default.Ibnoutampba
        Ibnoutampbb = default.Ibnoutampbb
        if extra_params.has_key('outAmpZero'):
            if extra_params['outAmpZero'] == True:
                self.outamp = numpy.zeros_like(default.outamp)
        myLogger.debug('outamp biases: ' + str(Ibnoutampba) +
                       ' / ' + str(Ibnoutampbb) + ' / ' + str(self.outamp))

        self.biasb = [default.vdtc, default.vdtc, default.vdtc, default.vdtc,
                      default.vcb, default.vcb, default.vcb, default.vcb,
                      default.vplb, default.vplb, default.vplb, default.vplb,
                      Ibnoutampba, Ibnoutampbb,
                      default.Ibcorrreadb]
        # bias current for all vout voltages - the more load the corresponding
        # vout has, the larger this value should be
        self.voutbias = default.voutbias
        # individual voutbias array - overwrites the global value
        # self.voutbias; initialized in: __init__ and mapNetworkToHardware
        self.voutbiases = default.voutbiases
        # obsolete value, remaining from spikey1
        self.probepad = default.probepad
        # obsolete value, remaining from spikey1
        self.probebias = default.probebias

        # a vector extracted from container stdParams is passed to the
        # SpikeyConfig object for initialization
        stdParamsList = [self.tsense, self.tpcsec, self.tpcorperiod, self.irefdac,
                         self.vcasdac, self.vm, self.vstart, self.vrest, self.adjdel, self.icb_base]
        stdParamsList.extend(self.biasb)
        stdParamsList.extend(self.outamp)
        stdParamsList.extend([self.voutbias, self.probepad, self.probebias])
        stdParamsList.extend([self.defaultParamValue])
        # create transfer vector
        paramsVec = pyhalc.vectorDouble()
        for p in stdParamsList:
            paramsVec.append(p)

        # transfer the prepared vector
        self.cfg.initialize(paramsVec)

        #######################################################################
        # dyncamic hardware parameters (i.e. adjustable by pyNN or involved in calibration routines #
        #######################################################################

        # arrays for calibration factors
        self.voltageDivider = 2.0  # 50 Ohm chip and 50 Ohm termination
        self.outputPinsFit = [[1.0, 0] for i in range(8)]
        self.neuronMemsFit = [[1.0, 0] for i in range(numNeurons)]
        self.calibTauMem = True
        self.iLeakCalib = numpy.ones(numNeurons)
        self.iLeakPoly = numpy.zeros(3 * numNeurons).reshape(numNeurons, 3)
        self.iLeakRange = numpy.zeros(2 * numNeurons).reshape(numNeurons, 2)
        self.iLeakResidual = numpy.zeros(2 * numNeurons).reshape(numNeurons, 2)
        self.iLeakResidualLimit = 0.1
        self.calibSynDrivers = True
        self.drvioutCalib = dict(exc=numpy.ones(
            numExternalInputs), inh=numpy.ones(numExternalInputs))
        self.drvifallCalib = dict(exc=numpy.ones(
            numExternalInputs), inh=numpy.ones(numExternalInputs))
        self.drviriseCalib = 1.0
        self.calibVthresh = True
        self.weightTrafoCalib = numpy.ones(numNeurons)
        self.calibIcb = True
        self.icbCalib = numpy.zeros(2 * numNeurons).reshape(numNeurons, 2)
        self.icbRange = numpy.zeros(2 * numNeurons).reshape(numNeurons, 2)
        self.bioDynrange = numpy.array(
            [numpy.ones(numNeurons), numpy.zeros(numNeurons)]).T

        # programmable voltages
        self.vouts = numpy.array(default.vouts)
        self.voutMins = []
        self.voutMaxs = []

        # amount of charge (vclra/c) loaded on capacitor for each spike pair
        # and threshold (vcthigh - vctlow) for charge on capacitor
        for i in range(8, 12):
            if not default.vouts[0][i] == default.vouts[1][i]:
                myLogger.warn(
                    'Vclra/c and Vcthigh/low are not the same for both blocks, values for left block are taken as default')
            for j in range(numBlocks):
                self.vouts[j][i] = default.vouts[0][i]

        # hardware to bio voltage translation
        self.hardwareToBioFactor = 0.0
        self.hardwareOffset = 0.0
        self.bioOffset = 0.0

        # transfer voutbiases (needed here to be available to
        # spikey::voutcalib)
        voutbiasVec = pyhalc.vectorVectorDouble()
        for i in range(2):
            voutbiasVec.append(pyhalc.vectorDouble())
            for j in xrange(numVout):
                voutbiasVec[i].append(self.voutbiases[i, j])
        # tranfer the prepared container to the SpikeyConfig object
        self.cfg.setVoltageBiases(voutbiasVec)

        # neuron membrane leakage conductance (for each neuron an individual
        # calibration factor will be multiplied to this value)
        self.iLeak_base = default.iLeak_base
        # synapse driver voltage ramp parameters
        # base value for upper limit (for each driver an individual calibration
        # factor will be multiplied to this value)
        self.drviout_base = default.drviout_base
        # rising ramp current
        self.drvirise = default.drvirise
        # falling ramp current
        self.drvifall_base = default.drvifall_base

        # delay of extra spike at the end of experiment (needed by Spikey
        # controller)
        self.dummyWaitTime_base = 20000 * 16  # 1s in bio time
        self.dummyWaitTime = self.dummyWaitTime_base

        # weight update controller (STDP) frequency
        self.autoSTDPFrequency = default.autoSTDPFrequency
        # same as above, but for synapse reset before experiment
        self.autoSTDPFrequencyPreExp = default.autoSTDPFrequency

        # bio->hardware weight transformation factors (for the excitatory
        # factor, an individual calibration factor will be multiplied to this
        # value)
        self.lsbPerMicroSiemens = dict(exc=1.0 / 0.001, inh=1.0 / 0.016)

        self.membraneReadoutPinOccupied = [
            False, False, False, False, False, False, False, False]

        # soft profiling
        self.timeInit = 0
        self.timeConfig = 0
        self.timeEncode = 0
        self.timeSend = 0
        self.timeRun = 0
        self.timeReceive = 0
        self.timeReadAdc = 0
        self.timeReadWeights = 0
        self.timeReadWeightsPyNN = 0
        self.timeMapNetwork = 0

    def checkHaveHardware(self):
        if not self.dryRun and not self.haveHardware:
            raise Exception('ERROR: Not connected to hardware!')

    def getBus(self, workStationName):
        '''Connect to hardware.'''

        # create slow control bus model
        myLogger.debug('Creating bus object')
        if not self.dryRun and not self.haveBus:
            self.bus = pyhalc.SC_Mem(0, workStationName)
            self.haveBus = True

            workStationNameTemp = self.bus.getWorkStationName()
            if workStationName != '':
                assert workStationName == workStationNameTemp
            else:
                myLogger.debug('Retrieved workStationName ' +
                               workStationNameTemp + ' from hardware')
        else:
            workStationNameTemp = 'station666'  # dummy data

        return workStationNameTemp

    # get access objects to hardware specified by the arguments
    # TP (27.05.2015): TODO: tidy up this function, split into init hardware
    # and load calibration
    def getHardware(self,
                    workStationName='',
                    spikeyNr=0,
                    spikeyClk=10.0,
                    voutMins=None,
                    voutMaxs=None,
                    calibOutputPins=False,
                    calibNeuronMems=True,
                    calibIcb=True,
                    calibTauMem=True,
                    calibSynDrivers=True,
                    calibVthresh=True,
                    # New calibration options
                    calibBioDynrange=True,
                    #                    calibWeightsExc=False,
                    #                    calibWeightsInh=False,
                    #
                    calibfileOutputPins=basePath + '/config/calibration/calibOutputPins.pkl',
                    calibfileIcb=basePath + '/config/calibration/calibIcb.dat',
                    calibfileTauMem=basePath + '/config/calibration/calibTauMem.dat',
                    calibfileSynDriverExc=basePath + '/config/calibration/calibSynDriver.dat',
                    calibfileSynDriverInh=basePath + '/config/calibration/calibSynDriver.dat',
                    calibfileVthresh=basePath + '/config/calibration/calibVthresh.dat',
                    # New calibration options
                    calibfileBioDynrange=basePath + '/config/calibration/calibBioDynrange.dat',
                    #                    calibfileWeightsExc=basePath+'/config/calibration/calibWeightsExc.dat',
                    #                    calibfileWeightsInh=basePath+'/config/calibration/calibWeightsInh.dat',
                    #
                    neuronPermutation=[],
                    mappingOffset=0,
                    ratioSuperthreshSubthresh=0.8):
        '''In case there has not yet been made a connection to a hardware device, the communication objects for the specified device are created.'''

        if self.dryRun:
            myLogger.warn("Hardware not used!")

        if mappingOffset >= numNeurons:
            raise Exception(
                'ERROR: Mapping offset is equal to or larger than available neuron number.')

        # mapping parameters
        self.neuronPermutation = numpy.array(neuronPermutation)
        self.mappingOffset = mappingOffset

        # clear mapping containers
        # bio to hardware
        self.hardwareIndexMap = numpy.ones(
            numNeurons, dtype=int) * -1  # -1 means not assigned yet
        # hardware to bio
        self.neuronIndexMap = numpy.ones(numNeurons, dtype=int) * -1

        # functions for unified logging messages
        def calibLoaded(name, filename):
            myLogger.debug(name + ' calibration loaded from ' + os.path.basename(
                filename) + '\n  in path ' + os.path.dirname(filename))

        def calibNotLoaded(name):
            myLogger.info('Calibrations' + name + ' NOT loaded!')

        def calibLoadError(name, filename):
            myLogger.warn('Could not load ' + name + ' calibration from ' +
                          os.path.basename(filename) + '\n  in path ' + os.path.dirname(filename))

        def getCalibExceptionStr(name, filename):
            return 'ERROR: Wrong number of ' + name + ' calibration values in file ' + filename.replace(basePath, self.dummyPynnhwPath) + '!'

        if not self.haveHardware:
            # create spikey model
            if os.environ.has_key('SPIKEYHALPATH'):
                calibfile = os.environ['SPIKEYHALPATH'] + '/spikeycalib.xml'
                myLogger.debug(
                    'Creating Spikey object (with clock period T = ' + str(spikeyClk) + 'ns)')
                startTime = time.time()
                if not self.dryRun:
                    self.sp = pyhalc.Spikey(
                        self.bus, spikeyClk, 0, spikeyNr, calibfile)
                self.timeInit += time.time() - startTime
                self.haveHardware = True
                self.calibOutputPins = calibOutputPins
                self.calibNeuronMems = calibNeuronMems
                if self.calibNeuronMems and self.calibOutputPins:
                    raise Exception(
                        'ERROR: enable either output pins OR neuron membrane calibration!')
                self.calibIcb = calibIcb
                self.calibTauMem = calibTauMem
                self.calibSynDrivers = calibSynDrivers
                self.calibVthresh = calibVthresh
                self.calibBioDynrange = calibBioDynrange
#                self.calibWeightsExc = calibWeightsExc
#                self.calibWeightsInh = calibWeightsInh
            else:
                raise Exception(
                    'ERROR: Environment variable SPIKEYHALPATH is not defined! It is needed for retrieval of Spikey calibration data.')

            # store vout min and max for this chip
            totalVouts = numVout * numBlocks
            if voutMins is None:
                voutMins = numpy.ones(totalVouts) * 0.5
            if voutMaxs is None:
                voutMaxs = numpy.ones(totalVouts) * 1.5
            self.voutMins = voutMins
            self.voutMaxs = voutMaxs

            # get maximum of v_rest, e_rev_I and v_reset values for the whole chip
            # vout indices of these voltages on left block are are v_rest: 0,
            # 1, e_rev_I: 2, 3, v_reset: 4, 5
            vlist = [0, 1, 2, 3, 4, 5]
            lowerVoltages = []
            for v in vlist:
                for block in [0, 1]:
                    lowerVoltages.append(voutMins[block * numVout + v])
            self.lowerBound = max(lowerVoltages)
            if self.lowerBound < 0.5:
                # according to JS, v_rest can not be set to values < 0.5 V on
                # Spikey
                self.lowerBound = 0.5
            elif self.lowerBound > 1.1:
                raise Exception(
                    'ERROR: For this chip, the minimum possible vout value for e_rev_I and v_reset is larger than the maximum threshold voltage (1.1 V) !')

            # get minimum of e_rev_E for the whole chip
            # vout indices of these voltages on left block are e_rev_E: 6, 7
            vlist = [6, 7]
            upperVoltages = []
            for v in vlist:
                for block in [0, 1]:
                    upperVoltages.append(voutMaxs[block * numVout + v])
            self.upperBound = min(upperVoltages)
            if self.upperBound < 1.1:
                raise Exception(
                    'ERROR: For this chip, the maximum possible vout value for e_rev_E is smaller than the maximum threshold voltage (1.1 V) !')

            # determine lower and upper bound for hardware membrane voltage range plus the hardware threshold voltage
            # relative to v_lower and v_upper, the following threshold value has been found to result in a reasonable strength of exc synapses (e.g. v_rest=0.8, e_rev_E=1.3, v_thresh=1.1):
            # assure that the ratio (v_upper-v_thresh)/(v_thresh-v_lower) == ratioSuperthreshSubthresh
            # and also assure that the hardware threshold voltage is not higher
            # than 1.1 V (acc. to J.Schemmel, this max membrane voltage should
            # be reachable on every chip)
            totrange = ratioSuperthreshSubthresh + 1.
            buf = self.lowerBound + \
                ((self.upperBound - self.lowerBound) * (1. / totrange))
            if buf > 1.1:
                self.voutThresh = 1.1
                self.upperBound = self.voutThresh + \
                    ((self.voutThresh - self.lowerBound)
                     * ratioSuperthreshSubthresh)
            else:
                self.voutThresh = buf
            myLogger.debug('Vouts corners [min / thresh / max]: ' + str(
                self.lowerBound) + ' / ' + str(self.voutThresh) + ' / ' + str(self.upperBound))

            # check if determined threshold vout fits into allowed range vor
            # the specific vout DAC
            if not ((self.voutThresh < min([voutMaxs[16], voutMaxs[17], voutMaxs[39], voutMaxs[40]])) and (self.voutThresh > max([voutMins[16], voutMins[17], voutMins[39], voutMins[40]]))):
                raise Exception(
                    'ERROR: For this chip, the determined vout value for v_thresh does not fit into its allowed range!')

            # string for calibs that are NOT loaded
            calibsNotLoaded = ''

            # get calibration data from files
            calibName = 'OutputPins'
            try:
                if self.calibOutputPins:
                    pickleFile = open(calibfileOutputPins, 'rb')
                    calibList = pickle.load(pickleFile)
                    pickleFile.close()

                    if (len(calibList['polyFitOutputPins'].keys()) != 4 and self.chipVersion() == 4) \
                            or (len(calibList['polyFitOutputPins'].keys()) != 8 and self.chipVersion() != 4):
                        raise Exception(
                            'ERROR: output pin calibration file invalid!')

                    for key in calibList['polyFitOutputPins'].keys():
                        keyInt = int(''.join(x for x in key if x.isdigit()))
                        self.outputPinsFit[keyInt] = calibList[
                            'polyFitOutputPins'][key]

                    calibLoaded(calibName, calibfileOutputPins)
                else:
                    calibsNotLoaded += ' ' + calibName
            except:
                calibLoadError(calibName, calibfileOutputPins)

            calibName = 'NeuronMems'
            try:
                if self.calibNeuronMems:
                    pickleFile = open(calibfileOutputPins, 'rb')
                    calibList = pickle.load(pickleFile)
                    pickleFile.close()

                    if (len(calibList['polyFitNeuronMems'].keys()) != 192 and self.chipVersion() == 4) \
                            or (len(calibList['polyFitNeuronMems'].keys()) != 384 and self.chipVersion() != 4):
                        raise Exception(
                            'ERROR: neuron membrane calibration file invalid!')

                    for key in calibList['polyFitNeuronMems'].keys():
                        keyInt = int(''.join(x for x in key if x.isdigit()))
                        self.neuronMemsFit[keyInt] = calibList[
                            'polyFitNeuronMems'][key]

                    calibLoaded(calibName, calibfileOutputPins)
                else:
                    calibsNotLoaded += ' ' + calibName
            except:
                calibLoadError(calibName, calibfileOutputPins)

            calibName = 'Icb'
            try:
                if self.calibIcb:
                    calibList = numpy.loadtxt(calibfileIcb)
                    if len(calibList) != numNeurons or calibList.shape[1] != 6:
                        raise Exception(getCalibExceptionStr(
                            calibName, calibfileIcb))
                    # calibList consists of entries in the form [neuron,
                    # param1, param2, lowerlimit, upperlimit]
                    self.icbCalib = calibList[:, 1:3]
                    self.icbRange = calibList[:, 3:5]
                    calibLoaded(calibName, calibfileIcb)
                else:
                    calibsNotLoaded += ' ' + calibName
            except:
                calibLoadError(calibName, calibfileIcb)

            calibName = 'TauMem'
            try:
                if self.calibTauMem:
                    calibList = numpy.loadtxt(calibfileTauMem)
                    if len(calibList) != numNeurons or calibList.shape[1] != 7:
                        raise Exception(getCalibExceptionStr(
                            calibName, calibfileTauMem))
                    # calibList consists of entries in the form [neuron,
                    # polycoeff1, polycoeff2, polycoeff3, lowerlimit,
                    # upperlimit]
                    self.iLeakPoly = calibList[:, 1:4]
                    self.iLeakRange = calibList[:, 4:6]
                    self.iLeakResidual = calibList[:, 6]
                    calibLoaded(calibName, calibfileTauMem)
                else:
                    calibsNotLoaded += ' ' + calibName
            except:
                calibLoadError(calibName, calibfileTauMem)

            for synType, calibfile in zip(('exc', 'inh'), (calibfileSynDriverExc, calibfileSynDriverInh)):
                calibName = synType + 'SynDrivers'
                try:
                    if self.calibSynDrivers:
                        calibList = numpy.loadtxt(calibfile)
                        if len(calibList) != (2 * numExternalInputs) + 4:
                            raise Exception(
                                getCalibExceptionStr(calibName, calibfile))
                        lsbIdx = dict(exc=0, inh=1)
                        self.lsbPerMicroSiemens[
                            synType] = calibList[lsbIdx[synType]]
                        self.drviout_base[synType] = calibList[2]
                        self.drvifall_base[synType] = calibList[3]
                        count = 0
                        for c in calibList[4:4 + numExternalInputs]:
                            self.drvioutCalib[synType][count] = c
                            count += 1
                        count = 0
                        for c in calibList[4 + numExternalInputs:]:
                            self.drvifallCalib[synType][count] = c
                            count += 1
                        calibLoaded(calibName, calibfile)
                    else:
                        calibsNotLoaded += ' ' + calibName
                except:
                    calibLoadError(calibName, calibfile)

            calibName = 'Vthresh'
            try:
                if self.calibVthresh:
                    calibList = numpy.loadtxt(calibfileVthresh)
                    if len(calibList) != numNeurons:
                        raise Exception(getCalibExceptionStr(
                            calibName, calibfileVthresh))
                    count = 0
                    for c in calibList:
                        self.weightTrafoCalib[count] = c
                        count += 1
                    calibLoaded(calibName, calibfileVthresh)
                else:
                    calibsNotLoaded += ' ' + calibName
            except:
                calibLoadError(calibName, calibfileVthresh)

            calibName = 'bioDynrange'
            try:
                if self.calibBioDynrange:
                    self.bioDynrange = numpy.loadtxt(calibfileBioDynrange)
                    calibLoaded(calibName, calibfileBioDynrange)
                else:
                    calibsNotLoaded += ' ' + calibName
            except:
                calibLoadError(calibName, calibfileBioDynrange)

            if calibsNotLoaded != '':
                calibNotLoaded(calibsNotLoaded)

        availableNeuronRange = range(default.numNeurons)
        if self.chipVersion() == 4:
            availableNeuronRange = range(
                default.neuronsPerBlock, default.numNeurons) + range(default.neuronsPerBlock)
        if len(self.neuronPermutation) > 0:
            exceptionString = 'ERROR: Neuron permutation list must include all hardware neuron indices: ' + \
                str(availableNeuronRange)
            # check for unique and complete neuron indices
            if len(self.neuronPermutation) != len(availableNeuronRange):
                raise Exception(exceptionString)
            for neuronIndex in availableNeuronRange:
                if not neuronIndex in self.neuronPermutation:
                    raise Exception(exceptionString)
            bioIndexList = self.neuronPermutation
            myLogger.info('Loaded neuron permutation list')
        else:
            bioIndexList = availableNeuronRange
        # apply mapping offset
        if self.chipVersion() == 4:
            bioIndexFirstBlock = bioIndexList[
                0:default.neuronsPerBlock]  # working block
            bioIndexSecondBlock = bioIndexList[
                default.neuronsPerBlock:default.numNeurons]  # broken block
            self.neuronPermutation = numpy.array(numpy.concatenate((bioIndexFirstBlock[
                                                 self.mappingOffset:], bioIndexFirstBlock[:self.mappingOffset], bioIndexSecondBlock)))
        else:
            self.neuronPermutation = numpy.array(numpy.concatenate(
                (bioIndexList[self.mappingOffset:], bioIndexList[:self.mappingOffset])))

        # disable global trigger signal (will be enabled if membrane is
        # recorded)
        if not self.dryRun:
            self.sp.setGlobalTrigger(False)

    def discretizeWeight(self, origWeight, maxValue=15.0001):
        # if transformed weight is not an integer, randomly round up or down,
        # but with probabilities such that the average weight is correct for
        # multiple connections
        weight = abs(origWeight)
        weight += numpy.random.random()
        weight = min(weight, maxValue)
        weight = int(weight)
        # if origWeight > 0: print origWeight, weight
        return weight

    # map an instance of bb.Network class to the hardware in terms of
    # connectivity and neuron/synapse parameters
    def mapNetworkToHardware(self, net, hardwareBinsPerBioSec=1, doFlush=True,
                             updateChip=True, updateDAC=True, updateParam=True, updateRowConf=True, updateColConf=True, updateWeight=True, avoidSpikes=False):
        '''Maps the abstract network description given by \'net\' to the hardware currently accessed by this class.'''

        timeStart = time.time()

        # check network container
        if net.__class__ != bb.Network().__class__:
            raise Exception(
                'ERROR: First argument has to be of type bb.Network.')

        # check if hardware is available
        self.checkHaveHardware()

        # reset counter for hardware neurons
        self.hwNeuronCounter = 0

        # initialize configuration
        self.hardwareBinsPerBioSec = hardwareBinsPerBioSec

        if updateColConf:
            # print printIndentation + 'INFO: (pyhal_config_s1v2) Clearing
            # column configuration.'
            self.cfg.clearColConfig()
            self.membraneReadoutPinOccupied = [
                False, False, False, False, False, False, False, False]
        if updateRowConf:
            # print printIndentation + 'INFO: (pyhal_config_s1v2) Clearing row
            # configuration.'
            self.cfg.clearRowConfig()
        if updateParam:
            # print printIndentation + 'INFO: (pyhal_config_s1v2) Clearing
            # voltages and currents.'
            self.cfg.clearParams()
            self.resetVoutNeuron()
        if updateWeight:
            # print printIndentation + 'INFO: (pyhal_config_s1v2) Clearing
            # weights.'
            self.cfg.clearWeights()

        # this bool array stores if the vouts for the four different location
        # types (left/right block, even/odd neuron) have been configured
        # already or not
        voutConfigured = [False, False, False, False]

        # prepare transfer weight container
        weightVec = pyhalc.vectorUbyte()

        externalInputWarningThrown = False

        # debug strings
        listOutOfRangeTauMem = []
        listBadFit = []
        listNoCalibTauMem = []
        listOutOfRangeIcb = []
        listNoCalibIcb = []

        # run through all neurons of the abstract network
        for n in net.neuron:
            # determine the neuron's hardware coordinates
            hardwareNeuron = self.hardwareIndex(n.index)
            myLogger.trace('bio neuron ID ' + str(n.index) +
                           ' is assigned to hardware neuron ID ' + str(hardwareNeuron))
            hardwareBlock = hardwareNeuron / neuronsPerBlock
            synapseBlockOffset = hardwareBlock * numPresyns

            # create an array for the hardware weights
            w = numpy.array([], int)
            w.resize(numPresyns)

            if updateParam or updateRowConf or updateWeight:
                # run through the neuron's inputs from within the network
                for c in n.incomingNeuronWeights.keys():
                    # determine the source's hardware coordinates
                    source = self.hardwareIndex(int(c))
                    # the synapse driver is determined by the block of the
                    # target neuron and by the source index
                    driver = synapseBlockOffset + (source % neuronsPerBlock)
                    # swap even and uneven feedback lines for feedback to
                    # adjacent block
                    if (source / neuronsPerBlock != hardwareBlock):
                        driver = driver + 1 - 2 * (driver % 2)
                    # determine the connection weight and type
                    synapse = n.incomingNeuronWeights[c]
                    synapseType = self.synapseStatusByte(synapse)

                    if synapse.weight >= 0.0:
                        hw_weight = synapse.weight * \
                            self.lsbPerMicroSiemens[
                                'exc'] * self.weightTrafoCalib[hardwareNeuron]
                        excOrInh = 'exc'
                    else:
                        hw_weight = synapse.weight * \
                            self.lsbPerMicroSiemens['inh']
                        excOrInh = 'inh'
                    discrete_weight = self.discretizeWeight(hw_weight)
                    # print 'INT:', source, '>>', n.index, 'driver:',driver,'w:',weight, \
                    #        'drviout:', self.drviout_base[excOrInh]*self.drvioutCalib[excOrInh][driver], \
                    #        'drvifall:', self.drvifall_base[excOrInh]*self.drvifallCalib[excOrInh][driver]

                    if updateParam:
                        self.cfg.setSynapseDriver(driver, 0, source, synapseType,
                                                  self.drviout_base[
                                                      excOrInh] * self.drvioutCalib[excOrInh][driver],
                                                  self.drvifall_base[excOrInh] * self.drvifallCalib[excOrInh][driver], self.drvirise * self.drviriseCalib, self.adjdel)
                    w[driver - synapseBlockOffset] = discrete_weight

                # run through the neuron's inputs from outside the network
                # for c in range(n.externalInputs.size):
                for source in n.externalWeights.keys():
                    # determine the source's hardware coordinates
                    #source = int(c)
                    # currently the number of possible external input sources
                    # is limited to numPresys in order to get consistent neuron
                    # configuration for both blocks on the chip!
                    if source >= numPresyns:
                        raise Exception('ERROR: Invalid source index!')
                    # the synapse driver is determined by the block of the
                    # target neuron and by the source index
                    driver = (numPresyns - 1 - (source %
                                                numPresyns)) + synapseBlockOffset
                    if driver % numPresyns <= hardwareNeuron % neuronsPerBlock and not externalInputWarningThrown:
                        myLogger.warn(
                            'Synapse driver lines of external inputs overlap with feedback connections of hardware neurons.')
                        externalInputWarningThrown = True
                    # determine the connection weight and type
                    synapse = n.externalWeights[source]
                    synapseType = self.synapseStatusByte(synapse)
                    if synapse.weight >= 0.0:
                        hw_weight = synapse.weight * \
                            self.lsbPerMicroSiemens[
                                'exc'] * self.weightTrafoCalib[hardwareNeuron]
                        excOrInh = 'exc'
                    else:
                        hw_weight = synapse.weight * \
                            self.lsbPerMicroSiemens['inh']
                        excOrInh = 'inh'
                    discrete_weight = self.discretizeWeight(hw_weight, 15)

                    # if n.index == 0: print printIndentation + 'INFO: (pyhal_config_s1v2) syn driver calib for driver',driver,'with drviout =',self.drviout_base,'is now',self.drvioutCalib[driver]
                    # print 'EXT:', source, '>>', n.index, 'driver:',driver,'w:',weight, \
                    #        'drviout:', self.drviout_base[excOrInh]*self.drvioutCalib[excOrInh][driver], \
                    #        'drvifall:', self.drvifall_base[excOrInh]*self.drvifallCalib[excOrInh][driver]

                    self.cfg.setSynapseDriver(driver, 1, source, synapseType,
                                              self.drviout_base[
                                                  excOrInh] * self.drvioutCalib[excOrInh][driver],
                                              self.drvifall_base[excOrInh] * self.drvifallCalib[excOrInh][driver], self.drvirise * self.drviriseCalib, self.adjdel)
                    w[driver - synapseBlockOffset] = discrete_weight

            # tranfer the prepared containers to the SpikeyConfig object
            if updateWeight:
                weightVec[:] = []
                for wi in w:
                    weightVec.append(int(wi))
                self.cfg.setWeights(hardwareNeuron, weightVec)

            if updateColConf:
                self.cfg.enableNeuron(hardwareNeuron, n.recordSpikes)

            if updateParam:
                # if fit parameters are nonzero, calculate desired iLeak from
                # fit
                calib_params = self.iLeakPoly[hardwareNeuron]
                if self.calibTauMem and (calib_params != 0).any():
                    # capacitance is 0.2nF
                    tauMem = 0.2 * 1e3 / n.parameters['g_leak']
                    iLeak_inv_calib = numpy.polyval(calib_params, x=tauMem)
                    iLeak_calib = 1. / iLeak_inv_calib
                    # fix to valid currents
                    iLeak_calib = numpy.max([0, numpy.min([2.5, iLeak_calib])])
                    self.iLeakCalib[hardwareNeuron] = iLeak_calib / \
                        (n.parameters['g_leak'] * self.iLeak_base)
                    min_iLeak = self.iLeakRange[hardwareNeuron][0]
                    max_iLeak = self.iLeakRange[hardwareNeuron][1]
                    # warn if desired membrane time constant is not in range of
                    # calibration
                    if iLeak_calib < min_iLeak or iLeak_calib > max_iLeak:
                        listOutOfRangeTauMem.append(hardwareNeuron)
                    if self.iLeakResidual[hardwareNeuron] > self.iLeakResidualLimit:
                        listBadFit.append(hardwareNeuron)
                else:
                    if self.calibTauMem:
                        listNoCalibTauMem.append(hardwareNeuron)
                self.cfg.setILeak(hardwareNeuron, n.parameters[
                                  'g_leak'] * self.iLeak_base * self.iLeakCalib[hardwareNeuron])

                #### calibrate tau_refrac ###
                calib_params = self.icbCalib[hardwareNeuron]
                if self.calibIcb and (calib_params != 0).any():
                    icb = numpy.log(
                        n.parameters['tau_refrac'] / calib_params[0]) * (-1. / calib_params[1])
                    min_icb = min(self.icbRange[hardwareNeuron])
                    max_icb = max(self.icbRange[hardwareNeuron])
                    # warn if desired refractory period is not in range of
                    # calibrated values
                    if icb < min_icb or icb > max_icb:
                        listOutOfRangeIcb.append(hardwareNeuron)
                    self.cfg.setIcb(hardwareNeuron, icb)
                else:
                    self.cfg.setIcb(hardwareNeuron, self.icb_base)
                    if self.calibIcb:
                        listNoCalibIcb.append(hardwareNeuron)

                # translate neuron voltages to vouts
                # determine the index of this neuron's voltages
                voutIndex = numBlocks * hardwareBlock + hardwareNeuron % 2
                if not voutConfigured[voutIndex]:
                    neuronVoltages = [n.parameters['v_reset'], n.parameters[
                        'e_rev_I'], n.parameters['v_rest'], n.parameters['v_thresh']]
                    if min(neuronVoltages) < -80.:
                        myLogger.warn('Biological voltages smaller than -80 mV may not be supported by the utilized chip! (One reference voltage for hardware neuron ' + str(
                            hardwareNeuron) + ' on block ' + str(hardwareBlock) + ' has value: ' + str(min(neuronVoltages)) + ')')
                    if max(neuronVoltages) > -55.:
                        myLogger.warn('Biological voltages larger than -55 mV (excitatory reversal potential excluded) may not be supported by the utilized chip! (One reference voltage for hardware neuron ' + str(
                            hardwareNeuron) + ' on block ' + str(hardwareBlock) + ' has value ' + str(max(neuronVoltages)) + ')')
                    bio_lowest_voltage = -80.
                    bio_highest_voltage = -55.
                    deltaVbio = bio_highest_voltage - bio_lowest_voltage
                    deltaVhw = self.voutThresh - self.lowerBound
                    factor = deltaVhw / deltaVbio
                    # save the conversion factors for later re-translation of
                    # hardware to bio voltage
                    self.hardwareToBioFactor = 1. / factor
                    self.hardwareOffset = self.lowerBound
                    self.bioOffset = bio_lowest_voltage
                    # start writing vouts
                    self.vouts[hardwareBlock, hardwareNeuron % 2 + 0] = (
                        n.parameters['e_rev_I'] - bio_lowest_voltage) * factor + self.lowerBound
                    self.vouts[hardwareBlock, hardwareNeuron % 2 + 2] = (
                        n.parameters['v_rest'] - bio_lowest_voltage) * factor + self.lowerBound
                    self.vouts[hardwareBlock, hardwareNeuron % 2 + 4] = (
                        n.parameters['v_reset'] - bio_lowest_voltage) * factor + self.lowerBound
                    self.vouts[hardwareBlock, hardwareNeuron %
                               2 + 6] = self.upperBound  # exc reversal potential
                    # neuron input cascode gate voltage
                    self.vouts[hardwareBlock,
                               18] = self.upperBound

                    if avoidSpikes:
                        self.vouts[hardwareBlock, hardwareNeuron %
                                   2 + 16] = 2.0
                    else:
                        self.vouts[hardwareBlock, hardwareNeuron % 2 + 16] = (
                            n.parameters['v_thresh'] - bio_lowest_voltage) * factor + self.lowerBound

                    # let special lowlevel voltages override biological values
                    if n.parameters['lowlevel_parameters'].has_key('e_rev_I'):
                        self.vouts[hardwareBlock, hardwareNeuron % 2 +
                                   0] = n.parameters['lowlevel_parameters'].get('e_rev_I')
                    if n.parameters['lowlevel_parameters'].has_key('v_rest'):
                        self.vouts[hardwareBlock, hardwareNeuron % 2 +
                                   2] = n.parameters['lowlevel_parameters'].get('v_rest')
                    if n.parameters['lowlevel_parameters'].has_key('v_reset'):
                        self.vouts[hardwareBlock, hardwareNeuron % 2 +
                                   4] = n.parameters['lowlevel_parameters'].get('v_reset')
                    if n.parameters['lowlevel_parameters'].has_key('e_rev_E'):
                        self.vouts[hardwareBlock, hardwareNeuron % 2 +
                                   6] = n.parameters['lowlevel_parameters'].get('e_rev_E')
                    if n.parameters['lowlevel_parameters'].has_key('v_thresh'):
                        self.vouts[hardwareBlock, hardwareNeuron % 2 +
                                   16] = n.parameters['lowlevel_parameters'].get('v_thresh')
                    if n.parameters['lowlevel_parameters'].has_key('permanentFiring') and n.parameters['lowlevel_parameters']['permanentFiring']:
                        myLogger.warn('Setting the firing threshold below the resting potential! (Hardware neuron ' + str(
                            hardwareNeuron) + ' on block ' + str(hardwareBlock) + ')')
                        buf = self.vouts[hardwareBlock, hardwareNeuron % 2 + 2]
                        buf -= 1. / numpy.e * (self.vouts[hardwareBlock, hardwareNeuron % 2 + 2] - self.vouts[
                                               hardwareBlock, hardwareNeuron % 2 + 4])
                        self.vouts[hardwareBlock, hardwareNeuron %
                                   2 + 16] = buf

                    voutConfigured[voutIndex] = True

        if len(listOutOfRangeTauMem) > 0:
            listOutOfRangeTauMem.sort()
            myLogger.warn(
                'Membrane time constant is outside calibrated range for neurons ' + str(listOutOfRangeTauMem))
        if len(listBadFit) > 0:
            listBadFit.sort()
            myLogger.warn('Bad calibration of membrane time constant for neurons ' +
                          str(listBadFit) + '; blacklist if necessary')
        if len(listNoCalibTauMem) > 0:
            listNoCalibTauMem.sort()
            myLogger.error(
                'No valid calibration available for membrane time constant of neurons ' + str(listNoCalibTauMem))

        if len(listOutOfRangeIcb) > 0:
            listOutOfRangeIcb.sort()
            myLogger.warn(
                'Refractory period is outside calibrated range for neurons ' + str(listOutOfRangeIcb))
        if len(listNoCalibIcb) > 0:
            listNoCalibIcb.sort()
            myLogger.error(
                'No valid calibration available for refractory period of neurons ' + str(listNoCalibIcb))

        # the following is done only once, not for every neuron
        if updateParam:
            # fill transfer container
            voutVec = pyhalc.vectorVectorDouble()
            for i in range(2):
                voutVec.append(pyhalc.vectorDouble())
                for j in xrange(numVout):
                    voutVec[i].append(self.vouts[i, j])
            # tranfer the prepared container to the SpikeyConfig object
            self.cfg.setVoltages(voutVec)
            # print printIndentation + 'INFO: (pyhal_config_s1v2) vouts in pyhal_config_s1v2::mapNetworkToHardware:'
            # print self.vouts

            # transfer voutbiases
            voutbiasVec = pyhalc.vectorVectorDouble()
            for i in range(2):
                voutbiasVec.append(pyhalc.vectorDouble())
                for j in xrange(numVout):
                    voutbiasVec[i].append(self.voutbiases[i, j])
            # tranfer the prepared container to the SpikeyConfig object
            self.cfg.setVoltageBiases(voutbiasVec)

            # transfer biasbs
            biasbVec = pyhalc.vectorFloat()
            for b in self.biasb:
                biasbVec.append(b)
            # tranfer the prepared container to the SpikeyConfig object
            self.cfg.setBiasbs(biasbVec)

        self.timeMapNetwork += time.time() - timeStart

        # transfer the SpikeyConfig object to the hardware
        if doFlush:
            self.applyConfig(updateChip, updateDAC, updateParam,
                             updateRowConf, updateColConf, updateWeight)

    def getWeightsHW(self, connList, synapseType, format, readHW):
        """Get hardware weights (in bits) before (readHW=False) and after (readHW=True) the experiment run"""

        startTimeWeights = time.time()

        if synapseType == 'inhibitory':
            def HWtoBioTranslation():  # n is pynn ID
                return self.minInhWeight()
        elif synapseType == 'excitatory':
            def HWtoBioTranslation():  # n is pynn ID
                return self.minExcWeight()
        else:
            raise Exception("Synapse type not supported!")

        for i in range(len(connList)):
            connList[i] = [self.hardwareSynapseIndex(
                connList[i][0]), self.hardwareIndex(connList[i][1])]
        connList = numpy.array(connList)

        # get hardware neuron IDs
        srcCollection = connList[:, 0]
        tgtCollection = connList[:, 1]

        # define subarray of synapse array that should be read out
        rowMin = int(srcCollection.min())
        columnMin = int(tgtCollection.min())
        rowMax = int(srcCollection.max())
        columnMax = int(tgtCollection.max())
        assert rowMin <= rowMax
        assert columnMin <= columnMax

        if readHW:  # read weights from HW
            myLogger.info('Reading weights (row ' + str(rowMin) + '->' + str(rowMax) + '; column ' +
                          str(columnMin) + '->' + str(columnMax) + ') from HW (after running experiment)')
            startTime = time.time()
            weights = numpy.array(self.sp.getSynapseWeights(
                rowMin, rowMax, columnMin, columnMax))
            self.timeReadWeights += time.time() - startTime
        else:  # read weights from class-internal containers
            myLogger.info('Reading weights (row ' + str(rowMin) + '->' + str(rowMax) + '; column ' + str(
                columnMin) + '->' + str(columnMax) + ') from config object (before running experiment)')
            weights = numpy.array(self.cfg.weight)
            assert len(weights) == default.numExternalInputs * \
                default.neuronsPerBlock
            weights.resize(default.numExternalInputs, default.neuronsPerBlock)
            # get left and right block next to each other
            weights = numpy.hstack(numpy.split(weights, 2))
            # selection
            # transform to hardware representation
            weights = numpy.flipud(weights)
            weights = weights[rowMin:rowMax + 1, columnMin:columnMax + 1]

        weightsReturn = []
        if format == 'array':
            # different synapse row indexing in pyNN and SpikeyHAL
            weights = numpy.flipud(weights)
            # from hardware weights to biological weights
            TranslationVector = numpy.ones(columnMax - columnMin + 1)
            for i in range(columnMin, columnMax + 1):
                TranslationVector[i - columnMin] = HWtoBioTranslation()
            weightsReturn = weights * TranslationVector
        elif format == 'list':
            weight_list = []
            for (src, tgt) in connList.tolist():
                weight_list.append(
                    weights[src - rowMin, tgt - columnMin] * HWtoBioTranslation())
            weightsReturn = weight_list
        else:
            raise Exception('Invalid format.')

        self.timeReadWeightsPyNN += time.time() - startTimeWeights

        return weightsReturn

    def initSTDP(self):
        """Initialize STDP"""
        # init synapse area used with STDP to minimum
        self.stdp['enable'] = True
        self.stdp['minR'] = default.numPresyns - 1
        self.stdp['maxR'] = 0
        if self.chipVersion() == 4:
            self.stdp['minC'] = default.neuronsPerBlock
        else:
            self.stdp['minC'] = 0
        self.stdp['maxC'] = default.numNeurons - 1

    def disableSTDP(self):
        """Disable STDP"""
        self.stdp['enable'] = False

    def setLUT(self, causalLUT, acausalLUT, first=False):
        """Set STDP look-up tables."""
        if len(self.getLUT()) == 0 and not first:
            myLogger.warn(
                'setLUT() should be called after pyNN.hardware.spikey.Projection(), otherwise it will be overwritten by default values')
        myLogger.debug('Setting look-up table:\n causal: ' +
                       str(causalLUT) + '\n acausal: ' + str(acausalLUT))
        lutVector = pyhalc.vectorInt()
        if len(causalLUT) != 16 or len(acausalLUT) != 16:  # TODO: TP: remove hard-coding
            raise Exception('ERROR: setLUT: 2*16 values required.')
        for i in acausalLUT:
            i = int(i)
            if i < 0 or i > 15:
                raise Exception('ERROR: setLUT: 0<=value<=15.')
            lutVector.append(i)
        for i in causalLUT:
            i = int(i)
            if i < 0 or i > 15:
                raise Exception('ERROR: setLUT: 0<=value<=15.')
            lutVector.append(i)
        self.sp.setLUT(lutVector)

    def getLUT(self):
        """Get STDP look-up tables."""
        return self.sp.getLUT()

    def fromMsToClockCycles(self, timeInMs):
        # TODO: TP: speedup factor and clock frequency still hardcoded
        # see __init__.py _hardwareParameters['speedup'] and
        # _hardwareParameters['spikeyClk']
        """From ms to Spikey clock cycles (default: 200MHz)."""
        return timeInMs / 1e3 / 1e4 * 200e6

    def setSTDPParamsCont(self, activate, expOffset, distance, distance_pre, rowmin, rowmax):
        """Set hardware parameters relevant for continuous STDP."""
        if rowmin > rowmax or rowmin < 0 or rowmax < 0 or rowmin > default.numPresyns or rowmax > default.numPresyns:
            raise Exception(
                'ERROR: Continuous STDP: combination of startrow and stoprow not allowed (0<=startrow<=stoprow<=255).')
        distance = int(self.fromMsToClockCycles(distance))
        distance = distance + distance % 2  # must be even
        distance_pre = int(self.fromMsToClockCycles(distance_pre))
        distance_pre = distance_pre + distance_pre % 2  # must be even
        first = expOffset >> 4  # clock cycle is one 4-bit time stamp
        self.sp.setSTDPParamsCont(
            bool(activate), first, distance, distance_pre, int(rowmin), int(rowmax))

    def setSTDPRowsCont(self, presynaptic_neurons):
        """Set range of synapse rows enabled for STDP"""
        # get hardware indices for neurons and synapses
        hwIndicesSrcVec = numpy.vectorize(self.hardwareSynapseIndex)
        srcCollection = hwIndicesSrcVec(presynaptic_neurons)

        # check for enlargement of synapse area (range of rows) used with STDP
        minRow = min(srcCollection.min(), self.stdp['minR'])
        maxRow = max(srcCollection.max(), self.stdp['maxR'])

        self.stdp['minR'] = minRow
        self.stdp['maxR'] = maxRow

        distance = int((self.stdp['maxR'] - self.stdp['minR'] + 1)
                       * self.fromMsToClockCycles(self.autoSTDPFrequency))
        distance = distance + distance % 2  # must be even
        # TP: should be 4 (4-bit time stamps), but this was too short
        addDummyWaitTime = (distance << 5)
        self.dummyWaitTime = self.dummyWaitTime_base + addDummyWaitTime
        myLogger.trace('Moved spike that marks end of experiment by ' +
                       str(addDummyWaitTime) + ' time stamp values')

    def getSTDPRowsCont(self):
        """Get range of synapse rows enabled for STDP"""
        return self.stdp['minR'], self.stdp['maxR']

    def setSTDPParams(self, vm, tpcsec, tpcorperiod, vclra, vclrc, vcthigh, vctlow, adjdel):
        """Set all hardware parameters relevant for STDP dynamics."""
        # fill class-internal containers with passed values
        self.vm = float(vm)
        self.tpcsec = int(tpcsec)
        self.tpcorperiod = int(tpcorperiod)
        self.adjdel = float(adjdel)
        for b in range(default.numBlocks):
            self.vouts[b, 8] = float(vclra)
            self.vouts[b, 9] = float(vclrc)
            self.vouts[b, 10] = float(vcthigh)
            self.vouts[b, 11] = float(vctlow)

        #_synapsesChanged = True #TODO: TP: needed here

    def setIcb(self, icb_base):
        """Set the neuron refractory period in hardware values (icb_base)."""
        # fill class-internal containers with passed values
        self.icb_base = icb_base
        for n in xrange(numNeurons):
            self.cfg.setIcb(n, icb_base)

    def _set_STP_globals(self, Vfac=None, Vstdf=None, Vdtc=None):
        '''
        Set global Short Term Plasticity parameters.
        If a value is not provided (or None) it remains unchanged.

        Vstdf (=C_2 Voltage V_max) : determines the maximal effect of fac and dep
        Vdtc: determines the timeconstant the (in)active partition discharges
        Vfac: While for depression V_I is compared to 0, facilitation's base voltage is Vfac.

        For each parameter there exist reasonable default values. So you might not need to touch them.
        '''
        # global _synapsesChanged #TP: still needed?
        #_synapsesChanged = True

        if Vfac is not None:
            self.vouts[0][12] = Vfac
            self.vouts[0][13] = Vfac
            self.vouts[1][12] = Vfac
            self.vouts[1][13] = Vfac
        if Vstdf is not None:
            self.vouts[0][14] = Vstdf
            self.vouts[0][15] = Vstdf
            self.vouts[1][14] = Vstdf
            self.vouts[1][15] = Vstdf
        if Vdtc is not None:
            self.biasb[0] = Vdtc
            self.biasb[1] = Vdtc
            self.biasb[2] = Vdtc
            self.biasb[3] = Vdtc

        # print '\nINFO: Stort term plasticity parameters changed. Current values are:'
        # print '\tVstdf: %.3f' % self.vouts[0][14],
        # print '\tVfac: %.3f' % self.vouts[0][12],
        # print '\tVdtc: %.3f' % self.biasb[0]

    def get_STP_globals(self):
        '''
        returns a dict containing current Vstdf, Vfac and Vdtc.
        '''

        d = dict(Vstdf=self.vouts[0][14], Vfac=self.vouts[
                 0][12], Vdtc=self.biasb[0])
        return d

    def synapseStatusByte(self, synapse):
        """Build statusbyte <type> for setSynapseDriver()-call. Checks for STD. Lowest two bits are not affected (=> sourceType)!"""

        status = neurotypes.neuronType['baseValue']
        if synapse.weight >= 0:
            status += neurotypes.neuronType["excitatory"]
        elif synapse.weight < 0:
            status += neurotypes.neuronType["inhibitory"]
        else:
            return status   # STP not important

        STP = synapse.STP   # get Short Term Plasticity params, might be None
        if STP:
            status += neurotypes.STPTypes['enable']

            if STP.has_key('U'):
                U = STP['U']
            else:
                raise Exception("STP active but no U provided.")

            # facilitating or depressing
            if (STP['tau_facil'] != 0 or STP['tau_rec'] != 0):
                if STP['tau_facil'] == 0:
                    # depression
                    status += neurotypes.STPTypes['dep']
                    stp_type = 'dep'
                    tau = STP['tau_rec']
                    # modify weight according to stp parameter translation
                    synapse.weight *= U
                else:
                    # facilitation
                    status += neurotypes.STPTypes['fac']
                    stp_type = 'fac'
                    tau = STP['tau_facil']
                    # modify weight according to stp parameter translation
                    synapse.weight *= (1 - U)

                if (U < 1. / 9):
                    cap = 0  # C2 = 1/8 * C1
                elif (U < 3. / 11):
                    cap = 2  # C2 = 3/8 * C1
                elif (U < 5. / 13):
                    cap = 4  # C2 = 5/8 * C1
                else:
                    cap = 6  # C2 = 7/8 * C1

                # capacitance C_2 (= strength)
                status += bool(4 & cap) * neurotypes.STPTypes['cap4']
                status += bool(2 & cap) * neurotypes.STPTypes['cap2']
                # print cap, status

                # setting STP globals
                tau_eff, Vdtc, Vstdf, Vfac = calibSTPglobals.get_stp_params(
                    stp_type, cap + 1, tau)
                # print "set global STP params to:", tau_eff, Vdtc, Vstdf, Vfac
                self._set_STP_globals(Vfac, Vstdf, Vdtc)

            else:
                raise Exception("STP problems!")
        # print ">> A Status for HWNeuron",target.index,":",status
        return status

    def monitorMembrane(self, neuron, value, doFlush=True, copyToTestPin=False):
        '''
        Enable/disable the connection between the network\'s neuron with index \'neuron\' and its dedicated output pin.
        return: pin
        '''

        hardwareNeuron = self.hardwareIndex(neuron)
        pin = hardwareNeuron % 4 + int(hardwareNeuron / neuronsPerBlock) * 4

        # check if pin is already occupied
        if value and (not self.cfg.membraneMonitorEnabled(hardwareNeuron)) and self.membraneReadoutPinOccupied[pin]:
            raise Exception('ERROR: You are trying to connect two membrane potentials to readout pin', pin,
                            'at the same time! Technically, this is possible, but the developers of PyHAL want to prevent this way of usage!')

        # adjust 'occupied' bit
        if value:
            self.membraneReadoutPinOccupied[pin] = True
        else:
            if self.cfg.membraneMonitorEnabled(hardwareNeuron):
                self.membraneReadoutPinOccupied[pin] = False

        # actually en- or disable neuron output
        self.cfg.enableMembraneMonitor(hardwareNeuron, value)
        if copyToTestPin:
            self.assignMembranePin2TestPin(pin)

        if doFlush:
            self.applyConfig(updateChip=False, updateDAC=False, updateParam=False,
                             updateRowConf=False, updateColConf=True, updateWeight=False)
        if value:
            myLogger.info('Monitoring hardware neuron ' + str(hardwareNeuron) +
                          ' (on membrane readout pin ' + str(pin) + ')')
        # else: print printIndentation + 'INFO: (pyhal_config_s1v2) Disabling
        # the monitoring of hardware neuron', hardwareNeuron
        return pin

    def assignMembranePin2TestPin(self, membranePin):
        '''Assigns the signal of membrane potential readout pin \'membranePin\' (value between 0 and 7) to the universal test pin    (ibTest).'''

        myLogger.debug('Muxing membrane readout pin ' +
                       str(membranePin % 8) + ' to test pin')
        if not self.dryRun:
            self.sp.assignMembranePin2TestPin(membranePin % 8)
            self.sp.setGlobalTrigger(True)

    def assignVoltage2TestPin(self, signal, block):
        '''Assigns the vOut voltage with index \'signal\' to the universal test pin    (ibTest).'''

        myLogger.info('Muxing voltage ' + str(signal) + ' to test pin.')
        if not self.dryRun:
            self.sp.assignVoltage2TestPin(signal, block)

    def assignMultipleVoltages2IBTest(self, vouts, leftBlock, rightBlock, pin4Mux):
        '''Assigns the vOut voltages (for seperate blocks) to ibTest. Switches the coaxial test pin to the membrane MUX!'''
        self.assignMembranePin2TestPin(pin4Mux)
        pyhalCVouts = pyhalc.vectorInt()
        for v in vouts:
            pyhalCVouts.append(v)
        myLogger.info("Connecting voltages " + str(vouts) + " ( leftBlock: " +
                      str(leftBlock) + " , rightBlock: " + str(rightBlock) + " ) to ibTest pin.")
        if not self.dryRun:
            self.sp.assignMultipleVoltages2IBTest(
                pyhalCVouts, leftBlock, rightBlock)

    def applyConfig(self, updateChip=True, updateDAC=True, updateParam=True, updateRowConf=True, updateColConf=True, updateWeight=True):
        '''Sends all prepared configuration data to the chip.'''

        msg = 'The following update flags are true:\n'
        if updateChip:
            msg += 'updateChip; '
        if updateDAC:
            msg += 'updateDAC; '
        if updateParam:
            msg += 'updateParam; '
        if updateRowConf:
            msg += 'updateRowConf; '
        if updateColConf:
            msg += 'updateColConf; '
        if updateWeight:
            msg += 'updateWeight; '
        myLogger.debug('Sending config object to spikey:\n' + msg)
        # print printIndentation + msg
        startTime = time.time()
        if not self.dryRun:
            self.sp.config(self.cfg, updateChip, updateDAC, updateParam,
                           updateRowConf, updateColConf, updateWeight)
        self.timeConfig += time.time() - startTime
        self.clearPlaybackMem()

    def writeConfigFile(self, filename):
        '''Writes the SpikeyConfig object to a file.'''

        # print printIndentation + 'INFO: (pyhal_config_s1v2) Writing
        # SpikeyConfig object to file',filename
        self.cfg.writeConfigFile(filename)

    def clearPlaybackMem(self):
        '''Clears the playback memory (all data stored in pb mem should be flushed before calling this function, otherwise it will be lost).'''

        if not self.dryRun:
            self.sp.clearPlaybackMem()

    def run(self, replay=False, expOffset=0):
        '''Runs the hardware.'''

        if self.debug:
            temparray = numpy.array(self.stin.data)
            # print printIndentation + 'data to be sent:'
            # print numpy.transpose(temparray)
            myLogger.info('Shape of data to be sent: ' +
                          str(numpy.shape(temparray)))
            myLogger.info('Send data to chip')

        if self.dryRun:
            myLogger.warn(
                "Hardware not used and hence experiment not executed.")
            return(len(self.stin.data[0]), len(self.stin.data[0]))

        self.bus.intClear()
        # if this run is just a replay of the last input spike train
        if replay:
            # self.bus.intClear()
            startTime = time.time()
            self.sp.resendSpikeTrain()    # calls Spikey::replayPB()
            myLogger.info("Resent...")
            # self.sp.flushPlaybackMemory() # trigger transmission of program to spikey
            # self.sp.Run()
            self.sp.waitPbFinished()
            myLogger.info("Waited for pb...")
            if self.debug:
                myLogger.info('Receive data from chip')
            self.timeRun += time.time() - startTime

            # receive output spikes
            startTime = time.time()
            self.sp.recSpikeTrain(self.stout)
            self.timeReceive += time.time() - startTime
            # self.sp.setLEDs(0)
            return (0, len(self.stin.data[0]))

        # create a temporal spike train object
        lostInputSpikes = pyhalc.SpikeTrain()
        # send input spike train plus a container for those spikes that can not
        # be delivered
        if self.stdp['enable']:
            # TP: TODO: redundant to vouts written via self.vout?
            # set STDP hardware parameters
            myLogger.debug('STDP parameters: vm: ' + str(self.vm) + ', tpcsec: ' + str(self.tpcsec) + ', tpcorperiod: ' + str(self.tpcorperiod) + ', vclra: ' + str(self.vouts[
                           0][8]) + ', vclrc: ' + str(self.vouts[0][9]) + ', vcthigh: ' + str(self.vouts[0][10]) + ', vctlow: ' + str(self.vouts[0][11]) + ', adjdel: ' + str(self.adjdel))
            # fill SpikeyConfig object: vm, tpcsec, tpcorperiod, vclra, vclrc,
            # vcthigh, vctlow, adjdel
            self.cfg.setSTDPParams(float(self.vm), int(self.tpcsec), int(self.tpcorperiod), float(self.vouts[0][
                                   8]), float(self.vouts[0][9]), float(self.vouts[0][10]), float(self.vouts[0][11]), float(self.adjdel))

            # calc frequency of STDP controller (dependent on number or rows)
            frequency = (
                self.stdp['maxR'] - self.stdp['minR'] + 1) * self.autoSTDPFrequency
            frequencyPreExp = (
                self.stdp['maxR'] - self.stdp['minR'] + 1) * self.autoSTDPFrequencyPreExp
            myLogger.info('STDP is enabled for row ' + str(self.stdp['minR']) + ' -> ' + str(self.stdp['maxR']) + ' and column ' + str(
                self.stdp['minC']) + ' -> ' + str(self.stdp['maxC']) + ' with frequency ' + str(round(1 / frequency * 1000.0, 3)) + 'Hz')

            # TP: TODO: beautify translation between PyNN and HAL synapse row indices
            # here to local parameter in case another projection is added and
            # network is re-run (if this is possible?)
            hwMinRow = int(default.numPresyns - self.stdp['maxR'] - 1)
            hwMaxRow = int(default.numPresyns - self.stdp['minR'] - 1)
            self.setSTDPParamsCont(
                self.contSTDP, expOffset, frequency, frequencyPreExp, hwMinRow, hwMaxRow)

            # last bools are 'changeWeights' (=True for continuous STDP, =False
            # for STDP curve recordings) and 'verbose'
            startTime = time.time()
            self.sp.sendSpikeTrainWithSTDP(self.stin, self.stout, lostInputSpikes,
                                           hwMinRow, hwMaxRow,
                                           int(self.stdp['minC']), int(
                                               self.stdp['maxC']),
                                           self.contSTDP, False)
            # this function reads back everything needed (including rec spiketrain)
            # no Run(), waitPbFinished()
            self.timeSend += time.time() - startTime
        else:
            myLogger.info('Transfer data to hardware system.')

            startTime = time.time()
            self.sp.sendSpikeTrain(self.stin, lostInputSpikes)
            self.timeEncode += time.time() - startTime
            startTime = time.time()
            self.sp.flushPlaybackMemory()  # trigger transmission of program to spikey
            self.timeSend += time.time() - startTime

            myLogger.info('Trigger network emulation.')
            startTime = time.time()
            self.sp.Run()  # execute program on spikey
            self.sp.waitPbFinished()  # wait for termination of program
            self.timeRun += time.time() - startTime

            myLogger.info('Receive spike data.')
            startTime = time.time()
            # receive output spikes (in case of non-STDP)
            self.sp.recSpikeTrain(self.stout)
            self.timeReceive += time.time() - startTime

        if self.debug:
            temparray = numpy.array(self.stout.data)
            # print printIndentation + 'received data:'
            # print numpy.transpose(temparray)
            myLogger.info('Shape of received data: ' +
                          str(numpy.shape(temparray)))
            myLogger.info("Number of lost input spikes " +
                          str(len(lostInputSpikes.data[0])))

        return (len(lostInputSpikes.data[0]), len(self.stin.data[0]))

    def numInputsPerNeuron(self):
        '''Returns the number of presynaptic inputs per neuron.'''

        return numPresyns

    def numNeurons(self):
        '''Returns the number of neurons per chip.'''

        return numNeurons

    def numNeuronsPerBlock(self):
        '''Returns the number of neurons per block.'''

        return neuronsPerBlock

    def numBlocks(self):
        '''Returns the number of neuron blocks per chip.'''

        return numBlocks

    def numVoutsPerBlock(self):
        '''Returns the number of vouts per chip block.'''

        return numVout

    def writeInSpikeTrain(self, filename):
        '''Writes the input spike train to file \'filename\'.'''

        self.stin.writeToFile(filename)

    def writeOutSpikeTrain(self, filename):
        '''Writes the output spike train to file \'filename\'.'''

        self.stout.writeToFile(filename)

    def minExcWeight(self):
        '''Returns the minimum excitatory weight larger than zero, i.e. the weight given in uS that corresponds to the discrete hardware weight value 1.'''
        self.checkHaveHardware()
        return 1.0 / (self.lsbPerMicroSiemens['exc'])

    def minInhWeight(self):
        '''Returns the minimum inhibitory weight larger than zero, i.e. the weight given in uS that corresponds to the discrete hardware weight value 1.'''
        self.checkHaveHardware()
        return 1.0 / (self.lsbPerMicroSiemens['inh'])

    def maxExcWeight(self, neuronIndex=0):
        '''Returns the maximum excitatory weight larger than zero, i.e. the weight given in uS that corresponds to the discrete hardware weight value 15.'''

        return 15. * self.minExcWeight()

    def maxInhWeight(self, neuronIndex=0):
        '''Returns the maximum inhibitory weight larger than zero, i.e. the weight given in uS that corresponds to the discrete hardware weight value 15.'''

        return 15. * self.minInhWeight()

    def chipVersion(self):
        '''Returns the Spikey version as an integer.'''
        self.checkHaveHardware()
        if not self.dryRun:
            if self._chipVersion is None:  # load chip version only once
                self._chipVersion = self.sp.version()
            return self._chipVersion
        else:
            return -1

    def getTemperature(self):
        '''Returns the temperature measured between Nathan and Spikey board'''
        self.checkHaveHardware()
        if not self.dryRun:
            return self.sp.getTemp()
        else:
            return 42.0  # dummy data

    def interruptActivity(self):
        '''Interrupts possible remaining activity from previous experiments by temporariliy setting all synaptic weights to zero.'''
        self.checkHaveHardware()
        if not self.dryRun:
            self.sp.interruptActivity(self.cfg)

    # TP: TODO: neuron resets are obsolete!
    def configNeuronReset(self, enable=True, constSeed=True, distance=196, duration=8):
        if (enable):
            myLogger.warn("This activated a workaround using neuron resets!")
            if (constSeed):
                myLogger.warn("A constant seed isn't supported yet!")
            assert (distance >= 16 and duration >= 2)
        if not self.dryRun:
            self.bus.configNeuronReset(enable, distance, duration)

    def translateToBioVoltage(self, hardwareVoltage):
        '''Translates the given hardware voltage into a biological interpretation.'''

        if self.hardwareToBioFactor == 0.0:
            raise Exception(
                'ERROR: Hardware to bio voltage conversion factor not yet defined!')

        bioV = self.bioOffset + \
            (hardwareVoltage - self.hardwareOffset) * self.hardwareToBioFactor
        return bioV

    def hardwareIndex(self, neuronIndex):
        '''Translates the given biological neuron index into the hardware neuron index.'''

        assert neuronIndex < numNeurons

        hwIndex = -1
        # this neuron is already mapped
        if self.hardwareIndexMap[neuronIndex] != -1:
            hwIndex = self.hardwareIndexMap[neuronIndex]
        else:
            # find next free hardware neuron
            while self.neuronIndexMap[self.neuronPermutation[self.hwNeuronCounter]] != -1:
                self.hwNeuronCounter += 1
                myLogger.trace(
                    'Hardware neuron ' + str(self.hwNeuronCounter) + ' is already mapped, skipping it.')
            hwIndex = self.neuronPermutation[self.hwNeuronCounter]
            assert self.hwNeuronCounter < numNeurons
            # fill look-up tables
            # should not be assigned to bio neuron yet
            assert self.hardwareIndexMap[neuronIndex] == -1
            self.hardwareIndexMap[neuronIndex] = hwIndex
            # should not be assigned to hw neuron yet
            assert self.neuronIndexMap[hwIndex] == -1
            self.neuronIndexMap[hwIndex] = neuronIndex
            self.hwNeuronCounter += 1

        return int(hwIndex)

    def hardwareIndexMax(self):
        '''Get highest hardware index of allocated neurons.'''
        return numpy.max(self.hardwareIndexMap)

    def neuronIndex(self, hwIndex):
        '''Translates the given hardware neuron index into the biological neuron index.'''

        return self.neuronIndexMap[hwIndex]

    def hardwareSynapseIndex(self, src):
        """Calculate synapse array indices from pyNN ID"""
        if src < 0:
            # TP: TODO: check, whether this works for 2 blocks
            srcHW = abs(src + 1) % default.numPresyns
        else:
            srcHW = (default.numPresyns - 1) - \
                (self.hardwareIndex(src) % default.neuronsPerBlock)
        return srcHW

    def setupUsbAdc(self, simtime):
        """Set duration for ADC sampling of membrane potential in ms"""
        startTime = time.time()
        if not self.dryRun:
            self.sp.setupFastAdc(simtime)
        self.timeReadAdc += time.time() - startTime

    def readUsbAdc(self, slope, offset):
        """Read ADC sampling data of membrane potential"""
        if not self.dryRun:
            startTime = time.time()
            mem = numpy.array(self.sp.readFastAdc(), numpy.float)
            self.timeReadAdc += time.time() - startTime
            mem = mem - offset
            mem = mem / slope
            mem = mem / 1e3  # from mV to V
        else:
            mem = numpy.arange(numpy.power(42, 5))  # dummy data
        return mem

    def delHardware(self):
        """Delete SpikeyHAL bus and Spikey object"""
        myLogger.debug('delete bus and spikey object')
        # this does not affect memory consumption :(
        del self.sp
        del self.bus
        #del self.cfg
        self.sp = None
        self.bus = None
        #self.cfg = None
        self.haveHardware = False
        # gc.collect() #did not help to free memory

    def getSoftProfiling(self):
        return {'initCpp': self.timeInit,
                'configCpp': self.timeConfig,
                'encodeCpp': self.timeEncode,
                'sendCpp': self.timeSend,
                'runCpp': self.timeRun,
                'receiveCpp': self.timeReceive,
                'adcCpp': self.timeReadAdc,
                'weightsCpp': self.timeReadWeights,
                'weightsPyNN': self.timeReadWeightsPyNN - self.timeReadWeights,
                'mapNetworkPyHAL': self.timeMapNetwork}

    def resetVoutNeuron(self):
        '''Reset all vout values affecting neuron parameters'''
        neuronParamIDs = range(8) + [16, 17]
        for blockID in range(default.numBlocks):
            for neuronParamID in neuronParamIDs:
                self.vouts[blockID][neuronParamID] = default.vouts[
                    blockID][neuronParamID]
