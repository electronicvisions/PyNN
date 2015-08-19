# ***************************************************************************
#
# This file contains the most important structures needed to create and
# administrate abstract representations of neural networks intended
# to be emulated on the Spikey neuromorphic system.
# The building blocks for such representations are 'Neurons' and 'Network's
# of 'Neuron's. These classes have methods for the creation of different types
# of neurons and for interconnecting them. A check for the feasibility to map
# these networks to the hardware is provided.
#
# ***************************************************************************

import pylogging as pylog
myLogger = pylog.get("PyN.bbl")

# these modules will be needed
import numpy
import string
import pyhal_neurotypes as neurotypes
import copy

# for nicer stdout messages
printIndentation = '    '


class Network:
    '''This class is an abstract representation of a neural network to be emulated on the Spikey neuromorphic system.'''

    def __init__(self, maxSize=384, maxExternalInputs=512, maxExternalInputsPerNeuron=256, debug=False):
        '''Constructor of class Network.'''

        self.netsize = 0
        self.neuron = []
        self.largestExternalInputIndex = -1
        self.maxSize = maxSize
        self.maxExternalInputs = maxExternalInputs
        self.maxExternalInputsPerNeuron = maxExternalInputsPerNeuron
        self.debug = debug

    def clear(self):
        '''Clears the network, i.e. removes all neurons and synapses.'''

        self.setSize(0)
        self.largestExternalInputIndex = -1

    def size(self):
        '''Returns number of neurons within the network.'''

        # print printIndentation + 'current network size is',self.netsize
        return self.netsize

    def setSize(self, newsize):
        '''Sets the number of neurons within the network to \'newsize\'.'''

        if newsize < 0 or newsize > self.maxSize:
            raise Exception("Invalid network size!")
        oldsize = self.netsize
        diff = newsize - oldsize
        if diff == 0:
            return
        elif newsize > 0:  # increase network size
            for i in range(diff):
                self.create()
        else:             # decrease network size
            for i in range(abs(diff)):
                self.neuron.pop()
        self.netsize = newsize
        myLogger.debug("New network size is now " + str(self.netsize))

    def create(self, paramDict=None, number=1):
        '''Creates \'number\' new neurons within the network.'''

        for i in range(number):
            # print paramDict
            n = Neuron(paramDict)
            n.index = self.netsize
            self.neuron.append(n)
            self.netsize += 1

        myLogger.debug("New network size is now " + str(self.netsize))

    def setNeuronParameters(self, neuronIndex, paramDict):
        '''Sets the parameters of the neuron with index \'neuronIndex\' to those found in dictionary \'paramDict\'.'''

        self.neuron[neuronIndex].setParameters(paramDict)

    def setNeuronType(self, neuronIndex, ntype):
        '''Sets the type of the neuron with index \'neuronIndex\' to \'ntype\'.'''

        if neuronIndex > self.netsize:
            raise Exception(
                "Given neuron index exceeds network size of", self.netsize, '!')
        self.neuron[neuronIndex].setType(ntype)

    def connect(self, sourceType, source, target, weight=1., delay=0., **extra_params):
        """
        Connects the neuron or spike source (determined by \'sourceType\')  with index \'source\' to the neuron with index \'target\' with connection weight \'weight\'.
        The delay is neglected for this version of the hardware.
        """

        # print printIndentation + 'trying to connect',source,'of
        # type',sourceType,'with',target,'with weight',weight,'and delay',delay
        if (target >= self.netsize) or (target < 0):
            raise Exception("Invalid target index!")
        if source < 0:
            raise Exception("Invalid source index!")
        if delay < 0:
            raise Exception("Invalid delay!")

        # transform the weight to a Synapse object
        synapse = Synapse(weight, STDP=extra_params[
                          'STDP'], STP=extra_params['STP'])

        # source is a neuron
        if sourceType == neurotypes.connectionType["internal"]:
            # print printIndentation + 'creating an internal connection'
            if source >= self.netsize:
                raise Exception("Invalid source index!")
            self.neuron[target].addIncomingConnection(
                sourceType, source, synapse)
            self.neuron[source].addOutgoingConnection(target, synapse)
            if weight >= 0.:
                self.neuron[source].setType(
                    neurotypes.neuronType["excitatory"])
            elif weight < 0.:
                self.neuron[source].setType(
                    neurotypes.neuronType["inhibitory"])

        # source is an external input
        elif sourceType == neurotypes.connectionType["external"]:
            # print printIndentation + 'creating an external connection'
            if source >= self.maxExternalInputs:
                raise Exception("Invalid source index!")
            self.neuron[target].addIncomingConnection(
                sourceType, source, synapse)
            if source > self.largestExternalInputIndex:
                self.largestExternalInputIndex = source
        else:
            raise Exception("Invalid source type!")

    def printConnectivity(self):
        '''Prints the connections of all neurons within the network to the std output.'''

        for n in self.neuron:
            n.printConnectivity()

    def largestInputIndex(self):
        '''Prints the largest index of all available external inputs into the network to the std output.'''

        return self.largestExternalInputIndex

    def generateNetlist(self, filename):
        '''Writes the netlist into a file with name \'filename\', if possible.'''

        output = open(filename, 'wb')
        for n in self.neuron:
            n.generateNetlistEntry(output)

        output.close()


class Synapse():
    '''Simple Synapse class: weight is stored as float. Additional STDP and STP parameters can be provided.'''

    weight = 0.0
    STDP = None
    STP = None

    def __init__(self, weight, STDP=None, STP=None):
        self.weight = weight
        self.STDP = STDP
        self.STP = STP


class Neuron:
    '''This class is an abstract representation of a neuron to be emulated on the Spikey neuromorphic system.'''

    def __init__(self, paramDict):
        '''Constructor of the class \'Neuron\'.'''

        self.index = 0
        self.recordSpikes = False

        self.externalWeights = {}
        self.incomingNeuronWeights = {}
        self.outgoingNeuronWeights = {}

        self.parameters = {
            'ntype': neurotypes.neuronType['excitatory'],
            'estimator_cm': 0.0,
            'estimator_e_rev_E': 0.0,
            'estimator_tau_syn_E': 0.0,
            'estimator_tau_syn_I': 0.0,
            'g_leak': 0.0,
            'tau_refrac': 0.0,

            'v_reset': 0.0,
            'e_rev_I': 0.0,
            'v_rest': 0.0,
            'v_thresh': 0.0,

            'lowlevel_parameters': {},
            'index': 0,
            'recordSpikes': False
        }

        if paramDict:
            self.setParameters(paramDict)

    def setParameters(self, paramDict):
        '''Sets the parameters of this neuron to those with matching keys in \'paramDict\'.'''

        for p in paramDict.keys():
            if self.parameters.has_key(p):
                self.parameters[p] = copy.deepcopy(paramDict[p])
            else:
                raise Exception('Neuron has no parameter named ' + p + '!')

    def printConnectivity(self):
        '''Prints the incoming connections of this neuron to the std output.'''

        for i in self.incomingNeuronWeights.keys():
            print printIndentation + "neuron", string.rjust(str(int(i)), 4), "----", string.rjust(str(self.incomingNeuronWeights[i]), 6), "----> neuron", string.rjust(str(int(self.index)), 4)
        for i in self.externalWeights.keys():
            print printIndentation + "input ", string.rjust(str(int(i)), 4), "----", string.rjust(str(self.externalWeights[i]), 6), "----> neuron", string.rjust(str(int(self.index)), 4)

    def generateNetlistEntry(self, filehandler):
        '''Writes all outgoing connections of this neuron to a open file given by \'filehandler\'.'''

        for i in self.outgoingNeuronWeights.keys():
            filehandler.write(str(int(self.index)) + "    " +
                              str(self.outgoingNeuronWeights[i]) + "    " + str(int(i)) + "\n")

    def setType(self, t):
        '''Sets the type of this neuron to \'t\'.'''

        if t in neurotypes.neuronType.values():
            self.parameters['ntype'] = t
        else:
            raise Exception("Unknown neuron type!")

    def addIncomingConnection(self, sourceType, source, synapse):
        '''Adds an incoming connection to this neuron from a neuron or spike source (determined by \'sourceType\') with index \'source\', with connection weight \'weight\'.'''

        # print 'adding source',source,'of type',sourceType,'to
        # neuron',self.parameters['index']
        if sourceType == 0:
            # source is a neuron
            self.incomingNeuronWeights[source] = synapse

        elif sourceType == 1:
            # source is an external input
            self.externalWeights[source] = synapse

        else:
            raise Exception("Invalid source type!")

    def addOutgoingConnection(self, target, weight):
        '''Adds an incoming connection to this neuron from a neuron or spike source (determined by \'sourceType\') with index \'source\', with connection weight \'weight\'.'''
        self.outgoingNeuronWeights[target] = weight
