import pylogging as pylog
myLogger = pylog.get("PyN.prj")

from pyNN import common
from pyNN import connectors
import numpy
from pyNN.random import RandomDistribution, NativeRNG
from math import *
import types

import pyNN.hardware.spikey


import hwconfig_default_s1v2 as default

# this function is needed below in order to provide pyNN version backwards
# compatibility


def pynnVersionLarger05():
    try:
        v = int(((pyNN.__version__.partition(' ')[0]).partition(
            '.')[2]).partition('.')[0])
        return (v > 5)
    except:
        return True


DEFAULT_BUFFER_SIZE = 10000


class WDManager(object):

    def getWeight(self, w=None):
        if w is not None:
            weight = w
        else:
            weight = 1.
        return weight

    def getDelay(self, d=None):
        if d is not None:
            delay = d
        else:
            delay = _min_delay
        return delay

    def convertWeight(self, w, synapse_type):
        if isinstance(w, list):
            w = numpy.array(w)
        if isinstance(w, RandomDistribution):
            weight = RandomDistribution(w.name, w.parameters, w.rng)
            if weight.name == "uniform":
                (w_min, w_max) = weight.parameters
                weight.parameters = (w_min, w_max)
            elif weight.name == "normal":
                (w_mean, w_std) = weight.parameters
                weight.parameters = (w_mean, w_std)
        else:
            weight = w

        if synapse_type == 'inhibitory':
            if isinstance(weight, RandomDistribution):
                if weight.name == "uniform":
                    myLogger.info(weight.name + ' ' + str(weight.parameters))
                    (w_min, w_max) = weight.parameters
                    if w_min >= 0 and w_max >= 0:
                        weight.parameters = (-w_max, -w_min)
                elif weight.name == "normal":
                    (w_mean, w_std) = weight.parameters
                    if w_mean > 0:
                        weight.parameters = (-w_mean, w_std)
                else:
                    myLogger.warn(
                        "No conversion of the inhibitory weights for this particular distribution")
#            elif weight > 0:
                #weight *= -1
#                weight = weight
        return weight


class Projection(common.Projection, WDManager):

    class ConnectionDict:
        """docstring needed."""

        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, id):
            """Returns a (source address,target address number) tuple."""
            assert isinstance(id, int)
            return (self.parent._sources[id], self.parent._targets[id])

    def __init__(self, presynaptic_population, postsynaptic_population,
                 method='allToAll', source=None,
                 target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.

        source - string specifying which attribute of the presynaptic cell signals action potentials

        target - string specifying which synapse on the postsynaptic cell to connect to
        If source and/or target are not given, default values are used.

        method - string indicating which algorithm to use in determining connections.
        Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
        'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
        'fromFile', 'fromList'

        method_parameters - dict containing parameters needed by the connection method,
        although we should allow this to be a number or string if there is only
        one parameter.

        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target,
                                   synapse_dynamics, label, rng)

        self._targets = []     # holds gids
        self._sources = []     # holds gids
        self._weights = []     # holds the connection weights
        self._delays = []      # holds the delays of the connections
        self.synapse_type = target
        self._method = method

        if synapse_dynamics != None and isinstance(synapse_dynamics.slow, pyNN.hardware.spikey.STDPMechanism):
            # set range of synapse rows enabled for STDP
            presynaptic_neurons = self.pre.cell.flatten()
            pyNN.hardware.spikey.hardware.setSTDPRowsCont(presynaptic_neurons)

        # Create connections

        # unfortunately, the class Connector has changed its host module between pyNN 0.5 -> pyNN 0.6
        # this is a workaround to guarantee backwards compatibility
        if pynnVersionLarger05():
            ConnectorClass = connectors.Connector
        else:
            ConnectorClass = common.Connector

        if isinstance(method, str):
            connection_method = getattr(self, '_%s' % method)
            self.nconn = connection_method(method_parameters)
        elif isinstance(method, ConnectorClass):
            self.nconn = method.connect(self)

        # Define a method to access individual connections
        self.connection = Projection.ConnectionDict(self)

    def __len__(self):
        """Return the total number of connections."""
        return len(self._sources)

    def connections(self):
        """for conn in prj.connections()..."""
        for i in xrange(len(self)):
            yield self.connection[i]

    # --- Connection methods -------------------------------------------------

    def _allToAll(self, parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True  # when pre- and post- are the same population,
        # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = AllToAllConnector(allow_self_connections)
        return c.connect(self)

    def _oneToOne(self, parameters=None):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """
        c = OneToOneConnector()
        return c.connect(self)

    def _fixedProbability(self, parameters):
        """
        For each pair of pre-post cells, the connection probability is constant.
        """
        allow_self_connections = True
        try:
            p_connect = float(parameters)
        except TypeError:
            p_connect = parameters['p_connect']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        c = FixedProbabilityConnector(p_connect, allow_self_connections)
        return c.connect(self)

    def _distanceDependentProbability(self, parameters):
        """
        For each pair of pre-post cells, the connection probability depends on distance.
        d_expression should be the right-hand side of a valid python expression
        for probability, involving 'd', e.g. "exp(-abs(d))", or "float(d<3)"
        """
        allow_self_connections = True
        if type(parameters) == types.StringType:
            d_expression = parameters
        else:
            d_expression = parameters['d_expression']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        c = DistanceDependentProbabilityConnector(d_expression,
                                                  allow_self_connections=allow_self_connections)
        return c.connect(self)

    def _fixedNumberPre(self, parameters):
        """Each presynaptic cell makes a fixed number of connections."""
        n = parameters['n']
        if parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = FixedNumberPreConnector(n, allow_self_connections)
        return c.connect(self)

    def _fixedNumberPost(self, parameters):
        """Each postsynaptic cell receives a fixed number of connections."""
        n = parameters['n']
        if parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = FixedNumberPostConnector(n, allow_self_connections)
        return c.connect(self)

    def _fromFile(self, parameters):
        """
        Load connections from a file.
        """
        if type(parameters) == types.FileType:
            fileobj = parameters
            # should check here that fileobj is already open for reading
            lines = fileobj.readlines()
        elif type(parameters) == types.StringType:
            filename = parameters
            # now open the file...
            f = open(filename, 'r', DEFAULT_BUFFER_SIZE)
            lines = f.readlines()
        elif type(parameters) == types.DictType:
            # dict could have 'filename' key or 'file' key
            # implement this...
            raise NotImplementedError("Argument type not yet implemented")

        # We read the file and gather all the data in a list of tuples (one per
        # line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[", 1)[1]
            tgt = "[%s" % tgt.split("[", 1)[1]
            src = eval(src)
            tgt = eval(tgt)
            input_tuples.append((src, tgt, float(w), float(d)))
        f.close()

        self._fromList(input_tuples)

    def _fromList(self, conn_list):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        for i in xrange(len(conn_list)):
            src, tgt, weight, delay = conn_list[i][:]
            src = self.pre[tuple(numpy.atleast_1d(src))]
            tgt = self.post[tuple(numpy.atleast_1d(tgt))]
            pyNN.hardware.spikey.connect(source=src, target=tgt, weight=weight, delay=delay,
                                         synapse_type=self.synapse_type, synapse_dynamics=self.synapse_dynamics)
            self._sources.append(src)
            self._targets.append(tgt)
            self._weights.append(weight)
            self._delays.append(delay)

    # --- Methods for setting connection parameters --------------------------

    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and uS for conductance-based
        synapses.
        """
        w = self.convertWeight(w, self.synapse_type)

        if not isinstance(w, (list, numpy.ndarray)):
            w = [w] * len(self._sources)

        for i in xrange(len(self._sources)):
            pyNN.hardware.spikey.connect(source=self._sources[i], target=self._targets[i], weight=w[
                                         i], synapse_type=self.synapse_type, synapse_dynamics=self.synapse_dynamics)
        self._weights = w

    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        self.setWeights(rand_distr.next(len(self)))

    def setDelays(self, d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        # print "WARNING: Method setDelays not implementable for the Spikey
        # hardware!"
        pass

    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        raise NotImplementedError("Method not implemented yet")

    def setThreshold(self, threshold):
        """
        Where the emission of a spike is determined by watching for a
        threshold crossing, set the value of this threshold.
        """
        # This is a bit tricky, because in NEST the spike threshold is a
        # property of the cell model, whereas in NEURON it is a property of the
        # connection (NetCon).
        raise Exception("Method deprecated.")

    def getDrviriseFactor(self):
        """
        Returns the value of the drvirise_calib parameter valid for all synaptic connections on the chip.
        """

        return pyNN.hardware.spikey.hardware.hwa.drviriseCalib

    def getDrviriseFactorsRange(self):
        return [default.currentMin, default.currentMax]

    def setDrviriseFactor(self, value):
        """
        Sets the value of the drvirise_calib parameter valid for all synaptic connection on the chip.
        """

        if value < default.currentMin or value > default.currentMax:  # TP: introduce drvirise_base
            raise Exception('ERROR: Drvirise factor outside of allowed range ' +
                            str(self.getDrviriseFactorsRange()))

        pyNN.hardware.spikey.hardware.hwa.drviriseCalib = value

    def scaleDrviriseFactor(self, factor):
        value = self.getDrviriseFactor() * factor
        self.setDrviriseFactor(value)

    def getDrvifallFactors(self):
        """
        Returns a list containing the active drvifall_calib parameter values for all synaptic connection within this Projection.
        Note that for every synapse driver two drvifall_calib values exist: one for excitatory, one for inhibotory configuration.
        Only the list of currently active parameters is returned (i.e. exc OR inh).
        """

        value_list = []
        if self.synapse_type == 'excitatory':
            syntype = 'exc'
        else:
            syntype = 'inh'

        for s, t in zip(self._sources, self._targets):
            targetHardwareNeuron = pyNN.hardware.spikey.hardware.hwa.hardwareIndex(
                t)
            targetHardwareBlock = targetHardwareNeuron / default.neuronsPerBlock
            targetSynapseBlockOffset = targetHardwareBlock * default.numPresyns
            if s.cell.__class__ == pyNN.hardware.spikey.IF_facets_hardware1:
                driverIndex = targetSynapseBlockOffset + \
                    (s % default.neuronsPerBlock)
            elif s.cell.__class__ in [pyNN.hardware.spikey.SpikeSourcePoisson, pyNN.hardware.spikey.SpikeSourceArray]:
                source = -1 - s
                driverIndex = (default.numPresyns - 1 - (source %
                                                         default.numPresyns)) + targetSynapseBlockOffset
            else:
                raise Exception(
                    'ERROR: Could not determine celltype of Projection source!')
            value_list.append(pyNN.hardware.spikey.hardware.hwa.drvifallCalib[
                              syntype][driverIndex])
        return value_list

    def getDrvifallFactorsRange(self, syntype):
        return [default.currentMin / default.drvifall_base[syntype], default.currentMax / default.drvifall_base[syntype]]

    def setDrvifallFactors(self, value_list):
        """
        Sets the currently active drvifall_calib parameter values for all synaptic connection within this Projection
        instance to the values found in value_list. This list must contain as many floating point numbers as there are
        connections within this Projection.
        """

        if self.synapse_type == 'excitatory':
            syntype = 'exc'
        else:
            syntype = 'inh'

        for i, x in enumerate(zip(self._sources, self._targets)):
            value = value_list[i]
            factorsRange = self.getDrvifallFactorsRange(syntype)
            if value < factorsRange[0] or value > factorsRange[1]:
                raise Exception(
                    'ERROR: Drvifall factor outside of allowed range ' + str(factorsRange))
            s = x[0]
            t = x[1]
            targetHardwareNeuron = pyNN.hardware.spikey.hardware.hwa.hardwareIndex(
                t)
            targetHardwareBlock = targetHardwareNeuron / default.neuronsPerBlock
            targetSynapseBlockOffset = targetHardwareBlock * default.numPresyns
            if s.cell.__class__ == pyNN.hardware.spikey.IF_facets_hardware1:
                driverIndex = targetSynapseBlockOffset + \
                    (s % default.neuronsPerBlock)
            source = -1 - s
            if s.cell.__class__ in [pyNN.hardware.spikey.SpikeSourcePoisson, pyNN.hardware.spikey.SpikeSourceArray]:
                driverIndex = (default.numPresyns - 1 - (source %
                                                         default.numPresyns)) + targetSynapseBlockOffset
            pyNN.hardware.spikey.hardware.hwa.drvifallCalib[
                syntype][driverIndex] = value

    def scaleDrvifallFactors(self, factor):
        value_list = numpy.array(self.getDrvifallFactors()) * factor
        self.setDrvifallFactors(value_list.tolist())

    def getDrvioutFactors(self):
        """
        Returns a list containing the active drviout_calib parameter values for all synaptic connection within this Projection.
        Note that for every synapse driver two drviout_calib values exist: one for excitatory, one for inhibotory configuration.
        Only the list of currently active parameters is returned (i.e. exc OR inh).
        """

        value_list = []
        if self.synapse_type == 'excitatory':
            syntype = 'exc'
        else:
            syntype = 'inh'

        for s, t in zip(self._sources, self._targets):
            targetHardwareNeuron = pyNN.hardware.spikey.hardware.hwa.hardwareIndex(
                t)
            targetHardwareBlock = targetHardwareNeuron / default.neuronsPerBlock
            targetSynapseBlockOffset = targetHardwareBlock * default.numPresyns
            if s.cell.__class__ == pyNN.hardware.spikey.IF_facets_hardware1:
                driverIndex = targetSynapseBlockOffset + \
                    (s % default.neuronsPerBlock)
            elif s.cell.__class__ in [pyNN.hardware.spikey.SpikeSourcePoisson, pyNN.hardware.spikey.SpikeSourceArray]:
                source = -1 - s
                driverIndex = (default.numPresyns - 1 - (source %
                                                         default.numPresyns)) + targetSynapseBlockOffset
            else:
                raise Exception(
                    'ERROR: Could not determine celltype of Projection source!')
            value_list.append(pyNN.hardware.spikey.hardware.hwa.drvioutCalib[
                              syntype][driverIndex])
        return value_list

    def getDrvioutFactorsRange(self, syntype):
        return [default.currentMin / default.drviout_base[syntype], default.currentMax / default.drviout_base[syntype]]

    def setDrvioutFactors(self, value_list):
        """
        Sets the currently active drviout_calib parameter values for all synaptic connection within this Projection
        instance to the values found in value_list. This list must contain as many floating point numbers as there are
        connections within this Projection.
        """

        if self.synapse_type == 'excitatory':
            syntype = 'exc'
        else:
            syntype = 'inh'

        for i, x in enumerate(zip(self._sources, self._targets)):
            value = float(value_list[i])
            factorsRange = self.getDrvioutFactorsRange(syntype)
            if value < factorsRange[0] or value > factorsRange[1]:
                raise Exception(
                    'ERROR: Drviout factor outside of allowed range ' + str(factorsRange))
            s = x[0]
            t = x[1]
            targetHardwareNeuron = pyNN.hardware.spikey.hardware.hwa.hardwareIndex(
                t)
            targetHardwareBlock = targetHardwareNeuron / default.neuronsPerBlock
            targetSynapseBlockOffset = targetHardwareBlock * default.numPresyns
            if s.cell.__class__ == pyNN.hardware.spikey.IF_facets_hardware1:
                driverIndex = targetSynapseBlockOffset + \
                    (s % default.neuronsPerBlock)
            source = -1 - s
            if s.cell.__class__ in [pyNN.hardware.spikey.SpikeSourcePoisson, pyNN.hardware.spikey.SpikeSourceArray]:
                driverIndex = (default.numPresyns - 1 - (source %
                                                         default.numPresyns)) + targetSynapseBlockOffset
            pyNN.hardware.spikey.hardware.hwa.drvioutCalib[
                syntype][driverIndex] = value

    def scaleDrvioutFactors(self, factor):
        value_list = numpy.array(self.getDrvioutFactors()) * factor
        self.setDrvioutFactors(value_list.tolist())

    # --- Methods for writing/reading information to/from file. --------------

    def _dump_connections(self):
        """For debugging."""
        print "Connections for Projection %s, connected with %s" % (self.label or '(un-labelled)',
                                                                    self._method)
        print "\tsource\ttarget"
        for conn in zip(self._sources, self._targets):
            print "\t%d\t%d" % conn

    def _get_connection_values(self, format, parameter_name, gather):
        assert format in (
            'list', 'array'), "`format` is '%s', should be one of 'list', 'array'" % format
        assert parameter_name in (
            'weight', 'delay'), "`parameter_name` is '%s', should be one of 'weight', 'delay'" % format
        if parameter_name == 'weight':
            parameter = self._weights
        elif parameter_name == 'delay':
            parameter = self._delays

        if format == 'list':
            return parameter[:]
        elif format == 'array':
            values = numpy.NaN * numpy.zeros((self.pre.size, self.post.size))
            for ((src, tgt), v) in zip(self.connections(), parameter):
                values[src - self.pre.first_id, tgt - self.post.first_id] = v
        return values

    def getWeights(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D weight array (with zero or None for non-existent
        connections).
        """
        weights = self._get_connection_values(format, 'weight', gather)

        return weights

    def getWeightsHW(self, format='list', gather=True, readHW=False):
        """
        Get hardware weights (in bits) before (readHW=False) and after (readHW=True)
        the experiment run
        """
        connList = []
        for (src, tgt) in self.connections():
            connList.append([src, tgt])

        return pyNN.hardware.spikey.hardware.getWeightsHW(connList, self.synapse_type, format, readHW)

    def getDelays(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D delay array (with None or 1e12 for non-existent
        connections).
        """
        return self._get_connection_values(format, 'delay', gather)

    def saveConnections(self, filename, gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        f = open(filename, 'w', DEFAULT_BUFFER_SIZE)

        fmt = "%s%s\t%s%s\t%s\t%s\n" % (self.pre.label, "%s", self.post.label,
                                        "%s", "%g", "%g")
        for i in xrange(len(self)):
            line = fmt % (self.pre.locate(self._sources[i]),
                          self.post.locate(self._targets[i]),
                          self._weights[i],
                          self._delays[i])
            line = line.replace('(', '[').replace(')', ']')
            f.write(line)
        f.close()

    def printWeights(self, filename, format='list', gather=True):
        """Print synaptic weights to file."""
        weights = self.getWeights(format=format, gather=gather)
        f = open(filename, 'w', DEFAULT_BUFFER_SIZE)
        if format == 'list':
            f.write("\n".join([str(w) for w in weights]))
        elif format == 'array':
            fmt = "%g " * len(self.post) + "\n"
            for row in weights:
                f.write(fmt % tuple(row))
        f.close()

    def weightHistogram(self, min=None, max=None, nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        bins = numpy.arange(min, max, (max - min) / nbins)
        # returns n, bins
        return numpy.histogram(self.getWeights(format='list', gather=True), bins)


class AllToAllConnector(connectors.AllToAllConnector, WDManager):

    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay = self.getDelay(self.delays)
        postsynaptic_neurons = projection.post.cell.flatten()
        target_list = postsynaptic_neurons.tolist()
        for pre in projection.pre.cell.flat:
            # if self connections are not allowed, check whether pre and post
            # are the same
            if not self.allow_self_connections:
                if pre in target_list:
                    target_list = postsynaptic_neurons.tolist()
                    target_list.remove(pre)
            N = len(target_list)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight] * N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)] * N
            projection._targets += target_list
            projection._sources += [pre] * N
            projection._weights += weights
            projection._delays += delays
            #projection._target_ports += get_target_ports(pre, target_list)

            for i in xrange(N):
                pyNN.hardware.spikey.connect(source=pre, target=target_list[i], weight=weights[i], delay=delays[
                                             i], synapse_type=projection.synapse_type, synapse_dynamics=projection.synapse_dynamics)

        return len(projection._targets)


class OneToOneConnector(connectors.OneToOneConnector, WDManager):

    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay = self.getDelay(self.delays)
        if projection.pre.dim == projection.post.dim:
            projection._sources = projection.pre.cell.flatten()
            projection._targets = projection.post.cell.flatten()
            N = len(projection._sources)

            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight] * N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)] * N

            projection._weights = weights
            projection._delays = delays

            for i in xrange(N):
                pyNN.hardware.spikey.connect(source=projection._sources[i], target=projection._targets[i], weight=weights[
                                             i], delay=delays[i], synapse_type=projection.synapse_type, synapse_dynamics=projection.synapse_dynamics)
            return N
        else:
            raise NotImplementedError(
                "Connection method not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes.")


class FixedProbabilityConnector(connectors.FixedProbabilityConnector, WDManager):

    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay = self.getDelay(self.delays)
        postsynaptic_neurons = projection.post.cell.flatten()
        npost = projection.post.size
        for pre in projection.pre.cell.flat:
            if projection.rng:
                rarr = projection.rng.uniform(
                    0, 1, (npost,))  # what about NativeRNG?
            else:
                rarr = numpy.random.uniform(0, 1, (npost,))
            target_list = numpy.compress(numpy.less(
                rarr, self.p_connect), postsynaptic_neurons).tolist()
            # if self connections are not allowed, check whether pre and post
            # are the same
            if not self.allow_self_connections and pre in target_list:
                target_list.remove(pre)
            N = len(target_list)

            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight] * N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)] * N

            projection._targets += target_list
            projection._sources += [pre] * N
            projection._weights += weights
            projection._delays += delays

            for i in xrange(N):
                pyNN.hardware.spikey.connect(source=pre, target=target_list[i], weight=weights[i], delay=delays[
                                             i], synapse_type=projection.synapse_type, synapse_dynamics=projection.synapse_dynamics)

        return len(projection._sources)


class DistanceDependentProbabilityConnector(connectors.DistanceDependentProbabilityConnector, WDManager):

    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay = self.getDelay(self.delays)
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is not None:
            dimensions = projection.post.dim
            periodic_boundaries = numpy.concatenate(
                (dimensions, numpy.zeros(3 - len(dimensions))))
        postsynaptic_neurons = projection.post.cell.flatten()  # array
        presynaptic_neurons = projection.pre.cell.flat  # iterator
        # what about NativeRNG?
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                myLogger.warn(
                    "Use of NativeRNG not implemented. Using NumpyRNG")
                rng = numpy.random
            else:
                rng = projection.rng
        else:
            rng = numpy.random
        rarr = rng.uniform(0, 1, (projection.pre.size * projection.post.size,))
        j = 0
        idx_post = 0
        for pre in presynaptic_neurons:
            target_list = []
            idx_post = 0
            distances = common.distances(pre, projection.post, self.mask,
                                         self.scale_factor, self.offset,
                                         periodic_boundaries)
            for post in postsynaptic_neurons:
                if self.allow_self_connections or pre != post:
                    # calculate the distance between the two cells :
                    d = distances[0][idx_post]
                    p = eval(self.d_expression)
                    if p >= 1 or (0 < p < 1 and rarr[j] < p):
                        target_list.append(post)
                        # projection._targets.append(post)
                        # projection._target_ports.append(nest.connect(pre_addr,post_addr))
                        #nest.ConnectWD([pre],[post], [weight], [delay])
                j += 1
                idx_post += 1
            N = len(target_list)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight] * N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)] * N
            projection._targets += target_list
            projection._sources += [pre] * N
            projection._weights += weights
            projection._delays += delays

            for i in xrange(N):
                pyNN.hardware.spikey.connect(source=pre, target=target_list[i], weight=weights[i], delay=delays[
                                             i], synapse_type=projection.synapse_type, synapse_dynamics=projection.synapse_dynamics)
        return len(projection._sources)


class FixedNumberPostConnector(connectors.FixedNumberPostConnector, WDManager):

    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay = self.getDelay(self.delays)

        postsynaptic_neurons = projection.post.cell.flatten()
        if projection.rng:
            rng = projection.rng
        else:
            rng = numpy.random
        for pre in projection.pre.cell.flat:
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            # if self connections are not allowed, check whether pre and post
            # are the same
            if not self.allow_self_connections and pre in postsynaptic_neurons:
                numpy.delete(postsynaptic_neurons, numpy.where(
                    postsynaptic_neurons == pre)[0])
            target_list = rng.permutation(postsynaptic_neurons)[0:n].tolist()

            N = len(target_list)

            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight] * N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)] * N

            for i in xrange(N):
                pyNN.hardware.spikey.connect(source=pre, target=target_list[i], weight=weights[i], delay=delays[
                                             i], synapse_type=projection.synapse_type, synapse_dynamics=projection.synapse_dynamics)

            projection._sources += [pre] * N
            projection._targets += target_list
            projection._weights += weights
            projection._delays += delays

        return len(projection._sources)


class FixedNumberPreConnector(connectors.FixedNumberPreConnector, WDManager):

    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay = self.getDelay(self.delays)

        presynaptic_neurons = projection.pre.cell.flatten()
        if projection.rng:
            rng = projection.rng
        else:
            rng = numpy.random
        for post in projection.post.cell.flat:
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            # if self connections are not allowed, check whether pre and post
            # are the same
            if not self.allow_self_connections and post in presynaptic_neurons:
                numpy.delete(presynaptic_neurons, numpy.where(
                    presynaptic_neurons == post)[0])
            source_list = rng.permutation(presynaptic_neurons)[0:n].tolist()

            N = len(source_list)

            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight] * N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)] * N

            for i in xrange(N):
                pyNN.hardware.spikey.connect(source=source_list[i], target=post, weight=weights[i], delay=delays[
                                             i], synapse_type=projection.synapse_type, synapse_dynamics=projection.synapse_dynamics)

            projection._sources += source_list
            projection._targets += [post] * N
            projection._weights += weights
            projection._delays += delays

        return len(projection._sources)


class FromListConnector(connectors.FromListConnector):

    def connect(self, projection):
        projection._fromList(self.conn_list)


class FromFileConnector(connectors.FromFileConnector):

    def connect(self, projection):
        projection._fromFile(self.filename)
