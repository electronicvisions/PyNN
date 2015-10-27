import pylogging as pylog
myLogger = pylog.get("PyN.pop")

from pyNN import common
import numpy
from pyNN.random import RandomDistribution, NativeRNG
from math import *
import types
import os
import hwconfig_default_s1v2 as default

import pyNN.hardware.spikey


################################
##  PyNN Object-Oriented API  ##
################################

class PopulationIterator:
    '''
    Implementation of an iterator supporting calls like 'for cell in population:'
    '''

    def __init__(self, target):
        self.index = 0
        self.target = target

    def __iter__(self):
        return self

    def next(self):
        if self.index >= self.target.size:
            raise StopIteration
        self.index = self.index + 1
        return self.target.cell.flatten().tolist()[self.index - 1]


class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """

    nPop = 0

    def __init__(self, dims, cellclass, cellparams=None, label=None):
        """
        dims should be a tuple containing the population dimensions, or a single
        integer, for a one-dimensional population.
        e.g., (10,10) will create a two-dimensional population of size 10x10.
        cellclass should either be a standardized cell class (a class inheriting
        from common.StandardCellType) or a string giving the name of the
        simulator-specific model that makes up the population.
        cellparams should be a dict which is passed to the neuron model
        constructor
        label is an optional name for the population.
        """
        self.dim = dims
        if isinstance(dims, int):  # also allow a single integer, for a 1D population
            self.dim = (self.dim,)
        else:
            assert isinstance(
                dims, tuple), "`dims` must be an integer or a tuple."
        self.label = label
        self.celltype = cellclass
        self.ndim = len(self.dim)
        if cellparams:
            self.cellparams = cellparams
        else:
            self.cellparams = cellclass.default_parameters
        self.size = self.dim[0]
        for i in range(1, self.ndim):
            self.size *= self.dim[i]
        # create and reshape cells
        self.cell = pyNN.hardware.spikey.create(
            self.celltype, self.cellparams, self.size)
        if type(self.cell) != type([]):
            self.cell = [self.cell]
        for c in self.cell:     # set parent
            c.parent = self
        self.cell = numpy.array([c for c in self.cell],
                                pyNN.hardware.spikey.ID)
        self.cell = numpy.reshape(self.cell, self.dim)

        self.first_id = int(self.cell.flatten().tolist()[0])
        self.last_id = int(self.cell.flatten().tolist()[-1])

        if not self.label:
            self.label = 'population%d' % Population.nPop
        Population.nPop += 1

    def __getitem__(self, addr):
        """
        Returns a representation of the cell with coordinates given by addr,
        suitable for being passed to other methods that require a cell id.
        Note that __getitem__ is called when using [] access, e.g.
          p = Population(...)
          p[2,3] is equivalent to p.__getitem__((2,3)).
        """
        if isinstance(addr, int):
            addr = (addr,)
        if len(addr) == self.ndim:  # check if adress is in correct range
            for a, d in zip(addr, self.dim):
                if a >= d:
                    raise Exception("Element" + str(addr) +
                                    " not in " + self.label)
            return self.cell[addr]
        else:
            raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (
                self.ndim, str(addr))

    def __len__(self):
        """Returns the total number of cells in the population."""
        return self.size

    def meanSpikeCount(self, gather=True):
        """Returns the mean number of spikes per neuron."""
        # ---- HARDWARE SPECIFIC IMPLEMENTATION ------------------------
        nRecNeurons = 0
        nSpikes = 0
        for id in self.cell.flatten().tolist():
            if pyNN.hardware.spikey.hardware.net.neuron[id].recordSpikes:
                nRecNeurons += 1
                nSpikes += len(pyNN.hardware.spikey.hardware.getOutputOfNeuron(id.cell))

        if nRecNeurons == 0:
            myLogger.warn(
                'No neuron has been recorded in population <' + self.label + '>.')
            return None
        else:
            myLogger.info('\nMean spike count for population <' + self.label + '> is based on '
                          + str(nRecNeurons) + ' out of ' + str(self.size) +
                          ' neurons.\n <spikes per neuron> = '
                          + str(float(nSpikes) / float(nRecNeurons)))
            return float(nSpikes) / float(nRecNeurons)

    def __iter__(self):
        '''Returns a PopulationIterator object.'''
        return PopulationIterator(self)

    def ids(self):
        '''Usage: iter = p.ids() ; iter.next() provides the next id in p.cell'''
        return self.__iter__()

    def locate(self, id):
        assert isinstance(id, int)
        try:
            return tuple([a.tolist()[0] for a in numpy.where(self.cell == id)])
        except:
            raise Exception(self.label + '.locate(): ID ' +
                            str(id) + ' not found.')

    def set(self, param, val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        if isinstance(param, str):
            paramDict = {param: val}
        elif isinstance(param, dict):
            paramDict = param
        else:
            raise common.InvalidParameterValueError
        cells = self.cell.flatten().tolist()
        pyNN.hardware.spikey.set(cells, self.celltype, paramDict)

    def tset(self, parametername, valueArray):
        """
        'Topographic' set. Sets the value of parametername to the values in
        valueArray, which must have the same dimensions as the Population.
        """
        if len(valueArray) != self.cell.shape[0]:
            raise Exception(
                'Population an valueArray must be of same dimensions!')

        for cell, value in zip(self, valueArray):
            pyNN.hardware.spikey.set(
                cell, cell.cellclass, parametername, val=value)

    def record(self, record_from=None, rng=None, to_file=False):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """

        # VERY SIMPLE TMP-FILENAME!!!
        if to_file:
            raise NotImplementedError(
                'Buffering population spike data in files (in order to save RAM) is not supported!')

        if record_from is None:
            self.recorded_cells = self.cell.flatten().tolist()
            myLogger.debug('Recording entire population <' + self.label + '>')
        elif isinstance(record_from, int):
            raise NotImplementedError(
                'Recording from randomly chosen cells not yet implemented.')
        elif isinstance(record_from, list):
            self.recorded_cells = record_from
            myLogger.debug(
                'Recording selected cells of population <' + self.label + '>')
        elif isinstance(record_from, numpy.ndarray):
            self.recorded_cells = record_from.flatten().tolist()
            myLogger.debug(
                'Recording selected cells of population <' + self.label + '>')
        else:
            myLogger.error('Can not record_from ' + str(type(record_from)))
            raise Exception('record_from must be of type int, list or None!')

        pyNN.hardware.spikey.record(self.recorded_cells, '')

    def record_v(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """

        raise NotImplementedError("""Recording membrane potentials from populations is not implemented for the hardare.
        The membrane potential of only one neuron can be recorded! Use the procedural record_v function instead!""")

    # -------------- HARDWARE SPECIFIC IMPLEMENTATION-----------------------
    def printSpikes(self, filename=None, gather=True, compatible_output=True):
        """
        Returns spiketimes and writes them to file.
        All recorded spikes of population's neurons (recorded via Population.record() ) are
        sorted by spiketime and - if filename is provided - written to a file.
        Further, the generated list is returned.
        File - format: spiketime <tab> int(neuronID) <newline>
        List - format: [..., [spiketime, int(neuronID) ], ...]
        """

        # unfortunately, PyNN defines a different column order for getSpikes
        # and printSpikes
        spikeArray = self.getSpikes(gather=gather)
        spikeArrayPrint = spikeArray[:, ::-1]

        if filename != None:
            numpy.savetxt(filename, spikeArrayPrint)

        return spikeArrayPrint

    def print_v(self, filename=None, gather=True, compatible_output=True):
        raise NotImplementedError("""Recording membrane potentials from populations is not implemented for the hardare.
        The membrane potential of only one neuron can be recorded! Use the procedural record_v function instead!""")

    def getSpikes(self, gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for recorded cells
        # gather is not relevant, but is needed for API consistency
        """

        spikeIDs = None
        spikeTimes = None
        lut = None
        maxInt = numpy.iinfo(numpy.uint32).max
        if self.celltype == pyNN.hardware.spikey.IF_facets_hardware1:
            if pyNN.hardware.spikey.spikeOutput == None:
                raise Exception(
                    'use pyNN.Population.record() to record spikes')
            spikeIDs, spikeTimes = pyNN.hardware.spikey.spikeOutput

            # create look-up table to replace all filtered IDs by max int
            # (quite fast)
            lut = numpy.ones(default.numNeurons, int) * maxInt
            for i in range(len(lut)):
                if i in self.recorded_cells:
                    lut[i] = i
        elif self.celltype in [pyNN.hardware.spikey.SpikeSourcePoisson, pyNN.hardware.spikey.SpikeSourceArray]:
            spikeIDs, spikeTimes = pyNN.hardware.spikey._appliedInputs

            # create look-up table to replace all filtered IDs by max int
            # (quite fast)
            lut = numpy.ones(default.numExternalInputs, int) * maxInt
            for i in range(len(lut)):
                if -i in self.recorded_cells:
                    lut[-i] = -i

        if len(spikeIDs) == 0:
            return numpy.zeros((0, 2))

        # only return spikes that were marked as recorded
        spikeIDs = numpy.take(lut, spikeIDs)
        mask = spikeIDs != maxInt  # no ID=nan anymore
        spikeIDs = spikeIDs[mask]
        spikeTimes = spikeTimes[mask]

        if spikeIDs.shape[0] < 1:  # no spikes left after filtering for IDs
            return numpy.zeros((0, 2))

        # sort spike times
        # the fastest way I found for numpy
        sortidx = numpy.argsort(spikeTimes)
        spikeIDs = spikeIDs[sortidx]
        spikeTimes = spikeTimes[sortidx]

        spikeArray = numpy.transpose(numpy.array([spikeIDs, spikeTimes]))

        return spikeArray
