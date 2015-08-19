from pyNN import common
from pyNN import cells

# ==============================================================================
# Implementation of Standard cells
# ==============================================================================


class IF_facets_hardware1(cells.IF_facets_hardware1):
    """
    Leaky integrate and fire model with fixed threshold and conductance-based synpases,
    describes the neuron model of the Spikey neuromorphic system.
    """

    translations = common.build_translations(
        ('v_reset',                  'v_reset'),
        ('v_rest',                   'v_rest'),
        ('v_thresh',                 'v_thresh'),
        ('e_rev_I',                  'e_rev_I'),
        ('g_leak',                   'g_leak'),
        ('tau_refrac',               'tau_refrac')
    )

    estimator_cm = 0.2
    estimator_e_rev_E = 0.0
    estimator_tau_syn_E = 5.0
    estimator_tau_syn_I = 5.0

    def __init__(self, parameters):

        # extend with Spikey specific neuron parameters
        cells.IF_facets_hardware1.__init__(self, parameters)
        self.parameters['lowlevel_parameters'] = {}
        self.parameters['estimator_cm'] = self.estimator_cm
        self.parameters['estimator_e_rev_E'] = self.estimator_e_rev_E
        self.parameters['estimator_tau_syn_E'] = self.estimator_tau_syn_E
        self.parameters['estimator_tau_syn_I'] = self.estimator_tau_syn_I


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    '''Spike source, generating spikes according to a Poisson process.'''

    translations = common.build_translations(
        ('rate',      'rate'),
        ('start',     'start'),
        ('duration',  'duration')
    )

    def __init__(self, parameters):

        cells.SpikeSourcePoisson.__init__(self, parameters)
        self.index = None


class SpikeSourceArray(cells.SpikeSourceArray):
    '''Spike source generating spikes at the times given in the spike_times array.'''

    translations = common.build_translations(
        ('spike_times',   'spike_times')
    )

    def __init__(self, parameters):

        cells.SpikeSourceArray.__init__(self, parameters)
        self.index = None
