import pylogging as pylog
myLogger = pylog.get("PyN.syn")

from pyNN import common
from pyNN import synapses
import numpy
from pyNN.random import RandomDistribution, NativeRNG
from math import *
import types
import hwconfig_default_s1v2 as default

import pyNN.hardware.spikey


class SynapseDynamics(common.SynapseDynamics):
    # common.SynapseDynamics sets:
    #  - self.fast
    #  - self.slow

    def __init__(self, fast=None, slow=None):
        common.SynapseDynamics.__init__(self, fast, slow)


class STDPMechanism(common.STDPMechanism):
    parameters = {}
    possible_models = set(['stdp_synapse'])

    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=0.0):
        if not isinstance(timing_dependence, SpikePairRule):
            raise Exception(
                "Setting timing_dependence != SpikePairRule isn't supported by Spikey.")
        if not voltage_dependence is None:
            raise Exception(
                "Voltage dependent STDP isn't supported by Spikey.")
        if not dendritic_delay_fraction == 0.0:
            raise Exception(
                "Setting a dendritic_delay_fraction != 0.0 isn't supported by Spikey.")
        self.timing_dependence = timing_dependence
        self.weight_dependence = weight_dependence
        self.voltage_dependence = None
        self.dendritic_delay_fraction = 0.0

        pyNN.hardware.spikey.hardware.initSTDP()


class TsodyksMarkramMechanism(synapses.TsodyksMarkramMechanism):
    translations = common.build_translations(
        ('U', 'U'),
        ('tau_rec', 'tau_rec'),
        ('tau_facil', 'tau_facil'),
        ('u0', 'u0'),
        ('x0', 'x0'),
        ('y0', 'y0')
    )
    possible_models = set([])

    def __init__(self, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        if tau_rec != 0 and tau_facil != 0:
            raise Exception(
                "Tsodyks-Markram-STP on hardware does allow only facilitation OR depression.")

        # need the dict to get a copy of locals. When running
        parameters = dict(locals())
        # through coverage.py, for some reason, the pop() doesn't have any
        # effect
        parameters.pop('self')
        self.parameters = self.translate(parameters)


class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    """
    The amplitude of the weight change is fixed for depression (`A_minus`)
    and for potentiation (`A_plus`).
    If the new weight would be less than `w_min` it is set to `w_min`. If it would
    be greater than `w_max` it is set to `w_max`.
    """
    parameters = {
        'w_min':   0.0,
        'w_max':   1.0,
        'A_plus':  1.0 / 16,
        'A_minus': 1.0 / 16
    }
    possible_models = set(['stdp_synapse'])

    def __init__(self, w_min=0.0, w_max=1.0, A_plus=1.0 / 16, A_minus=1.0 / 16):
        myLogger.info(
            "Additive STDP is implemented as initial weight +1 (for potentiation) -1 (for depression)")
        causalLUT = range(1, 16) + [15]
        acausalLUT = [0] + range(0, 15)
        pyNN.hardware.spikey.hardware.setLUT(causalLUT, acausalLUT, first=True)


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Dw propto w-w_min
    For potentiation, Dw propto w_max-w
    """
    parameters = {
        'w_min':   0.0,
        'w_max':   1.0,
        'A_plus':  1.0 / 16,
        'A_minus': 1.0 / 16
    }
    possible_models = set(['stdp_synapse'])

    def __init__(self, w_min=0.0, w_max=1.0, A_plus=1.0 / 16, A_minus=1.0 / 16):
        raise Exception(
            "Multiplicative weight dependence not available yet. Work in progress.")


class SpikePairRule(synapses.SpikePairRule):
    translations = common.build_translations(
        ('tau_plus',  'tau_plus'),
        ('tau_minus', 'tau_minus')
    )
    possible_models = set(['stdp_synapse'])

    def __init__(self, tau_plus=2.5, tau_minus=2.5):
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = self.translate(parameters)

        # TP: time constants of STDP can be adjusted by the hardware parameter
        # vclra/c, see Bachelor thesis by Ole Schmidt
        if self.parameters['tau_plus'] != 2.5 or self.parameters['tau_minus'] != 2.5:
            myLogger.warn(
                "The configuration of STDP time constants (spike pair rule) to other values than 2.5ms is not possible via PyNN")
