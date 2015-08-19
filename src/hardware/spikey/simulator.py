# encoding: utf-8
"""
Implementation of some "low-level" functionality used by the common implementation of the API.
Functions and classes useable by the common implementation:
Attributes:
    state -- a singleton instance of the _State class.
$Id$
"""
import pyNN.hardware.spikey
from pyNN import common, recording, random


# --- For implementation of get_time_step() and similar functions --------
class _State(object):
    """Represent the simulator state."""

    def __init__(self):
        self.initialized = False
        self.running = False

    @property
    def t(self):
        return 0.0  # DB: this is just a workaround

    def setHardwareTimestep(timestep):
        pyNN.hardware.spikey._dt = timestep

    dt = property(fget=lambda self: pyNN.hardware.spikey._dt,
                  fset=lambda self, timestep: setHardwareTimestep(timestep))

    @property
    def min_delay(self):
        return pyNN.hardware.spikey._dt

    @property
    def max_delay(self):
        return pyNN.hardware.spikey._dt

    @property
    def num_processes(self):
        return 1

    @property
    def mpi_rank(self):
        return 1

    @property
    def num_threads(self):
        return 1

# --- Initialization, and module attributes ------------------------------
state = _State()  # a Singleton, so only a single instance ever exists
del _State
