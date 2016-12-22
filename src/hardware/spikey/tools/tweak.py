import types

# # # # # # # # # # # # # # # # # # # # # # # #
# # #   M O D U L E   V A R I A B L E S   # # #
# # # # # # # # # # # # # # # # # # # # # # # #

# General
INITIALIZED = False
VERBOSE = None
pyNN = None
hwa = None
cfg = None

# Printing
GREEN = '\033[95m'
ENDC = '\033[0m'
INFO = GREEN + "TWEAK " + ENDC

# Targets
tCHIP = "chip 0"
tBLOCK = "block 0..1"
tNEURON = "neuron 0..383"
tDRIVER = "driver 0..511"
VALID_TARGETS = {}      # target index ranges, determined in init()

# Scopes
sCHIP = "chip-wide"
sBLOCK = "chip half"
sSHAREDN = "shared by neurons (left/right even/odd)"
sSHAREDD = "shared by drivers (left/right lower/upper) "
sINDIVIDUAL = "individual"


# # # # # # # # # # # # # # # # # # # # # # # # #
# # #   G E N E R A L   F U N C T I O N S   # # #
# # # # # # # # # # # # # # # # # # # # # # # # #

# Initialize module
def init(pynn, verbose=False):
    global VERBOSE
    VERBOSE = verbose
    global pyNN
    global hwa
    global cfg
    global INITIALIZED
    assert isinstance(pynn, types.ModuleType), "pynn is no python module!"
    assert pynn.hardware.hwa is not None, "No HWAccess object. Call pynn.setup() first."
    assert pynn.hardware.hwa.haveHardware, "No hardware. Call pynn.setup() first."
    assert pynn.hardware.hwa.cfg is not None, "No SpikeyConfig low level access. Call pynn.setup() first."
    pyNN = pynn
    hwa = pynn.hardware.hwa
    cfg = pynn.hardware.hwa.cfg
    VALID_TARGETS[tCHIP] = xrange( 1 )
    VALID_TARGETS[tBLOCK] = xrange( hwa.numBlocks() )
    VALID_TARGETS[tNEURON] = xrange( hwa.numBlocks() * hwa.numNeuronsPerBlock() )
    VALID_TARGETS[tDRIVER] = xrange( hwa.numBlocks() * hwa.numInputsPerNeuron() )
    INITIALIZED = True
    print INFO + "pyhal_tweak initialized."


# A decorator to set function names in function generators
def setfuncname(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


# Determine correct function calls from the scope and target index
def kwargs_from_scope(scope, target):
    kwargs = dict()
    if scope is sCHIP:
        assert target == 0, "Target of chip-wide parameter must be 0!"
    elif scope is sBLOCK:
        raise NotImplemented
    elif scope is sSHAREDN:
        assert isinstance(target, int),\
            INFO + 'Invalid target type for scope "%s": %s (expected int)' % (scope, str(target))
        kwargs['block'] = target / hwa.numNeuronsPerBlock()
        kwargs['relidx'] = target % 2
    elif scope is sSHAREDD:
        assert isinstance(target, int),\
            INFO + 'Invalid target type for scope "%s": %s (expected int)' % (scope, str(target))
        kwargs['block'] = target / hwa.numInputsPerNeuron()
        kwargs['relidx'] = int( ( hwa.numInputsPerNeuron() / 2 ) <= ( target % hwa.numInputsPerNeuron() ) )
    elif scope is sINDIVIDUAL:
        assert isinstance(target, int),\
            INFO + 'Invalid target type for scope "%s": %s (expected int)' % (scope, str(target))
        kwargs['target'] = target
    return kwargs



# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #   G E T   /   S E T   F U N C T I O N S   # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# Generators for vout access functions
def gen_set_vout(baseidx):
    @setfuncname("set_vout[baseidx=%d]" % baseidx)
    def set_vout(block, relidx, value):
        cfg.vout[block * hwa.vouts.shape[1] + baseidx + relidx] = value
    return set_vout

def gen_get_vout(baseidx):
    @setfuncname("get_vout[baseidx=%d]" % baseidx)
    def get_vout(block, relidx):
        return cfg.vout[block * hwa.vouts.shape[1] + baseidx + relidx]
    return get_vout


# Generators for voutbias access functions
def gen_set_voutbias(baseidx):
    @setfuncname("set_voutbias[baseidx=%d]" % baseidx)
    def set_voutbias(block, relidx, value):
        cfg.voutbias[block * hwa.vouts.shape[1] + baseidx + relidx] = value
    return set_voutbias

def gen_get_voutbias(baseidx):
    @setfuncname("get_voutbias[baseidx=%d]" % baseidx)
    def get_voutbias(block, relidx):
        return cfg.voutbias[block * hwa.vouts.shape[1] + baseidx + relidx]
    return get_voutbias


# Generators for biasb access functions
def gen_set_biasb(baseidx):
    @setfuncname("set_biasb[baseidx=%d]" % baseidx)
    def set_biasb(block, relidx, value):
        cfg.biasb[baseidx + hwa.numBlocks() * block + relidx] = value
    return set_biasb

def gen_get_biasb(baseidx):
    @setfuncname("get_biasb[baseidx=%d]" % baseidx)
    def get_biasb(block, relidx):
        return cfg.biasb[baseidx + hwa.numBlocks() * block + relidx]
    return get_biasb


# Generators for synapse analog currents access functions
def gen_set_driverCurrent(drviname):
    @setfuncname("setSynapseDriver[%s]" % drviname)
    def set_driverCurrent(target, value):
        #drvifunc = dict(
            #drviout=cfg.setSynapseDriverDrviout,
            #adjdel=cfg.setSynapseDriverAdjdel,
            #drvifall=cfg.setSynapseDriverDrvifall,
            #drvirise=cfg.setSynapseDriverDrvirise,
            #)
        #drvifunc[drviname](target, value)
        kwargs = {drviname: value}
        print kwargs
        cfg.setSynapseDriver(target, **kwargs)
    return set_driverCurrent

def gen_get_driverCurrent(drviname):
    @setfuncname("getSynapseDriver[%s]" % drviname)
    def get_driverCurrent(target):
        idx = dict(
            drviout=0,
            adjdel=3,
            drvifall=1,
            drvirise=2,
            )[drviname]
        return cfg.getSynapseDriver(target)[idx]
        #return drvifunc[drviname](target)
    return get_driverCurrent


# Generators for neuron analog currents access functions
def gen_set_neuronCurrent(iname):
    @setfuncname("set_Neuron%s" % iname.capitalize())
    def set_neuronCurrent(target, value):
        ifunc = dict(
            ileak=cfg.setILeak,
            icb=cfg.setIcb,
            )
        ifunc[iname](target, value)
    return set_neuronCurrent

def gen_get_neuronCurrent(iname):
    @setfuncname("get_Neuron%s" % iname.capitalize())
    def get_neuronCurrent(target):
        ifunc = dict(
            ileak=cfg.getILeak,
            icb=cfg.getIcb,
            )
        return ifunc[iname](target)
    return get_neuronCurrent



# THE MAIN GET FUNCTION
def get(pname, target):
    assert pname in PARM, INFO + pname + " not supported by pyhal_tweak!"
    targettype = PARM[pname]["targettype"]
    assert target in VALID_TARGETS[targettype],\
        INFO + "Invalid target for %s: %s (target type: %s)" % (pname, str(target), targettype)
    scope = PARM[pname]["scope"]
    getf = PARM[pname]["getf"]
    kwargs = kwargs_from_scope(scope, target)
    lowlevelstr = getf.__name__ + "("
    for k,v in kwargs.iteritems():
        lowlevelstr += "%s=%s, " % (k, str(v))
    lowlevelstr += "\b\b)"
    value = getf(**kwargs)
    if VERBOSE:
        print INFO + "get('%s', target=%s)  -->  %s  -->  RETURN: %s" % (pname, target, lowlevelstr, value)
    return value


# THE MAIN SET FUNCTION
def set(pname, target, value):
    assert pname in PARM, INFO + pname + " not supported by pyhal_tweak!"
    targettype = PARM[pname]["targettype"]
    assert target in VALID_TARGETS[targettype],\
        INFO + "Invalid target for %s: %s (target type: %s)" % (pname, str(target), targettype)
    valuetype = PARM[pname]["valuetype"]
    assert isinstance(value, valuetype),\
        INFO + "Invalid value for %s: %s (expected %s)" % (pname, str(value), valuetype.__name__)
    scope = PARM[pname]["scope"]
    setf = PARM[pname]["setf"]
    kwargs = kwargs_from_scope(scope, target)
    lowlevelstr = setf.__name__ + "("
    for k,v in kwargs.iteritems():
        lowlevelstr += "%s=%s, " % (k, str(v))
    kwargs['value'] = value
    lowlevelstr += "%s=%s)" % ('value', str(value))
    if VERBOSE:
        print INFO + "set('%s', target=%s, value=%s)  -->  %s" % (pname, target, value, lowlevelstr)
    setf(**kwargs)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #   L I S T   O F   S U P P O R T E D   P A R A M E T E R S   # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# LIST OF SUPPORTED PARAMETERS pname=dict(target, scope, ptype, setf, getf)
# Both docstring and get() / set() rely on the PARM dictionary.
PARM = dict(
    # Neuron vouts
    e_rev_I=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_vout(baseidx=0),  getf=gen_get_vout(baseidx=0)),
    v_rest=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_vout(baseidx=2),  getf=gen_get_vout(baseidx=2)),
    v_reset=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_vout(baseidx=4),  getf=gen_get_vout(baseidx=4)),
    e_rev_E=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_vout(baseidx=6),  getf=gen_get_vout(baseidx=6)),
    v_thresh=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_vout(baseidx=16),  getf=gen_get_vout(baseidx=16)),
    # Neuron voutbiases
    e_rev_I_bias=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_voutbias(baseidx=0),  getf=gen_get_voutbias(baseidx=0)),
    v_rest_bias=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_voutbias(baseidx=2),  getf=gen_get_voutbias(baseidx=2)),
    v_reset_bias=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_voutbias(baseidx=4),  getf=gen_get_voutbias(baseidx=4)),
    e_rev_E_bias=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_voutbias(baseidx=6),  getf=gen_get_voutbias(baseidx=6)),
    v_thresh_bias=dict(targettype=tNEURON, scope=sSHAREDN, valuetype=float, setf=gen_set_voutbias(baseidx=16),  getf=gen_get_voutbias(baseidx=16)),
    # Neuron analog currents
    ileak=dict(targettype=tNEURON, scope=sINDIVIDUAL, valuetype=float, setf=gen_set_neuronCurrent('ileak'),  getf=gen_get_neuronCurrent('ileak')),
    icb=dict(targettype=tNEURON, scope=sINDIVIDUAL, valuetype=float, setf=gen_set_neuronCurrent('icb'),  getf=gen_get_neuronCurrent('icb')),
    # Driver vouts
    Vfac=dict(targettype=tDRIVER, scope=sSHAREDD, valuetype=float, setf=gen_set_vout(baseidx=12),  getf=gen_get_vout(baseidx=12)),
    Vstdf=dict(targettype=tDRIVER, scope=sSHAREDD, valuetype=float, setf=gen_set_vout(baseidx=14),  getf=gen_get_vout(baseidx=14)),
    # Driver voutbiases
    Vfac_bias=dict(targettype=tDRIVER, scope=sSHAREDD, valuetype=float, setf=gen_set_voutbias(baseidx=12),  getf=gen_get_voutbias(baseidx=12)),
    Vstdf_bias=dict(targettype=tDRIVER, scope=sSHAREDD, valuetype=float, setf=gen_set_voutbias(baseidx=14),  getf=gen_get_voutbias(baseidx=14)),
    # Driver biasb
    Vdtc=dict(targettype=tDRIVER, scope=sSHAREDD, valuetype=float, setf=gen_set_biasb(baseidx=0),  getf=gen_get_biasb(baseidx=0)),
    Vcb=dict(targettype=tDRIVER, scope=sSHAREDD, valuetype=float, setf=gen_set_biasb(baseidx=4),  getf=gen_get_biasb(baseidx=4)),
    Vplb=dict(targettype=tDRIVER, scope=sSHAREDD, valuetype=float, setf=gen_set_biasb(baseidx=8),  getf=gen_get_biasb(baseidx=8)),
    # Driver analog currents
    drviout=dict(targettype=tDRIVER, scope=sINDIVIDUAL, valuetype=float, setf=gen_set_driverCurrent('drviout'),  getf=gen_get_driverCurrent('drviout')),
    adjdel=dict(targettype=tDRIVER, scope=sINDIVIDUAL, valuetype=float, setf=gen_set_driverCurrent('adjdel'),  getf=gen_get_driverCurrent('adjdel')),
    drvifall=dict(targettype=tDRIVER, scope=sINDIVIDUAL, valuetype=float, setf=gen_set_driverCurrent('drvifall'),  getf=gen_get_driverCurrent('drvifall')),
    drvirise=dict(targettype=tDRIVER, scope=sINDIVIDUAL, valuetype=float, setf=gen_set_driverCurrent('drvirise'),  getf=gen_get_driverCurrent('drvirise')),
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #   A U T O M A T E D   M O D U L E   D O C S T R I N G   # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__doc__ = """
  The pyhal_tweak module is a developer extension to the pyNN.hardware.spikey module
  to enable low level hardware manipulation within pyNN-controlled experiment execution.

  The standard call of pyhal_tweak is via

    pyhal_tweak.get(pname, target)

  and

    pyhal_tweak.set(pname, target, value)

  with
    pname : parameter name (type str, see list below)
    target : Hardware index of the neuron or synapse driver; 0 for chip-wide parameters (type int)
    value : value to be set (type parameter-dependent)

  BASIC USAGE WITHIN PYNN:
    import pyNN.hardware.spikey as pyNN
    import pyhal_tweak as tweak
    pyNN.setup()
    tweak.init(pyNN, verbose=True/False)
    [..all pyNN network and experiment definitions..]
    pyNN.run(runtime, interruptRunAfterMapping=True)
    tweak.set(pname, target, value)
    pynn.run(runtime)
  
    
  SUPPORTED PARAMETERS:"""

keys = PARM.keys()
keys.sort()
for pname in keys:
    d = PARM[pname]
    s = "\n    %-*s  target type: %s, scope: %s, value type: %s" % (16, pname, d['targettype'], d['scope'], d['valuetype'].__name__)
    __doc__ += s
