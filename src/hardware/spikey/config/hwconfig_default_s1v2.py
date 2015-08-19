import numpy

#######################################################
# # # # # # # # # constants describing the Spikey neuromorphic system # # # # # # # # #
#######################################################


numPresyns = 256
numBlocks = 2
neuronsPerBlock = 192
numExternalInputs = numPresyns * numBlocks
numNeurons = neuronsPerBlock * numBlocks
numVout = 25


recordableNeurons = range(192, 384, 1)

#############################
# # # # # # # # # # # # # fixed hardware parameters # # # # # # # # # # # # #
#############################

# minimum allowed current, lower currents can not be programmed precisely
currentMin = 0.02
currentMax = 2.5                # maximum allowed current
voltageMin = 0                  # minimum allowed voltage
voltageMax = 2.0                # maximum allowed voltage


# spikey internal timing (given in external chip clock periods): after
# this period the output of synapse ram bitline reading is valid
tsense = 150.0

# pre-charge time for secondary read when processing correlations (given
# in external chip clock periods)
tpcsec = 30.0

# minimum time used for the correlation processing of a single row (given
# in external chip clock periods)
tpcorperiod = 360.0

# DAC reference current determining possible hardware currents: minimal
# programmable current possible = irefdac / 10. * (1./1024.)
irefdac = 25.0
# irefdac given in uA                                           maximal
# programmable current possible = irefdac / 10. * (1023./1024.)

# cascode DAC voltage (given in V): Never touch this value!
vcasdac = 1.6

# A reference voltage for STDP measurement (given in V): The larger vm,
# the smaller is the charge
vm = 0.3
# which is stored per measured pre-/post-synaptic correlation -> STDP curve amplitude gets smaller
# vm = 0.3 corresponds to tau=2.5ms for a speed-up factor of 10^4

# start value of rising voltage ramp (given in V), influence on integral
# over PSP is weak
vstart = 0.25

# baseline of voltage ramp (given in V), stronly influences efficacy of
# ramp, i.e. huge impact on PSP!
vrest = 0.0
# set very low if you want weak PSPs

# "adjust delay": this value controls the delay between a spike running into a synapse driver
adjdel = 2.5
# and the triggered voltage up/down ramp

# threshold comparator bias current: the larger this value, the faster is
# the comparator
icb_base = 0.2
# TP: for proper STDP curves set icb_base to a high value (2.5 muA),
# because pulse of refractory period is also used for correlation
# measurements

outamp = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,  # outamp 0..7: bias current for 50 ohm membrane voltage monitors, \
          0.0]                                     # outamp 8: current memory for ibtest_pin, should be 0.0!

# bias current for all vout voltages - the more load the corresponding
# vout has, the larger this value should be
voutbias = 0.02
# SEMI-OBSOLETE: used for first init in HWAccess.__init__(), but is
# overwritten later by voutbiases !

probepad = 0.8                 # obsolete value, remaining from spikey1
probebias = 0.2                 # obsolete value, remaining from spikey1
# time needed to process STDP for one synapse row in ms biological time
# (speed-up factor 10^4) / 12.0 ms was too short!
autoSTDPFrequency = 15.0


# ~~~~~~~~~~~~~~~~~~~~~~ B I A S B ~~~~~~~~~~~~~~~~~~~~~~

# biasb  0.. 3: Vdtc0=0,Vdtc1=1,Vdtc2=2,Vdtc3=3,
# biasb  4.. 7: Vcb0=4,Vcb1=5,Vcb2=6,Vcb3=7 ,
# biasb  8..11: Vplb0=8,Vplb1=9,Vplb2=10,Vplb3=11,
# biasb 12..13: Ibnoutampba=12,Ibnoutampbb=13,
# biasb     14: Ibcorrreadb=14,

vdtc = 0.7       # maybe 0.2   # short term plasticity time constant for spike history, higher current->shorter averaging windowtime

vcb = 1.25      # maybe 1.0   # spike driver comparator bias

vplb = 0.15      # maybe 0.5   # spike driver pulse length bias, higher current->shorter internal pulse,important for short term plasticity

Ibnoutampba = 0.5              # both add together to the neuronoutampbias
# for values < 0.1 the output amplifier is too slow, membrane peak to
# peak, e.g. in test_spikes.py becomes smaller
Ibnoutampbb = 0.5

Ibcorrreadb = 2.0               # correlation read out bias
# TP: should be high for fast readout

##########################################################################
# # # # # # # # # # # # # dyncamic hardware parameters (i.e. adjustable by pyNN or involved in calibration routines # # # # # # # # # # # # #
##########################################################################


iLeak_base = 0.1 / \
    10.                   # neuron membrane leakage conductance (for each neuron an individual calibration factor will be multiplied to this value)
# divisor 10 is due to the fact that ileak is multiplied by the desired value for g_leak
# (a typical value is g_leak = 10nS, which - assuming C_m to be 0.2 nF - makes tau_mem = 20ms)

# synapse driver voltage ramp parameters
# base value for upper limit (for each driver an individual calibration
# factor will be multiplied to this value)
drviout_base = {'exc': 1.5, 'inh': 1.5}
drvirise = 1.0                           # rising ramp current
# falling ramp current - this value should be chosen rather low because it
# parasitically
drvifall_base = {'exc': 1.0, 'inh': 1.0}
# couples into vrest, which then increases the synaptic leakage current
# across the whole chip

# ~~~~~~~~~~~~~~~~~~~~~~ V O U T S ~~~~~~~~~~~~~~~~~~~~~~
# Explanation of the "vout" voltages:

#    Ei0=0,Ei1=1,                inhibitory reversal potential
#    El0=2,El1=3,                leakage reversal potential
#    Er0=4,Er1=5,                reset potential
#    Ex0=6,Ex1=7,                excitatory reversal potential
#    Vclra=8,                    storage clear bias synapse array acausal (higher bias->smaller amount stored on cap)
#    Vclrc=9,                    dito, causal
#    Vcthigh=10,                 correlation threshold high
#    Vctlow=11,                  correlation threshold low
#    Vfac0=12,Vfac1=13,          short term facilitation reference voltage
#    Vstdf0=14,Vstdf1=15,        short term capacitor high potential
#    Vt0=16,Vt1=17,              neuron threshold voltage
#    Vcasneuron=18,              neuron input cascode gate voltage
#    Vresetdll=19,               dll reset voltage
#    aro_dllvctrl=20,            dll control voltage readout (only spikey v2, only bias important)
#    aro_pre1b=21,               spike input buf 1 presyn (only spikey v2, only bias important)
# aro_selout1hb=22,           spike input buf 1 selout (only spikey v2,
# only bias important)

# Changing the default "vout" values below might have no effect, because the configuration routines affect various vout entries,
# as for example the leakage, resting, threshold and reversal potentials...
# Some of the voltages have specific constraints, such as Vcthigh and
# Vctlow, which must not be set to values larger than 1V

vouts = numpy.array([[1.0, 1.0,   # inhibitory reversal potential\
                      1.0, 1.0,   # leakage reversal potential\
                      1.0, 1.0,   # reset potential\
                      1.3, 1.3,   # excitatory reversal potential\
                      1.2,        # storage clear bias synapse array acausal (higher bias->smaller amount stored on cap)\
                      1.2,        # dito, causal\
                      1.0,        # <1V ; correlation threshold high\
                      0.85,        # <1V ; correlation threshold low\
                      0.02, 0.02,  # short term facilitation reference voltage\
                      1.12, 1.12,  # short term capacitor high potential\
                      1.1, 1.1,   # neuron threshold voltage\
                      1.6,        # neuron input cascode gate voltage\
                      0.75,       # dll reset voltage\
                      1.0,        # dll control voltage readout (only spikey v2, only bias important)\
                      1.0,        # spike input buf 1 presyn (only spikey v2, only bias important)\
                      1.0,        # spike input buf 1 selout (only spikey v2, only bias important)\
                      1.0,        # new for spikey4: test vout, to be combined with neighbor\
                      1.0],       # new for spikey4: test vout, to be combined with neighbor\
                                  \
                     [1.0, 1.0,   \
                      1.0, 1.0,   \
                      1.0, 1.0,   \
                      1.3, 1.3,   \
                      1.2,       \
                      1.2,       \
                      1.0,        # <1V \
                      0.85,        # <1V \
                      0.02, 0.02, \
                      1.12, 1.12, \
                      1.1, 1.1,   \
                      1.6,        \
                      0.75,       \
                      1.0,        \
                      1.0,        \
                      1.0,
                      1.0,
                      1.0]])


voutbiases = numpy.array([[
    2.5, 2.5,  # inhibitory reversal potential\
    2.5, 2.5,  # leakage reversal potential\
    2.5, 2.5,  # reset potential\
    2.5, 2.5,   # excitatory reversal potential\
    2.5,       # storage clear bias synapse array acausal (higher bias->smaller amount stored on cap)\
    2.5,       # dito, causal\
    2.5,       # <1V ; correlation threshold high\
    2.5,       # <1V ; correlation threshold low\
    2.5, 2.5,  # short term facilitation reference voltage\
    2.5, 2.5,  # short term capacitor high potential\
    2.5, 2.5,  # neuron threshold voltage\
    2.5,       # neuron input cascode gate voltage\
    2.5,       # dll reset voltage\
    2.5,       # dll control voltage readout (only spikey v2, only bias important)\
    2.5,       # spike input buf 1 presyn (only spikey v2, only bias important)\
    2.5,       # spike input buf 1 selout (only spikey v2, only bias important)\
    2.5,       # new for spikey4: test vout, to be combined with neighbor\
    2.5],      # new for spikey4: test vout, to be combined with neighbor\

    [2.5, 2.5,  # inhibitory reversal potential\
     2.5, 2.5,  # leakage reversal potential\
     2.5, 2.5,  # reset potential\
     2.5, 2.5,   # excitatory reversal potential\
     2.5,       # storage clear bias synapse array acausal (higher bias->smaller amount stored on cap)\
     2.5,       # dito, causal\
     2.5,       # <1V ; correlation threshold high\
     2.5,       # <1V ; correlation threshold low\
     2.5, 2.5,  # short term facilitation reference voltage\
     2.5, 2.5,  # short term capacitor high potential\
     2.5, 2.5,  # neuron threshold voltage\
     2.5,       # neuron input cascode gate voltage\
     2.5,       # dll reset voltage\
     2.5,       # dll control voltage readout (only spikey v2, only bias important)\
     2.5,       # spike input buf 1 presyn (only spikey v2, only bias important)\
     2.5,       # spike input buf 1 selout (only spikey v2, only bias important)\
     2.5,       # new for spikey4: test vout, to be combined with neighbor\
     2.5]])     # new for spikey4: test vout, to be combined with neighbor
