# this files contains Python dictionaries for neural entities' type definitions

connectionType = {"i": 0, "e": 1, "int": 0,
                  "ext": 1, "internal": 0, "external": 1, }

# compare statusbyte 'type' in pyspikeyconfig.cpp::setSynapseDriver
neuronType = {"baseValue": 0,   # other bits will be added to this one (= disabled)
              "disabled": 0,   # for compatibility only
              "excitatory": 4,
              "inhibitory": 8
              }

STPTypes = {"enable":  16,
            "fac":   0,
            "dep":  32,
            "cap2":  64,
            "cap4": 128,
            }
