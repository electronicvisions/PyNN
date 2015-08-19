import unittest


class test_001_mappingoffset(unittest.TestCase):
    """
    mappingoffsets larger 192 cause wrong configuration for spikey version 4
    """

    def test(self):
        vrest1 = {'v_rest': -80.0}
        vrest2 = {'v_rest': -70.0}

        import copy
        import pyNN.hardware.spikey as pynn

        def withMappingOffset(mappingOffset, vrest):
            pynn.setup(mappingOffset=mappingOffset)
            pynn.Population(1, pynn.IF_facets_hardware1, vrest)
            pynn.run(1000.0)
            vout = copy.deepcopy(pynn.hardware.hwa.vouts[1, 2:4])
            pynn.end()
            return vout

        def withDummyNeurons(mappingOffset, vrest):
            pynn.setup()
            if mappingOffset > 0:
                pynn.Population(mappingOffset, pynn.IF_facets_hardware1)
            pynn.Population(1, pynn.IF_facets_hardware1, vrest)
            pynn.run(1000.0)
            vout = copy.deepcopy(pynn.hardware.hwa.vouts[1, 2:4])
            pynn.end()
            return vout

        vout1a = withMappingOffset(0, vrest1)
        vout2a = withMappingOffset(0, vrest2)
        map0a = withMappingOffset(0, vrest1)
        map1a = withMappingOffset(1, vrest2)
        map2a = withMappingOffset(2, vrest2)
        map3a = withMappingOffset(2, vrest1)

        self.assertTrue(vout1a[0] == map0a[0])
        self.assertTrue(vout2a[0] == map1a[1])
        self.assertTrue(vout2a[0] == map2a[0])
        self.assertTrue(vout1a[0] == map3a[0])

        vout1b = withMappingOffset(192, vrest1)
        vout2b = withMappingOffset(192, vrest2)
        map0b = withMappingOffset(192, vrest1)
        map1b = withMappingOffset(193, vrest2)
        map2b = withMappingOffset(194, vrest2)
        map3b = withMappingOffset(194, vrest1)

        self.assertTrue(vout1b[0] == map0b[0])
        self.assertTrue(vout2b[0] == map1b[1])
        self.assertTrue(vout2b[0] == map2b[0])
        self.assertTrue(vout1b[0] == map3b[0])

        self.assertTrue(vout1a[0] == vout1b[0])
        self.assertTrue(vout1a[1] == vout1b[1])
        self.assertTrue(vout2a[0] == vout2b[0])
        self.assertTrue(vout2a[1] == vout2b[1])
        self.assertTrue(map0a[0] == map0b[0])
        self.assertTrue(map0a[1] == map0b[1])
        self.assertTrue(map1a[0] == map1b[0])
        self.assertTrue(map1a[1] == map1b[1])
        self.assertTrue(map2a[0] == map2b[0])
        self.assertTrue(map2a[1] == map2b[1])
        self.assertTrue(map3a[0] == map3b[0])
        self.assertTrue(map3a[1] == map3b[1])

        #vout1 = withDummyNeurons(0, vrest1)
        #vout2 = withDummyNeurons(0, vrest2)
        #map0 = withDummyNeurons(0, vrest1)
        #map1 = withDummyNeurons(1, vrest2)
        #map2 = withDummyNeurons(2, vrest2)
        #map3 = withDummyNeurons(2, vrest1)

        #self.assertTrue(vout1[0] == map0[0])
        #self.assertTrue(vout2[0] == map1[1])
        #self.assertTrue(vout2[0] == map2[0])
        #self.assertTrue(vout1[0] == map3[0])

if __name__ == "__main__":
    unittest.main()
