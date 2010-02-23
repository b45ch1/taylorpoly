from numpy.testing import *
import numpy

from taylorpoly.utps import UTPS, mul 

class Test_Binary_Operators(TestCase):
    
    def test_mul(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        
        z = mul(x,y)
        
        assert_array_almost_equal([5.,17.,40.], z.data)
        
    def test_mul2(self):
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        z = mul(x,y)
        
        assert_array_almost_equal([5.,17.,40., 21., 36.], z.data)

if __name__ == "__main__":
    run_module_suite()



