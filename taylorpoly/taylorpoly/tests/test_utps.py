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
        
        # assert_array_almost_equal([5.,17.,40.], z.data)        
        
        
        # for d in range(D):
            # zd = numpy.sum(numpy.convolve(x.data[:d+1],y.data[:d+1],mode='same'))
            # print zd
            # print z.data[d]
            # # assert_array_almost_equal(zd ,z.data[d])    
    
   
    # def test_mul(self):
        # D,P = 3,1
        # x = UTPS(numpy.random.rand( 1 + (D-1)*P), D,P)
        # y = UTPS(numpy.random.rand( 1 + (D-1)*P), D,P)
        
        # z = mul(x,y)
        
        # for d in range(D):
            # zd = numpy.sum(numpy.convolve(x.data[:d+1],y.data[:d+1],mode='same'))
            # print zd
            # print z.data[d]
            # # assert_array_almost_equal(zd ,z.data[d])



if __name__ == "__main__":
    run_module_suite()



