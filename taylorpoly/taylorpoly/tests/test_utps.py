from numpy.testing import *
import numpy

from taylorpoly.utps import UTPS, mul, add

class Test_Binary_Operators(TestCase):
    
    def test_add(self):
        """ test z = x + y"""
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        
        z = add(x,y)
        
        assert_array_almost_equal([6.,9.,14.], z.data)
    
    def test_iadd(self):
        """ test y += x"""
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        
        y = add(y,x,y)
        
        assert_array_almost_equal([6.,9.,14.], y.data)
        
    def test_add_vectorized(self):
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        z = add(x,y)
        
        assert_array_almost_equal([6.,9.,14., 5., 8.], z.data)
        
    def test_iadd_vectorized(self):
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        y = add(y,x,y)
        
        assert_array_almost_equal([6.,9.,14., 5., 8.], y.data)
        
        
    
    def test_mul(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        
        z = mul(x,y)
        
        assert_array_almost_equal([5.,17.,40.], z.data)
        
    def test_imul(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        
        y = mul(y,x,y)
        assert_array_almost_equal([5.,17.,40.], y.data)
        
        
    def test_mul_vectorized(self):
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        z = mul(x,y)
        
        assert_array_almost_equal([5.,17.,40., 21., 36.], z.data)
        
    def test_imul_vectorized(self):
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        y = mul(y,x,y)
        
        assert_array_almost_equal([5.,17.,40., 21., 36.], y.data)
        
        

if __name__ == "__main__":
    run_module_suite()



