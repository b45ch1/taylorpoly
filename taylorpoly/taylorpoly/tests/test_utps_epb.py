from numpy.testing import TestCase, assert_array_almost_equal, assert_almost_equal, assert_equal
import numpy

from taylorpoly.utps import UTPS, add, sub, mul, div, log, exp, pow, sin_cos, sin, cos, sqrt, amul
from taylorpoly.utps_epb import epb_add, epb_sub, epb_mul

class Test_global_funcs(TestCase):
    def test_add_epb(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        z = add(x,y)
        
        zbar = UTPS(numpy.random.rand(3), P = 1, D = 3)
        xbar = UTPS(numpy.zeros(3), P = 1, D = 3)
        ybar = UTPS(numpy.zeros(3), P = 1, D = 3)
        
        epb_add(x,y,z, zbar, xbar, ybar)
        assert_array_almost_equal(zbar.data, xbar.data)
        assert_array_almost_equal(zbar.data, ybar.data)
        
    def test_sub_epb(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        z = sub(x,y)
        
        zbar = UTPS(numpy.random.rand(3), P = 1, D = 3)
        xbar = UTPS(numpy.zeros(3), P = 1, D = 3)
        ybar = UTPS(numpy.zeros(3), P = 1, D = 3)
        
        epb_sub(x,y,z, zbar, xbar, ybar)
        assert_array_almost_equal(zbar.data, xbar.data)
        assert_array_almost_equal(zbar.data, -ybar.data)
        
    def test_mul_epb(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        z = mul(x,y)
        
        zbar = UTPS(numpy.random.rand(3), P = 1, D = 3)
        xbar = UTPS(numpy.zeros(3), P = 1, D = 3)
        ybar = UTPS(numpy.zeros(3), P = 1, D = 3)
        
        epb_mul(x,y,z, zbar, xbar, ybar)
        
        assert_array_almost_equal( (zbar*x).data, ybar.data)
        assert_array_almost_equal( (zbar*y).data, xbar.data)


        
