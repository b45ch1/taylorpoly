from numpy.testing import TestCase, assert_array_almost_equal, assert_almost_equal, assert_equal
import numpy

from taylorpoly.utps import UTPS, add, sub, mul, div, log, exp, pow, sin_cos, sin, cos, sqrt, amul
from taylorpoly.utps_epb import epb_add, epb_sub, epb_mul, epb_div

class Test_vector_forward_scalar_reverse(TestCase):
    def test_add_epb(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1)
        z = add(x,y)
        
        zbar = UTPS(numpy.random.rand(3), P = 1)
        xbar = UTPS(numpy.zeros(3), P = 1)
        ybar = UTPS(numpy.zeros(3), P = 1)
        
        epb_add(x,y,z, zbar, xbar, ybar)
        assert_array_almost_equal(zbar.data, xbar.data)
        assert_array_almost_equal(zbar.data, ybar.data)
        
    def test_sub_epb(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1)
        z = sub(x,y)
        
        zbar = UTPS(numpy.random.rand(3), P = 1)
        xbar = UTPS(numpy.zeros(3), P = 1)
        ybar = UTPS(numpy.zeros(3), P = 1)
        
        epb_sub(x,y,z, zbar, xbar, ybar)
        assert_array_almost_equal(zbar.data, xbar.data)
        assert_array_almost_equal(zbar.data, -ybar.data)
        
    def test_mul_epb(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1)
        z = mul(x,y)
        
        zbar = UTPS(numpy.random.rand(3), P = 1)
        xbar = UTPS(numpy.zeros(3), P = 1)
        ybar = UTPS(numpy.zeros(3), P = 1)
        
        epb_mul(x,y,z, zbar, xbar, ybar)
        
        assert_array_almost_equal( (zbar*x).data, ybar.data)
        assert_array_almost_equal( (zbar*y).data, xbar.data)

    def test_div_epb(self):
        x = UTPS(numpy.array([1.,2.,3.]), P = 1)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1)
        z = div(x,y)
        
        zbar = UTPS(numpy.random.rand(3), P = 1)
        xbar = UTPS(numpy.zeros(3), P = 1)
        ybar = UTPS(numpy.zeros(3), P = 1)
        
        xbar2 = zbar/y
        ybar2 = - zbar*z/y
        epb_div(x,y,z, zbar, xbar, ybar)
        
        assert_array_almost_equal(xbar2.data, xbar.data)
        assert_array_almost_equal(ybar2.data, ybar.data)
        
        
    def test_compare_pushforward_pullback_derivatives(self):
        x = UTPS(numpy.array([7.,1.,0.]), P = 1)
        y = UTPS(numpy.array([13.,0.,0.]), P = 1)
        v1 = mul(x,y)
        v2 = mul(x,v1)
        
        v2bar = UTPS([1.,0.,0.], P = 1)
        v1bar = UTPS(numpy.zeros(3), P = 1)
        xbar = UTPS(numpy.zeros(3), P = 1)
        ybar = UTPS(numpy.zeros(3), P = 1)
        
        epb_mul(x,v1,v2, v2bar, xbar, v1bar)
        epb_mul(x,y,v1, v1bar, xbar, ybar)
        
        assert_array_almost_equal(xbar.data[0], v2.data[1])
        assert_array_almost_equal(xbar.data[1], 2*v2.data[2])
        
    def test_compare_pushforward_pullback_derivatives2(self):
        x = UTPS(numpy.array([1.,0.,0.,0.]), P = 1)
        y = UTPS(numpy.array([2.,1.,0.,0.]), P = 1)
        z = div(x,y)
        
        zbar = UTPS([1.,0.,0., 0.], P = 1)
        xbar = UTPS(numpy.zeros(4), P = 1)
        ybar = UTPS(numpy.zeros(4), P = 1)
        
        epb_div(x,y,z, zbar, xbar, ybar)
        
        facs = numpy.array([1.,1.,2.,6.])
        
        assert_array_almost_equal((z.data*facs)[1:], (ybar.data*facs)[:-1])


class Test_vector_forward_vector_reverse(TestCase):
    def test_add_epb(self):
        
        Q,P,D = 3,1,3
        x = UTPS(numpy.random.rand(1+P*(D-1)), P = P)
        y = UTPS(numpy.random.rand(1+P*(D-1)), P = P)
        z = add(x,y)
        
        zbar = UTPS(numpy.random.rand( Q*(1+P*(D-1))), Q = Q, P = P)
        xbar = UTPS(numpy.random.rand( Q*(1+P*(D-1))), Q = Q, P = P)
        ybar = UTPS(numpy.random.rand( Q*(1+P*(D-1))), Q = Q, P = P)
        
        xbar2 = xbar.copy()
        ybar2 = ybar.copy()
        zbar2 = ybar.copy()
        
        epb_add(x,y,z, zbar, xbar, ybar)
        
        assert_array_almost_equal(xbar2.data + zbar.data, xbar.data)
        assert_array_almost_equal(ybar2.data + zbar.data, ybar.data)
        
    def test_sub_epb(self):
        Q,P,D = 3,1,3
        x = UTPS(numpy.random.rand(1+P*(D-1)), P = P)
        y = UTPS(numpy.random.rand(1+P*(D-1)), P = P)
        z = sub(x,y)
        
        zbar = UTPS(numpy.random.rand( Q*(1+P*(D-1))), Q = Q, P = P)
        xbar = UTPS(numpy.random.rand( Q*(1+P*(D-1))), Q = Q, P = P)
        ybar = UTPS(numpy.random.rand( Q*(1+P*(D-1))), Q = Q, P = P)
        
        xbar2 = xbar.copy()
        ybar2 = ybar.copy()
        zbar2 = ybar.copy()
        
        epb_sub(x,y,z, zbar, xbar, ybar)
        
        assert_array_almost_equal(xbar2.data + zbar.data, xbar.data)
        assert_array_almost_equal(ybar2.data - zbar.data, ybar.data)

