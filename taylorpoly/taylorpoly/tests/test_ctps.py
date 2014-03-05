from numpy.testing import TestCase
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal, assert_equal
import numpy

from taylorpoly.ctps import CTPS, add, sub, mul, div


class Test_global_funcs(TestCase):

    def test_add_K1(self):
        """ test z = x + y"""
        x = CTPS(numpy.array([1.,2.]))
        y = CTPS(numpy.array([5.,7.]))
        z = CTPS(numpy.array([0.,0.]))
        add(x, y, z)

        expected = numpy.array([6.,9.])
        assert_array_almost_equal(expected, z.data)

    def test_add_K2(self):
        """ test z = x + y"""
        x = CTPS(numpy.array([1.,2.,3.,4.]))
        y = CTPS(numpy.array([5.,7.,9.,1.]))
        z = CTPS(numpy.array([0.,0.,0.,0.]))
        add(x, y, z)

        expected = numpy.array([6.,9.,12.,5.])
        assert_array_almost_equal(expected, z.data)


    def test_sub_K1(self):
        """ test z = x - y"""
        x = CTPS(numpy.array([1.,2.]))
        y = CTPS(numpy.array([5.,7.]))
        z = CTPS(numpy.array([0.,0.]))
        sub(x, y, z)

        expected = numpy.array([-4.,-5.])
        assert_array_almost_equal(expected, z.data)

    def test_sub_K2(self):
        """ test z = x - y"""
        x = CTPS(numpy.array([1.,2.,3.,4.]))
        y = CTPS(numpy.array([5.,7.,9.,1.]))
        z = CTPS(numpy.array([0.,0.,0.,0.]))
        sub(x, y, z)

        expected = numpy.array([-4.,-5.,-6.,3.])
        assert_array_almost_equal(expected, z.data)

    def test_mul_K1(self):
        """ test z = x * y"""
        x = CTPS(numpy.array([1.,2.]))
        y = CTPS(numpy.array([5.,7.]))
        z = CTPS(numpy.array([0.,0.]))
        mul(x, y, z)

        expected = numpy.array([5.,17.])
        assert_array_almost_equal(expected, z.data)

    def test_mul_K2(self):
        """ test z = x * y"""
        x = CTPS(numpy.array([1.,2.,3.,4.]))
        y = CTPS(numpy.array([5.,7.,9.,1.]))
        z = CTPS(numpy.array([0.,0.,0.,0.]))
        mul(x, y, z)

        expected = numpy.array([5.,17.,24.,1.+20. + 18. + 21.])
        assert_array_almost_equal(expected, z.data)

    def test_div_K1(self):
        """ test z = x / y"""
        x = CTPS(numpy.array([1.,2.]))
        y = CTPS(numpy.array([5.,7.]))
        z = CTPS(numpy.array([0.,0.]))
        v = CTPS(numpy.array([0.,0.]))
        div(x, y, z)
        mul(z, y, v)

        assert_array_almost_equal(v.data, x.data)

    def test_div_K2(self):
        """ test z = x / y"""
        x = CTPS(numpy.array([1.,2.,3.,4.]))
        y = CTPS(numpy.array([5.,7.,9.,1.]))
        z = CTPS(numpy.array([0.,0.,0.,0.]))
        v = CTPS(numpy.array([0.,0.,0.,0.]))
        div(x, y, z)
        mul(z, y, v)

        assert_array_almost_equal(v.data, x.data)




if __name__ == "__main__":
    run_module_suite()



