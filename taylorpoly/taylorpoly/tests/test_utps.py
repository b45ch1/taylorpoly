from numpy.testing import *
import numpy

from taylorpoly.utps import UTPS, add, sub, mul, div, log, exp, pow

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
         
    def test_sub(self):
        """ test z = x - y"""
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        
        z = sub(x,y)
        
        assert_array_almost_equal([-4.,-5.,-8.], z.data)
    
    def test_isub(self):
        """ test y -= x"""
        x = UTPS(numpy.array([1.,2.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,7.,11.]), P = 1, D = 3)
        
        y = sub(y,x,y)
        
        assert_array_almost_equal([4.,5.,8.], y.data)
        
    def test_sub_vectorized(self):
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        z = sub(y,x)
        
        assert_array_almost_equal([4.,5.,8., -3., -4.], z.data)
        
    def test_iadd_vectorized(self):
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        y = sub(y,x,y)
        
        assert_array_almost_equal([4.,5.,8., -3., -4.], y.data)
      
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
        
    def test_div(self):
        try:
            import sympy
        
        except:
            return
        
        x = UTPS(numpy.array([1.,4.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,1, 7.]), P = 1, D = 3)
        
        z = div(x,y)
        assert_array_almost_equal([1.,4.,3.], x.data)
        assert_array_almost_equal([5.,1, 7.], y.data)
        
        t = sympy.symbols('t')
        sx = x.data[0] + x.data[1]*t + x.data[2]*t**2
        sy = y.data[0] + y.data[1]*t + y.data[2]*t**2
        sz = sx/sy
        
        correct = numpy.array([ x.data[0]/y.data[0],
                                sz.series(t).coeff(t).evalf(),
                                sz.series(t).coeff(t**2).evalf()])
        assert_array_almost_equal(correct, z.data)
        
    def test_idiv(self):
        try:
            import sympy
        
        except:
            return
        
        x  = UTPS(numpy.array([1.,4.,3.]), P = 1, D = 3)
        x2 = UTPS(numpy.array([1.,4.,3.]), P = 1, D = 3)
        y = UTPS(numpy.array([5.,1, 7.]), P = 1, D = 3)
        
        x = div(x,y,x)
        assert_array_almost_equal([5.,1, 7.], y.data)
        
        t = sympy.symbols('t')
        sx = x2.data[0] + x2.data[1]*t + x2.data[2]*t**2
        sy = y.data[0] + y.data[1]*t + y.data[2]*t**2
        sz = sx/sy
        
        correct = numpy.array([ x2.data[0]/y.data[0],
                                sz.series(t).coeff(t).evalf(),
                                sz.series(t).coeff(t**2).evalf()])
        assert_array_almost_equal(correct, x.data)
        
    def test_div_vectorized(self):
        try:
            import sympy
        
        except:
            return
        
        
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        z = div(x,y)
        
        t = sympy.symbols('t')
        sx1 = x.data[0] + x.data[1]*t + x.data[2]*t**2
        sy1 = y.data[0] + y.data[1]*t + y.data[2]*t**2
        sz1 = sx1/sy1
        
        sx2 = x.data[0] + x.data[3]*t + x.data[4]*t**2
        sy2 = y.data[0] + y.data[3]*t + y.data[4]*t**2
        sz2 = sx2/sy2
        
        
        correct1 = numpy.array([ x.data[0]/y.data[0],
                        sz1.series(t).coeff(t).evalf(),
                        sz1.series(t).coeff(t**2).evalf()])
        
        correct2 = numpy.array([ x.data[0]/y.data[0],
                                sz2.series(t).coeff(t).evalf(),
                                sz2.series(t).coeff(t**2).evalf()])

        assert_array_almost_equal(correct1, z.data[[0,1,2]])
        assert_array_almost_equal(correct2, z.data[[0,3,4]])
        
        
    def test_idiv_vectorized(self):
        try:
            import sympy
        
        except:
            return
        
        x = UTPS(numpy.array([1.,2.,3, 4.,6.]),P = 2, D = 3)
        y = UTPS(numpy.array([5.,7.,11, 1.,2.]),P = 2, D = 3)
        
        t = sympy.symbols('t')
        sx1 = x.data[0] + x.data[1]*t + x.data[2]*t**2
        sy1 = y.data[0] + y.data[1]*t + y.data[2]*t**2
        sz1 = sx1/sy1
        
        sx2 = x.data[0] + x.data[3]*t + x.data[4]*t**2
        sy2 = y.data[0] + y.data[3]*t + y.data[4]*t**2
        sz2 = sx2/sy2
        
        correct1 = numpy.array([ x.data[0]/y.data[0],
                        sz1.series(t).coeff(t).evalf(),
                        sz1.series(t).coeff(t**2).evalf()])
        
        correct2 = numpy.array([ x.data[0]/y.data[0],
                                sz2.series(t).coeff(t).evalf(),
                                sz2.series(t).coeff(t**2).evalf()])
        
        x = div(x,y,x)

        assert_array_almost_equal(correct1, x.data[[0,1,2]])
        assert_array_almost_equal(correct2, x.data[[0,3,4]])

    def test_log(self):
        x = UTPS(numpy.array([1.,2.,3.]),P = 1, D = 3)
        y = log(x)
                
        correct = numpy.array([ numpy.log(x.data[0]),
                                x.data[1]/x.data[0],
                                (2*x.data[2]*x.data[0] - x.data[1]**2)/(2*x.data[0]**2)])
        assert_array_almost_equal(correct, y.data)

    def test_log_vectorized(self):
        x = UTPS(numpy.array([1.,2.,3.,4.,5.]),P = 2, D = 3)
        y = log(x)
        
        correct1 = numpy.array([ numpy.log(x.data[0]),
                                x.data[1]/x.data[0],
                                (2*x.data[2]*x.data[0] - x.data[1]**2)/(2*x.data[0]**2)])
        
        correct2 = numpy.array([ numpy.log(x.data[0]),
                                x.data[3]/x.data[0],
                                (2*x.data[4]*x.data[0] - x.data[3]**2)/(2*x.data[0]**2)])
        
        assert_array_almost_equal(correct1, y.data[[0,1,2]])
        assert_array_almost_equal(correct2, y.data[[0,3,4]])

    def test_exp(self):
        x = UTPS(numpy.array([1.,2.,3.]),P = 1, D = 3)
        y = exp(x)
                
        correct = numpy.array([ numpy.exp(x.data[0]),
                                x.data[1]*numpy.exp(x.data[0]),
                                (2*x.data[2] + x.data[1]**2) *numpy.exp(x.data[0])/2.])
        
        assert_array_almost_equal(correct, y.data)
        
    def test_exp_vectorized(self):
        x = UTPS(numpy.array([1.,2.,3.,4.,5.]),P = 2, D = 3)
        x1 = UTPS(numpy.array([1.,2.,3.]),P = 1, D = 3)
        x2 = UTPS(numpy.array([1.,4.,5.]),P = 1, D = 3)
        
        y  = exp(x)
        y1 = exp(x1)
        y2 = exp(x2)
       
        assert_array_almost_equal(y1.data, y.data[[0,1,2]])
        assert_array_almost_equal(y2.data, y.data[[0,3,4]])
        
    def test_pow(self):
        x = UTPS(numpy.array([2.,2.,3.,7.]),P = 1, D = 4)
        y = pow(x,2.)
        
        y2 = mul(x,x);
        assert_array_almost_equal(y2.data, y.data)
        
    def test_pow(self):
        x = UTPS(numpy.array([2.,2.,3.,7., 1., 3., 2.]),P = 2, D = 4)
        x1 = UTPS(numpy.array([2.,2.,3.,7.]),P = 1, D = 4)
        x2 = UTPS(numpy.array([2.,1., 3., 2.]),P = 1, D = 4)
        y = pow(x,2.)
        y1 = mul(x1,x1)
        y2 = mul(x2,x2);
        
        assert_array_almost_equal(y1.data, y.data[[0,1,2,3]])
        assert_array_almost_equal(y2.data, y.data[[0,4,5,6]])
        

if __name__ == "__main__":
    run_module_suite()



