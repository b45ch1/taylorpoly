"""
This module provides algorithms for Univariate Taylor Propagation over Scalars (UTPS)
in the forward and reverse mode of Algorithmic Differentiation (AD).

More specifically, algorithms are provided that take Univariate Taylor Polynomials
over Scalars ( also abbreviated UTPS) as inputs and have UTPS instances as output.

For convenience, a class UTPS is provided that wraps the data structure of the
univariate Taylor polynomials over scalars.

The algorithms take objects of the class UTPS as input. The class UTPS is a simple
wrapper of the data structure.


Example 1:

    Compute the gradient: df/dx (x,y)

    Code::
    
        import numpy
        from taylorpoly.utps import UTPS, exp
        
        x,y = 2,3
        v0 = UTPS([x,1.,0.], P = 2)
        v1 = UTPS([y,0.,1.], P = 2)
        
        v2 = v0 * v1
        v3 = exp(v2)
        f = v3 + v0
        
        # finite differences gradient
        delta = 10**-8
        g1_fd = ((numpy.exp((x + delta) * y) + x + delta) - ((numpy.exp(x * y) + x)))/delta
        g2_fd = ((numpy.exp((y + delta) * x) + x) - ((numpy.exp(x * y) + x)))/delta
        
        print 'forward mode AD gradient = [ %f, %f] '%(f.data[1], f.data[2])
        print 'symbolic        gradient = [ %f, %f] '%(y*numpy.exp(x*y) + 1, x*numpy.exp(x*y))
        print 'finite diff.    gradient = [ %f, %f] '%(g1_fd, g2_fd)
    


"""


import os
import ctypes
import numpy

_utps = numpy.ctypeslib.load_library('libutps', os.path.dirname(__file__))

double_ptr =  ctypes.POINTER(ctypes.c_double)
argtypes1 = [ctypes.c_int, ctypes.c_int, double_ptr, double_ptr, double_ptr]
argtypes2 = [ctypes.c_int, ctypes.c_int, double_ptr, double_ptr]
argtypes3 = [ctypes.c_int, ctypes.c_int, double_ptr, ctypes.c_double, double_ptr]


_utps.utps_add.argtypes = argtypes1
_utps.utps_sub.argtypes = argtypes1
_utps.utps_mul.argtypes = argtypes1
_utps.utps_div.argtypes = argtypes1
_utps.utps_sqrt.argtypes = argtypes2
_utps.utps_log.argtypes = argtypes2
_utps.utps_exp.argtypes = argtypes2
_utps.utps_pow.argtypes = argtypes3
_utps.utps_sin_cos.argtypes = argtypes1



class UTPS:
    """
    UTPS = Univariate Taylor Polynomial over Scalars
    Implements the factor ring  R[[t]]/ R[[t]]t^{D},
    where R[[t]] is the formal power series over the field of real numbers R.
    
    
    Rationale:
    
        UTPS instances are a convenient way to do Taylor arithmetic.
        Taylor arithmetic is a fundamental tool for Algorithmic Differentiation.
        Using the algorithms in `taylorpoly` it is possible to easily create new 
        Algorithmic Differentiation tools.
        
        
    Example:
    
        Compute the gradient g of the function `f(x) = sin(x)**2.3 - x` in x = 3.
    
        Code::
        
            from taylorpoly import UTPS, sin
            
            x = UTPS([3.,1.])
            f = sin(x)**2.3 - x
            g = f.data[1]
            
            print 'computed gradient is g=',g
    
    
    
    Internal Data Structure:
    
        To allow vectorized operations (i.e. propagating several directional derivatives at once),
        the following memory layout is used:
        
        x = [x_0, x_{1,1}, x_{1,2}, ..., x_{1,P}, ..., x_{D-1,P}]
        i.e. x.size = (D-1)*P+1
    
    """
    
    def __init__(self, data, P = 1):
        """
        x = [x_0, x_{1,1}, x_{1,2}, ..., x_{1,P}, ..., x_{D-1,P}]
        """
        
        D = (numpy.size(data)-1)//P + 1
        
        if (numpy.size(data)-1) % P != 0:
            err_str  = 'size of input data does not match the value of P\n'
            err_str += 'inputs have to satisfy data.size = (D-1)*P+1\n'
            err_str += 'but provided data.size = %d and P = %d\n'%(numpy.size(data), P)
            raise ValueError(err_str)
        
        self.data = numpy.ascontiguousarray(data, dtype=float)
        self.D = D
        self.P = P
        
        self.coeff = self.Coeff(self)
        
        
    class Coeff:
        """
        helper class for UTPS that allows to extract the array of direction p
        and degree d from UTPS.data as numpy array with the correct shape.
        """
        def __init__(self, x):
            self.x = x
            
        def __getitem__(self, sl):
            p,d = sl
            if d >= self.x.D:
                raise ValueError('d is too large')
            if p >= self.x.P:
                raise ValueError('p is too large')
                
            if d == 0:
                return self.x.data[0]
            
            else:
               return self.x.data[d + p*(self.x.D-1)]
                
        def __setitem__(self, sl, value):
            p,d = sl
            if d >= self.x.D:
                raise ValueError('d is too large')
            if p >= self.x.P:
                raise ValueError('p is too large')
                
            if d == 0:
                self.x.data[0] = value
            
            else:
                self.x.data[d + p*(self.x.D-1)] = value
        
        
    def __str__(self):
        """ return human readable string representation"""
        return str(self.data)
        
    def __repr__(self):
        """ return data that allows to reconstruct the instance"""
        ret_str = 'UTPS(%s, %d, %d)'%(str(self.data), self.P, self.D)
        return ret_str

    def __zeros_like__(self):
        """ returns a copy of self with all elements set to zero"""
        return self.__class__(numpy.zeros_like(self.data), self.P)
        
    def copy(self):
        """ copies all data in self to a new instance """
        return self.__class__(self.data.copy(), self.P)
        
    def __neg__(self):
        return neg(self)
        
    def __add__(self, other):
        return add(self,other)
        
    def __sub__(self, other):
        return sub(self,other)
        
    def __mul__(self, other):
        return mul(self,other)
        
    def __div__(self, other):
        return div(self,other)
        
    def __radd__(self, other):
        return add(other, self)
        
    def __rsub__(self, other):
        return sub(other, self)
        
    def __rmul__(self, other):
        return mul(other, self)
        
    def __rdiv__(self, other):
        return div(other, self)

    def __iadd__(self, other):
        return add(self,other, out = self)
        
    def __isub__(self, other):
        return sub(self,other, out = self)
        
    def __imul__(self, other):
        return mul(self,other, out = self)
        
    def __idiv__(self, other):
        return div(self,other, out = self)
        
    def __pow__(self, r):
        return pow(self,r)
        
    def sin(self):
        return sin(self)
        
    def cos(self):
        return cos(self)
        
    def exp(self):
        return exp(self)
        
    def __abs__(self):
        if self.data[0] < 0:
            return self.__class__( -1.* self.data, self.P)
        else:
            return self.copy()
            
    def __le__(self, other):
        return self.data[0] <= other.data[0]
        
    def __ge__(self, other):
        return self.data[0] >= other.data[0]
        
    def __lt__(self, other):
        return self.data[0] < other.data[0]
        
    def __gt__(self, other):
        return self.data[0] > other.data[0]

def convert2UTPS(x,y):
    """
    If either x or y is not an UTPS instance but a scalar quantity (e.g. a Float)
    this function converts either x or y to an UTPS instance such that x and y
    have the same P and same D.
    
    Rationale:
    This function allows a workaround to do computations like::
    
        x = UTPS([1,2,3])
        y = 3. * x
        
    by using algorithms that assume arithmetic on homogenous (i.e. same D same P)
    Taylor polynomials.
    """
    
    if not isinstance(x, UTPS):
        out = y.__zeros_like__()
        out.data[0] = x
        x = out
        
    elif not isinstance(y, UTPS):
        out = x.__zeros_like__()
        out.data[0] = y
        y = out
        
    return x,y
    

def neg(x, out = None):
    """ computes y = -x in Taylor arithmetic """
    if out == None:
        out = x.copy()
        
    out.data *= -1.
    
    return out

def add(x,y, out = None):
    """ computes z = x+y in Taylor arithmetic
    """
    x,y = convert2UTPS(x,y)
    if out == None:
        out = x.__zeros_like__()
        
    if id(x) == id(y) and id(y) == id(out):
        out.data *= 2
        return out
    
    _utps.utps_add(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def sub(x,y, out = None):
    """ computes z = x-y in Taylor arithmetic
    """
    x,y = convert2UTPS(x,y)

    if out == None:
        out = x.__zeros_like__()
        
    if id(x) == id(y) and id(y) == id(out):
        out.data *= 0
        return out
        
    _utps.utps_sub(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out

def mul(x,y,out = None):
    """ computes z = x*y in Taylor arithmetic
    """
    x,y = convert2UTPS(x,y)
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_mul(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def div(x,y,out = None):
    """ computes z = x/y in Taylor arithmetic
    """
    x,y = convert2UTPS(x,y)
    if out == None:
        out = x.__zeros_like__()
        
    if id(x) == id(y) and id(y) == id(out):
        out.data *= 0
        out.data[0] = 1.
        return out
        
    _utps.utps_div(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def amul(x,y,out = None):
    """ computes z += x*y in Taylor arithmetic
    """
    x,y = convert2UTPS(x,y)
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_amul(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def sqrt(x,out = None):
    """ computes y = sqrt(x) in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_sqrt(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def log(x,out = None):
    """ computes y = log(x) in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_log(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def exp(x,out = None):
    """ computes y = exp(x) in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_exp(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def pow(x,r, out = None):
    """ computes y = pow(x,r) in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_pow(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    r,
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def sin_cos(x, out = None):
    """ computes s = sin(x) in Taylor arithmetic
    """
    
    if out == None:
        s = x.__zeros_like__()
        c = x.__zeros_like__()
    
    else:
        s,c = out
    
    _utps.utps_sin_cos(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    s.data.ctypes.data_as(double_ptr),
    c.data.ctypes.data_as(double_ptr))
    
    return s,c
    
def sin(x, out = None):
    """ computes s = sin(x) in Taylor arithmetic
    
    Remark: If you also need to compute cos(x) then use the function sin_cos.
    """
    
    if out == None:
        s = x.__zeros_like__()
        
    c = x.__zeros_like__()
    
    _utps.utps_sin_cos(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    s.data.ctypes.data_as(double_ptr),
    c.data.ctypes.data_as(double_ptr))
    
    return s
    
    
def cos(x, out = None):
    """ computes s = cos(x) in Taylor arithmetic
    
    Remark: If you also need to compute sin(x) then use the function sin_cos.
    """
    
    if out == None:
        c = x.__zeros_like__()
        
    s = x.__zeros_like__()
    
    _utps.utps_sin_cos(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    s.data.ctypes.data_as(double_ptr),
    c.data.ctypes.data_as(double_ptr))
    
    return c

