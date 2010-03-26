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
    Implements the factor ring  R[t]/ R[t]t^{D},
    where R[t] is the polynomial ring over the field of real numbers R.
    
    
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
    
    def __init__(self, data, P = None):
        """
        x = [x_0, x_{1,1}, x_{1,2}, ..., x_{1,P}, ..., x_{D-1,P}]
        """
        
        if P == None:
            P = 1
        
        D = (numpy.size(data)-1)//P + 1
        
        if (numpy.size(data)-1) % P != 0:
            err_str  = 'size of input data does not match the value of P\n'
            err_str += 'inputs have to satisfy data.size = (D-1)*P+1\n'
            err_str += 'but provided data.size = %d and P = %d\n'%(numpy.size(data), P)
            raise ValueError(err_str)
        
        self.data = numpy.ascontiguousarray(data, dtype=float)
        self.D = D
        self.P = P
        
    def __str__(self):
        """ return human readable string representation"""
        return str(self.data)
        
    def __repr__(self):
        """ return data that allows to reconstruct the instance"""
        ret_str = 'UTPS(%s, %d, %d)'%(str(self.data), self.P)

    def __zeros_like__(self):
        """ returns a copy of self with all elements set to zero"""
        return self.__class__(numpy.zeros_like(self.data), self.P)
        
    def copy(self):
        """ copies all data in self to a new instance """
        return self.__class__(self.data.copy(), self.P)
        
    def __add__(self, other):
        return add(self,other)
        
    def __sub__(self, other):
        return sub(self,other)
        
    def __mul__(self, other):
        return mul(self,other)
        
    def __div__(self, other):
        return div(self,other)

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

def extract(x,p,d):
    """
    Extracts the d'th coefficients from a numpy.ndarray x of dtype object (UTPS)
    and returns an numpy.ndarray of dtype=float.
    """
    shp = numpy.shape(x)
    xr = numpy.ravel(x)
    y = numpy.array([xn.data[(d>0)*(1 + p*d)] for xn in xr], dtype=float)
    y.reshape(shp)
    return y
    


def add(x,y, out = None):
    """ computes z = x+y in Taylor arithmetic
    """
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

