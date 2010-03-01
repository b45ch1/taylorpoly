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
_utps.utps_log.argtypes = argtypes2
_utps.utps_exp.argtypes = argtypes2
_utps.utps_pow.argtypes = argtypes3
_utps.utps_sin_cos.argtypes = argtypes1



class UTPS:
    """
    UTPS = Univariate Taylor Polynomial over Scalars
    Implements the factor ring  R[t]/<t^{D}>.
    
    To allow vectorized operations (i.e. propagating several directional derivatives at once),
    the following memory layout is used:
    
    x = [x_0, x_{1,1}, x_{1,2}, ..., x_{1,P}, ..., x_{D-1,P}]
    
    i.e. x.size = (D-1)*P+1
    
    """
    
    def __init__(self, data, P, D):
        """
        x = [x_0, x_{1,1}, x_{1,2}, ..., x_{1,P}, ..., x_{D-1,P}]
        """
        if numpy.size(data) != (D-1)*P+1:
            err_str = 'size(data) should be (D-1)*P+1\n'
            err_str = 'but provided size(data) = %d\n'%numpy.size(data)
            err_str = ' (D-1)*P+1 = (%d-1)*%d+1 = %d'%(D,P,(D-1)*P+1)
            raise ValueError(err_str)
        
        self.data = numpy.ascontiguousarray(data)
        self.D = D
        self.P = P
        
    def __str__(self):
        return str(self.data)
        
    def __repr__(self):
        ret_str = 'UTPS(%s, %d, %d)'%(str(self.data), self.P, self.D)

    def __zeros_like__(self):
        return self.__class__(numpy.zeros_like(self.data), self.P, self.D)
        
    def copy(self):
        return self.__class__(self.data.copy(), self.P, self.D)
        
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
        
    def __abs__(self):
        if self.data[0] < 0:
            return self.__class__( -1.* self.data, self.P, self.D)
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
        

def add(x,y, out = None):
    """
    computes z = x*y in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_add(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def sub(x,y, out = None):
    """
    computes z = x*y in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_sub(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out

def mul(x,y,out = None):
    """
    computes z = x*y in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_mul(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def div(x,y,out = None):
    """
    computes z = x/y in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_div(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def log(x,out = None):
    """
    computes y = log(x) in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_log(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def exp(x,out = None):
    """
    computes y = exp(x) in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_exp(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def pow(x,r, out = None):
    """
    computes y = pow(x,r) in Taylor arithmetic
    """
    if out == None:
        out = x.__zeros_like__()
    
    _utps.utps_pow(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    r,
    out.data.ctypes.data_as(double_ptr))
    
    return out
    
def sin_cos(x, out = None):
    """
    computes s = sin(x) in Taylor arithmetic
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
    """
    computes s = sin(x) in Taylor arithmetic
    
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
    """
    computes s = cos(x) in Taylor arithmetic
    
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

