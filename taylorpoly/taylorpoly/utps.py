import os
import ctypes
import numpy

_utps = numpy.ctypeslib.load_library('libutps', os.path.dirname(__file__))

double_ptr =  ctypes.POINTER(ctypes.c_double)
argtypes1 = [ctypes.c_int, ctypes.c_int, double_ptr, double_ptr, double_ptr]
argtypes2 = [ctypes.c_int, ctypes.c_int, double_ptr, double_ptr]

_utps.utps_add.argtypes = argtypes1
_utps.utps_sub.argtypes = argtypes1
_utps.utps_mul.argtypes = argtypes1
_utps.utps_div.argtypes = argtypes1
_utps.utps_log.argtypes = argtypes2
_utps.utps_exp.argtypes = argtypes2


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
        
        self.data = numpy.asarray(data)
        self.D = D
        self.P = P
        
    def __str__(self):
        return str(self.data)
        
    def __repr__(self):
        ret_str = 'UTPS(%s, %d, %d)'%(str(self.data), self.P, self.D)

    def __zeros_like__(self):
        return self.__class__(numpy.zeros_like(self.data), self.P, self.D)
    

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
