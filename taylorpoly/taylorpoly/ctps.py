import os
import ctypes
import numpy

_ctps = numpy.ctypeslib.load_library('libctps', os.path.dirname(__file__))

double_ptr =  ctypes.POINTER(ctypes.c_double)
argtypes1 = [ctypes.c_int, double_ptr, double_ptr, double_ptr]

_ctps.ctps_add.argtypes = argtypes1
_ctps.ctps_sub.argtypes = argtypes1
_ctps.ctps_mul.argtypes = argtypes1
_ctps.ctps_div.argtypes = argtypes1

class CTPS(object):
    """
    CTPS = Cross Derivative Taylor Polynomial
    Implements the factor ring  R[t1,...,tK]/<t1^2,...,tK^2>

    Calls C functions internally without type and memory checks.
    """

    def __init__(self, data):
        self.data = numpy.array(data)

    @classmethod
    def __scalar_to_data__(cls, xdata, x):
        xdata[0] = x

    @classmethod
    def zeros_like(cls, x):
        return cls(numpy.zeros_like(x.data))

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

def data_add(lhs_data, rhs_data, retval_data):
    h = retval_data.size
    _ctps.ctps_add(h,
    lhs_data.ctypes.data_as(double_ptr),
    rhs_data.ctypes.data_as(double_ptr),
    retval_data.ctypes.data_as(double_ptr))

def data_sub(lhs_data, rhs_data, retval_data):
    h = retval_data.size
    _ctps.ctps_sub(h,
    lhs_data.ctypes.data_as(double_ptr),
    rhs_data.ctypes.data_as(double_ptr),
    retval_data.ctypes.data_as(double_ptr))

def data_mul(lhs_data, rhs_data, retval_data):
    h = retval_data.size
    _ctps.ctps_mul(h,
    lhs_data.ctypes.data_as(double_ptr),
    rhs_data.ctypes.data_as(double_ptr),
    retval_data.ctypes.data_as(double_ptr))

def data_div(lhs_data, rhs_data, retval_data):
    h = retval_data.size
    _ctps.ctps_div(h,
    lhs_data.ctypes.data_as(double_ptr),
    rhs_data.ctypes.data_as(double_ptr),
    retval_data.ctypes.data_as(double_ptr))

def add(x, y, z):
    data_add(x.data, y.data, z.data)

def sub(x, y, z):
    data_sub(x.data, y.data, z.data)

def mul(x, y, z):
    data_mul(x.data, y.data, z.data)

def div(x, y, z):
    data_div(x.data, y.data, z.data)

