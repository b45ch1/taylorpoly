import ctypes
from taylorpoly.utps import _utps, double_ptr

argtypes = [ctypes.c_int, ctypes.c_int,
            double_ptr, double_ptr, double_ptr,
            double_ptr, double_ptr, double_ptr]

_utps.utps_pbe_add.argtypes = argtypes

def epb_add(x,y,z, zbar, xbar, ybar):
    """ computes the pullback of the extended linear form
    i.e. [zbar] d[z] = [zbar] d[x] + [zbar] d[y]
    """
    
    _utps.utps_pbe_add(x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    z.data.ctypes.data_as(double_ptr),
    zbar.data.ctypes.data_as(double_ptr),
    xbar.data.ctypes.data_as(double_ptr),
    ybar.data.ctypes.data_as(double_ptr))
    
    return (xbar, ybar)
