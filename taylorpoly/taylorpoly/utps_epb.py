import ctypes
from taylorpoly.utps import _utps, double_ptr

argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
            double_ptr, double_ptr, double_ptr,
            double_ptr, double_ptr, double_ptr]

_utps.utps_epb_add.argtypes = argtypes
_utps.utps_epb_sub.argtypes = argtypes
_utps.utps_epb_mul.argtypes = argtypes
_utps.utps_epb_div.argtypes = argtypes



def epb_add(x,y,z, zbar, xbar, ybar):
    """ computes the pullback of the extended linear form
    i.e. [zbar] d[z] = [zbar] d[x] + [zbar] d[y]
    """
    
    _utps.utps_epb_add(zbar.Q, x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    z.data.ctypes.data_as(double_ptr),
    zbar.data.ctypes.data_as(double_ptr),
    xbar.data.ctypes.data_as(double_ptr),
    ybar.data.ctypes.data_as(double_ptr))
    
    return (xbar, ybar)
    
def epb_sub(x,y,z, zbar, xbar, ybar):
    """ computes the pullback of the extended linear form
    i.e. [zbar] d[z] = [zbar] d[x] - [zbar] d[y]
    """
    
    _utps.utps_epb_sub(zbar.Q, x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    z.data.ctypes.data_as(double_ptr),
    zbar.data.ctypes.data_as(double_ptr),
    xbar.data.ctypes.data_as(double_ptr),
    ybar.data.ctypes.data_as(double_ptr))
    
    return (xbar, ybar)
    
def epb_mul(x,y,z, zbar, xbar, ybar):
    """ computes the pullback of the extended linear form
    i.e. [zbar] d[z] = [zbar] [y] d[x] + [zbar] [x] d[y]
    """
    
    _utps.utps_epb_mul(zbar.Q, x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    z.data.ctypes.data_as(double_ptr),
    zbar.data.ctypes.data_as(double_ptr),
    xbar.data.ctypes.data_as(double_ptr),
    ybar.data.ctypes.data_as(double_ptr))
    
    return (xbar, ybar)
    
def epb_div(x,y,z, zbar, xbar, ybar):
    """ computes the pullback of the extended linear form
    i.e. [zbar] d[z] = [zbar] [z] d[x] + [zbar] [z] [z] d[y]
    
    Warning: this function changes the value of z!
    """
    
    err_code = _utps.utps_epb_div(zbar.Q, x.P,x.D,
    x.data.ctypes.data_as(double_ptr),
    y.data.ctypes.data_as(double_ptr),
    z.data.ctypes.data_as(double_ptr),
    zbar.data.ctypes.data_as(double_ptr),
    xbar.data.ctypes.data_as(double_ptr),
    ybar.data.ctypes.data_as(double_ptr))
    
    if err_code != 0:
        raise Exception('error in epb_div')
    
    return (xbar, ybar)
