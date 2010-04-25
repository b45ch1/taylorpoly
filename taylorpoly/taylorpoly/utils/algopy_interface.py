import algopy
import numpy


def extract(x,p,d):
    """
    Extracts the d'th coefficients from a numpy.ndarray x of dtype object (UTPS)
    and returns an numpy.ndarray of dtype=float.
    """
    shp = numpy.shape(x)
    
    xr = numpy.ravel(x)
    y = numpy.array([xn.data[(d>0)*(d + p*(xr[0].D-1))] for xn in xr], dtype=float)
    y = y.reshape(shp)
    return y


def taylorpoly_utps2algopy_utpm(x):
    """
    converts a numpy array of taylorpoly.UTPS instances into a an algopy.UTPM instance
    """
    
    shp = numpy.shape(x)
    
    if not isinstance(shp,tuple):
        shp = (shp,)
    
    xr = numpy.ravel(x)
    P,D = xr[0].P,xr[0].D
    out = algopy.UTPM(numpy.zeros( (D,P) + shp))
    
    out.data[0,:] = extract(x, 0, 0)
    
    for d in range(1,D):
        for p in range(P):
            out.data[d,p] = extract(x, p, d)
    
    return out
    
        
