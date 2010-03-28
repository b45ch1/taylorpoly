"""
This module provides algorithms for Univariate Taylor Propagation over Matrices (UTPM)
in the forward and reverse mode of Algorithmic Differentiation (AD).

More specifically, algorithms are provided that take Univariate Taylor Polynomials
over Matrices ( also abbreviated UTPM) as inputs and have UTPM instances as output.

For convenience, a class UTPM is provided that wraps the data structure of the
univariate Taylor polynomials over matrices.

The algorithms take objects of the class UTPM as input. The class UTPM is a simple
wrapper of the data structure.

"""


import os
import ctypes
import numpy

_utpm = numpy.ctypeslib.load_library('libutpm.so', os.path.dirname(__file__))

c_double_ptr = ctypes.POINTER(ctypes.c_double)
c_int_ptr    = ctypes.POINTER(ctypes.c_int)
c_int        = ctypes.c_int
c_double     = ctypes.c_double

_utpm.utpm_daxpy.argtypes = [c_int, c_int, c_int, c_double, c_double_ptr, c_int, c_double_ptr, c_int]
_utpm.utpm_dgesv.argtypes = [c_int, c_int, c_int, c_int, c_int, c_double_ptr, c_int, c_int_ptr, c_double_ptr, c_int]

class UTPM:
    """
    UTPM = Univariate Taylor Polynomial over Matrices
    Implements the the matrices over R[[t]]/ R[[t]]t^{D},
    where R[[t]] are the formal power series over the field of real numbers R.
    """
    
    def __init__(self, data, shape = (), P = 1):
        """
        data = [x_0, x_{1,1}, x_{1,2}, ..., x_{1,P}, ..., x_{D-1,P}]
        is a flat, contiguous array of
        x_{d,p} is an array of arbitrary shape shp.
        
        
        For convenience, if P = 1, it is possible to provide data as a
        (P,D) + X.shape
        
        """
        
        if P != 1 or shape == ():
            raise NotImplementedError('should implement that')
        
        tmp = numpy.prod(shape)
        D = (numpy.size(data)//tmp-1)//P + 1
        
        if not isinstance(shape,tuple):
            shape = (shape,)
        
        self._shape = shape
        self.D = D
        self.P = P
        self.data = numpy.ravel(data)
        
        
    def __str__(self):
        """ return human readable string representation"""
        ret_str =  'zero coefficient, d = 0:\n'
        ret_str += str(self.data[:numpy.prod(self._shape)].reshape(self._shape)) + '\n'
        ret_str += 'higher order coefficients, d >0:\n'
        ret_str += str(self.data[numpy.prod(self._shape):].reshape((self.P,self.D-1) + self._shape))
        return ret_str
        
    def __repr__(self):
        """ return data that allows to reconstruct the instance"""
        ret_str = 'UTPM(%s, %s, %d)'%(str(self.data), str(self._shape), self.P)

    def copy(self):
        """ copies all data in self to a new instance """
        return self.__class__(self.data.copy(), shape = self._shape, P = self.P)
        
    def __zeros_like__(self):
        """ returns a copy of self with all elements set to zero"""
        return self.__class__(numpy.zeros_like(self.data), shape = self._shape, P = self.P)


def add(x,y, out = None):
    """ computes z = x+y in Taylor arithmetic
    
    if out = y, then  y = x + y is computed.
    """
    if out == None:
        out = y.copy()
        
    if id(x) == id(y) and id(y) == id(out):
        out.data *= 2
        return out
    
    P,D,N = x.P, x.D, x._shape[0]
    
    # int P, int D, const int N, const double alpha, const double *x,
    #              const int incx, double *y, const int incy
    _utpm.utpm_daxpy(P, D, N, 1., x.data.ctypes.data_as(c_double_ptr), 1,
    out.data.ctypes.data_as(c_double_ptr), 1)
    
    return out


def cauchy_product(d,A,B, out = None):
    P,D,N,K = A.P, A.D, A._shape
    K,M = B._shape
    
    if out == None:
        out = UTPM( numpy.zeros(N*M*P*(D-1)+N*M), P = P, shape = (N,M))
    

def solve(A,B):
    """
    solves A X = B in Taylor arithmetic
    """
    
    A = A.copy()
    B = B.copy()
    
    P,D = A.P,A.D
    order = 101 # row major
    
    N = A._shape[0]
    NRHS = B._shape[1]
    lda = N
    ipiv = numpy.zeros(N,dtype=int)
    ldb = N

# utpm_dgesv(int P, int D, const enum CBLAS_ORDER Order,
#                   const int N, const int NRHS,
#                   double *A, const int lda, int *ipiv,
#                   double *B, const int ldb){

    # ipiv.ctypes.data_as(c_int_ptr)
    # A.data.ctypes.data_as(c_double_ptr)
    _utpm.utpm_dgesv(P,D, order, N, NRHS, A.data.ctypes.data_as(c_double_ptr),
        lda, ipiv.ctypes.data_as(c_int_ptr), B.data.ctypes.data_as(c_double_ptr), ldb)
    
    return B
