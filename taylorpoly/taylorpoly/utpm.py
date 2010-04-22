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

import utils

_utpm = numpy.ctypeslib.load_library('libutpm.so', os.path.dirname(__file__))

c_double_ptr = ctypes.POINTER(ctypes.c_double)
c_int_ptr    = ctypes.POINTER(ctypes.c_int)
c_int        = ctypes.c_int
c_double     = ctypes.c_double

_utpm.imul.argtypes = [c_int, c_int, c_double, c_double_ptr, c_int, c_double_ptr, c_int, c_double_ptr, c_int]
_utpm.utpm_imul.argtypes = [c_int, c_int, c_int, c_int, c_double_ptr, c_int, c_double_ptr, c_int]


_utpm.utpm_dgemm.argtypes = [c_int, c_int, c_int,  c_int, c_int,  c_int, c_int, c_int, c_double, c_double_ptr, c_int, c_double_ptr, c_int, c_double, c_double_ptr, c_int]
_utpm.utpm_dgemm_residual.argtypes = [c_int, c_int, c_int, c_int,  c_int, c_int,  c_int, c_int, c_int, c_double, c_double_ptr, c_int, c_double_ptr, c_int, c_double, c_double_ptr, c_int]
_utpm.utpm_daxpy.argtypes = [c_int, c_int, c_int, c_double, c_double_ptr, c_int, c_double_ptr, c_int]
_utpm.utpm_dgesv.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_double_ptr, c_int, c_int_ptr, c_double_ptr, c_int]


_utpm.utpm_dot.argtypes = [c_int, c_int, c_int, c_int, c_int, c_double, c_double_ptr,
c_int_ptr, c_double_ptr, c_int_ptr, c_double, c_double_ptr, c_int_ptr]

_utpm.utpm_solve.argtypes = [c_int, c_int, c_int, c_int, c_int_ptr, c_double_ptr, c_int_ptr, c_double_ptr, c_int_ptr]
# int utpm_solve(int P, int D, int N, int NRHS, int *ipiv, double *A, int *Astrides,
#                   double *B, int *Bstrides)


class UTPM:
    """
    UTPM = Univariate Taylor Polynomial over Matrices
    Implements the the matrices over R[[t]]/ R[[t]]t^{D},
    where R[[t]] are the formal power series over the field of real numbers R.
    """
    
    def __init__(self, data, shape = (), P = 1, allstrides = None):
        """
        data = [x_0, x_{1,1}, x_{1,2}, ..., x_{1,P}, ..., x_{P,D-1}]
        is a flat, contiguous array of
        x_{d,p} is an array of arbitrary shape shp in column major memory layout.
        
        
        For convenience, if P = 1, it is possible to provide data as a
        (P,D) + X.shape
        
        """
        
        data = numpy.asarray(data)
        
        if shape == ():
            if P != 1:
                raise ValueError('the general case with P>1 only works by setting the shape manually')
            
            if numpy.ndim(data) != 3:
                err_str = 'data should be a 3D array\n'
                err_str += 'but provided data.ndim = %d\n'%numpy.ndim(data)
                raise ValueError(err_str)
                
            shp = numpy.shape(data)
            self.D = shp[0]
            self.P = P
            self._shape = numpy.array(shp[1:],dtype=int)
            self.data = numpy.ravel(data.transpose((0,2,1)))
            
        else:
            tmp = numpy.prod(shape)
            D = (numpy.size(data)//tmp-1)//P + 1
            
            self._shape = numpy.array(shape,dtype=int)
            self.D = D
            self.P = P
            self.data = numpy.ravel(data)
            
        self._Dstride = 8*numpy.prod(self._shape)
        self._strides = 8*numpy.array([numpy.prod(self._shape[:i]) for i in range(self.ndim)], dtype=int)
        
        if allstrides != None:
            self._Dstride = allstrides[0]
            self._strides = allstrides[1:].copy()
        
        self.coeff = self.Coeff(self)
    
    @property
    def is_transposed(self):
        return (self.strides[-1] < self.strides[-2])
    
    @property
    def cblas_transpose_code(self):
        if self.is_transposed:
            return 112
        else:
            return 111
            
    @property
    def cblas_leadim(self):
        if self.is_transposed == True:
            return self._strides[-2] // 8
        else:
            return self._strides[-1] // 8
            
    @property
    def allstrides(self):
        return numpy.asarray( numpy.concatenate(([self._Dstride], self.strides)), dtype=ctypes.c_int)
               
    @property
    def T(self):
        return transpose(self)
        
    def get_ndim(self):
        return len(self._shape)
        
    def get_shape(self):
        return self._shape
        
    def get_strides(self):
        return self._strides
        
    ndim = property(get_ndim)
    shape = property(get_shape)
    strides = property(get_strides)

    class Coeff:
        """
        helper class for UTPM that allows to extract the array of direction p
        and degree d from UTPM.data as numpy array with the correct shape.
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
                return utils.as_strided(self.x.data[:numpy.prod(self.x._shape)], shape = self.x._shape, strides = self.x._strides)
            
            else:
                N = numpy.prod(self.x._shape)
                D = self.x.D
                start = N + p * N * (D-1) + N*(d-1)
                stop  = N + p * N * (D-1) + N*d
                return utils.as_strided(self.x.data[start:stop], shape = self.x._shape, strides = self.x._strides)
                
        def __setitem__(self, sl, value):
            p,d = sl
            if d >= self.x.D:
                raise ValueError('d is too large')
            if p >= self.x.P:
                raise ValueError('p is too large')
                
            if d == 0:
                self.x.data[:numpy.prod(self.x._shape)].reshape(self.x._shape[::-1]).T.__setitem__(Ellipsis, value)
            
            else:
                N = numpy.prod(self.x._shape)
                D = self.x.D
                start = N + p * N * (D-1) + N*(d-1)
                stop  = N + p * N * (D-1) + N*d
                self.x.data[start:stop].reshape(self.x._shape[::-1]).T.__setitem__(Ellipsis, value)

        
    def __str__(self):
        """ return human readable string representation"""
        ret_str =  '['
        ret_str += str(utils.as_strided(self.data[:numpy.prod(self._shape)], shape = self._shape, strides = self._strides)) + '],\n'
        ret_str += '['+ str(self.data[numpy.prod(self._shape):]) + ']'
        return ret_str
        
    def __repr__(self):
        """ return data that allows to reconstruct the instance"""
        ret_str = 'UTPM(%s, %s, %d)'%(str(self.data), str(self._shape), self.P)

    def copy(self):
        """ copies all data in self to a new instance """
        return self.__class__(self.data.copy(), shape = self._shape, P = self.P, allstrides = self.allstrides)
        
    def __zeros_like__(self):
        """ returns a copy of self with all elements set to zero"""
        return self.__class__(numpy.zeros_like(self.data), shape = self._shape, P = self.P)

    def __add__(self, other):
        return add(self, other)
        
    def __sub__(self, other):
        return sub(self, other)
        
    def __mul__(self, other):
        return mul(self, other)
        
    def dot(self, other):
        return dot(self, other)
        
        
def view(x):
    """ returns a copy of the object but no copy of the data"""
    return x.__class__(x.data, shape = x._shape, P = x.P)
        
def transpose(x, axes = None):
    """Permute the dimensions of an UTPM instance.
    
    Parameters
    ----------
    x : UTPM instance
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.
        
    How it works
    ------------
    x.shape   = x.shape[list(axes)]
    x.strides = x.strides[list(axes)]
    """
    
    if axes == None:
        axes = numpy.arange(x.ndim)[::-1]
        
    y = view(x)
    
    y._shape   = y._shape[list(axes)]
    y._strides = y._strides[list(axes)]
        
    return y
    
    


def add(x,y, out = None):
    """ computes z = x+y in Taylor arithmetic
    """
    if out == None:
        out = y.copy()
        
    if id(x) == id(y) and id(y) == id(out):
        out.data *= 2
        return out
    
    P,D,N = x.P, x.D, numpy.prod(x._shape)

    _utpm.utpm_daxpy(P, D, N, 1., x.data.ctypes.data_as(c_double_ptr), 1,
    out.data.ctypes.data_as(c_double_ptr), 1)
    
    return out
    
def sub(x,y, out = None):
    """ computes z = x-y in Taylor arithmetic

    """
    if out == None:
        out = x.copy()
        
    if id(x) == id(y) and id(y) == id(out):
        out.data *= 0
        return out
    
    P,D,N = x.P, x.D, numpy.prod(x._shape)

    _utpm.utpm_daxpy(P, D, N, -1., y.data.ctypes.data_as(c_double_ptr), 1,
    out.data.ctypes.data_as(c_double_ptr), 1)
    
    return out

def mul(x, y, out = None):
    """ computes z = x * y
    """
    
    P,D = x.P, x.D
    M,N = x._shape
    
    if out == None:
        out = y.copy()
    
    _utpm.utpm_imul(P, D, M, N, out.data.ctypes.data_as(c_double_ptr), M,
    x.data.ctypes.data_as(c_double_ptr), M)
    
    return out

    
# def dot(x,y, out = None):
#     """ computes z = dot(x,y) in Taylor arithmetic

#     """
    
#     if len(x._shape) != 2 or len(y._shape) != 2:
#         raise NotImplementedError('only 2d arrays work right now')
        
#     if id(x) == id(y):
#         raise ValueError('x and y may not be the same')
        
#     P,D,M,K,N = x.P, x.D, x._shape[0], x._shape[1], y._shape[1]
    
#     if x._shape[1] != y._shape[0]:
#         raise ValueError('shape of x does not match shape of y')
        
#     if out == None:
#         out = UTPM(numpy.zeros(N*M + (D-1) * N*M * P), P=P, shape = (M,N))
        
        
#     print 'cblas_transpose_code= ',x.cblas_transpose_code, y.cblas_transpose_code

#     A,B,C = x, y, out
#     order = 102 # column major
#     lda, ldb, ldc = x.cblas_leadim, y.cblas_leadim, M
    
#     print lda,ldb
        
#     _utpm.utpm_dgemm(P, D, order, x.cblas_transpose_code, y.cblas_transpose_code, M, N, K, 1., A.data.ctypes.data_as(c_double_ptr),
#         lda, B.data.ctypes.data_as(c_double_ptr), ldb, 0., C.data.ctypes.data_as(c_double_ptr), ldc)
    
#     return out
    
def dot(x,y, out = None):
    """ computes z = dot(x,y) in Taylor arithmetic

    """
    
    if len(x._shape) != 2 or len(y._shape) != 2:
        raise NotImplementedError('only 2d arrays work right now')
        
    if id(x) == id(y):
        raise ValueError('x and y may not be the same')
        
    P,D,M,K,N = x.P, x.D, x._shape[0], x._shape[1], y._shape[1]
    
    if x._shape[1] != y._shape[0]:
        raise ValueError('shape of x does not match shape of y')
        
    if out == None:
        out = UTPM(numpy.zeros(N*M + (D-1) * N*M * P), P=P, shape = (M,N))

    A,B,C = x, y, out
    
    Astrides = A.allstrides
    Bstrides = B.allstrides
    Cstrides = C.allstrides
        
    _utpm.utpm_dot(P, D, M, N, K, 1.,
        A.data.ctypes.data_as(c_double_ptr),Astrides.ctypes.data_as(c_int_ptr),
        B.data.ctypes.data_as(c_double_ptr),Bstrides.ctypes.data_as(c_int_ptr),
        0,
        C.data.ctypes.data_as(c_double_ptr),Cstrides.ctypes.data_as(c_int_ptr))
    
    return out
    
    

def dot_residual(p, d, x,y, out = None):
    """
    x,y are UTPM instances
    
    p, direction
    d, current degree
    """
    
    if len(x._shape) != 2 or len(y._shape) != 2:
        raise NotImplementedError('only 2d arrays work right now')
        
    P,D,M,K,N = x.P, x.D, x._shape[0], x._shape[1], y._shape[1]
    
    if x._shape[1] != y._shape[0]:
        raise ValueError('shape of x does not match shape of y')
    
    if out == None:
        out = numpy.zeros((M,N), order = 'F' )

    A,B,C = x, y, out
    order = 102 # column major
    trans = 111 # no trans
    lda, ldb, ldc = M, K, M
        
    _utpm.utpm_dgemm_residual(p, D, d, order, trans, trans, M, N, K, 1., A.data.ctypes.data_as(c_double_ptr),
        lda, B.data.ctypes.data_as(c_double_ptr), ldb, 0., C.ctypes.data_as(c_double_ptr), ldc)
    
    return out


# def solve(A,B):
#     """
#     solves A X = B in Taylor arithmetic
#     """
    
#     A = A.copy()
#     B = B.copy()
    
#     P,D = A.P,A.D
#     order = 102 # col major 102 # row major 101
#     trans = 111 # no trans
    
#     N = A._shape[0]
#     NRHS = B._shape[1]
#     lda = N
#     ipiv = numpy.zeros(N,dtype=int)
#     ldb = N

#     _utpm.utpm_dgesv(P,D, order,trans, N, NRHS, A.data.ctypes.data_as(c_double_ptr),
#         lda, ipiv.ctypes.data_as(c_int_ptr), B.data.ctypes.data_as(c_double_ptr), ldb)
    
#     return B


def solve(A,B, fulloutput = False):
    """
    solves A X = B in Taylor arithmetic
    """
    
    A = A.copy()
    B = B.copy()
    
    P,D = A.P,A.D
    
    N = A._shape[0]
    NRHS = B._shape[1]
    
    Astrides = A.allstrides
    Bstrides = B.allstrides
    
    ipiv = numpy.zeros(N,dtype=ctypes.c_int)
    
    # print 'Astrides = ', Astrides
    
    _utpm.utpm_solve(P, D, N, NRHS, ipiv.ctypes.data_as(c_int_ptr),
        A.data.ctypes.data_as(c_double_ptr), Astrides.ctypes.data_as(c_int_ptr),
        B.data.ctypes.data_as(c_double_ptr), Bstrides.ctypes.data_as(c_int_ptr))
        
    if(fulloutput == True):
        return (B,A,ipiv)
    else:
        return B
    
    
    
    
    
