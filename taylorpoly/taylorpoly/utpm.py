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



_utpm.utpm_dgemm.argtypes = [c_int, c_int, c_int,  c_int, c_int,  c_int, c_int, c_int, c_double, c_double_ptr, c_int, c_double_ptr, c_int, c_double, c_double_ptr, c_int]

_utpm.utpm_daxpy.argtypes = [c_int, c_int, c_int, c_double, c_double_ptr, c_int, c_double_ptr, c_int]
_utpm.utpm_dgesv.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_double_ptr, c_int, c_int_ptr, c_double_ptr, c_int]





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
        x_{d,p} is an array of arbitrary shape shp in column major memory layout.
        
        
        For convenience, if P = 1, it is possible to provide data as a
        (P,D) + X.shape
        
        """
        
        if shape == ():
            raise NotImplementedError('should implement that')
        
        tmp = numpy.prod(shape)
        D = (numpy.size(data)//tmp-1)//P + 1
        
        if not isinstance(shape,tuple):
            shape = (shape,)
        
        self._shape = shape
        self.D = D
        self.P = P
        self.data = numpy.ravel(data)
        self.coeff = self.Coeff(self)
        
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
                return self.x.data[:numpy.prod(self.x._shape)].reshape(self.x._shape[::-1]).T
            
            else:
                N = numpy.prod(self.x._shape)
                D = self.x.D
                start = N + p * N * (D-1) + N*(d-1)
                stop  = N + p * N * (D-1) + N*d
                return self.x.data[start:stop].reshape(self.x._shape[::-1]).T
                
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
        ret_str += str(self.data[:numpy.prod(self._shape)].reshape(self._shape)) + '],\n'
        ret_str += '['
        ret_str += str(self.data[numpy.prod(self._shape):].reshape((self.P,self.D-1) + self._shape)) + ']'
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

    def __add__(self, other):
        return add(self,other)
        
    def __sub__(self, other):
        return sub(self,other)


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
    
    
def dot(x,y, out = None):
    """ computes z = dot(x,y) in Taylor arithmetic

    """
    
    if len(x._shape) != 2 or len(y._shape) != 2:
        raise NotImplementedError('only 2d arrays work right now')
        
    if id(x) == id(y):
        raise ValueError('x and y may not be the same')
        
    P,D,M,K,N = x.P, x.D, x._shape[0], x._shape[1], y._shape[1]
    
    if out == None:
        out = UTPM(numpy.zeros(N*M + (D-1) * N*M * P), P=P, shape = (M,N))

    A,B,C = x, y, out
    order = 102 # column major
    trans = 111 # no trans
    lda, ldb, ldc = M, K, M
        
    _utpm.utpm_dgemm(P, D, order, trans, trans, M, N, K, 1., A.data.ctypes.data_as(c_double_ptr),
        lda, B.data.ctypes.data_as(c_double_ptr), ldb, 0., C.data.ctypes.data_as(c_double_ptr), ldc)
    
    return out


def solve(A,B):
    """
    solves A X = B in Taylor arithmetic
    """
    
    A = A.copy()
    B = B.copy()
    
    P,D = A.P,A.D
    order = 102 # col major 102 # row major 101
    trans = 111 # no trans
    
    N = A._shape[0]
    NRHS = B._shape[1]
    lda = N
    ipiv = numpy.zeros(N,dtype=int)
    ldb = N

    _utpm.utpm_dgesv(P,D, order,trans, N, NRHS, A.data.ctypes.data_as(c_double_ptr),
        lda, ipiv.ctypes.data_as(c_int_ptr), B.data.ctypes.data_as(c_double_ptr), ldb)
    
    return B
