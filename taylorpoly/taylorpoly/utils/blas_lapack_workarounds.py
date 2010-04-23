import numpy

from taylorpoly.utpm import _utpm, c_double_ptr

def l_times_u(alpha, A,L,U):
    """ computes A := A + alpha * L * U, where L lower triangular and U upper triangular,
        for A,L,U (N,N) arrays in column major memory layout.
    """
    N = A.shape[0]
    
    A = numpy.asfortranarray(A)
    L = numpy.asfortranarray(L)
    U = numpy.asfortranarray(U)
    
    _utpm.l_times_u(N, alpha,
        A.ctypes.data_as(c_double_ptr), N,
        L.ctypes.data_as(c_double_ptr), N,
        U.ctypes.data_as(c_double_ptr), N)
    
    return A 
