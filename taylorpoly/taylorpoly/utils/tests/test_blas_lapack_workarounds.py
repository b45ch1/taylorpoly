from numpy.testing import TestCase, assert_array_almost_equal, assert_almost_equal, assert_equal
import numpy
import scipy.linalg

from taylorpoly.utpm import *
from taylorpoly.utils import *

class test_helper_functions(TestCase):
    
    def test_l_times_u(self):
        N = 10
        A = numpy.random.rand(N,N)
        P,L,U = scipy.linalg.lu(A)
        C = numpy.dot(P.T,A)
        
        F = L + U
        for n in range(N):
            F[n,n] -= 1.
        
        assert_array_almost_equal(C, numpy.dot(L,U))
        A = l_times_u(1., numpy.zeros((N,N)), L,U)
        B = l_times_u(1., numpy.zeros((N,N)), F,F)
        
        assert_array_almost_equal(A, numpy.dot(L,U))
        assert_array_almost_equal(B, numpy.dot(L,U))


if __name__ == "__main__":
    run_module_suite()

