from numpy.testing import TestCase, assert_array_almost_equal, assert_almost_equal, assert_equal
import numpy
import scipy.linalg

from taylorpoly.utps import *
from taylorpoly.utpm import *
from taylorpoly.utils import *

class Taylorpoly2Algopy(TestCase):
    
    def test_utps_to_utpm(self):
        P, D, M , N = 11,3, 1,1
        x = numpy.array([UTPS(numpy.random.rand(P*(D-1) + 1), P = P) for i  in range(M*N)]).reshape((M,N))
        y = taylorpoly_utps2algopy_utpm(x)
        
        for p in range(P):
            # print 'p=',p
            for d in range(D):
                # print 'd=',d
                for m in range(M):
                    for n in range(N):
                        assert_array_almost_equal(x[m,n].coeff[p,d], y.data[d,p,m,n])


if __name__ == "__main__":
    run_module_suite()
 
