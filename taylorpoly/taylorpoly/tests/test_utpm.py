from numpy.testing import TestCase, assert_array_almost_equal, assert_almost_equal, assert_equal
import numpy

from taylorpoly.utpm import UTPM, solve

class Test_UTPM_methods(TestCase):
    def test_constructor(self):
        P,D,N = 1,1,3
        A = UTPM(numpy.random.rand(P,D,N,N), shape = (N,N))
        b = UTPM(numpy.random.rand(P,D,N,1), shape = (N,1))
        # print A
        
        x = solve(A,b)
        
        print A.data
        print b.data
        print x.data
        
        print numpy.dot(A.data.reshape((N,N)), x.data) - b.data
        
        


if __name__ == "__main__":
    run_module_suite()



