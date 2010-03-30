from numpy.testing import TestCase, assert_array_almost_equal, assert_almost_equal, assert_equal
import numpy

from taylorpoly.utpm import UTPM, solve, add, sub, dot

class test_global_functions(TestCase):
    def test_add(self):
        P,D,N = 3,4,3
        x = UTPM(numpy.random.rand(P,D,N,N), shape = (N,N))
        y = UTPM(numpy.random.rand(P,D,N,N), shape = (N,N))
        z = add(x,y)
        assert_array_almost_equal(z.data, x.data + y.data)

    def test_sub(self):
        P,D,N = 3,4,3
        x = UTPM(numpy.random.rand(P,D,N,N), shape = (N,N))
        y = UTPM(numpy.random.rand(P,D,N,N), shape = (N,N))
        z = sub(x,y)
        assert_array_almost_equal(z.data, x.data - y.data)

    def test_dot(self):
        P,D,M,K,N = 3,2,5,2,3
        x = UTPM(numpy.random.rand(P,D,M,K), shape = (M,K), P = P)
        y = UTPM(numpy.random.rand(P,D,K,N), shape = (K,N), P = P)
        
        z = dot(x,y)
        
        assert_array_almost_equal(z.coeff[0,0], numpy.dot(x.coeff[0,0], y.coeff[0,0]))
        for p in range(P):
            assert_array_almost_equal(z.coeff[p,1], numpy.dot(x.coeff[p,1], y.coeff[p,0]) + numpy.dot(x.coeff[p,0], y.coeff[p,1]))
        
    def test_solve(self):
        P,D,N,M = 3,3,6,3
        A = UTPM(numpy.random.rand((P*(D-1)+1)*N*N), shape = (N,N), P = P)
        b = UTPM(numpy.random.rand((P*(D-1)+1)*N*M), shape = (N,M), P = P)
        
        x = solve(A,b)
        
        assert_array_almost_equal(dot(A,x).data,b.data)
        

class Test_UTPM_methods(TestCase):
    
    def test_constructor(self):
        P,D,N = 1,1,3
        A = UTPM(numpy.random.rand(P,D,N,N), shape = (N,N))
        b = UTPM(numpy.random.rand(P,D,N,1), shape = (N,1))
        # print A
        
        # x = solve(A,b)
        
        # print A.data
        # print b.data
        # print x.data
        
        # print numpy.dot(A.data.reshape((N,N)), x.data) - b.data
    
    

        
        
    # def test_add(self):
    #     P,D,N = 3,7,3
    #     x = UTPM(numpy.random.rand(P,D,N), shape = (N,))
    #     y = UTPM(numpy.random.rand(P,D,N), shape = (N,))
        
    #     z = add(x,y)
    #     assert_array_almost_equal(z.data, x.data + y.data)
        
        


if __name__ == "__main__":
    run_module_suite()



