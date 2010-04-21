from numpy.testing import TestCase, assert_array_almost_equal, assert_almost_equal, assert_equal
import numpy

from taylorpoly.utpm import UTPM, solve, add, sub, mul, dot, transpose, dot_residual, dot2

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

    def test_mul(self):
        P,D,N,M = 4,3,3,2
        x = UTPM(numpy.random.rand((P*(D-1)+1)*M*N), shape = (M,N), P = P)
        y = UTPM(numpy.random.rand((P*(D-1)+1)*M*N), shape = (M,N), P = P)
        z = mul(x,y)


        # for p in range(P):
        #     print 'p=',p
        #     print z.coeff[p,0] - x.coeff[p,0] * y.coeff[p,0]
        #     print z.coeff[p,1] - x.coeff[p,0] * y.coeff[p,1] - x.coeff[p,1] * y.coeff[p,0]
        #     print z.coeff[p,2] - x.coeff[p,0] * y.coeff[p,2] - x.coeff[p,1] * y.coeff[p,1] - x.coeff[p,2] * y.coeff[p,0]
        
        for p in range(P):
            assert_array_almost_equal( z.coeff[p,0], x.coeff[p,0] * y.coeff[p,0])
            assert_array_almost_equal( z.coeff[p,1],  x.coeff[p,0] * y.coeff[p,1] + x.coeff[p,1] * y.coeff[p,0])
            assert_array_almost_equal( z.coeff[p,2],  x.coeff[p,0] * y.coeff[p,2] + x.coeff[p,1] * y.coeff[p,1] + x.coeff[p,2] * y.coeff[p,0])
        
        

    def test_dot(self):
        P,D,M,K,N = 3,2,5,2,3
        x = UTPM(numpy.random.rand(P,D,M,K), shape = (M,K), P = P)
        y = UTPM(numpy.random.rand(P,D,K,N), shape = (K,N), P = P)
        
        z = dot(x,y)
        
        assert_array_almost_equal(z.coeff[0,0], numpy.dot(x.coeff[0,0], y.coeff[0,0]))
        for p in range(P):
            assert_array_almost_equal(z.coeff[p,1], numpy.dot(x.coeff[p,1], y.coeff[p,0]) + numpy.dot(x.coeff[p,0], y.coeff[p,1]))
    
    def test_dot_transposed(self):
        P,D,M,K,N = 3,2,5,2,3
        x = UTPM(numpy.random.rand(P,D,M,K), shape = (M,K), P = P)
        y = UTPM(numpy.random.rand(P,D,K,N), shape = (N,K), P = P).T
        
        z = dot(x,y)
        
        assert_array_almost_equal(z.coeff[0,0], numpy.dot(x.coeff[0,0], y.coeff[0,0]))
        for p in range(P):
            assert_array_almost_equal(z.coeff[p,1], numpy.dot(x.coeff[p,1], y.coeff[p,0]) + numpy.dot(x.coeff[p,0], y.coeff[p,1]))

    def test_dot2(self):
        P,D,M,K,N = 3,2,5,2,3
        x = UTPM(numpy.random.rand(P,D,M,K), shape = (M,K), P = P)
        y = UTPM(numpy.random.rand(P,D,K,N), shape = (K,N), P = P)
                                     
        z = dot2(x,y)
        
        assert_array_almost_equal(z.coeff[0,0], numpy.dot(x.coeff[0,0], y.coeff[0,0]))
        for p in range(P):
            assert_array_almost_equal(z.coeff[p,1], numpy.dot(x.coeff[p,1], y.coeff[p,0]) + numpy.dot(x.coeff[p,0], y.coeff[p,1]))
            
    
        x = UTPM(numpy.random.rand(P,D,M,K), shape = (M,K), P = P)
        y = UTPM(numpy.random.rand(P,D,K,N), shape = (N,K), P = P).T
        
        z = dot2(x,y)
        
        assert_array_almost_equal(z.coeff[0,0], numpy.dot(x.coeff[0,0], y.coeff[0,0]))
        for p in range(P):
            assert_array_almost_equal(z.coeff[p,1], numpy.dot(x.coeff[p,1], y.coeff[p,0]) + numpy.dot(x.coeff[p,0], y.coeff[p,1]))
                    
        x = UTPM(numpy.random.rand(P,D,M,K), shape = (K,M), P = P).T
        y = UTPM(numpy.random.rand(P,D,K,N), shape = (K,N), P = P)
        
        z = dot2(x,y)
        
        assert_array_almost_equal(z.coeff[0,0], numpy.dot(x.coeff[0,0], y.coeff[0,0]))
        for p in range(P):
            assert_array_almost_equal(z.coeff[p,1], numpy.dot(x.coeff[p,1], y.coeff[p,0]) + numpy.dot(x.coeff[p,0], y.coeff[p,1]))

        x = UTPM(numpy.random.rand(P,D,M,K), shape = (K,M), P = P).T
        y = UTPM(numpy.random.rand(P,D,K,N), shape = (N,K), P = P).T
        
        z = dot2(x,y)
        
        assert_array_almost_equal(z.coeff[0,0], numpy.dot(x.coeff[0,0], y.coeff[0,0]))
        for p in range(P):
            assert_array_almost_equal(z.coeff[p,1], numpy.dot(x.coeff[p,1], y.coeff[p,0]) + numpy.dot(x.coeff[p,0], y.coeff[p,1]))
    

    def test_dot_residual(self):
        P,D,N,M = 3,4,3,2
        x = UTPM(numpy.random.rand((P*(D-1)+1)*M*N), shape = (M,N), P = P)
        y = UTPM(numpy.random.rand((P*(D-1)+1)*M*N), shape = (N,M), P = P)
        
        for d in range(D):
            for p in range(P):
                tmp = numpy.zeros((M,M))
                for k in range(1,d):
                    tmp +=  numpy.dot(x.coeff[p,d-k],  y.coeff[p,k])
                assert_array_almost_equal(tmp, dot_residual(p, d, x,y))

    def test_solve(self):
        P,D,N,M = 3,3,6,3
        A = UTPM(numpy.random.rand((P*(D-1)+1)*N*N), shape = (N,N), P = P)
        b = UTPM(numpy.random.rand((P*(D-1)+1)*N*M), shape = (N,M), P = P)
        
        x = solve(A,b)
        
        assert_array_almost_equal(dot(A,x).data,b.data)
        

class Test_UTPM_methods(TestCase):
    
    def test_constructor(self):
        P,D,M,N = 5,3,2,7
        tmp = numpy.arange( M*N*(1+ (D-1)*P))
        A = UTPM(tmp , shape = (M,N), P = P)
        
        assert_equal(P, A.P)
        assert_equal(D, A.D)
        
        assert_array_almost_equal(tmp, A.data)
        
    def test_coeff(self):
        P,D,M,N = 1,3,2,7
        tmp = numpy.arange( M*N*(1+ (D-1)*P)).reshape((D,M,N))
        A = UTPM(tmp)
        
        for d in range(D):
            assert_array_almost_equal(tmp[d], A.coeff[0,d])
    
    def test_transpose(self):
        P,D,N,M = 1,2,3,2
        x = UTPM(numpy.random.rand((P*(D-1)+1)*M*N), shape = (M,N), P = P)
        y = transpose(x)
        
        for p in range(P):
            for d in range(D):
                assert_array_almost_equal(x.coeff[p,d].T, y.coeff[p,d])

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



