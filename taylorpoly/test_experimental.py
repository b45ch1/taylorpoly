from numpy.testing import *
import numpy
from taylorpoly import *





P,D,N,M = 3,3,6,3
A = UTPM(numpy.random.rand((P*(D-1)+1)*N*N), shape = (N,N), P = P)
b = UTPM(numpy.random.rand((P*(D-1)+1)*N*M), shape = (N,M), P = P)

x = solve(A,b)

assert_array_almost_equal(dot(A,x).data,b.data)

