from numpy.testing import *
import numpy
from taylorpoly import *
from taylorpoly.utpm import *
from taylorpoly.utils import *

P,D,N,M = 2,2,2,1
A = UTPM(numpy.random.rand((P*(D-1)+1)*N*N), shape = (N,N), P = P)
b = UTPM(numpy.random.rand((P*(D-1)+1)*N*M), shape = (N,M), P = P)

# x = solve2(A,b)

print b
print b.data
print b._strides
print b.strides
# print x


# assert_array_almost_equal(dot(A,x).data,b.data)
