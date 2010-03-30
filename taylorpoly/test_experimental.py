from numpy.testing import *
import numpy
from taylorpoly import *


P,D,N = 3,3,3
A = UTPM(numpy.random.rand((P*(D-1)+1)*N*N), shape = (N,N), P = 3)
b = UTPM(numpy.random.rand((P*(D-1)+1)*N*1), shape = (N,1), P = 3)

# A.coeff[0,1] *= 0
# b.coeff[0,1] *= 0

x = solve(A,b)

assert_array_almost_equal(numpy.dot(A.coeff[0,0], x.coeff[0,0]), b.coeff[0,0])
assert_array_almost_equal(x.coeff[0,1], numpy.linalg.solve(A.coeff[0,0], b.coeff[0,1] - numpy.dot(A.coeff[0,1],x.coeff[0,0])))
assert_array_almost_equal(dot(A,x).data,b.data)

