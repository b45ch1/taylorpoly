from numpy.testing import *
import numpy
from taylorpoly import *






# P,D,N = 1,2,3
# A = UTPM(numpy.random.rand(P,D,N,N), shape = (N,N))
# b = UTPM(numpy.random.rand(P,D,N,1), shape = (N,1))

# b.coeff(0,1) *= 0
# tmp = b.coeff[0,1]
# tmp *= 0

# A.coeff[0,1] *= 0
# b.coeff[0,1] *= 0

# print b.data.shape

# x = solve(A,b)

# print x.coeff[0,1]
# print numpy.linalg.solve(A.coeff[0,0], b.coeff[0,1])
# print numpy.dot(A.coeff[0,1], x.coeff[0,0])

# print x.coeff[0,1]

# print numpy.dot(A.coeff[0,0], x.coeff[0,0]) - b.coeff[0,0]
# print numpy.dot(A.coeff[0,1], x.coeff[0,1]) - b.coeff[0,1]
