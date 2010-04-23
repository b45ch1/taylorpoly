from numpy.testing import *
import numpy
import scipy.linalg as linalg
from taylorpoly import *
from taylorpoly.utpm import *
from taylorpoly.utils import *

P,D,N,M = 1,3,3,20
A = UTPM(numpy.random.rand((P*(D-1)+1)*N*N), shape = (N,N), P = P)
B = lu(A).coeff[0,0]

# print B.coeff[0,0]
A = numpy.random.rand(3,3)
P,L,U = linalg.lu(A)
C = numpy.dot(P.T,A)

F = L + U
for n in range(N):
    F[n,n] -= 1.

# print numpy.dot(P.T,P)

assert_array_almost_equal(C, numpy.dot(L,U))

A = l_times_u(1., numpy.zeros((3,3)), L,U)
B = l_times_u(1., numpy.zeros((3,3)), F,F)


print A - numpy.dot(L,U)
print B - numpy.dot(L,U)


# print A
# print L
# print U

# C2 = numpy.dot(P,A.coeff[0,1])

# print C - numpy.dot(L,U)

# print numpy.dot(L,U)





# print A-B
