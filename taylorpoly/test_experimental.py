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



# print A
# print L
# print U

# C2 = numpy.dot(P,A.coeff[0,1])

# print C - numpy.dot(L,U)

# print numpy.dot(L,U)





# print A-B
