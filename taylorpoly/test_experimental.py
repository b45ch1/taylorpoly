from numpy.testing import *
import numpy
from taylorpoly import *





P,D,N,M = 3,3,6,3
x = UTPM(numpy.random.rand((P*(D-1)+1)*N*N), shape = (N,N), P = P)
y = UTPM(numpy.random.rand((P*(D-1)+1)*N*N), shape = (N,N), P = P)

mul(x,y)

