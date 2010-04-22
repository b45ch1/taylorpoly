from numpy.testing import *
import numpy
from taylorpoly import *
from taylorpoly.utpm import *
from taylorpoly.utils import *



P,D,M,K,N = 3,2,5,2,3
x = UTPM(numpy.random.rand(P,D,M,K), shape = (M,K), P = P)

print x.strides
print x.allstrides
