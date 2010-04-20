from numpy.testing import *
import numpy
from taylorpoly import *
from taylorpoly.utpm import *
from taylorpoly.utils import *


P,D,M,K,N = 3,2,5,2,3
x = UTPM(numpy.random.rand(P,D,M,K), shape = (M,K), P = P)
y = UTPM(numpy.random.rand(P,D,K,N), shape = (N,K), P = P).T

z = dot(x,y)

assert_array_almost_equal(z.coeff[0,0], numpy.dot(x.coeff[0,0], y.coeff[0,0]))
for p in range(P):
    assert_array_almost_equal(z.coeff[p,1], numpy.dot(x.coeff[p,1], y.coeff[p,0]) + numpy.dot(x.coeff[p,0], y.coeff[p,1]))



