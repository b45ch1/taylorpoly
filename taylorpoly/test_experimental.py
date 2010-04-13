from numpy.testing import *
import numpy
from taylorpoly import *

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

