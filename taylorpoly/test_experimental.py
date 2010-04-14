from numpy.testing import *
import numpy
from taylorpoly import *

P,D,N,M = 1,2,3,2
x = UTPM(numpy.random.rand((P*(D-1)+1)*M*N), shape = (M,N), P = P)

# x0 = x.coeff[0,0].copy()
# x1 = x.coeff[0,1].copy()

z = transpose(x)

print x
z.data *= 0
print x


# z0 = z.coeff[0,0].copy()
# z1 = z.coeff[0,1].copy()

# assert_equal(id(x.data), id(z.data))
# assert_array_almost_equal(x0.T, z0)
# assert_array_almost_equal(x1.T, z1)

# print x._shape
# print len(x.data)
# print len(x.data[numpy.prod(x._shape):])
# print x.P * (x.D - 1) * numpy.prod(x._shape[::-1])
# x.data[numpy.prod(x._shape):].reshape((x.P,x.D-1) + tuple(x._shape[::-1])).transpose((0,1,3,2))
# print x

# y = view(x)
# z = transpose(x)
# print z.shape
# print z.strides


# v = as_strided(numpy.arange(3*5*7), shape = (3,5,7), strides = (8,3*8,3*8*5))

# print v

# print x.coeff[0,0]
# z.coeff[0,0] += 30
# print x.coeff[0,0]
# print z
# print x._shape

# print x.strides
# print x.ndim
# print x.shape



# transpose(x, (1,0))
# transpose(x)


# print x.strides
# print x.ndim
# print x.shape
