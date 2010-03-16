import numpy
from taylorpoly.utps import UTPS, exp

x = UTPS([5.,1.,0.], P = 2)
y = UTPS([3.,0.,1.], P = 2)

v1 = x * y
v2 = exp(v1)
f = v2 + x

print 'forward mode AD gradient = [ %f, %f] '%(f.data[1], f.data[2])
print 'symbolic        gradient = [ %f, %f] '%(y.data[0]*numpy.exp(x.data[0]) + 1, x.data[0]*numpy.exp(y.data[0]))
    
     
