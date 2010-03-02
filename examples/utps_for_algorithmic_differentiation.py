"""
This example shows how Taylor arithmetic can be used to compute derivatives of functions.
To be more specific, we compute here the gradient of a computer program that contains a
for loop in the so-called forward mode of Algorithmic Differentiation. 
"""

import numpy

from taylorpoly.utps import UTPS, extract

def f(x, p, y ):
    """ some arbitrary function """
    N = numpy.size(x)

    for n in range(1,N):
        y[n] += numpy.sin(x[n-1])**2.3
        y[n] *= p
        
    return y

# finite differences
delta = numpy.sqrt(2**-53)

x = numpy.random.rand(100)
y = numpy.zeros(100)
p = 3.
y1 = f(x, p, y)

y = numpy.zeros(100)
y2 = f(x, p + delta, y)

g_fd = (y2 - y1)/delta

# compute on UTPS
x = numpy.array([ UTPS([xn,0.]) for xn in x])
y = numpy.array([ UTPS([0,0]) for xn in x])
p = UTPS([p,1.])
y = f(x, p, y)

g_ad = extract(y,0,1)

print 'comparing FD gradient g_fd and AD gradient g_ad'
print g_fd - g_ad



