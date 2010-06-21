"""
Implementation of truncated Taylor Polynomial algorithms, i.e. of the algebraic
structure:

    R[t]/ t^D R[t],
    
where R[t] the ring of formal power series and t^D R[t] an ideal.
The algorithms add, sub, mul, div, etc.  are the fundamental building blocks of the
forward mode of AD using Taylor polynomials.

The module also features a class UTPS (Univariate Taylor Polynomial over Scalars) that uses
operator overloading to allow convenient arithmetic operations on the algebraic class, i.e.
to perform operations like for example `z = sin(x * y)` in Taylor arithmetic.

This module also implements the operations that are necessary for the reverse mode
of AD as it is implemented in the C++ AD tool ADOL-C.

"""


from taylorpoly.utps import UTPS, add, sub, mul, div, amul, sqrt, log, exp, pow, sin_cos, sin, cos
from taylorpoly.utps_epb import epb_add, epb_sub, epb_mul, epb_div
#from taylorpoly.utils import *

#from taylorpoly.utpm import UTPM, add, sub, mul, dot, solve, transpose, view

