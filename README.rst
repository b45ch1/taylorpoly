Description:

    This project hosts an ANSI C implementation of vectorized truncated Taylor polynomial operations. The underlying data structure is a simple array 
    `[x]_{D,P} := [x_0, x_{1,1},...,x_{1,D-1},x_{2,1},...,x_{P,D-1}]`.

    We believe the algorithms would be a nice addition to numpy.
    One should think of [x]_{D,P} as a data type, just as the real numbers or complex numbers.
    

Rationale:

    The algebraic structure "truncated Taylor polynomials" play a fundamental role, e.g. for higher order algorithmic differentiation algorithms or Taylor polynomial integrators.
    For efficiency, portability and other reasons the algorithms are implemented in ANSI C and wrapped by python.ctypes using the numpy.ctypes interface.


Licence:
    BSD