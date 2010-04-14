"""
A collection of useful functions that are either missing in numpy and workarounds.

Parts of it have been adapted from:
    * http://projects.scipy.org/numpy/browser/trunk/numpy/lib/stride_tricks.py
"""

import numpy

__all__ = ['as_strided']

class DummyArray(object):
    """ Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """
    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base

def as_strided(x, shape=None, strides=None):
    """ Make an ndarray from the given array with the given shape and strides.
    """
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)
    return numpy.asarray(DummyArray(interface, base=x)) 
