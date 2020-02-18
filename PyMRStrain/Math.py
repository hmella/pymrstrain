import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift

MEGA = 1e+06
KILO = 1e+03
MILI = 1e-02
MICRO = 1e-06

def itok(x, axes=[0,1]):
    return fftshift(fftn(ifftshift(x,axes=axes),axes=axes),axes=axes)

def ktoi(x, axes=[0,1]):
    return fftshift(ifftn(ifftshift(x,axes=axes),axes=axes),axes=axes)

def wrap(x, value):
    return np.mod(x + 0.5*value, value) - 0.5*value
