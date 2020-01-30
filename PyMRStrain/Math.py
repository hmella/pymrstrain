from numpy.fft import fftshift, fftn, ifftn, ifftshift
import numpy as np

def itok(x, axes=None):
    return fftshift(fftn(ifftshift(x,axes=axes),axes=axes),axes=axes)

def ktoi(x, axes=None):
    return fftshift(ifftn(ifftshift(x,axes=axes),axes=axes),axes=axes)

def wrap(x, value):
    return np.mod(x + 0.5*value, value) - 0.5*value