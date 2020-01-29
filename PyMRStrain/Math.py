from numpy.fft import fftshift, fftn, ifftn, ifftshift

def itok(x, axes=None):
  return fftshift(fftn(ifftshift(x,axes=axes),axes=axes),axes=axes)

def ktoi(x, axes=None):
  return fftshift(ifftn(ifftshift(x,axes=axes),axes=axes),axes=axes)
