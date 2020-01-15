from numpy.fft import fftshift, fftn, ifftn, ifftshift

def FFT(x, axes=None):
  return fftshift(fftn(ifftshift(x,axes=axes),axes=axes),axes=axes)

def iFFT(x, axes=None):
  return fftshift(ifftn(ifftshift(x,axes=axes),axes=axes),axes=axes)
