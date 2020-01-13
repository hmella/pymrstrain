from numpy.fft import fftshift, fftn, ifftn, ifftshift

def FFT(x):
  return fftshift(fftn(ifftshift(x)))

def iFFT(x):
  return fftshift(ifftn(ifftshift(x)))
