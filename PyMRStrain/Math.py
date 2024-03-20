import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift

MEGA = 1e+06
KILO = 1e+03
MILI = 1e-02
MICRO = 1e-06

def itok(x, axes=None):
  if axes is None:
    axes = [i for i in range(len(x.shape)) if i < 3]
  return fftshift(fftn(ifftshift(x, axes=axes), axes=axes), axes=axes)

def ktoi(x, axes=None):
  if axes is None:
    axes = [i for i in range(len(x.shape)) if i < 3]
  return fftshift(ifftn(ifftshift(x, axes=axes), axes=axes), axes=axes)

def wrap(x, value):
  return np.mod(x + 0.5*value, value) - 0.5*value

def Rx(tx):
  return np.array([[1, 0, 0],
                    [0, np.cos(tx), -np.sin(tx)],
                    [0, np.sin(tx), np.cos(tx)]])

def Ry(ty):
  return np.array([[np.cos(ty), 0, np.sin(ty)],
                    [0, 1, 0],
                    [-np.sin(ty), 0, np.cos(ty)]])

def Rz(tz):
  return np.array([[np.cos(tz), -np.sin(tz), 0],
                    [np.sin(tz), np.cos(tz), 0],
                    [0, 0, 1]])