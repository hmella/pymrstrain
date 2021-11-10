import numpy as np


# Get tissue and background indicators
def get_segmentation_mask(u_image, tol=1.0e-15):
    '''
    '''
    # Background pixels
    condition = np.abs(u_image) < tol
    tissue      = np.where(~condition)
    background  = np.where(condition)

#    # Get np.pixels with tissue
#    condition = u_image != 0.0
#    tissue      = np.where(condition)
#    background  = np.where(~condition)
#
    return tissue, background


# Add complex noise
def add_cpx_noise(I, mask=[], std=[], rstd=[], SNR=20, ref=0, recover_noise=False):

  # Standard deviation
  if not rstd:
    sigma = std
  else:  	   
    peak = max([np.abs(I[...,i,0]).max() for i in range(I.shape[3])])
    sigma = rstd*peak

  # Noise generation and addition
  noise = np.random.normal(0, sigma, I.shape) + 1j*np.random.normal(0, sigma, I.shape)
  In = I + noise*mask

  if recover_noise:
    return In, noise
  else:
    return In
