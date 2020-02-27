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
def add_cpx_noise(I, mask=[], sigma=[], SNR=20, ref=0, recover_noise=False):

  # Standard deviation
  if not sigma:
    pos = mask[...,ref].astype(np.bool_)
    I0_ref = I[...,0,ref]
    I1_ref = I[...,1,ref]
    mu = 0.5*(np.abs(I0_ref) + np.abs(I1_ref))
    mean  = np.mean(mu[pos])
    sigma = np.sqrt(2)*mean/SNR
  else:
    peak = max([np.abs(I[...,0,0]).max(), np.abs(I[...,1,0]).max()])
    sigma = sigma*peak

  # Check mask argument
  if (mask is None) or (mask is []):
      mask = 1

  # Noise generation and addition
  noise = np.random.normal(0, sigma, I.shape) + 1j*np.random.normal(0, sigma, I.shape)
  In = I + noise*mask

  if recover_noise:
    return In, noise
  else:
    return In