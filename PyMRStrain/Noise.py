import numpy as np
from PyMRStrain.MPIUtilities import rank, comm

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


# Add noise to displacement images
def add_noise_to_DENSE(ke, sigma, u, mask, sigma_noise=[0.1, 0.25], wrap=False):

  # Time steps
  n = u.shape[-1]

  # Add noise
  sigma_noise = [sigma*sn for sn in sigma_noise]
  for i in range(n):
    # Get tissue and background indicators
    tissue, background = get_segmentation_mask(mask[...,i])

    # Tissue noise
    sigma = (1.0/ke/2.0/np.pi)*((sigma_noise[1] - sigma_noise[0])*float(i)/(n-1) + sigma_noise[0])
    noise = [sigma*np.random.standard_normal(u[...,0,i].shape) for l in range(u.shape[-2])]

    # Background noise
    e_sigma = (1.0/ke/2.0/np.pi)*float(i)/(0.5*(n-1))
    e_noise = [e_sigma*np.random.uniform(-np.pi, np.pi, u[...,0,i].shape) for l in range(u.shape[-2])]

    # Add Gaussian noise to tissue
    if u.shape[-2] is 2:
      for j in range(u.shape[-2]):
        u[tissue[0],tissue[1],j,i] += (noise[j])[tissue[0],tissue[1]]
    else:
      for j in range(u.shape[-2]):
        u[tissue[0],tissue[1],tissue[2],j,i] += (noise[j])[tissue[0],tissue[1],tissue[2]]

    # Add uniform noise to background
    if u.shape[-2] is 2:
      for j in range(u.shape[-2]):
        u[background[0],background[1],j,i] += (e_noise[j])[background[0],background[1]]
    else:
      for j in range(u.shape[-2]):
        u[background[0],background[1],background[2],j,i] += (e_noise[j])[background[0],background[1],background[2]]

    # Wrap velocity
    if wrap:
      for j in range(u.shape[-2]):
         u[...,j,i] = np.mod(u[...,j,i] + 0.5/ke, 1.0/ke) - 0.5/ke


# Add noise to PC-MR images
def add_noise_to_PC(VENC, sigma, v, mask, wrap=False):
  # Coefficient for uniform distribution and wrapnp.ping
  alpha = VENC/np.pi

  # Check if velocity is below of venc value
  below_venc = [np.all((np.abs(v[...,i,:]) - VENC) < 0.0) for i in range(v.shape[-2])]

  # Standard deviation for velocity
  v_max = np.zeros([v.shape[-2],], dtype=float)
  for i in range(v_max.size):
    if below_venc[i]:
      v_max[i] = np.abs(v[...,i,:]).max()
    else:
      v_max[i] = VENC
  sigma = sigma*v_max

  # Add noise
  for i in range(v.shape[-1]):
    # Get tissue and background indicators
    tissue, background = get_segmentation_mask(mask[...,i])

    # Tissue noise
    noise = [sigma[l]*np.random.normal(0.0, 1.0, v[...,0,i].shape) for l in range(v.shape[-2])]

    # Noise for background spaces
    e_noise = [VENC*np.random.uniform(-1.0, 1.0, v[...,0,i].shape) for l in range(v.shape[-2])]

    # Add Gaussian noise to tissue
    if v.shape[-2] is 2:
      for j in range(v.shape[-2]):
        v[tissue[0],tissue[1],j,i] += (noise[j])[tissue[0],tissue[1]]
    else:
      for j in range(v.shape[-2]):
        v[tissue[0],tissue[1],tissue[2],j,i] += (noise[j])[tissue[0],tissue[1],tissue[2]]

    # Add uniform noise to background
    if v.shape[-2] is 2:
      for j in range(v.shape[-2]):
        v[background[0],background[1],j,i] += (e_noise[j])[background[0],background[1]]
    else:
      for j in range(v.shape[-2]):
        v[background[0],background[1],background[2],j,i] += (e_noise[j])[background[0],background[1],background[2]]

    # Wrap velocity
    if wrap:
      for j in range(v.shape[-2]):
        v[...,j,i] = np.mod(v[...,j,i] + VENC, 2*VENC) - VENC


# Add noise to displacement images
def add_noise_to_SPAMM(T, mask, sigma):
    # Noise standard deviation
    sigma = sigma*np.abs(T).max()

    # Image dimension
    if len(T.shape) is 4:
      d = 3
    else:
      d = 2

    # Add noise
    for i in range(T.shape[-1]):
        # Get tissue and background indicators
        tissue, background = get_segmentation_mask(mask[...,i])

        # Noise
        for j in range(T.shape[-2]):
          mean  = np.zeros([2,])
          cov   = np.eye(2, 2)
          shape = T[...,j,i].shape
          N = sigma*np.random.multivariate_normal(mean, cov, shape)
          Nr, Ni = N[...,0], N[...,1]

          # Add noise
          T[...,j,i] += Nr + 1j*Ni

# Add noise to displacement images
def add_noise_to_SPAMM_(T, mask, sigma=[], SNR=20):

  # Noisy image
  Tn = np.zeros(T.shape, dtype=T.dtype)

  # Standard deviation
  if sigma == []:
    tt = T[...,0,0]
    sigma = 0.5*np.abs(tt.max()-tt.min())/SNR
  else:
    tt = T[...,0,0]
    sigma = sigma*np.abs(tt.max()-tt.min())

  # Image dimension
  if len(T.shape) is 4:
    d = 3
  else:
    d = 2

  # Add noise
  for i in range(T.shape[-1]):

    # Noise
    for j in range(T.shape[-2]):
      Nr = np.random.normal(0, sigma, T[...,j,i].shape)
      Ni = np.random.normal(0, sigma, T[...,j,i].shape)

      # Add noise
      Tn[...,j,i] = T[...,j,i] + Nr + 1j*Ni

  return Tn


def add_noise_to_DENSE_(u, mask, sigma=[], SNR=20, ref=0, recover_noise=False):

  # Noisy image
  un = np.zeros(u.shape,dtype=u.dtype)

  # Noise image
  if recover_noise:
    noise = np.zeros(u.shape,dtype=u.dtype)

  # Standard deviation
  if not sigma:
    pos = mask[...,ref].astype(np.bool_)
    u0_ref = u[...,0,ref]
    u1_ref = u[...,1,ref]
    mu = 0.5*(np.abs(u0_ref) + np.abs(u1_ref))
    mean  = np.mean(mu[pos])
    sigma = np.sqrt(2)*mean/SNR
  else:
    # peak = np.sqrt(sum([np.power(np.abs(u[...,i,0]),2) for i in range(u.shape[-2])])).max()
    peak = max([np.abs(u[...,0,0]).max(), np.abs(u[...,1,0]).max()])
    sigma = sigma*peak

  # Time steps
  n = u.shape[-1]

  # Add noise
  for i in range(n):
    # Tissue noise
    noise_r = [np.random.normal(0, sigma, u[...,0,i].shape) for l in range(u.shape[-2])]
    noise_i = [np.random.normal(0, sigma, u[...,0,i].shape) for l in range(u.shape[-2])]

    for j in range(u.shape[-2]):
      un[...,j,i] = u[...,j,i] + noise_r[j] + 1j*noise_i[j]

      if recover_noise:
        noise[...,j,i] = noise_r[j] + 1j*noise_i[j]

  if recover_noise:
    return un, noise
  else:
    return un
