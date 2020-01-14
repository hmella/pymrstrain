import numpy as np

def Parameters_2D(mesh_resolution=1e-03, time_steps=20, decimals=8):
    ''' Generate parameters for simulations
    '''
    # Time stepping parameters
    t     = 0.0
    t_end = 1.0
    dt = t_end/time_steps

    # Ventricle geometry
    h = 0.02
    tau  = np.round(np.random.uniform(0.0075, 0.0125),decimals=decimals)
    # R_en = np.random.uniform(0.01, 0.03)
    R_en = np.round(np.random.uniform(0.02, 0.03),decimals=decimals)
    R_ep = R_en+tau

    # Temporal modulation
    tA = np.random.uniform(0.05, 0.15)
    tB = np.random.uniform(0.35, 0.45)
    tC = np.random.uniform(0.5, 0.6)

    # Displacemets
    sigma  = np.round(np.random.uniform(0.25, 2.0),decimals=decimals)                     #
    S_en   = np.round(np.random.uniform(0.6, 0.8),decimals=decimals)                      # end-systolic endo. scaling
    S_ar   = np.round(np.random.uniform(0.9, 1.1),decimals=decimals)                      # end-systolic area scaling
    phi_en = np.round(np.random.uniform(-15.0*np.pi/180.0, 15.0*np.pi/180.0),decimals=decimals)  # end-systolic epi. twist
    phi_ep = np.round(np.random.uniform(min([phi_en,0]), max([phi_en,0])),decimals=decimals)

    # Pacient parameters
    psi = np.round(np.random.uniform(0.0, 2.0*np.pi),decimals=decimals)
    xi  = np.round(np.random.uniform(0.0, 1.0),decimals=decimals)

    # Image parameters
    FOV = [0.1, 0.1]                     # Field of view in x-direction  [m]

    # DENSE acquisition parameters
    ke   = 0.1                     # [cycle/mm]
    ke   = 0.1*1000.0              # [cycle/m]
    ke   = 2.0*np.pi*ke            # [rad/m]
    sigma_noise = [0.1, 0.25]      # [rad]

    # Create dict
    parameters = {'t': t,
              'dt': dt,
              't_end': t_end,
              'h': h,
              'tau': tau,
              'R_en': R_en,
              'R_ep': R_en+tau,
              'tA': tA,
              'tB': tB,
              'tC': tC,
              'sigma': sigma,
              'S_en': S_en,
              'S_ar': S_ar,
              'phi_ep': phi_ep,
              'phi_en': phi_en,
              'psi': psi,
              'xi': xi,
              'mesh_resolution': mesh_resolution,
              'sigma_noise': sigma_noise,
              'time_steps': time_steps}

    return parameters

# TODO
def Parameters_3D():
  return True
