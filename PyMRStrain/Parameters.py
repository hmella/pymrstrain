import numpy as np
from PyMRStrain.MPIUtilities import MPI_comm, MPI_rank


class Parameters:
    def __init__(self,time_steps=20,**kwargs):
        # default parameters
        p = self.default_parameters(time_steps)
        for (key,value) in p.items():
            self.__dict__[key] = value
        self.__dict__.update(kwargs)

    def default_parameters(self,time_steps):
        ''' Generate parameters for simulations
        '''
        if MPI_rank==0:
            # Time stepping parameters
            t     = 0.0
            t_end = 1.0
            dt = t_end/time_steps

            # Ventricle geometry
            h = 0.1
            tau  = np.random.uniform(0.0075, 0.0125)
            center = np.array([0.0,0.0,0.0])
            # R_en = np.random.uniform(0.01, 0.03)
            R_en = np.random.uniform(0.02, 0.03)
            R_ep = R_en+tau
            R_inner = R_en-tau
            while R_inner <= 0:
                R_inner = 0.9*R_inner
            R_outer = R_ep+tau

            # Temporal modulation
            tA = np.random.uniform(0.05, 0.15)
            tB = np.random.uniform(0.35, 0.45)
            tC = np.random.uniform(0.5, 0.6)

            # Displacemets
            sigma  = np.random.uniform(0.25, 2.0)               #
            S_en   = np.random.uniform(0.6, 0.8)                # end-systolic endo. scaling
            S_ar   = np.random.uniform(0.9, 1.1)                # end-systolic area scaling
            phi_en = np.random.uniform(-30.0*np.pi/180.0, 0)    # end-systolic epi. twist
            phi_ep = np.random.uniform(phi_en, 0)

            # Pacient parameters
            psi = np.random.uniform(0.0, 2.0*np.pi)
            xi  = np.random.uniform(0.0, 1.0)

            # Create dict
            parameters = {'t': t,
                      'dt': dt,
                      't_end': t_end,
                      'h': h,
                      'tau': tau,
                      'center': center,
                      'R_en': R_en,
                      'R_ep': R_ep,
                      'R_inner': R_inner,
                      'R_outer': R_outer,
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
                      'time_steps': time_steps}
        else:
            parameters = None
        parameters = MPI_comm.bcast(parameters, root=0)

        return parameters