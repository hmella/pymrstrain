from PyMRStrain import *
import numpy as np

# Create image
I0 = DENSEImage(FOV=np.array([0.15, 0.15, 0.2]),
                resolution=np.array([55, 55, 25]),
                center=np.array([0.01,-0.01,-0.03]),
                encoding_frequency=np.array([125,125,115]),
                T1=0.85,
                flip_angle=15*np.pi/180.0)

print(I0.field_of_view())