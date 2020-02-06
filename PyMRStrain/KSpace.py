import numpy as np

from PyMRStrain.Math import itok, ktoi


# kspace class
class kspace:
    def __init__(self, artifact, acq_matrix, rawg, oversampling_factor):
        self.acq_matrix = acq_matrix
        self.rawg = rawg
        self.oversampling_factor = oversampling_factor
        self.artifact = artifact
        self.k = []
        self.filter = []

   def to_img(self):
        return ktoi(self.k)

   def gen_to_acq(self):
        return 1
