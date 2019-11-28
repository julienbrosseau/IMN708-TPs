import nibabel as nib
import numpy as np
import math
import os

class Utils():

    def __init__(self):
        # Initalisation de la classe      
        self.path       = "../data"
        self.ideal_file = "ideal.txt"
        self.fmri_file  = "fmri.nii.gz"

    def get_mean(self, x):
        # Retourne la moyenne du paramètre
        y = np.sum(x) / np.size(x)
        
        return y

    def get_corr2(self, a, b):
        # Retroune la coorelation entre les deux parametres
        a = a - self.get_mean(a)
        b = b - self.get_mean(b)

        corr2 = np.sum(a*b) / math.sqrt(np.sum(a*a) * np.sum(b*b))
        
        return corr2

    def get_ideal(self):
        # Retourne le vecteur "ideal"
        ideal_data = []
        with open(os.path.join(self.path, self.ideal_file), 'r') as file:
            for data in file.readlines():
                ideal_data.append(float((data)))

        #print(ideal_data)
        return ideal_data

    def open_fmri(self):
        # Ouvre le fichier Nifti "fmri.nii"
        img = nib.load(os.path.join(self.path, self.fmri_file))
        
        return(img.get_data())