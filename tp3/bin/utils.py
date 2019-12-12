
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from matplotlib import cm
from scipy import ndimage

class Utils():

    def __init__(self):
        # Initalisation de la classe      
        self.path       = "../data"
        self.ideal_file = "ideal.txt"
        self.fmri_file  = "fmri.nii.gz"

    def open_fmri(self):
        # Ouvre le fichier Nifti "fmri.nii"
        img = nib.load(os.path.join(self.path, self.fmri_file))
        
        return img

    def get_ideal(self):
        # Retourne le vecteur "ideal"
        ideal_data = []
        with open(os.path.join(self.path, self.ideal_file), 'r') as file:
            for data in file.readlines():
                ideal_data.append(float((data)))

        #print(ideal_data)
        return ideal_data

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

    def normalize(self, arr):
        # Normalisation de l'image pour le calcul des contrastes et l'affichage de l'histogramme
        arr_min = np.min(arr)
        
        return (arr-arr_min)/(np.max(arr)-arr_min)

    def show_histogram(self, values):
        # Création et affichage de l'histogramme
        n, bins, patches = plt.hist(values.reshape(-1), 50, density=1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        for c, p in zip(self.normalize(bin_centers), patches):
            plt.setp(p, 'facecolor', cm.viridis(c))

        plt.show()
    
    def median_filter(self, data, sigma):
        # Filtage médian
        return ndimage.median_filter(data, size=sigma)
    
    def gaussian_filter(self, data, sigma):

        return ndimage.gaussian_filter(data, sigma=sigma)