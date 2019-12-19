
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from matplotlib import cm
from scipy import ndimage
from astropy.convolution import convolve, Gaussian2DKernel

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
    
    def lissage(self, img_4d):
        # Lissage à chaque temps a l'aide d'une gaussienne
        gauss_kernel = Gaussian2DKernel(1)

        for i in range(img_4d.shape[3]):
            for j in range(img_4d.shape[0]):
                img_4d[j, :, :, i] = convolve(img_4d[j, :, :, i], gauss_kernel) 

        return img_4d