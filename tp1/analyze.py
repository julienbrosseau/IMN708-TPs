import os
import sys
import nibabel as nib
import numpy as np
import utils as utils

# Enumération des axes et attribution d'un index
axes_enum = {'sagittal':0, 'coronal':1,'axial':2}

# Récupération du nom de l'image et de l'axe choisi dans les arguments de la commande 
path = sys.argv[1]
img_filename = sys.argv[2]
axe = sys.argv[3]

img = nib.load(os.path.join(path, img_filename))
img_data = img.get_data()

# Affichage de la taille des voxels
print ("Taille des voxels : ", img.header.get_zooms())

# Affichage des dimensions de l'image
print ("Dimensions de l'image : ", img_data.shape)

# Normalisation de l'image pour le calcul des contrastes normalisés
img_norm = utils.normalize(0.5 * (img_data[:-1] + img_data[1:]))

min_I = np.min(img_data)
max_I = np.max(img_data)

min_I_n = np.min(img_norm)
max_I_n = np.max(img_norm)

# Calcul du constrate de Michelson
c_Michelson = (max_I - min_I) / (max_I + min_I)
c_Michelson_n = (max_I_n - min_I_n) / (max_I_n + min_I_n)

# Calcul du contraste root mean square (RMS)
c_rms = np.sqrt(np.mean(img_data**2))
c_rms_n = np.sqrt(np.mean(img_norm**2))

print ("Constraste Michelson (non normalisé) : ", c_Michelson, " ; Contraste RMS (non normalisé) : ", c_rms)
print ("Constraste Michelson (normalisé) : ", c_Michelson_n, " ; Contraste RMS (normalisé) : ", c_rms_n)

# Affichage de l'histogramme
utils.show_histogram(img_data)

# SNR
def get_snr(data, noise, signal):
    # Déterminer une zone de bruit (zone homogène -- pas forcément le fond)
    bruit = np.std(noise)

    # Déterminer une zone à traiter
    mean = np.mean(signal)

    return mean / bruit