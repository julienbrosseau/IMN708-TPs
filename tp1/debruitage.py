import os
import sys

import nibabel as nib
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from scipy import ndimage
import utils as utils

# Paramètres pour la diffusion non-linéaire
NB_ITER = 10
COEFF_CONDUCTION = 20
GAMMA = 0.1
PAS = (1.,1.)
EQUATION = 1

# Paramètres pour le filtrage médian
SIGMA = 20

# Paramètres pour le filtrage moyen non-local
path  = sys.argv[1]
file  = sys.argv[2]
section = sys.argv[3]
index = int(sys.argv[4])

axis_enum = {'sagittal':0, 'coronal':1,'axial':2}

# Filtage anisotrope
def no_linear_filter(data, nbIter, coeffConduction, gamma, pas, equation):
    # Initialisation de l'image de sortie
    data = data.astype('float32')
    imgout = data.copy()

    # Initialisation des variables qui vont nous permettre de débruiter l'image
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for i in range(nbIter):

            # Calcul de la diffusion
            deltaS[:-1,: ] = np.diff(imgout, axis=0)
            deltaE[: ,:-1] = np.diff(imgout, axis=1)

            # Calcul des gradients de condution et mise à l'échelle des x et des y
            if equation == 1:
                    gS = np.exp(-(deltaS/coeffConduction)**2.)/pas[0]
                    gE = np.exp(-(deltaE/coeffConduction)**2.)/pas[1]
            elif equation == 2:
                    gS = 1./(1.+(deltaS/coeffConduction)**2.)/pas[0]
                    gE = 1./(1.+(deltaE/coeffConduction)**2.)/pas[1]

            # Recalcul des matrices
            E = gE*deltaE
            S = gS*deltaS

            # Extraction du bruit
            NS[:] = S
            EW[:] = E
            NS[1:,:] -= S[:-1,:]
            EW[:,1:] -= E[:,:-1]

            # Mise à jour de l'image en tenant compte du gamma
            imgout += gamma*(NS+EW)
    return imgout

# Filtage médian
def median_filter(data, sigma):
    return ndimage.median_filter(data, size=sigma)

# Filtrage moyen non-local
def nlmeans_filter(data, sigma, mask=None):
    return nlmeans(data, sigma, mask, rician=True)

# Main
img = nib.load(os.path.join(path, file))
img_3d = np.squeeze(img.get_data())
print (img_3d.shape)
img_2d = utils.get_slice(img_3d, section, index)

# Débruitage
#no_linear_filter = no_linear_filter(img_2d, NB_ITER, COEFF_CONDUCTION, GAMMA, PAS, EQUATION)
#median_filter = median_filter(img_3d, SIGMA)
nlmeans_filter = nlmeans_filter(img_3d, SIGMA)

# Affichage de la différence de bruit après débruitage
#utils.compare_image(img_2d, no_linear_filter)
#utils.compare_image(utils.get_slice(img_3d, section, index), utils.get_slice(median_filter, section, index))
utils.compare_image(utils.get_slice(img_3d, section, index), utils.get_slice(nlmeans_filter, section, index))

# Enregistre le débruitage
#nib.save(nib.Nifti1Image(nlmeans_filter, img.affine, img.header),'debruitage.nii')