# TP2 - Recalage
# Trucs  à pas coder :
#   Extrapolation / Interpolation
# Il faut prendre en compte les bords (contour de l'image) soit 0 soit median de l'image
# Bruit gaussien sur les images à recaler (il faudrait mieux débruiter avant le traitement)
# On peut rajouter un courbe d'évolution de SSD pour voir que c'est décroissant
# Brain1 > Brain4 faire des astuces de reclage sinon ça convergera jamais

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm

path = "../data"
img1 = "I2.jpg"
img2 = "I2.jpg"

img1_2d = mpimg.imread(os.path.join(path, img1), 0)
img2_2d = mpimg.imread(os.path.join(path, img2), 0)

print(img1_2d.shape, img2_2d.shape)

# Histogramme conjoint
def jointHist(img1, img2, bin):
    for i in range(img1_2d.shape[1]):
        #print(i)
        plt.hist2d(img1[:,i], img2[:,i], bins=bin, norm=LogNorm())
    plt.show() 

# Affichage de l histogramme
jointHist(img1_2d, img2_2d, 50)