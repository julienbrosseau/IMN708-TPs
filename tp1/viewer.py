# Pour lancer : dans le terminal taper :
# Coupe axial >>
#   python viewer.py viewer ../data/14971938/nifti 14971938_T2_AX_FS_20110309130206_8.nii
# Coupe coronal >>
#   python viewer.py viewer ../data/14971938/nifti 14971938_T2_STIR_CORO_20110309130206_7.nii
# Coupe sagittal >>
#   python viewer.py viewer ../data/14971938/nifti 14971938_T1_TSE_SAG_FS_GADO_20110309130206_13.nii

import os
import sys

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils as utils
from matplotlib.widgets import Slider, Button, RadioButtons

# Récupération des différents arguments passés dans la commade
cmd = sys.argv[1]
path  = sys.argv[2]
file  = sys.argv[3]
#section = sys.argv[4]

section = "axial"

# On récupère en plus les index de début et de fin pour les max et min projections
if cmd == 'maxP' or cmd == 'minP':
    startIndex = int(sys.argv[4])
    endIndex = int(sys.argv[5])

# Enumération de tous les axes possibles
axis_enum = {'sagittal':0, 'coronal':1,'axial':2}

# Récupère la coupe 2D à l'index et sur l'axe passés en paramètres
def viewer(img_3d, section, slice):
    img_2d = utils.get_slice(img_3d, section, slice)
    return img_2d

# Affichage de l'image en maximum projection selon l'axe et les index passés en argument
def get_max_projection(img_data, startIndex, endIndex, axe):
    if axe == 'sagittal':
        img_max = np.max(img_data[startIndex:endIndex, :, :], axis = axis_enum[axe])
    elif axe == 'coronal':
        img_max = np.max(img_data[:, startIndex:endIndex, :], axis = axis_enum[axe])
    elif axe == 'axial':
        img_max = np.max(img_data[:, :, startIndex:endIndex], axis = axis_enum[axe])
    plt.imshow(np.rot90(img_max))
    plt.show()

# Affichage de l'image en maximum projection selon l'axe et les index passés en argument
def get_min_projection(img_data, startIndex, endIndex, axe):
    if axe == 'sagittal':
        img_min = np.min(img_data[startIndex:endIndex, :, :], axis = axis_enum[axe])
    elif axe == 'coronal':
        img_min = np.min(img_data[:, startIndex:endIndex, :], axis = axis_enum[axe])
    elif axe == 'axial':
        img_min = np.min(img_data[:, :, startIndex:endIndex], axis = axis_enum[axe])
    plt.imshow(np.rot90(img_min))
    plt.show()

name, ext = os.path.splitext(file)

if ext == '.png' or ext == '.jpg' or ext == '.pgm':
    if cmd == 'viewer':
        img = mpimg.imread(path + '/' + file)
        plt.suptitle("2D image")
        plt.imshow(img, cmap='gray')
    else:
        raise Exception('Seule la commande viewer est disponible pour les images en 2 dimensions.')
else:
    img = nib.load(os.path.join(path, file))
    img_3d = np.squeeze(img.get_data())
    print (img_3d.shape)
    init = img_3d.shape[axis_enum[section]] // 2

    if cmd == 'viewer':
        current_slice = plt.imshow(viewer(img_3d, section, init), cmap='gray')

        plt.suptitle("Section " + section + "e")
        print(current_slice)

        delta_s = 1
        ax_slice = plt.axes([0.15, 0.01, 0.72, 0.01])
        s_slice = Slider(ax_slice, 'Slice', 0, img_3d.shape[axis_enum[section]] -1, valinit=init, valstep=delta_s)

        def update(val):
            slice = s_slice.val
            slice_of_section = viewer(img_3d, section, int(slice))
            current_slice.set_data(slice_of_section)

        s_slice.on_changed(update)
    elif cmd == 'minP':
        get_min_projection(img_3d, startIndex, endIndex, section)
    elif cmd == 'maxP':
        get_max_projection(img_3d, startIndex, endIndex, section)
    else:
        raise Exception('Commande non reconnue. Disponibles : viewer, minP, maxP')
plt.show()