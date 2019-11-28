import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import cm

# Récupération de la slice correspond à l'axe et à l'index passés en paramètres
def get_slice(img_3d, section, slice):
    if section == 'sagittal':
        try:
            img_2d = np.rot90(img_3d[slice,:,:])
        except:
            print("Error : no section 'sagittal' \n")
    if section == 'coronal':
        try:
            img_2d = np.rot90(np.flip(img_3d, axis=1)[:,slice,:])
        except:
            print("Error : no section 'coronal' \n")
    if section == 'axial':
        try:
            img_2d = np.rot90(np.flip(img_3d, axis=2)[:,:,slice])
        except:
            print("Error : no section 'axial' \n")
    return img_2d

# Affichage de l'image bruitée, l'image débruitée et la différence des deux. 
def compare_image(img_1, img_2, filename=None):
    NBR_IMG = 3
    fig, ax_arr = plt.subplots(nrows=1, ncols=NBR_IMG)

    img_list = [img_1, img_2, img_1 - img_2]
    img_title = ['Before', 'After', 'Difference']

    for i in range(NBR_IMG):
        ax_arr[i].imshow(img_list[i], cmap='gray')
        ax_arr[i].set_title(img_title[i])
        ax_arr[i].axis('off')

    plt.tight_layout()
    if filename is None:
    	plt.show()
    else:
    	plt.savefig(filename, dpi=300)
    plt.close()

# Normalisation de l'image pour le calcul des contrastes et l'affichage de l'histogramme
def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

# Création et affichage de l'histogramme
def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()