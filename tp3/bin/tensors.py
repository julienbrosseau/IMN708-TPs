import os
import sys
import math
import nibabel as nib
import numpy as np
import dipy
import random
import scipy
from skimage.restoration import denoise_nl_means
from scipy.ndimage import median_filter
from fury.colormap import line_colors
from dipy.viz import window, actor

# Récupération des informations relatives au fichier image à charger
path  = "Data"
file  = sys.argv[1]

# Chargement de l'image
img_diffusion = nib.load(os.path.join(path, file))
print (img_diffusion.shape)

img_data = img_diffusion.get_data()

# Débruitage avec NLM
#denoise = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2], img_data.shape[3]), dtype = np.int8)
#for i in range (img_data.shape[2]):
#    denoise[:, :, :, i] = denoise_nl_means(img_data[:, :, :, i], 7, 9, 0.08, multichannel = True)

# Débruitage avec filtre médian
#for i in range (img_data.shape[2]):
#    denoise[:, :, :, i] = scipy.signal.medfilt(img_data[:, :, :, i])
denoise = img_data
print ("Denoising ok")

# Génération d'un masque et d'une liste de voxels valides selon une intensité minimale donnée en paramètre
def getMask(low_intensity):
    mask = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]), dtype = np.int8)
    valid_list = []
    for x in range (img_data.shape[0]):
        for y in range (img_data.shape[1]):
            for z in range (img_data.shape[2]):
                if(img_data[x, y, z, 0] > low_intensity):
                    mask[x, y, z] = 1
                    valid_list.append([x, y, z])
    return mask, valid_list

# Calcul de la matrice B et de la bValue selon le fichier txt de bvalues passé en paramètre
def getB(bValuesFileName):
    bValuesFile = open(bValuesFileName, "r")
    bValuesRaw = bValuesFile.read().splitlines()
    bValues = []
    for bValue in bValuesRaw:
        splitted = [float(i) for i in bValue.split("\t")[:-1]]
        bValues.append(splitted)

    B = []
    for i in range (1, len(bValues)):
        B.append([bValues[i][0] * bValues[i][0], bValues[i][0] * bValues[i][1], bValues[i][0] * bValues[i][2], 
                bValues[i][1] * bValues[i][1], bValues[i][1] * bValues[i][2], bValues[i][2] * bValues[i][2]])

    return B, bValues[1][3]

# Calcul de la valeur du tenseur D pour un voxel x, y, z et une bValue donnés
def getD(x, y, z, B, bValue):
    S = img_data[x, y, z, :]
    S0 = S[0]

    X = []
    for s in S[1:]:
        if s == 0 or S0 == 0:
            return [0, 0, 0, 0, 0, 0]
        else:
            X.append((-1 / bValue) * math.log(s / S0))

    return np.dot(np.linalg.inv(np.dot(np.transpose(B), B)), np.dot(np.transpose(B), X))

B, bValue = getB(path + "/gradient_directions_b-values.txt")
print ("B & bValue ok")

theMask, valid_list = getMask(100)
print("Mask ok")

# Reconstitution de la matrice D barre avec la matrice D[xx, xy, xz, yy, yz, zz]
def getD_(D):
    return [[D[0], D[1], D[2]],
            [D[1], D[3], D[4]],
            [D[2], D[4], D[5]]]

# Calcul de la valeur de tous les tenseurs des voxels valides du masque
def getTensors(img, mask):
    tensors = np.zeros((img.shape[0], img.shape[1], img.shape[2], 6))
    for x in range (img.shape[0]):
        for y in range (img.shape[1]):
            for z in range (img.shape[2]):
                if(mask[x, y, z] == 1):
                    tensors[x][y][z] = getD(x, y, z, B, bValue)
    return tensors

tensors = getTensors(denoise, theMask)

# Save du maping de tenseurs dans un fichier nifti
tensors_save = nib.Nifti1Image(tensors, img_diffusion.affine, img_diffusion.header)
nib.save(tensors_save, "tensors.nii.gz")

print("Tensors ok")

# Calcul de la FA selon les valeurs propres passées en paramètre
def getFA(values):
    up = np.sqrt(((values[0] - values[1]) * (values[0] - values[1])) + ((values[1] - values[2]) * (values[1] - values[2])) + ((values[2] - values[0]) * (values[2] - values[0])))
    down = np.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])

    return np.sqrt(1 / 2) * (up / down)

# Calcul de toutes les valeurs de FA des voxels valides du masque
def getFAs(img, mask, tensors):
    fa = np.zeros((img.shape[0], img.shape[1], img.shape[2], 1))
    for x in range (img.shape[0]):
        for y in range (img.shape[1]):
            for z in range (img.shape[2]):
                if(mask[x, y, z] == 1):
                    D = tensors[x][y][z]
                    eig , vec = np.linalg.eig(getD_(D))
                    fa[x, y, z] = getFA(eig)
    return fa
                
fa = getFAs(denoise, theMask, tensors)
fa[np.isnan(fa)] = 0
fa = np.clip(fa, 0, 1)

fa_save = nib.Nifti1Image(fa, img_diffusion.affine, img_diffusion.header)
nib.save(fa_save, "fa.nii.gz")

print("FA ok")
#print (fa)

# Calcul de toutes les valeurs d'ADC des voxels valides du masque
def getADC(img, mask, tensors):
    adc = np.zeros((img.shape[0], img.shape[1], img.shape[2], 1))
    for x in range (img.shape[0]):
        for y in range (img.shape[1]):
            for z in range (img.shape[2]):
                if(mask[x, y, z] == 1):
                    D = tensors[x][y][z]
                    eig , vec = np.linalg.eig(getD_(D))
                    adc[x][y][z] = np.mean(eig)
    return adc

#adc = getADC(denoise, theMask, tensors)
#adc_save = nib.Nifti1Image(adc, img_diffusion.affine, img_diffusion.header)
#nib.save(adc_save, "adc.nii.gz")

print("ADC ok")

########### TRACTO ###########

# Retourne toutes les fibres trouvées selon le masque, les tenseurs et l'angle donnés en paramètre
def tracto(img, mask, tensors, theta):
    streamlines = []
    for i in range (10):
        pos = random.choice(valid_list)
        
        x = int(pos[0])
        y = int(pos[1])
        z = int(pos[2])
        chemin = []

        print(fa[x, y, z])
        print(mask[x, y, z])

        while ((fa[x, y, z] >= 0.15) and (mask[x, y, z] == 1)):
            chemin.append(pos)

            D = tensors[x, y, z]
            eig , vec = np.linalg.eig(getD_(D))
            x = np.round(x + vec[0])
            y = np.round(y + vec[1])
            z = np.round(z + vec[2])

            pos = [x, y, z]

            #flip du vecteur pour partir dans l'autre sens et recommencer en ajoutant les éléments du chemin au début de la liste

        print(len(chemin))
        streamlines.append(chemin)
            
    return streamlines

streamlines = tracto(denoise, theMask, tensors, 45)

# Coloriage puis affichage des fibres 
#color = line_colors(streamlines)
#streamlines_actor = actor.line(streamlines, line_colors(streamlines))

#r = window.ren()
#r.add(streamlines_actor)

#window.show(r)