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
from dipy.viz import window, actor
from fury.colormap import line_colors

ANGLE_MAX_LEFT = 0.785398 # 45° en rad
ANGLE_MAX_RIGHT = 2.35619 # 135° en rad

# Récupération des informations relatives au fichier image à charger
path  = "Data"
file  = sys.argv[1]

# Chargement de l'image
img_diffusion = nib.load(os.path.join(path, file))
print (img_diffusion.shape)

img_data = img_diffusion.get_data()

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
        B.append([bValues[i][0] * bValues[i][0], 
                  2 * bValues[i][0] * bValues[i][1], 
                  2 * bValues[i][0] * bValues[i][2], 
                  bValues[i][1] * bValues[i][1], 
                  2 * bValues[i][1] * bValues[i][2], 
                  bValues[i][2] * bValues[i][2]])

    return B, int(bValues[1][3])

# Calcul de la valeur du tenseur D pour un voxel x, y, z et une bValue donnés
def getD(x, y, z, B, bValue):
    S = img_data[x, y, z, :]
    S0 = S[0]

    X = []
    for s in S[1:]:
        if s == 0 or S0 == 0:
            return [0, 0, 0, 0, 0, 0]
        else:
            X.append(math.log(s / S0))

    return np.linalg.solve(np.dot(np.transpose(B), B), np.dot(np.transpose(B), (-1 / bValue) * np.transpose(X)))

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

# Calcul de la FA selon les valeurs propres passées en paramètre
def getFA(values):
    up = np.sqrt(((values[0] - values[1]) * (values[0] - values[1])) + 
                 ((values[1] - values[2]) * (values[1] - values[2])) + 
                 ((values[2] - values[0]) * (values[2] - values[0])))
    down = np.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])
    if down == 0:
        return 0
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

def getAngleFromVec(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# Débruitage avec NLM
denoise = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2], img_data.shape[3]), dtype = np.int8)
#for i in range (img_data.shape[2]):
#    denoise[:, :, :, i] = denoise_nl_means(img_data[:, :, :, i], 7, 9, 0.08, multichannel = True)

# Débruitage avec filtre médian
for i in range (img_data.shape[2]):
    denoise[:, :, :, i] = scipy.ndimage.median_filter(img_data[:, :, :, i], 3)

#np.save("saves/denoise", denoise)
denoise = np.load("saves/denoise.npy")
print ("Débruitage terminé")

B, bValue = getB(path + "/gradient_directions_b-values.txt")
print ("Calcul de B & bValue terminé")

theMask, valid_list = getMask(150)
print("Calcul du masque terminé")

mask_save = nib.Nifti1Image(theMask, img_diffusion.affine, img_diffusion.header)
nib.save(mask_save, "mask.nii.gz")

tensors = getTensors(denoise, theMask)

# Save du maping de tenseurs dans un fichier nifti
tensors_save = nib.Nifti1Image(tensors, img_diffusion.affine, img_diffusion.header)
nib.save(tensors_save, "tensors.nii.gz")

np.save("saves/tensors", tensors)
#tensors = np.load("saves/tensors.npy")
print("Calcul des tenseurs terminé")
                
fa = getFAs(denoise, theMask, tensors)
fa = np.clip(fa, 0, 1)

fa_save = nib.Nifti1Image(fa, img_diffusion.affine, img_diffusion.header)
nib.save(fa_save, "fa.nii.gz")
np.save("saves/fa", fa)
#fa = np.load("saves/fa.npy")

print("Calcul de la FA terminé")

#adc = getADC(denoise, theMask, tensors)
#adc_save = nib.Nifti1Image(adc, img_diffusion.affine, img_diffusion.header)
#nib.save(adc_save, "adc.nii.gz")

print("Calcul de l'ADC terminé")

########### TRACTO ###########

# Retourne toutes les fibres trouvées selon le masque, les tenseurs et l'angle donnés en paramètre
def tracto(img, mask, tensors, theta):
    streamlines = []
    for i in range (100000):
        pos = random.choice(valid_list)
        #pos = [65, 78, 29]
        x = int(pos[0])
        y = int(pos[1])
        z = int(pos[2])

        chemin = []

        D = tensors[x, y, z]
        eig , vec = np.linalg.eig(getD_(D))

        max_index = np.argmax(eig)
        currentVec = vec[:, max_index]
        oldVec = currentVec

        floatx = x
        floaty = y
        floatz = z

        # Si la FA est > 0.15, si on est toujours dans le masque et si l'angle est valide (entre 0 et 45° ou > 135 (faut flip))
        while ((fa[x, y, z] >= 0.15) and (mask[x, y, z] == 1) and (getAngleFromVec(oldVec, currentVec) <= ANGLE_MAX_LEFT or getAngleFromVec(oldVec, currentVec) >= ANGLE_MAX_RIGHT)):
            chemin.append([floatx, floaty, floatz])

            # Si l'angle est > 135° cela signifie que le vecteur n'est pas dans le bon sens, on le retourne donc
            if(getAngleFromVec(oldVec, currentVec) >= ANGLE_MAX_RIGHT):
                currentVec = currentVec * -1

            floatx += currentVec[0] * 2
            floaty += currentVec[1] * 2
            floatz += currentVec[2] * 2

            x = int(np.round(floatx))                
            y = int(np.round(floaty))
            z = int(np.round(floatz))
            
            # Si les x, y et z calculés sont toujours dans le masque, on continue
            if (x > -1 and x < img.shape[0] and y > -1 and y < img.shape[1] and z > -1 and z < img.shape[2]):
                D = tensors[x, y, z]
                eig , vec = np.linalg.eig(getD_(D))

                # Mise à jour de l'ancien vecteur et de l'actuel pour comparer les angles
                oldVec = currentVec
                max_index = np.argmax(eig)
                currentVec = vec[:, max_index]
            else:
                break

        
        x = int(pos[0])
        y = int(pos[1])
        z = int(pos[2])

        floatx = x
        floaty = y
        floatz = z

        D = tensors[x, y, z]
        max_index = np.argmax(eig)
        currentVec = -1 * vec[:, max_index]
        oldVec = currentVec

        while ((fa[x, y, z] >= 0.15) and (mask[x, y, z] == 1) and (getAngleFromVec(oldVec, currentVec) <= ANGLE_MAX_LEFT or getAngleFromVec(oldVec, currentVec) >= ANGLE_MAX_RIGHT)):

            if(getAngleFromVec(oldVec, currentVec) >= ANGLE_MAX_RIGHT):
                currentVec = currentVec * -1

            floatx += currentVec[0] * 2
            floaty += currentVec[1] * 2
            floatz += currentVec[2] * 2

            x = int(np.round(floatx))                
            y = int(np.round(floaty))
            z = int(np.round(floatz))

            if (x > -1 and x < img.shape[0] and y > -1 and y < img.shape[1] and z > -1 and z < img.shape[2]):
                D = tensors[x, y, z]
                eig , vec = np.linalg.eig(getD_(D))

                oldVec = currentVec
                max_index = np.argmax(eig)
                currentVec = vec[:, max_index]
            else:
                break

            chemin.insert(0, [floatx, floaty, floatz])
        
        if len(chemin) > 8:
            streamlines.append(np.array(chemin))
    return streamlines

streamlines = tracto(denoise, theMask, tensors, 90)

#print(streamlines)
# Coloriage puis affichage des fibres 

r = window.ren()
r.add(actor.line(streamlines, line_colors(streamlines)))

window.show(r)