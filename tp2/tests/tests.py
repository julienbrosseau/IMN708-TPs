# Implementation de la descente de gradient pour translation et rotation

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.interpolate import griddata
import math
from scipy import ndimage

img1 = plt.imread("../data/BrainMRI_1.jpg")
img2 = plt.imread("../data/BrainMRI_2.jpg")

fig, ax = plt.subplots(ncols=3)

ax[1].imshow(img2)

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def ssd(I, J):
    return np.sum(np.square(I - J))

def descente_gradient(img1, img2, p, q, theta, epsi):
    pas_trans = 20
    pas_rotat = 0.2

    # Descente du gradient pour une translation
    deriv_x = translation(img1, pas_trans, 0)
    deriv_y = translation(img1, 0, pas_trans)

    d_ssd_p = 2*np.sum((img1 - img2)*deriv_x)
    d_ssd_q = 2*np.sum((img1 - img2)*deriv_y)

    new_p = p - epsi*d_ssd_p
    new_q = q - epsi*d_ssd_q

    # Descente du gradient pour une rotation
    deriv_theta = rotation(img1, pas_rotat)

    d_ssd_theta = 2*np.sum((img1 - img2)*deriv_theta)

    new_theta = theta - epsi*d_ssd_theta

    return new_p, new_q, new_theta

def rotation(I, theta):
    nx, ny = I.shape[1], I.shape[0]
    new_I = np.zeros((nx*2, ny*2), dtype=np.float32)
    
    matrix = [
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]]
    
    for i in range(nx):
        for j in range(ny):
            v = [i,j,1]
            v_t = np.dot(matrix, v)

            try:    
                new_I[int(v_t[0]+nx), int(v_t[1]+ny)] = I[i, j]
            except:
                #print("Pas possible pour le point : ",v_t[0], v_t[1])
                pass
    
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1)) 

    ix = np.random.randint(nx*2, size=1000000)
    iy = np.random.randint(ny*2, size=1000000)
    samples = new_I[iy,ix]
    new_I = griddata((iy-ny, ix-nx), samples, (Y, X), method='nearest')
    """
    for i in range(new_I.shape[0]):
        for j in range(new_I.shape[1]):
            if math.isnan(new_I[i, j]):
                new_I[i, j] = median
    """
    return new_I

def translation(I, p, q):
    nx, ny = I.shape[1], I.shape[0]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1)) 

    ix = np.random.randint(nx, size=100000)
    iy = np.random.randint(ny, size=100000)
    samples = I[iy,ix]
    new_I = griddata((iy+p, ix+q), samples, (Y, X), method='nearest')
    """
    for i in range(new_I.shape[0]):
        for j in range(new_I.shape[1]):
            if math.isnan(new_I[i, j]):
                new_I[i, j] = median
    """
    return new_I

# Normalisation des images et passage en float32
img1 = normalize(np.array(img1, dtype=np.float32))
img2 = normalize(np.array(img2, dtype=np.float32))

# Variables
median = np.median(img1)
p = 0
q = 0
theta = 0
epsi = 0.001
iter = 0

# Main
pre_ssd = math.inf
post_ssd = ssd(img1, img2)

while(pre_ssd > post_ssd or iter < 2):
    if pre_ssd > post_ssd:
        iter += 1

    p, q, theta = descente_gradient(img2, img1, p, q, theta, epsi)
    print("new_p :", p)
    print("new_q :", q)
    print("new_theta :", theta)

    img2 = rotation(img2, theta)
    img2 = translation(img2, p, q)
    
    pre_ssd = post_ssd
    post_ssd = ssd(img1, img2)

ax[0].imshow(img1)
ax[2].imshow(img2)

plt.show()