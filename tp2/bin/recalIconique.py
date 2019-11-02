# 4. Recalage iconique 2D simple

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import ndimage
import math

path = "../data"
img = "BrainMRI_1.jpg"
img_test = "BrainMRI_2.jpg"  

img_2d = plt.imread(os.path.join(path, img))
img_test = plt.imread(os.path.join(path, img_test))

def median_filter(img_2d, sigma):
    return ndimage.median_filter(img_2d, size=sigma)

def ssd(img1, img2):
    ssd_totale = 0

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            ssd_totale += (int(img1[i, j]) - int(img2[i, j]))**2
    
    return ssd_totale

def translation(I, p, q):
    nx, ny = I.shape[1], I.shape[0]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1)) 

    ix = np.random.randint(nx, size=100000)
    iy = np.random.randint(ny, size=100000)
    samples = I[iy,ix]
    new_I = griddata((iy+p, ix+q), samples, (Y, X), method='cubic')

    return new_I

def rotation(I, theta):
    nx, ny = I.shape[1], I.shape[0]
    new_I = np.copy(I)
    matrix = [
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]]
    """
    for i in range(nx):
        for j in range(ny):
            v = [i,j,1]
            v_t = np.dot(matrix, v)

            try:    
                new_I[int(v_t[0]), int(v_t[1])] = I[i, j]
            except:
                print("pas possible pour ce point",v_t[0], v_t[1])
    """
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1)) 

    ix = np.random.randint(nx, size=100000)
    iy = np.random.randint(ny, size=100000)
    samples = I[iy,ix]
    new_I = griddata((iy, ix), samples, (Y, X), method='cubic')
    
    return new_I

def recalage(img1, img2, median, p, q):
    evol_ssd = []
    iter = 0

    pre_ssd = math.inf
    post_ssd = ssd(img1, img2)
    evol_ssd.append(post_ssd)
    
    while(pre_ssd > post_ssd):        
        img1 = translation(img1, p, q)
        
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if math.isnan(img1[i, j]):
                    img1[i, j] = median
        
        pre_ssd = post_ssd
        post_ssd = ssd(img1, img2)
        iter += p
        evol_ssd.append(post_ssd)

    print(iter)
           
    return img1, evol_ssd

# Debruitage des images
sigma = 4

debruit_img1 = median_filter(img_2d, sigma)
debruit_img2 = median_filter(img_test, sigma)

median = np.median(debruit_img1)

#new_I = translation(debruit_img1, 10, 0)
#new_I = rotation(debruit_img1, 10)
#print("SSD :", ssd(img_test, img_2d))

new_I, evol_ssd = recalage(debruit_img1, debruit_img2, median, 1, 0)
print("SSD's :", evol_ssd)

fig, ax = plt.subplots(ncols=3)

ax[0].imshow(debruit_img1)
ax[1].imshow(debruit_img2)
ax[2].imshow(new_I)

plt.show()