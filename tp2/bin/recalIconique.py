# 4. Recalage iconique 2D simple

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import ndimage
import math
import cv2 as cv

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

def des_gradient(img1, p, q, epsi):
    sobelx = cv.Sobel(img1,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(img1,cv.CV_64F,0,1,ksize=5)

    d_ssd_p = 2*np.sum((img1 - img2)*sobelx)
    d_ssd_q = 2*np.sum((img1 - img2)*sobely)

    new_p = p - epsi*d_ssd_p
    new_q = q - epsi*d_ssd_q

    return new_p, new_q

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
    new_I = np.zeros((nx*2, ny*2), dtype=np.uint8)
    # plt.imshow(new_I)
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
                print("Pas possible pour le point : ",v_t[0], v_t[1])
    
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1)) 

    ix = np.random.randint(nx*2, size=1000000)
    iy = np.random.randint(ny*2, size=1000000)
    samples = new_I[iy,ix]
    new_I = griddata((iy-ny, ix-nx), samples, (Y, X), method='linear')
    
    return new_I

def recalage(img1, img2, type, median, p, q, theta):
    evol_ssd = []
    iter = 0

    pre_ssd = math.inf
    post_ssd = ssd(img1, img2)
    evol_ssd.append(post_ssd)

    p, q = des_gradient(img1, 0, 0, 0.00000001)
    
    while(pre_ssd > post_ssd):  
        if type == "translation":      
            img1 = translation(img1, p, q)
        elif type == "rotation":
            img1 = rotation(img1, theta)
        
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if math.isnan(img1[i, j]):
                    img1[i, j] = median
        pre_ssd = post_ssd
        post_ssd = ssd(img1, img2)
        iter += 1
        evol_ssd.append(post_ssd)

    print("Nombre d'iterations :", iter)
           
    return img1, evol_ssd

# Debruitage des images
sigma = 4

debruit_img1 = median_filter(img_2d, sigma)
debruit_img2 = median_filter(img_test, sigma)

# Recuperation de la mediane de l image
median = np.median(debruit_img1)

#new_I = translation(debruit_img1, 10, 0)
#new_I = rotation(debruit_img1, 1)
#print("SSD :", ssd(img_test, img_2d))

new_I, evol_ssd = recalage(debruit_img1, debruit_img2, "translation", median, 1, 0, -0.05)
print("SSD's :", evol_ssd)
x = evol_ssd
y = range(len(evol_ssd))

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0,0].imshow(debruit_img1)
ax[0,1].imshow(debruit_img2)
ax[1,0].imshow(new_I)
ax[1,1].plot(y, x)

plt.show()