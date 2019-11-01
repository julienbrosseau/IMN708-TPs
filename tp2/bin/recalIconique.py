# 4. Recalage iconique 2D simple

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import math

path = "../data"
img = "BrainMRI_1.jpg"
img_test = "BrainMRI_2.jpg"

img_2d = plt.imread(os.path.join(path, img))
img_test = plt.imread(os.path.join(path, img_test))

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
    new_I = griddata((iy+p, ix+q), samples, (Y, X), method='nearest')

    return new_I

def recalage(img1, img2, p, q):
    evol_ssd = []
    pre_ssd = math.inf
    post_ssd = ssd(img1, img2)
    evol_ssd.append(post_ssd)
    iter = 0

    while(p >= 0.5):
        if(pre_ssd > post_ssd):
            img1 = translation(img1, p, q)
            pre_ssd = post_ssd
            post_ssd = ssd(img1, img2)
            evol_ssd.append(post_ssd)
            iter += p
        else:
            p = p/2
        
    print(iter)
    return img1, evol_ssd

#new_I = translation(img_2d, 24.5, 0)
new_I, evol_ssd = recalage(img_2d, img_test, 2, 0)
print(evol_ssd)

fig, ax = plt.subplots(ncols=3)

print("SSD :", ssd(img_test, new_I))

ax[0].imshow(img_2d)
ax[1].imshow(img_test)
ax[2].imshow(new_I)

plt.show()