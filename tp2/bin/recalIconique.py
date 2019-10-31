# 4. Recalage iconique 2D simple

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.interpolate import Rbf

path = "../data"
img = "BrainMRI_1.jpg"
img_test = "BrainMRI_2.jpg"

img_2d = plt.imread(os.path.join(path, img))
img_test = plt.imread(os.path.join(path, img_test))

def translation(I, p, q):
    nx, ny = I.shape[1], I.shape[0]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1)) 

    ix = np.random.randint(nx, size=100000)
    iy = np.random.randint(ny, size=100000)
    samples = I[iy,ix]
    new_I = griddata((iy+p, ix+q), samples, (Y, X), method='linear')

    return new_I

new_I = translation(img_2d, 25, 0)

fig, ax = plt.subplots(ncols=2)

ax[0].imshow(new_I)
ax[1].imshow(img_test)

plt.show()