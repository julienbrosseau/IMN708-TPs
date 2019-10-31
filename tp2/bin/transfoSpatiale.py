# 3. Transformations spaciales

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import numpy as np
import math

fig = plt.figure()
ax = fig.gca(projection='3d')

# Generation de l'ensemble des points
x, y, z = np.meshgrid(np.arange(0, 20, 1),
                      np.arange(0, 20, 1),
                      np.arange(0, 5, 1))

n_x, n_y, n_z = np.meshgrid(np.arange(0, 20, 1),
                            np.arange(0, 20, 1),
                            np.arange(0, 5, 1))

def trans_rigide(theta=None, omega=None, phi=None, p=None, q=None, r=None):
    matrix = [[]]

    if theta != None:
        # Rotation sur l axe des x
        matrix = [
            [1, 0, 0, 0],
            [0, math.cos(theta), -math.sin(theta), 0],
            [0, math.sin(theta), math.cos(theta), 0],
            [0, 0, 0, 1]]

    elif omega != None:
        # Rotation sur l axe des y
        matrix = [
            [math.cos(omega), 0, -math.sin(omega), 0],
            [0, 1, 0, 0],
            [math.sin(omega), 0, math.cos(omega), 0],
            [0, 0, 0, 1]]
    
    elif phi != None:
        # Rotation sur l axe des z
        matrix = [
            [math.cos(phi), -math.sin(phi), 0, 0],
            [math.sin(phi), math.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
    
    else:
        # Translation
        matrix = [
            [1, 0, 0, p],
            [0, 1, 0, q],
            [0, 0, 1, r],
            [0, 0, 0, 1]]

    return matrix

def similitude(s=1, theta=None, omega=None, phi=None, p=None, q=None, r=None):
    matrix = [[]]

    if theta != None:
        # Rotation sur l axe des x
        matrix = [
            [s, 0, 0, 0],
            [0, s*math.cos(theta), -math.sin(theta), 0],
            [0, math.sin(theta), s*math.cos(theta), 0],
            [0, 0, 0, 1]]

    elif omega != None:
        # Rotation sur l axe des y
        matrix = [
            [s*math.cos(omega), 0, -math.sin(omega), 0],
            [0, s, 0, 0],
            [math.sin(omega), 0, s*math.cos(omega), 0],
            [0, 0, 0, 1]]
    
    elif phi != None:
        # Rotation sur l axe des z
        matrix = [
            [s*math.cos(phi), -math.sin(phi), 0, 0],
            [math.sin(phi), s*math.cos(phi), 0, 0],
            [0, 0, s, 0],
            [0, 0, 0, 1]]
    
    else:
        # Translation
        matrix = [
            [s, 0, 0, p],
            [0, s, 0, q],
            [0, 0, s, r],
            [0, 0, 0, 1]]

    return matrix

#matrix = trans_ridge(theta = 10)
#matrix = trans_ridge(omega = 10)
#matrix = trans_ridge(phi = 10)
#matrix = trans_ridge(p=1, q=1, r=1)

#matrix = similitude(s=2, theta = 10)
#matrix = similitude(s=2, omega = 10)
#matrix = similitude(s=2, phi = 10)
matrix = similitude(s=2, p=1, q=1, r=1)


for i in range(x[0][:,0].size):
    for j in range(y[0][:,0].size):
        for l in range(z[0][0].size):
            v = [i,j,l,1]
            v_t = np.dot(matrix, v)

            n_x[i][j][l] = v_t[0]
            n_y[i][j][l] = v_t[1]
            n_z[i][j][l] = v_t[2]

ax.scatter(x, y, z)

ax.scatter(n_x, n_y, n_z)

plt.show()



