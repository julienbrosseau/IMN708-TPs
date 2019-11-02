import numpy as np

def jointHist(I, J, bin):
    hist = np.zeros((bin, bin))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            intensity1 = I[i, j]
            intensity2 = J[i, j]
            binIntensity1 = int(intensity1 / (255 / (bin - 1)))
            binIntensity2 = int(intensity2 / (255 / (bin - 1)))
            hist[binIntensity2, binIntensity1] = hist[binIntensity2, binIntensity1] + 1
            print(hist[binIntensity2, binIntensity1])
    return hist

def SSD(I, J):
    sum = 0
    for x in range(I.shape[0]):
        for y in range(I.shape[1]):
            sum += (int(I[x, y]) - int(J[x, y]))**2
    return sum

def SSD2(I, J):
    return np.sum(np.square(I - J))

def CR(I, J):
    i = I.mean()
    j = J.mean()
    top = np.sum((I - i) * (J - j))
    downI = np.sum(np.square(I - i))
    downJ = np.sum(np.square(J - j))
    return (top / (np.sqrt(downI)*np.sqrt(downJ)))