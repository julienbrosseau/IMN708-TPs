import numpy as np
import matplotlib.pyplot as plt

import utils
ut = utils.Utils()

img_4d = ut.open_fmri()
print(img_4d.shape)

ideal_data = ut.get_ideal()

slices_sagittal = np.arange(64) #np.arange(20, 40)
slices_coronal  = np.arange(64) #np.arange(20, 40)
slices_axial    = np.arange(50) #np.arange(15, 35)

active_area = []

time = np.arange(85)
freq = np.fft.fftfreq(time.shape[-1])
 
#ut.show_histogram(img_4d[32, :, :, 0])

for sl_s in slices_sagittal:
    for sl_c in slices_coronal:
        for sl_a in slices_axial:
            img_tps = img_4d[sl_s, sl_c, sl_a, :]
            #print(img_tps)

            # On visualise la FFT pour chaque voxel
            sp = np.fft.fft(img_tps)
            #plt.plot(freq, fft.real)
            # On enleve les fréquences correspondant au coeur
            """
            for i, f in enumerate(freq):
                if abs(f) >= 0.06:
                    sp[i] = 0. + 0.j
            
            smoothed = np.fft.ifft(sp)
            """
            smoothed = img_tps
            # Par rapport a l'histogramme, on ne prend en compte que les voxels correspondant au cerveau
            if img_4d[sl_s, sl_c, sl_a, 0] > 300:
                # Correlation entre l'ideale et notre image
                corr2 = ut.get_corr2(ideal_data, smoothed.real)

                # On recupere seulement les voxels qui "enregistrent" l'activité
                if corr2 < -0.7 or corr2 > 0.7:
                    #print("Correlation entre 'ideal' et le voxel :", format(corr2, '.2f'), "% pour le voxel :", sl_s, sl_c, sl_a)
                    #plt.plot(time, smoothed)
                    active_area.append([sl_s, sl_c, sl_a, format(corr2, '.2f')])

            else:
                img_4d[sl_s, sl_c, sl_a, 0] = 0

print(active_area)
plt.show()