import numpy as np
import matplotlib.pyplot as plt

import utils
ut = utils.Utils()

img_4d = ut.open_fmri()
print(img_4d.shape)

ideal_data = ut.get_ideal()

slices_sagittal = np.arange(20, 40)
slices_coronal  = np.arange(20, 40)
slices_axial    = np.arange(15, 35)

time = np.arange(85)
freq = np.fft.fftfreq(time.shape[-1])
 
ut.show_histogram(img_4d[32, :, :, 0])

for sl_s in slices_sagittal:
    for sl_c in slices_coronal:
        for sl_a in slices_axial:

            # Par rapport a l'histogramme, on ne prend en compte que les voxels correspondant au cerveau
            if img_4d[sl_s, sl_c, sl_a, 0] > 300:
                img_tps = img_4d[sl_s, sl_c, sl_a, :]
                #print(img_tps)
                
                # On visualise la FFT pour chaque voxel
                #fft = np.fft.fft(img_tps)
                #plt.plot(freq, fft.real)

                # Correlation entre l'ideale et notre image
                corr2 = ut.get_corr2(ideal_data, img_tps)

                # On recupere seulement les voxels qui "enregistrent" l'activité
                if corr2 < -0.4 or corr2 > 0.4:
                    print("Correlation entre 'ideal' et le voxel :", format(corr2, '.2f'), "% pour le voxel :", sl_s, sl_c, sl_a)
                    plt.plot(time, img_tps)

            else:
                img_4d[sl_s, sl_c, sl_a, 0] = 0

plt.show()