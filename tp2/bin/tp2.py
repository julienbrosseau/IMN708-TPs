# TP2 - Recalage
# Trucs  à pas coder :
#   Extrapolation / Interpolation
# Il faut prendre en compte les bords (contour de l'image) soit 0 soit median de l'image
# Bruit gaussien sur les images à recaler (il faudrait mieux débruiter avant le traitement)
# On peut rajouter un courbe d'évolution de SSD pour voir que c'est décroissant
# Brain1 > Brain4 faire des astuces de reclage sinon ça convergera jamais

print("start \n")

def jointHist(img1, img2, bin):
    return 0

# Histogramme - Bruit présent
def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()

#show_histogram(img)