import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import numpy as np
import cv2

image = cv2.imread('inuits.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap="gray")

# naive split
idxs = np.linspace(0, image.shape[0], 4).astype(int)
ims = [ image[x:y] for x, y in zip(idxs, idxs[1:]) ]

crop = (np.array(ims[0].shape) * [.09, .08]).astype(int)
def preprocess_for_correlation(a):
    a = a[crop[0]:-crop[0], crop[1]:-crop[1]]
    return a.astype('float32')
keys = list(map(preprocess_for_correlation, ims))



h, w = keys[0].shape
keys[1] = cv2.resize(keys[1], (w, h))
keys[2] = cv2.resize(keys[2], (w, h))

R, G, B = keys

GRcorr = convolve2d(R, np.flip(np.flip(G, axis=0), axis=1), mode="same")

plt.imshow(GRcorr)
plt.show()
