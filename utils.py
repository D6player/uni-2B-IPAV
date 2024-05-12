import numpy as np

def pad_arrays(R, G, B):
    Ry, Rx = R.shape
    Gy, Gx = G.shape
    By, Bx = B.shape

    max_x = max([Rx, Gx, Bx])
    max_y = max([Ry, Gy, By])

    R = np.pad(R, [(0, max_y - Ry), (0, max_x - Rx)], mode="constant")
    G = np.pad(G, [(0, max_y - Gy), (0, max_x - Gx)], mode="constant")
    B = np.pad(B, [(0, max_y - By), (0, max_x - Bx)], mode="constant")
    
    return (R, G, B)

def correlation(a, b):
    a_fft = np.fft.fft2(a)
    b_fft = np.fft.fft2(b)
    ab_sfft = a_fft * np.conjugate(b_fft)
    ab_sfft /= np.abs(ab_sfft)
    ab_corr = np.fft.ifft2(ab_sfft).real
    
    return ab_corr
