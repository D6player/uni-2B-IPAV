#!/usr/bin/env python3
import sys
from PIL import Image
import numpy as np

# parse args, load image
args = sys.argv[1:]
if len(args) != 2:
    print(f'Usage: {sys.argv[0]} <input image> <output image>', file=sys.stderr)
    exit(1)
infile, outfile = args

image = np.array(Image.open(infile))
assert image.dtype == 'uint8' and image.ndim == 2

# split panels naively
idxs = np.linspace(0, image.shape[0], 4).astype(int)
ims = [ image[x:y] for x, y in zip(idxs, idxs[1:]) ]

# preprocess for correlation
crop = (np.array(ims[0].shape) * [.09, .08]).astype(int)
def preprocess_for_correlation(a):
    a = a[crop[0]:-crop[0], crop[1]:-crop[1]]
    return a.astype('float32')
keys = list(map(preprocess_for_correlation, ims))

# obtain offsets
def get_offset(a, b):
    shape = np.maximum(a.shape, b.shape)
    convfreq = np.fft.fft2(a, shape) * np.fft.fft2(b, shape).conj()
    conv = np.fft.ifft2(convfreq).real
    offset = np.unravel_index(np.argmax(conv), shape)
    offset = tuple( (x + w//2) % w - w//2 for x, w in zip(offset, shape) )
    assert all(abs(x) < w * 0.1 for x, w in zip(offset, shape)), f'offset too big'
    return offset
offsets = [(0, 0)] + [ get_offset(keys[0], keys[i]) for i in range(1, 3) ]

# stitch image
shapes = np.array([ im.shape for im in ims ])
bounds = np.array([np.amax(offsets, axis=0), np.amin(offsets + shapes, axis=0)])
panelbounds = ( bounds - offset for offset in offsets )
stitched = [ im[b[0][0]:b[1][0], b[0][1]:b[1][1]] for im, b in zip(ims, panelbounds) ]
stitched = np.moveaxis(stitched[::-1], 0, -1)

Image.fromarray(stitched).save(outfile)
