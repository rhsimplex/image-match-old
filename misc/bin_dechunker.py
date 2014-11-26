"""
Utility script for the 80-million-tiny-images dataset:

http://horatio.cs.nyu.edu/mit/tiny/data/index.html

The data is unhelpfully provided in a giant binary with
only matlab scripts to get images out of it. This extracts
all the images and dumps them into an 'images' directory.
"""
from skimage.io import imsave
import numpy as np
import array

filename = 'tiny_images.bin'
target_dir = 'images/' 
TOTAL_NUM_IMGS = 79302017
sx = 32
n_bytes_per_image = sx * sx * 3

with open(filename, 'rb') as f:
    for i in range(TOTAL_NUM_IMGS):
        a = array.array('B')
        a.fromfile(f, n_bytes_per_image)
        res = np.array(a, dtype='uint8').reshape((sx,sx,3), order='F')
        imsave(''.join([target_dir, str(i), '.png']), res)

