from astropy.io import fits
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

ffile = fits.open('../Data for Deep Learning/BrightComets/C_2007 Q3/C:2007 Q3/00805b/089/00805b089-w1-int-1b.fits')

ffile.info()

imdata = ffile[0].data

print(type(imdata))
print(imdata.shape)

plt.imshow(imdata, cmap='gray')