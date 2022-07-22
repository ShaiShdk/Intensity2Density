# Testing

import Intensity2Density as I2D
import matplotlib.pyplot as plt

path_to_image = 'Intensity_profile.tiff'
img = plt.imread(path_to_image)
I2D.Convert_Pattern_to_Points(img, 200 , [10,10])
