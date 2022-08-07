#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in June 2021
@author: Shahriar Shadkhoo -- CALTECH
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def Convert_Pattern_to_Points(img , density , XY_lens , save_converted=True, output_name='converted2density.png'):

    """
        A function that takes a hand-drawn shape in the form of gray-scale image
        and outputs a random seeded points with determined density in the region of interest

        Input Args:
            - img:             The input image in grayscale or rgb format
            - density:         The density scale, determined by the maximum desired density in the output
            - XY_lens:         The size of the box onto which the original image is to be mapped; type: list or numpy.array
            - save_converted:  Optional argument for saving the output image.

        Outpuy Args:
            - R_cnts:          The (x,y) coordinates of the generated particles
            - N_tot:           Total number of generated particles
            - img_converted:   The output image
            - dens:            The re-scaled of after mapping

    """

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    img_gray = img

    if img.ndim >= 3:
        img_gray = np.around(rgb2gray(img))

    Ly,Lx = img_gray.shape

    RegInds = list(np.where(img_gray.ravel() != 0)[0])
    Area    = len(RegInds)
    dens    = (density) * np.prod(np.asarray(XY_lens))/Area
    pks     = (dens/np.max(img_gray)) * img_gray.ravel()[RegInds]

    img_converted = np.zeros((Lx*Ly,1))
    img_converted[RegInds] = np.random.binomial(1,pks,Area).reshape((Area,1))
    img_converted = np.reshape(img_converted, [Ly,Lx])

    y_cnts,x_cnts = np.where(img_converted != 0)
    Ntot    = len(x_cnts)

    R_cnts  = np.concatenate((x_cnts.reshape(Ntot,1),y_cnts.reshape(Ntot,1)),axis=1)
    R_cnts  = (R_cnts) * np.array([1,-1])
    R_cnts -= np.mean(R_cnts, axis=0).astype(int)
    R_cnts  = R_cnts.astype(float)
    x_cnts  = R_cnts[:,0]
    y_cnts  = R_cnts[:,1]

    fig = plt.figure(facecolor='k',figsize=(10,10),dpi=100)
    plt.scatter(x_cnts , y_cnts, color=[.8,.8,.8], s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('black')
    plt.xlim([1.2*np.min(x_cnts) , 1.2*np.max(x_cnts)])
    plt.ylim([1.2*np.min(y_cnts) , 1.2*np.max(y_cnts)])
    if save_converted:
        plt.savefig(output_name)
    plt.show()
    plt.close()

    return R_cnts , Ntot , img_converted , dens

if __name__ == "__main__":
    path_to_image = 'Intensity_profile.tiff'
    img = plt.imread(path_to_image)
    Convert_Pattern_to_Points(img, 200 , [10,10])
    
    



