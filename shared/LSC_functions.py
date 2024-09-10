# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:45:22 2023

@author: goethas
"""

import os

from numpy import *

from matplotlib.pyplot import *
import matplotlib.image as mpimg
import matplotlib.patches as pcs
from matplotlib import image
import pandas as pd
from scipy.signal import fftconvolve
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, interp2d

__all__ = ['local_spatial_contrast', 'receptive_field']

def local_spatial_contrast(image, ellipse_points, ellipse_weights):
    '''
    Measuring the local spatial contrast in the RF following Liu 2022's paper.
    
    '''
    ellipse_luminance = image[ellipse_points[:,1], ellipse_points[:,0]]

    # Weber contrast in ellipse
    im_mean_luminance = np.mean(image)
    ellipse_weber = [(ellipse_luminance[k]-im_mean_luminance)/im_mean_luminance \
                for k in range(len(ellipse_points[:,0]))]

    # Weighted mean of the pixels values
    ellipse_mean = sum(ellipse_weights * ellipse_weber)/len(ellipse_weber)
    mean_intensity = ellipse_mean

    # Local contrast
    ellipse_contrasts = 0
    for k in range(len(ellipse_points[:,0])):
        contrast = (ellipse_weights[k] * ellipse_weber[k] - ellipse_mean)**2
        ellipse_contrasts += contrast

    local_contrast = np.sqrt(ellipse_contrasts/(len(ellipse_weber)-1))
    
#     # Local contrast 0.5
#     ellipse_contrasts_05 = 0
#     for k in range(len(ellipse_points[:,0])):
#         contrast_05 = ellipse_weights[k]**2 * (ellipse_luminance[k] - 0.5)**2
#         ellipse_contrasts_05 += contrast_05

#     local_contrast_05 = np.sqrt(ellipse_contrasts_05/(len(ellipse_weber)-1))
    
    return mean_intensity, local_contrast

def receptive_field(k, sigma_c, sigma_s, filter_size, x0, y0, OFF=False, plotting=False):

    x = np.arange(-filter_size/2 + x0, filter_size/2 + x0, 1)
    y = np.arange(-filter_size/2 + y0, filter_size/2 + y0, 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    
    r = sqrt(xx**2 + yy**2) 
    rfc = exp(-((r/sigma_c)**2)) 
    rfs = exp(-((r/sigma_s)**2)) 
    rf = rfc/sum(abs(rfc)) - k * rfs/sum(abs(rfs))
    if OFF:
        rf = -rf
    
    if plotting:
        figure(figsize=(10,5))
        
        subplot(121)
        h = contourf(x, y, rf, cmap='RdBu_r', levels=50)
        axis('scaled')
        xlabel('x (pix)')
        ylabel('y (pix)')
        
        subplot(122)
        plot(x, rf[int(len(rf)/2)], 'k', label='receptive field')
#         plot(x, rfc[int(len(rf)/2)], 'C3', label='center')
#         plot(x, rfs[int(len(rf)/2)], 'C0', label='surround')
#             xlim(-filter_size/2, filter_size/2)
        xlabel('x (pix)')
        legend(frameon=False)
        suptitle ('ON receptive field')
        
        show()
    
    return x, y, rf, rfc, rfs
