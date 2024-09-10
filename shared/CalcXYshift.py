#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.pyplot import *
from matplotlib import image
from numpy import *
import os.path
from scipy.signal import fftconvolve

__all__ = ['ind2sub', 'CalcXYshift']

def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind[0] // array_shape[1])
    cols = (ind[1] % array_shape[1])
    return (rows, cols)

def CalcXYshift(Pic, convPic, D0, border_thickness = 10):
    # Gaussian high pass filter
    im = Pic
    m, n = shape(im)
    # range of variables
    u = linspace(0, m+1, m)
    v = linspace(0, n+1, n)
    # Compute the indices for use in meshgrid.
    idx = where(u > m/2)[0]
    u[idx] = u[idx]-m
    idy = where(v > n/2)[0]
    v[idy] = v[idy]-n
    # compute the meshgrid array
    V, U = meshgrid(v, u)
    # transfer function
    D = sqrt(U**2 + V**2)
    H = fft.ifftshift(exp(-(D**2)/(2*(D0**2))))
    Hgh = 1-H
    #images to filter
    im = Pic
    F_u_v = fft.fft2(im)
    F_u_v = fft.fftshift(F_u_v)
    g = fft.ifft2(Hgh * F_u_v)
    
    im = convPic
    F_u_v = fft.fft2(im)
    F_u_v = fft.fftshift(F_u_v)
    gc = fft.ifft2(Hgh * F_u_v)
    
    #calculate correlation without border effects (10 pixels out from each border)
    ar = flipud(fliplr(abs(g[border_thickness:-border_thickness,border_thickness:-border_thickness])))
    cc = fftconvolve(abs(gc[border_thickness:-border_thickness,border_thickness:-border_thickness]), ar.conj(), mode='same')
    
    idx_max = unravel_index(cc.argmax(), cc.shape)
    true_max = amax(cc)
    max_cc = cc[idx_max]
    # print ('true max', true_max)
#     [x,y] = ind2sub(shape(cc), idx_max)
    x, y = idx_max
    print (max_cc, idx_max, x, y)

#     xshift = y - shape(Pic)[0]+20 # plus 20 from borders taken out
#     yshift = x - shape(Pic)[1]+20 # plus 20 from borders taken out

    xshift = y + 2*border_thickness - shape(Pic)[0]/2 # plus 20 from borders taken out
    yshift = x + 2*border_thickness - shape(Pic)[1]/2 # plus 20 from borders taken out

#     figure(figsize=(12,4))
#     subplot(131)
#     imshow(abs(g))
#     subplot(132)
#     imshow(abs(gc))
#     subplot(133)
#     imshow(cc)

    return int(xshift), int(yshift)