import torch
import torchvision.ops
from torch import nn
import numpy as np
from numpy import dot
from numpy.linalg import norm
import cv2
import time
from osgeo import gdal, osr
import os
import argparse
from torchvision.ops import nms
import glob
import scipy
from scipy import signal
import math
import gc
pading = 1000

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))


def contrast(arr):# this function for calculating contrast
    ui = np.sqrt(np.mean((np.abs(arr)**2 - np.mean(np.abs(arr)**2))**2))
    bi = np.mean(np.abs(arr)**2)

    return ui/bi

def amplitude(SLC_data):
    band1 = np.real(SLC_data)
    band2 = np.imag(SLC_data)

    return  np.sqrt(band1**2 + band2**2)

def sub_ap_um(SLC_data):#input is file path
    print(SLC_data.shape)
    #band_complex = da.from_array(SLC_data, chunks=(1000, 1000)) # Adjust chunk size as needed

    # Rechunk so that there is only one chunk along the axis you're performing the FFT on
    #band_complex = band_complex.rechunk({0: -1, 1: -1})

    range_window_size = round(SLC_data.shape[0] / 1)
    azimuth_window_size = round(SLC_data.shape[1] / 2) #8
    wx = signal.windows.kaiser(azimuth_window_size,0.5)
    wy = signal.windows.kaiser(range_window_size,0.5)

    masky, maskx = np.meshgrid(wx, wy, indexing='ij')
    masky, maskx = masky.swapaxes(1,0),maskx.swapaxes(1,0)
    window2D=masky * maskx ## gabor transform for extract specific frequency

    data_SLC_2D = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.fft.fft(SLC_data, axis=0), axes=0), axis=1), axes=1).astype('complex64')#.compute()

    window2D_pad = np.zeros((SLC_data.shape[0], SLC_data.shape[1]))
    azimuth_space_start = [0, math.floor(SLC_data.shape[1] / 2)]

    print(azimuth_space_start)
    del SLC_data,maskx,masky,wx,wy ;gc.collect()

    for num in range(1, len(azimuth_space_start) + 1):
        start_azimuth = (azimuth_space_start[num - 1])
        start_range = 0  # Adjust as needed

        window2D_pad_new = np.copy(window2D_pad)
        window2D_pad_new[start_range:start_range + range_window_size, start_azimuth:start_azimuth + azimuth_window_size] = window2D

        data_SLC_2D_multiply = data_SLC_2D * window2D_pad_new
        #data_SLC_2D_multiply = data_SLC_2D_multiply.rechunk({0: -1, 1: -1})

        #mem_usage()
        if num==1:
            del window2D_pad_new;gc.collect()
            data_SLC_2D_multiply_1 = np.fft.ifft2(data_SLC_2D_multiply).astype('complex64')
            #mem_usage()
        else:
            del window2D_pad,window2D_pad_new;gc.collect()
            data_SLC_2D_multiply_2 = np.fft.ifft2(data_SLC_2D_multiply).astype('complex64')

    return data_SLC_2D_multiply_1, data_SLC_2D_multiply_2
