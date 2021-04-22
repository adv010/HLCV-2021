# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import scipy
from scipy.signal import convolve2d as conv2

def gauss(sigma):
    # x = np.arange(-3*sigma, 3*sigma)      #does not include endpoint
    start = np.ceil(-3*sigma)
    end = np.floor(3*sigma)
    x = np.linspace(start, end, end-start+1)
    Gx = np.power(math.e,-np.multiply(x,x)/(2*sigma*sigma)) / (math.sqrt(2*math.pi)*sigma)
    return Gx, x

def gaussderiv(img, sigma):
    Dx, _ = gaussdx(sigma)
    Dx = Dx.reshape(1, Dx.size)
    imgDx = conv2(img, Dx, 'same')
    imgDy = conv2(img, Dx.T, 'same')

    return imgDx, imgDy

def gaussdx(sigma):
    start = np.ceil(-3*sigma)
    end = np.floor(3*sigma)
    x = np.linspace(start, end, end-start+1)
    D = -1 * np.multiply(x, np.power(math.e,-np.multiply(x,x)/(2*sigma*sigma))) / (math.sqrt(2*math.pi)*sigma*sigma*sigma)
    return D, x

def gaussianfilter(img, sigma):
    Gx, _ = gauss(sigma)

    # print(x)
    # Gx_2d = np.tile(Gx, (len(Gx),1))
    # print(Gx_2d)
    # Gy_2d = np.transpose(Gx_2d)
    # print(Gy_2d)
    # outimage = scipy.signal.convolve2d(img, Gx_2d*Gy_2d)

    # print(Gx)
    Gx = Gx.reshape(1, Gx.size)
    # print(Gx)
    outimage = conv2(conv2(img, Gx, 'same'), Gx.T, 'same')

    return outimage
