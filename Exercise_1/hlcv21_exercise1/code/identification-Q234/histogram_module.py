import numpy as np
from numpy import histogram as hist
import sys
sys.path.insert(0,'../filter-Q1')
import gauss_module
#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # # your code here
    # bin_size = 256/num_bins
    # hists = np.zeros(num_bins)
    # #converting image into 1-D 
    # img_gray_flattened = img_gray.flatten()
    # for i in range(len(img_gray_flattened)):
    #   bins = np.floor(img_gray_flattened//bin_size) + 1
    #   hists[bins]+=1 

    # hists = hists/hists.sum()
    # return hists, bins
    hists =[]
    #hists = np.zeros(num_bins)


    # print(img_gray.shape)
    img_gray = img_gray.flatten()
    bin_size = 256/num_bins

    start = 0.0
    end = bin_size

    bins = [start]

    while end <= 255:    
      bins_count = 0
      for i in img_gray:
        if i >= start and i <= end:
          bins_count += 1 
      #appending histogram and bin list
      hists.append(bins_count)
      bins.append(end)
      
      start += bin_size
      end += bin_size
      

    hists = np.array(hists)
    hists = hists / hists.sum()

    bins = np.array(bins)

    return hists,bins

#  compute joint histogram for each color channel in the image, histogram should be normalized so that sum of all values equals 1
#  assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
def rgb_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((int(num_bins), int(num_bins), int(num_bins)))
    
    bin_size = 256/num_bins

    # execute the loop for each pixel in the image 
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            # ...
            r = np.floor(img_color[i,j,0]/bin_size)
            g = np.floor(img_color[i,j,1]/bin_size)
            b = np.floor(img_color[i,j,2]/bin_size)
            hists[int(r),int(g),int(b)] += 1
            # pass

    # normalize the histogram such that its integral (sum) is equal 1
    # your code here

    hists = hists.reshape(hists.size)
    hists = hists/hists.sum()
    return hists

#  compute joint histogram for r/g values
#  note that r/g values should be in the range [0, 1];
#  histogram should be normalized so that sum of all values equals 1
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
def rg_hist(img_color, num_bins):

    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'
  
    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    bin_size = (1-0)/(num_bins-1) # to solve index error 
    # your code here
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            r = img_color[i,j,0]/(img_color[i,j,0]+img_color[i,j,1]+img_color[i,j,2])
            g = img_color[i,j,1]/(img_color[i,j,0]+img_color[i,j,1]+img_color[i,j,2])
            r = r/bin_size
            g = g/bin_size
            #bounding  r and g between 1 and num_bins
            if r > num_bins:
              r = num_bins
            elif r < 1:
              r = 1
            if g > num_bins:
              g = num_bins
            elif g < 1:
              g = 1            
            hists[int(r),int(g)] += 1

    hists = hists.reshape(hists.size)
    hists = hists/hists.sum()
    return hists


#  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
#  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
#  histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input grayvalue image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  note: you can use the function gaussderiv.m from the filter exercise.
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # compute the first derivatives
    img_gray_dx , img_gray_dy = gauss_module.gaussderiv(img_gray,sigma=7)

    hists = np.zeros((num_bins, num_bins))

    # quantize derivatives to "num_bins" number of values
    # values in -30 and 30 so quantization should be between -32 and 32
    lower_quant_range = -32
    upper_quant_range = 32  



    bin_size = (upper_quant_range - lower_quant_range +1 )/num_bins #including 0 and -32 to 32

    # define a 2D histogram  with "num_bins^2" number of entries
    for i in range(img_gray.shape[0]):
      for j in range(img_gray.shape[1]):
        dx = np.floor((img_gray_dx[i,j]+32)/bin_size)
        dy = np.floor((img_gray_dy[i,j]+32)/bin_size)
        hists[int(dx),int(dy)] += 1
    
    
    
    hists = hists.reshape(hists.size)
    hists = hists / (img_gray.shape[0]*img_gray.shape[1])  # TO solve IndexError: index 3 is out of bounds for axis 0 with size 3
    hists_sum  = hists.sum()
    hists = hists/hists_sum
    return hists

def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
  if dist_name == 'grayvalue':
    return normalized_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rgb':
    return rgb_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rg':
    return rg_hist(img1_gray, num_bins_gray)
  elif dist_name == 'dxdy':
    return dxdy_hist(img1_gray, num_bins_gray)
  else:
    assert 'unknown distance: %s'%dist_name
  
