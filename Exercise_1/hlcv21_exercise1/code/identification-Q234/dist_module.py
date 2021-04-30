import numpy as np
# 
# compute chi2 distance between x and y
#
def dist_chi2(x,y):
  num = (x-y)**2
  denom = x+y
  dist = np.zeros(num.shape)
  for i in range(num.shape[0]):
    if denom[i] != 0:
      dist[i] = num[i]/denom[i]
  return dist.sum()

# compute l2 distance between x and y
def dist_l2(x,y):
  dist = np.linalg.norm(x - y)
  return dist

# 
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similart histograms
#
def dist_intersect(x,y):
  d_intersection = np.minimum(x,y)
  d_intersection = d_intersection.sum()
  d_intersection = 1-d_intersection
  return d_intersection

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert 'unknown distance: %s'%dist_name
  
