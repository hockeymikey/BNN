import os

import numpy as np

from data_prep.lbl_utils import get_sorted_veg_from_excel, get_veg_label_of_img


def get_veg_dateset(filex="./Veg/2016_veg_data.xlsx", cols=None, dn='./Extraction/imgs_veg/train', size=-1, l_c={}):
  if cols == None:
    cols = get_sorted_veg_from_excel(file=filex)
  
  dir = os.listdir(dn)
  # Check type on the array if it should be float or what float32
  d = None
  
  # dir.__len__()
  #l = np.array([], dtype=np.int32)

  # l_c is used to convert from onehot int to label value
  # l_s is to count number of each label
  l_s = {}
  i = 0
  u = 0
  
  for file in dir:
    if file.endswith('.tif') and (size == -1 or i < size):
      
      # d[:,i] = np.reshape(ti.asarray(), -1, order='F')
      # d = np.vstack((d, np.reshape(ti.asarray(), -1, order='F')))
      
      
      pp = get_veg_label_of_img(file.split('.')[0], cols)
      
      if isinstance(pp, str):
        # t2 value or ti.asarray can be used.
        
        l_s.setdefault(pp, 0)
        l_s[pp] += 1
        
        if not (l_c.__contains__(pp)):
          l_c.setdefault(pp, u)
          l_c.setdefault(u, pp)
          u += 1
        
        #l = np.append(l, l_c[pp])
      
      i += 1
    elif size != -1 and i > size:
      break
  #return (d, l, l_s, l_c)
  for x in l_s:
    print(x)
    #print('')
    
get_veg_dateset()