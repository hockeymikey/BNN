#import numpy as np
#import cv2
#from osgeo import gdal, ogr, osr
#import os
#from os.path import dirname, abspath
#import tifffile as tiff
#import tensorflow as tf

#from tflow.cnn_mnist import get_dateset

#img = cv2.imread(r"20160714_wh-vh_01_75m_transparent_mosaic_group1.tif",1)
#print(type(img))
#ig = img[50:50,56:56]
#from tflow.cnn_mnist import get_dateset

# img = cv2.imread(r"20160714_wh-vh_01_75m_transparent_mosaic_group1.tif",1)
# print(type(img))
# ig = img[50:50,56:56]
# from tflow.cnn_mnist import get_dateset
from osgeo._gdalconst import GA_ReadOnly

from data_prep.data_utils import *
from data_prep.custom_classes import Mosaic_n
from data_prep.moz_utils import get_bounding_box
from data_prep.tif_utils import OpenTif_latlong


# from tflow.cnn_mnist import get_dateset

def cords2pixel(bbox,name="../../Mosaics/2016/20160714_wh-vh_01_75m_transparent_mosaic_group1.tif"):
  mos = Mosaic_n(name)
  
  con_y = (mos.max_lat - mos.min_lat) / mos.y
  con_x = (mos.max_long - mos.min_long) / mos.x
  
  n_b = []
  for point in bbox:
    n_y = int(math.ceil((point[0] - mos.min_lat) / con_y))
    n_x = int(math.ceil((point[1] - mos.min_long) / con_x))
    n_b.append([n_y,n_x])
  #50 = lat
  return n_b


y = Decimal('58.71931837720249161712328510')
x = Decimal('-93.44822577360472280246432206')
bbox = get_bounding_box(y, x, 0.5)
pixels = cords2pixel(bbox,"../../Mosaics/2016/20160714_wh-vh_01_75m_transparent_mosaic_group1.tif")
tmp = OpenTif_latlong("../../Mosaics/2016/20160714_wh-vh_01_75m_transparent_mosaic_group1.tif")


ds = gdal.Open("../../Mosaics/2016/20160714_wh-vh_01_75m_transparent_mosaic_group1.tif", GA_ReadOnly)
print('')


'''
kk = 'test'[1:]
print('test')
a = tiff.TiffFile('imgs_veg/WHI11_WHJ21-1.tif')
b = a.asarray()
c= np.reshape(b,-1,order='F')
a.close()

data, labels, unique, l_c = get_dateset()
n = 2
i = 1
while (n*2) < unique.__len__():
  n *= 2
  i+=1

#tmp = np.eye(6)[labels]
er = tf.one_hot(labels,i,1.0,0.0)
print(er)'''