import os

import numpy as np
#import cv2
#from osgeo import gdal, ogr, osr
#import os
#from os.path import dirname, abspath
import tifffile as tiff
import tensorflow as tf
#from Extraction.Utils import *
from tflow.cnn_mnist import get_dateset

#img = cv2.imread(r"20160714_wh-vh_01_75m_transparent_mosaic_group1.tif",1)
#print(type(img))
#ig = img[50:50,56:56]
#from tflow.cnn_mnist import get_dateset

#a = tiff.TiffFile('./imgs_veg/train/WHM71_WHL81-55.tif')
#b = a.asarray()
dn = "./imgs_veg"

def check_folder_alpha(dir, dirn):
  ii = 1
  li = str(dir.__len__())
  for file in dir:
    print(dirn +": "+str(ii)+"/"+li+"\r", end="")
    if file.endswith('.tif'):
      check_img_alpha(file.split('.')[0], dir=dirn)
    ii+=1
  print("")


def check_img_alpha(name,dn="./imgs_veg/",dir="train"):
  a = tiff.TiffFile(dn + dir + "/"+name+".tif")
  b = a.asarray()
  a.close()
  for i in range(b.shape[0]):
    t = b[i]
    for x in range(t.shape[0]):
      p = t[x]
      if p[0] == 0 and p[1] == 0 and p[2] == 0 and p[3] == 0:
        os.rename(dn + dir + "/"+name+".tif", dn + "trash/" + name+".tif")
        print('Moved: '+name+" to trash.")
        return
      
check_folder_alpha(os.listdir(dn + "/train"),"train")
check_folder_alpha(os.listdir(dn + "/test"),"test")

#c = tiff.TiffFile('./imgs_veg/train/WHM71_WHL81-56.tif')
#d = c.asarray()
#print('')
