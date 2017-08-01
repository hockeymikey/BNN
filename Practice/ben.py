import os

import cupy as cp
import math
import numpy as np
from decimal import Decimal

import pathlib
from numpy.lib.stride_tricks import as_strided

import data_prep.tif_utils as tif_size

from data_prep.moz_utils import get_step_size, get_bounding_box, latLonToPixel, Moz_contains
from data_prep.lbl_utils import get_sorted_veg_from_excel, get_veg_label_of_img
from data_prep.custom_classes import v_point
import tifffile as tiff

def sort_to_labeled_dirs(dir,xl_file="../Veg/2016_veg_data.xlsx",toGrab=['unsafe','test'],clean_mode='tiny'):
  cols = get_sorted_veg_from_excel(file=xl_file)
  
  for root, dirs, files in os.walk(dir):
    #dirs[:] = [d for d in dirs if d in toGrab]
    for f in [file for file in files if file.endswith(('.tif'))]:
      pl = pathlib.Path(root)
      stem = pl.stem
      parent = str(pl.parent)

      tc = f.split('.')
      pp = get_veg_label_of_img(tc[0], cols, mode=clean_mode)
      
      if not os.path.exists(parent+'/'+pp):
        os.makedirs(parent+'/'+pp)
      os.rename(str(pl)+'/'+f,parent+'/'+pp+'/'+f)
      print('Moved: '+f)

sort_to_labeled_dirs('../veg_imgs/40x',clean_mode='tiny')

def old():
  file = open('./Extraction/output_KML.csv')
  
  # v = file.readline().replace('\n', '').split(',')
  
  points = {}
  cMs = {}
  # 128
  # JGBB3_JGBA4
  Mos = tif_size.MinMax_Extract('../Mosaics')
  mm = Mos['20160714_wh-vh_01_75m_transparent_mosaic_group1.tif']
  
  img = tiff.TiffFile('../Mosaics/2016/20160714_wh-vh_01_75m_transparent_mosaic_group1.tif')
  for line in file.readlines():
    if line.startswith('JGBB2'):
      v = line.replace('\n', '').split(',')
      a = v_point(v[0], [v[1], v[2]])
    if line.startswith('JGBA3'):
      v = line.replace('\n', '').split(',')
      b = v_point(v[0], [v[1], v[2]])
    if line.startswith('JGBA2'):
      v = line.replace('\n', '').split(',')
      tl = v_point(v[0], [v[1], v[2]])
  cols = get_sorted_veg_from_excel(file="../Veg/2016_veg_data.xlsx")
  
  step_size = get_step_size(a, b, cols, tl)
  de = abs(((a.x - b.x) ** 2) + ((b.y - a.y) ** 2).sqrt())
  ttt = de / step_size
  
  Xa = a.x
  Ya = a.y
  Xb = b.x
  Yb = b.y
  d = ((Xb - Xa) ** 2) + ((Yb - Ya) ** 2).sqrt()
  m = (Ya - Yb) / (Xa - Xb)
  
  d2 = Decimal(0.000010782184135314802)
  leg = 129
  
  if Xa > Xb:
    Xc = Xa - (((d2 * leg) * (Xa - Xb)) / d)
  else:
    Xc = Xa + (((d2 * leg) * (Xa - Xb)) / d)
  if Ya > Yb:
    Yc = Ya - (((d2 * leg) * (Ya - Yb)) / d)
  else:
    Yc = Ya + (((d2 * leg) * (Ya - Yb)) / d)
  
  pixels = latLonToPixel(mm.root + mm.name, get_bounding_box(Yc, Xc, 0.5))
  
  if not (abs(pixels[1][1] - pixels[0][1]) == 46):
    pixels[1][1] = (46 - (pixels[1][1] - pixels[0][1])) + pixels[1][1]
  if not (abs(pixels[1][0] - pixels[0][0]) == 46):
    pixels[1][0] = (46 - (pixels[1][0] - pixels[0][0])) + pixels[1][0]
  conMoz = Moz_contains(mm, Xc, Yc)
  nimg = img.asarray()[pixels[0][1]:pixels[1][1], pixels[0][0]:pixels[1][0]]
  
  kk = {"kk": 'op', "op": 'kk'}
  img.close()
  op = list(kk.keys())
  oop2 = list(kk)
  
  pp = np.get_include()
  name = v_point('WHI1')
  ii = str(int(name.row) + 1)
  
  with cp.cuda.Device(0):
    batch_mask = cp.random.choice(1000, 100)
    x_cpu = np.array([1, 2, 3])
    x_gpu = cp.asarray(x_cpu)
    x = np.array([1, 2, 3, 4], dtype=np.int16)
    x_gpu = cp.asnumpy(x_gpu)
    pi = np.array(x_gpu)
    xx = as_strided(x_gpu, strides=(2 * 2,), shape=(2,))
    pass