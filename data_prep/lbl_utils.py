import math
import os
import pickle
import re

import pandas as pd

from data_prep.custom_classes import v_seg, v_point
from data_prep.moz_utils import points_2_direction

#basic list of all labels.  Never used really as I moved to regex.
lbls = ['A','A1','B1+','B1-','B2+','B2-','C1','C2'
        ,'C3','C4','C5','D','E','F','F*','G','H',
        'I','S','Sb','Sc','Sp','Sl','Sm','Sr','Mg',
        'Bg','L','Hv','Ps','Tm','Tp','Ro','Ma',
        'P','B']
#+ p

def extract_values_from_point_name(name):
  '''
  Given a name of a point (WHM1) returns list with extracted values (WH, M, 1)
  :param name:
  :return:
  '''
  extrated = {}

  #Zone, Col, Row
  if not(name.startswith('WH')) or not(name[3].isdigit()):
    extrated['zone'] = (name[0:3])
    extrated['col'] = (name[3])
    extrated['row'] = (name[4])
  else:
    extrated['zone'] = (name[0:2])
    extrated['col'] = (name[2])
    extrated['row'] = (name[3])
  return extrated


def get_sorted_veg_from_excel(file ="../../Veg/2016_veg_data.xlsx", root="./cache/"):
  '''
  Given an excel file of the veg data, sorts the points and line segments between them where data was measured.
  Sorted by: Zone -> Col -> Row -> Segment (NW-SE or NE-SW) -> [v_seg object, [Label of veg with index being step number]]
  '''
  
  if not(os.path.isfile(root+'sorted_veg_labels.pkl')):
    cols = {}
    xl = pd.ExcelFile(file)
  
    for sh in xl.sheet_names:
      if sh == "Veg Codes":
        continue
      dfl = xl.parse(sh)
    
      for c in dfl.items():
        if c[0] != "Cell":
          head = c[0]
          size = 0
        
          for po, v in c[1].items():
            # Direction col
            if po == 0:
              zone = str(sh).split('-')
              dir = v
              cols.setdefault(zone[0], {}).setdefault(zone[1], {}).setdefault(c[0][1], {}).setdefault(dir, [
                v_seg(zone[0] + zone[1] + c[0][1], 1), []])
              continue
            elif v == "NaN" or (type(v) is float and math.isnan(float(v))):
              continue
            else:
              size = size + 1
              zone = str(sh).split('-')
            
              # Zone                      Column                    Row                     Seg (direction)
              cols.setdefault(zone[0], {}).setdefault(zone[1], {}).setdefault(c[0][1], {}).setdefault(dir, [
                v_seg(zone[0] + zone[1] + c[0][1], 1), []])[1].append(v)
  
    with open(root+'sorted_veg_labels.pkl', 'wb') as fi:
      pickle.dump(cols, fi)
  else:
    #root
    with open('./cache/'+'sorted_veg_labels.pkl', 'rb') as f:
      cols = pickle.load(f)
  return cols


def clean_label(lbl,mode='tiny'):
  '''
  Cleans an img label, generalizing many classes into one.
  Different modes:
    complete - Raw class name
    full - includes d,D tailing specification
    med - includes tailing number and +,- or star
    mini - Same as tiny?
    tiny - Generalizes as far as it can, getting down to basic class
  :param lbl: label to clean
  :param mode:
  :return: cleaned label
  '''
  lbl = str(lbl[:1]).upper()+lbl[1:]
  try:
    if mode == 'complete':
      kl = lbl
    elif mode == 'full':
      kl = re.search('[A-Z]([a-c]|[e-z]|[1-9]|\*?)(\+|\-|d|D?)', lbl).group(0)
    elif mode == 'med':
      kl = re.search('[A-Z]([a-c]|[e-z]|[1-9]|\*?)(\+|\-?)', lbl).group(0)
    elif mode == 'tiny':
      kl = re.search('(([A-R]|[T-Z])[a-z]?)|S', lbl).group(0)
    elif mode == 'mini':
      kl = re.search('(([A-R]|[T-Z])[a-z]?)|S', lbl).group(0)
  except Exception as ep:
    print(ep)
  return kl


def get_veg_label_of_img(name, cols,mode='tiny'):
  '''
  Given a Tiff File name that was extracted (WHI11_WHJ21-1)
  cols is the sorted collection returned by get_dataset
  returns the veg label at that specific step as a string.
  '''
  
  # WHI11_WHJ21-1
  try:
    namet = name.split('_')
    tmp = namet[2].split('-')
    namet = [namet[1], tmp[0], tmp[1]]
  
    a = v_point(namet[0])
    b = v_point(namet[1])

    # tl = topleft_of_points(a,b)
  
    left, right, dir = points_2_direction(a, b)

  
    if dir == "NE-SW":
      pp = "first"
      dd = cols[left.zone][chr(ord(left.col)-1)][left.row][dir]
      dd = dd[1]
    else:
      pp = "Second"
      dd = cols[left.zone][left.col][left.row][dir]
      dd = dd[1]
      
    dd = dd[int(namet[2])-1]
    
    if dd.startswith('B') and dd.__len__() == 2:
      dd = dd + "+"
      
    return clean_label(dd,mode)
  except Exception as ex:
    print(str(ex) + '---> ' + namet[2])