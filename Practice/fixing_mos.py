from decimal import Decimal

import math

from data_prep.custom_classes import Mosaic_n, v_point


def cords2pixel_Test(mos, bbox):
  '''
  Given a Mosaic_n object and a list of tuple/list cords (lat/long ordered), returns the pixels based on the mosaic
  '''
  # Pixels/Points
  con_y = Decimal(mos.y) / (mos.max_lat - mos.min_lat)
  con_x = Decimal(mos.x) / abs(mos.min_long - mos.max_long)
  
  #con_x = Decimal(mos.x/mos.y) * con_y
  
  n_b = []
  for point in bbox:
    # Cords * Pixel/Cords
    n_y = abs(point[0] - mos.max_lat) * con_y
    n_y = abs(int(math.ceil(n_y)))
    
    #n_x = abs((point[1]-mos.max_long) *con_x) - abs((point[1]-mos.min_long) *con_x)
    n_x = abs((point[1] - mos.max_long))
    
    n_x *= con_x
    n_x = abs(int(math.ceil(n_x)))
    
    n_b.append([n_y, n_x])
    # 50 = lat
  return n_b
p = v_point(str("JGBB4_JGBA5"),[Decimal('-93.447183'),Decimal('58.723496')])
pix = cords2pixel_Test(Mosaic_n('../Mosaics/2016/20160714_wh_vg_02_75m_transparent_mosaic_group1.tif','02'), [[p.y, p.x]])
print('')