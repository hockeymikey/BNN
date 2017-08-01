import gc
import time

# from libtiff import TIFF
from decimal import Decimal

import data_prep.moz_utils
from data_prep import tif_utils
from data_prep.data_utils import *
from data_prep.custom_classes import v_point, v_step
from data_prep.img_extract import new_points, new_points_old
from data_prep.moz_utils import cords2pixel, points_2_direction, get_bounding_box, latLonToPixel, Moz_contains
from data_prep.tif_utils import has_pure_alpha
from data_prep.lbl_utils import extract_values_from_point_name, get_sorted_veg_from_excel

from data_prep.moz_utils import decdeg2dms

image_size = 24
total_extracted_imgs = 0
tmp_watch = []
total_skipped = 0
total_failed = 0

def normal_box(a, b, img, leg, mm, t_steps,dir='./imgs_veg/train/'):
  global total_extracted_imgs
  global total_skipped
  global total_failed
  
  global image_size
  rname = mm.shortened + '_' + a.name + '_' + b.name + '-' + str(leg + 1)
  
  if not (os.path.isfile(dir + rname + '.tif')):
    
    # d = ((Xb - Xa) ** 2) + ((Yb - Ya) ** 2).sqrt()
    # m = (Ya - Yb) / (Xa - Xb)
    if leg == 0:
      Xc = a.x
      Yc = a.y
    else:
      
      #This is ASSSUMING WRONG!!!
      #What if it's a NE-SW or something
      #The ordering on the X will be different!
      #Fix this.  Trust data will be fed right
      if a.x > b.x:
        Xc = a.x - (((a.x - b.x) / t_steps) * leg)
      else:
        Xc = a.x + (((a.x - b.x) / t_steps) * leg)
      if a.y > b.y:
        Yc = a.y - (((a.y - b.y) / t_steps) * leg)
      else:
        Yc = a.y + (((a.y - b.y) / t_steps) * leg)
      
      '''if Xa > Xb:
        Xc = Xa - (((c_d) * (Xa - Xb)) / t_d)
      else:
        Xc = Xa + (((c_d) * (Xa - Xb)) / t_d)
      if Ya > Yb:
        Yc = Ya - (((c_d) * (Ya - Yb)) / t_d)
      else:
        Yc = Ya + (((c_d) * (Ya - Yb)) / t_d)'''
    
    bbox = get_bounding_box(Yc, Xc, 0.25)
    
    # Long is negative and "lower" negative (-93 vs -90) is the max.  Comparison is flipped how you think it would be
    if (bbox[0][0] > mm.min_lat and bbox[1][0] < mm.max_lat) and (
        bbox[0][1] < mm.min_long and bbox[1][1] > mm.max_long):
      
      # pixels = latLonToPixel(mm.root + mm.name, bbox)
      # cords2pixel always returns as y(lat), x(long)
      pix = cords2pixel(mm, [[Yc,Xc]])[0]
      pixels = [[pix[0]-image_size,pix[1]-image_size],[pix[0]+image_size,pix[1]+image_size]]
      
      if image_size == -1:
        image_size = max(abs(pixels[1][1] - pixels[0][1]), abs(pixels[1][0] - pixels[0][0]))
      
      if not (abs(pixels[1][1] - pixels[0][1]) == image_size):
        pixels[1][1] = (image_size - (pixels[1][1] - pixels[0][1])) + pixels[1][1]
      if not (abs(pixels[1][0] - pixels[0][0]) == image_size):
        pixels[1][0] = (image_size - (pixels[1][0] - pixels[0][0])) + pixels[1][0]

      x_max = max(pixels[0][1], pixels[1][1])
      y_max = max(pixels[0][0], pixels[1][0])

      x = min(pixels[0][1], pixels[1][1])
      y = min(pixels[0][0], pixels[1][0])
      
      #left-right
      while y <= y_max:
        img[y,pixels[0][1]] = np.array([255,0,0])
        img[y, pixels[0][1]+1] = np.array([255,0,0])
        img[y, pixels[0][1]+2] = np.array([255,0,0])

        img[y, pixels[1][1]] = np.array([255,0,0])
        img[y, pixels[1][1]-1] = np.array([255,0,0])
        img[y, pixels[1][1]-2] = np.array([255,0,0])
        y+=1
      #top-bottom
      while x <= x_max:
        img[pixels[0][0],x] = np.array([255,0,0])
        img[pixels[0][0]-1, x] = np.array([255,0,0])
        img[pixels[0][0]-2, x] =np.array([255,0,0])

        img[pixels[1][0], x] = np.array([255,0,0])
        img[pixels[1][0]+1, x] = np.array([255,0,0])
        img[pixels[1][0]+2, x] = np.array([255,0,0])
        x += 1
        
    else:
      print(rname + " was offmap. Skipped.")
      total_skipped += 1
  else:
    print(rname + '.tif exists, skipping..')
    total_skipped += 1


def JourneyToPoint(a, b, img, m, cols, topLeft):
  # step_size = get_step_size(a,b,cols,topLeft)
  leg = 0
  d = Decimal(0.0)
  de = abs(((a.x - b.x) ** 2) + ((b.y - a.y) ** 2).sqrt())
  t_steps = len(cols[topLeft.zone][topLeft.col][topLeft.row][points_2_direction(a, b)[2]][1])
  
  while leg < t_steps:
    normal_box(a, b, img, leg, m, Decimal(t_steps))
    # d = d + step_size
    leg += 1
    # d = step_size * Decimal(leg)


def crisscross(image_size=24,dir='./data_prep/imgs_veg/train/'):
  
  cols = get_sorted_veg_from_excel(file="./Veg/2016_veg_data.xlsx", root="./cache/")
  
  Mos = tif_utils.get_mos_from_dir(dir='../Mosaics')
  # cols = size_of_col()
  
  file = open('./output/new_points_KML.csv')

  p_tmp = {}
  for line in file.readlines():
    v = line.replace('\n', '').split(',')

    ex_p = extract_values_from_point_name(v[0])
    
    if str(v[0]).__contains__('WH') and ex_p['col'] == 'I' and ex_p['row'] == '1':
      print('')
    p_tmp.setdefault(ex_p['zone'], {}).setdefault(ex_p['col'], {}).setdefault(ex_p['row'], v_point(v[0], [
      Decimal(v[2]),Decimal(v[1])]))
    
  points = {}
  
  for z, zv in p_tmp.items():
    lastC = list(zv.keys())
    lastC.sort(key=lambda x:(not x.islower(), x))
    lastC = lastC[zv.__len__() - 1]
    # Col
    for c, cv in zv.items():
      lastR = list(cv.keys())
      lastR.sort(key=int)
      lastR = lastR[cv.__len__() - 1]
      # Row, rv is the point
      for r, rv in cv.items():
        if rv.col != lastC and rv.row != lastR:
          # Nw-SE
          try:
            rv2 = p_tmp[z][chr(ord(c) + 1)][str(int(r) + 1)]
          except:
            pass
          #NW-SE
          d = points_2_direction(rv, rv2)[2]
          points.setdefault(z, {}).setdefault(c, {}).setdefault(r, rv).v_steps.setdefault(d, {})
          t_ben = cols[rv.zone][rv.col][rv.row][d][1]
          t_steps = len(t_ben)
          lk = 0
          while lk < t_steps:
            lv = t_ben[lk]
            rname = rv.name + '_' + rv2.name + '-' + str(lk + 1)
            
            #You subtract t_steps by 1 because it includes the first point which isn't a "step"
            #So you don't count it
            Yc, Xc = new_points(rv, rv2, t_steps-1, lk)
            
            if lk == 0 or lk+1 == t_steps:
              head = True
            else:
              head = False
  
            points[z][c][r].v_steps[d].setdefault(lk, v_step(rname, {'y': Yc, 'x': Xc}, lv,head=head))
            lk += 1
            
          # NE-SW
          p3 = p_tmp[z][chr(ord(c) + 1)][r]
          p4 = p_tmp[z][c][str(int(r) + 1)]

          d = points_2_direction(p3, p4)[2]
          points.setdefault(z, {}).setdefault(c, {}).setdefault(r, rv).v_steps.setdefault(d, {})

          t_ben = cols[rv.zone][rv.col][rv.row][d][1]
          t_steps = len(t_ben)
          lk = 0
          while lk < t_steps:
            lv = t_ben[lk]
            rname = p3.name + '_' + p4.name + '-' + str(lk + 1)
            Yc, Xc = new_points(p3, p4, t_steps-1, lk)
            
            if lk == 0 or lk+1 == t_steps:
              head = True
            else:
              head = False
  
            points[z][c][r].v_steps[d].setdefault(lk, v_step(rname, {'y': Yc, 'x': Xc}, lv,head=head))
            lk += 1

  f_p = {}
  cMs = {}
  
  for k, m in Mos.items():
    for z, zv in points.items():
      for c, cv in zv.items():
        # Row, rv is the point
        for r, rv in cv.items():
          for d, dv in rv.v_steps.items():
            for s, sv in dv.items():
              if z == 'WH' and c == 'I' and r == '1' and k == '20160714_wh-vh_01_75m_transparent_mosaic_group1.tif':
                print('',end="")
              if Moz_contains(m, sv.x, sv.y):
                ex_p = extract_values_from_point_name(v[0])
                n_rv = v_point(rv.name, rv.cords)
                f_p.setdefault(m.name, {}).setdefault(z, {}).setdefault(c, {}).setdefault(r, n_rv).v_steps.setdefault(d,{}).setdefault(s,sv)
                cMs.setdefault(m.name, m)

  # Loop over all Mosaics
  for k, cM in cMs.items():
    try:
      print("Loading: " + cM.name)
      img = tiff.TiffFile(cM.root + cM.name)
    except Exception as ex:
      print("Can't open the image. " + cM.name)
      print(ex)
    
    n_img = img.asarray()[:,:,:3]
    img.close()
    criss_cross_run(f_p[cM.name], n_img, cols, cM,image_size)
    print('Saving: '+cM.name)
    with tiff.TiffWriter('./n_'+cM.name+'.tif',bigtiff=True,software='Ben LLC') as tif:
      tif.save(n_img)
    #tiff.imsave('./n_'+cM.name+'.tif',n_img)
    del n_img

    # del img
  file.close()
  global total_extracted_imgs
  global total_failed
  global total_skipped
  print(str(total_extracted_imgs) + " images were extracted")
  print(str(total_skipped) + " skipped")
  print(str(total_failed) + " failed")


def cords2pixel_Test(mos, bbox):
  '''
  Given a Mosaic_n object and a list of tuple/list cords (lat/long ordered), returns the pixels based on the mosaic
  '''
  # Pixels/Points
  con_y = Decimal(mos.y) / (mos.max_lat - mos.min_lat)
  con_x = Decimal(mos.x) / abs(mos.min_long - mos.max_long)

  # con_x = Decimal(mos.x/mos.y) * con_y
  
  n_b = []
  for point in bbox:
    # Cords * Pixel/Cords
    n_y = abs(point[0] - mos.max_lat) * con_y
    n_y = abs(int(math.ceil(n_y)))

    # n_x = abs((point[1]-mos.max_long) *con_x) - abs((point[1]-mos.min_long) *con_x)
    n_x = abs((point[1] - mos.max_long))
    
    n_x *= con_x
    n_x = abs(int(math.ceil(n_x)))
    
    n_b.append([n_y, n_x])
    # 50 = lat
  return n_b


def cords2pixel_fixed(mos, bbox):
  '''
  Given a Mosaic_n object and a list of tuple/list cords (lat/long ordered), returns the pixels based on the mosaic
  '''
  # Pixels/Points
  con_y = Decimal(mos.y) / (mos.max_lat - mos.min_lat)
  con_x = Decimal(mos.x) / abs(mos.min_long - mos.max_long)
  
  # con_x = Decimal(mos.x/mos.y) * con_y
  
  n_b = []
  for point in bbox:
    # Cords * Pixel/Cords
    n_y = (Decimal(mos.y)* abs(point[0] - mos.max_lat)) / (mos.max_lat - mos.min_lat)
    n_y = abs(int(math.ceil(n_y)))
    
    # n_x = abs((point[1]-mos.max_long) *con_x) - abs((point[1]-mos.min_long) *con_x)
    n_x = (Decimal(mos.x) * abs((point[1] - mos.max_long))) / abs(mos.min_long - mos.max_long)

    n_x = abs(int(math.ceil(n_x)))
    
    n_b.append([n_y, n_x])
    # 50 = lat
  return n_b

def draw_point(a,img,cm,image_size=24):

  #pixels = cords2pixel(cm,get_bounding_box(a.y, a.x, 0.25))
  pix = cords2pixel_Test(cm, [[a.y, a.x]])[0]
  pix2 = cords2pixel_fixed(cm, [[a.y, a.x]])[0]
  
  if pix[0] != pix2[0] or pix[1] != pix2[1]:
    print('Different')
  
  pixels = [[pix[0] - image_size, pix[1] - image_size], [pix[0] + image_size, pix[1] + image_size]]

  x_max = max(pixels[0][0],pixels[1][0])
  y_max = max(pixels[0][1],pixels[1][1])
  
  y = min(pixels[0][1],pixels[1][1])
  x = min(pixels[0][0],pixels[1][0])
  
  # left-right
  while x <= x_max:
    while y <= y_max:
      try:
        img[x, y] = np.array([255,0,255])
      except IndexError:
        pass
      y+=1
    y = min(pixels[0][1],pixels[1][1])
    x+=1

  y = min(pixels[0][1], pixels[1][1])
  x = min(pixels[0][0], pixels[1][0])

  # left-right
  while x <= x_max:
    try:
      img[x, pixels[0][1]] = np.array([255, 215, 0])
    except IndexError as ex:
      pass
    try:
      img[x, pixels[0][1] + 1] = np.array([255, 215, 0])
    except IndexError:
      pass
    try:
      img[x, pixels[0][1] + 2] = np.array([255, 215, 0])
    except IndexError:
      pass
    
    
    try:
      img[x, pixels[1][1]] = np.array([255, 215, 0])
    except IndexError:
      pass
    try:
      img[x, pixels[1][1] - 1] = np.array([255, 215, 0])
    except IndexError:
      pass
    try:
      img[x, pixels[1][1] - 2] = np.array([255, 215, 0])
    except IndexError:
      pass

    x += 1
  # top-bottom
  while y <= y_max:
    try:
      img[pixels[0][0], y] = np.array([255, 215, 0])
    except IndexError:
      pass
    try:
      img[pixels[0][0] - 1, y] = np.array([255, 215, 0])
    except IndexError:
      pass
    try:
      img[pixels[0][0] - 2, y] = np.array([255, 215, 0])
    except IndexError:
      pass
    
    try:
      img[pixels[1][0], y] = np.array([255, 215, 0])
    except IndexError:
      pass
    try:
      img[pixels[1][0] + 1, y] = np.array([255, 215, 0])
    except IndexError:
      pass
    try:
      img[pixels[1][0] + 2, y] = np.array([255, 215, 0])
    except IndexError:
      pass

    y += 1
    
  return pix
  
def criss_cross_run(points, img, cols, cM,image_size):
  file = open(cM.name+'_output.log','w')
  print('CC-Running: '+cM.name)
  for z, zv in points.items():
    for c, cv in zv.items():
      # Row, rv is the point
      for r, rv in cv.items():
        for d, dv in rv.v_steps.items():
          lasts = list(dv.keys())
          lasts.sort(key=int)
          firsts = lasts[0]
          lasts = lasts[dv.__len__() - 1]
          for s, sv in dv.items():
            if sv.head:
              pix = draw_point(sv,img,cM,image_size)
            else:
              pix = single_box_draw(sv,img,cM,image_size)
            file.write(sv.name+' -> '+str(pix[0])+', '+str(pix[1])+'\n')
            file.flush()
  file.close()


def single_box_draw(a, img,mm, dir='./imgs_veg/train/',image_size=24):
  global total_extracted_imgs
  global total_skipped
  global total_failed

  rname = a.name
  pix = cords2pixel_Test(mm, [[a.y, a.x]])[0]
  pix2 = cords2pixel_fixed(mm, [[a.y, a.x]])[0]

  if pix[0] != pix2[0] or pix[1] != pix2[1]:
    print('Different')
  #pix = latLonToPixel(mm.address,[[a.y, a.x]])

  # pixels = latLonToPixel(mm.root + mm.name, bbox)
  # cords2pixel always returns as y(lat), x(long)

  pixels = [[pix[0] - image_size, pix[1] - image_size], [pix[0] + image_size, pix[1] + image_size]]

  if- image_size == -1:
    image_size = max(abs(pixels[1][1] - pixels[0][1]), abs(pixels[1][0] - pixels[0][0]))

  if not (abs(pixels[1][1] - pixels[0][1]) == image_size):
    pixels[1][1] = (image_size - (pixels[1][1] - pixels[0][1])) + pixels[1][1]
  if not (abs(pixels[1][0] - pixels[0][0]) == image_size):
    pixels[1][0] = (image_size - (pixels[1][0] - pixels[0][0])) + pixels[1][0]

  x_max = max(pixels[0][1], pixels[1][1])
  y_max = max(pixels[0][0], pixels[1][0])

  x = min(pixels[0][1], pixels[1][1])
  y = min(pixels[0][0], pixels[1][0])

  # left-right
  while y <= y_max:
    try:
      img[y, pixels[0][1]] = np.array([255, 0, 0])
    except IndexError:
      pass
    try:
      img[y, pixels[0][1] + 1] = np.array([255, 0, 0])
    except IndexError:
      pass
    try:
      img[y, pixels[0][1] + 2] = np.array([255, 0, 0])
    except IndexError:
      pass
    
    
    try:
      img[y, pixels[1][1]] = np.array([255, 0, 0])
    except IndexError:
      pass
    try:
      img[y, pixels[1][1] - 1] = np.array([255, 0, 0])
    except IndexError:
      pass
    try:
      img[y, pixels[1][1] - 2] = np.array([255, 0, 0])
    except IndexError:
      pass
    y += 1
  # top-bottom
  while x <= x_max:
    try:
      img[pixels[0][0], x] = np.array([255, 0, 0])
    except IndexError:
      pass
    try:
      img[pixels[0][0] - 1, x] = np.array([255, 0, 0])
    except IndexError:
      pass
    try:
      img[pixels[0][0] - 2, x] = np.array([255, 0, 0])
    except IndexError:
      pass
    
    try:
      img[pixels[1][0], x] = np.array([255, 0, 0])
    except IndexError:
      pass
    try:
      img[pixels[1][0] + 1, x] = np.array([255, 0, 0])
    except IndexError:
      pass
    try:
      img[pixels[1][0] + 2, x] = np.array([255, 0, 0])
    except IndexError:
      pass
    x += 1
  return pix

if __name__ == "__main__":
  # mass_extract()
  crisscross()

'''
  for line1, line2 in itertools.izip_longest(*[file] * 2):
    sl1 = line1.replace('\n', '').split(',')
    sl2 = line2.replace('\n', '').split(',')

    leg = 1
    d = 0.0
    de = abs(math.sqrt(((float(sl2[1])-float(sl1[1])) ** 2) + ((float(sl2[2])-float(sl1[2])) ** 2)))

    while d < de:
      normal_box(sl1, sl2, img, leg,m)
      d = d + M_2_CORD
      leg = leg +1
'''