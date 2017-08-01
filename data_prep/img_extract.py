import copy
import gc
import time

# from libtiff import TIFF
from decimal import Decimal

import data_prep.moz_utils
from data_prep import tif_utils
from data_prep.data_utils import *
from data_prep.custom_classes import v_point, v_step
from data_prep.moz_utils import cords2pixel, points_2_direction, get_bounding_box, latLonToPixel, Moz_contains
from data_prep.tif_utils import has_pure_alpha
from data_prep.lbl_utils import extract_values_from_point_name, get_sorted_veg_from_excel

def mass_extract():
  d = 0.0
  
  Mos = tif_utils.MinMax_Extract()
  
  file = open('../../Circular_Plot_Coordinates_LSGO_2016.csv')
  

  for k, m in Mos.iteritems():
    time.sleep(5)
    print(k)
    d3 = 0.000009090*30.0
    file.seek(0)
    
    for line in file.readlines():
      v = line.replace('\n', '').split(',')

      #print(v)
      # print(m.corners)
      vX = float(v[2])
      vY = float(v[1])

      if Moz_contains(m, vX, vY):
          thefile = m.root + m.name
          try:
            try:
              img
            except NameError:
              print('Loading >> ' + thefile)
              img = tiff.TiffFile(thefile)
            #[[vY + d3,vX - d3], [vY - d3, vX + d3]]
            pixels = data_prep.moz_utils.latLonToPixel(thefile, get_bounding_box(vY, vX, 15.0))
            #newImg = img[pixels[0][0]:pixels[0][1], pixels[1][0]:pixels[1][1]]
            
            tiff.imsave('imgs/'+m.name+'--'+v[0]+ '.tif', img.asarray()[pixels[0][1]:pixels[1][1], pixels[0][0]:pixels[1][0]],description=m.name+'--'+v[0], metadata=m.name+'--'+v[0])
            print("Saved: "+v[0])
            
          except Exception as ex:
            print("<**> " + thefile + " cannot be opened with TiffFile.")
            print(ex)
      else:
        #print('None were found.')
        pass
    try:
      img.close()
    except Exception as ex:
      # print(ex)
      pass
    finally:
      try:
        del img
        gc.collect()
      except Exception as ex1:
        #print(ex1)
        pass
      
  file.close()

def rotated_box(sl1, sl2,img,leg, mm):
  d2 = 0.0
  d22 = d2 / 2.0
  
  Xa = float(sl1[1])
  Ya = float(sl1[2])
  Xb = float(sl2[1])
  Yb = float(sl2[2])
  d = math.sqrt(((Xb - Xa) ** 2) + ((Yb - Ya) ** 2))
  m = (Ya - Yb) / (Xa - Xb)
  
  if Xa > Xb:
    Xc = Xa - ((d2 * (Xa - Xb)) / d)
  else:
    Xc = Xa + ((d2 * (Xa - Xb)) / d)
  if Ya > Yb:
    Yc = Ya - ((d2 * (Ya - Yb)) / d)
  else:
    Yc = Ya + ((d2 * (Ya - Yb)) / d)
  
  if Xa > Xb:
    Xm1 = Xc - ((d22 * (Xc - Xb)) / (d - d2))
    
    Xm2 = Xc + ((d22 * (Xc - Xa)) / d2)
  
  else:
    Xm1 = Xc + ((d22 * (Xc - Xb)) / (d - d2))
    
    Xm2 = Xc - ((d22 * (Xc - Xa)) / d2)
  
  Xm1_1 = Xm1 + d22 / (math.sqrt(1 + (1 / (m * m))))
  Xm1_2 = Xm1 - d22 / (math.sqrt(1 + (1 / (m * m))))
  
  Xm2_1 = Xm2 + d22 / (math.sqrt(1 + (1 / (m * m))))
  Xm2_2 = Xm2 - d22 / (math.sqrt(1 + (1 / (m * m))))
  
  if Ya > Yb:
    Ym1 = Yc - ((d22 * (Yc - Yb)) / (d - d2))
    
    Ym2 = Yc + ((d22 * (Yc - Ya)) / d2)
  else:
    Ym1 = Yc - ((d22 * (Yc - Yb)) / d2)
    
    Ym2 = Yc + ((d22 * (Yc - Ya)) / (d - d2))
  
  Ym1_1 = Ym1 + d22 / (math.sqrt(1 + (1 / (m * m))))
  Ym1_2 = Ym1 - d22 / (math.sqrt(1 + (1 / (m * m))))
  
  Ym2_1 = Ym2 + d22 / (math.sqrt(1 + (1 / (m * m))))
  Ym2_2 = Ym2 - d22 / (math.sqrt(1 + (1 / (m * m))))
  
  pixels = latLonToPixel(mm.root + mm.name, [[Xm1_1, Ym1_1], [Xm1_2, Ym1_2], [Xm2_1, Ym2_1], [Xm2_2, Ym2_2]])
  
  # cv2.minAreaRect(pixels)
  
  newImg = img[pixels]
  
  #cv2.imwrite('./imgs/' + sl1[0]+'_'+sl2[0]+'-'+leg + '.tif')
total_extracted_imgs = 0
tmp_watch = []
total_skipped = 0
total_failed = 0

status_imgs = {'Success':0,'Index_Fail':0,'G_Fail':0,'Alpha_Skip':0,'Exists':0}
status_pretty_names = {'Success':'Successful: ','Index_Fail':'Index Fail: ','G_Fail':'Failed: ',
                       'Alpha_Skip':'Alpha: ','Exists':'Existed: '}


def new_points_old(a, b, t_steps, leg):
  leg = Decimal(leg)
  if a.x > b.x:
    Xc = a.x - ((abs(a.x - b.x) / t_steps) * leg)
  else:
    Xc = a.x + ((abs(b.x - a.x) / t_steps) * leg)
  if a.y > b.y:
    Yc = a.y - ((abs(a.y - b.y) / t_steps) * leg)
  else:
    Yc = a.y + ((abs(a.y - b.y) / t_steps) * leg)
  
  return Yc, Xc


def new_points(a, b, t_steps, leg):
  leg = Decimal(leg)
  
  if leg == 0:
    return a.y,a.x
  
  if a.x > b.x:
    Xc = a.x - ((abs(a.x - b.x)*leg)  / t_steps)
  else:
    Xc = a.x + ((abs(a.x - b.x)*leg)  / t_steps)
  if a.y > b.y:
    Yc = a.y - ((abs(a.y - b.y)*leg)  / t_steps)
  else:
    Yc = a.y + ((abs(a.y - b.y)*leg)  / t_steps)
  
  return Yc, Xc
def normal_shape(img,image_size):
  norm = (image_size,image_size,4)
  re =  img.shape == norm
  return re

def normal_box(a,img,mm,dir,image_size):
  global status_imgs
  rname = a.name

  if tmp_watch.__contains__(rname):
    print('error!'+rname)
  else:
    tmp_watch.append(rname)
  
  if not(os.path.isfile(dir + rname + '.tif')):
  
    # pixels = latLonToPixel(mm.root + mm.name, bbox)
    # cords2pixel always returns as y(lat), x(long)
    pix = cords2pixel(mm,[[a.y,a.x]])[0]
    if image_size % 2 == 0:
      pixels = [[pix[0] - ((image_size/2)), pix[1] - ((image_size/2))], [pix[0] + ((image_size/2)), pix[1] + ((image_size/2))]]
    else:
      pixels = [[pix[0] - ((image_size-1)/2), pix[1] - ((image_size-1)/2)], [pix[0] + ((image_size-1)/2), pix[1] + ((image_size-1)/2)]]
    
    try:
      newimg = img[int(pixels[0][0]):int(pixels[1][0]), int(pixels[0][1]):int(pixels[1][1])]
    except IndexError:
      print('Index error on: '+rname+', skipped.')
      status_imgs['Index_Fail'] = status_imgs['Index_Fail']+1
      return
  
    if normal_shape(newimg,image_size):
      if not(has_pure_alpha(newimg)):
        try:
          with tiff.TiffWriter(dir + rname + '.tif', software='Becomi LLC') as tif:
            tif.save(newimg[:, :, :3], description=a.label)
            status_imgs['Success'] = status_imgs['Success'] + 1
            print('Saved: ' + rname)
        except Exception as ex:
          print('Failed on: ' + rname)
          print(ex)
          status_imgs['G_Fail'] = status_imgs['G_Fail'] + 1
      else:
        print(rname + " has alpha. Skipped")
        status_imgs['Alpha_Skip'] = status_imgs['Alpha_Skip'] + 1
    else:
      print(rname+' non-normal shape.')
  else:
    print(rname + '.tif exists, skipping..')
    status_imgs['Exists'] = status_imgs['Exists'] + 1
  
def criss_cross_run(points, o_img, cols, cM,dir,image_size):
  img = o_img.asarray()
  o_img.close()
  
  print('CC-Running: '+cM.name)
  
  for z, zv in points.items():
    for c, cv in zv.items():
      # Row, rv is the point
      for r, rv in cv.items():
        for d, dv in rv.v_steps.items():
          for s, sv in dv.items():
            normal_box(sv,img,cM,dir,image_size)

def crisscross(image_size=24,dir='./data_prep/imgs_veg/train/', moz_dir='../Mosaics'):
  
  if not os.path.exists(dir):
    os.makedirs(dir)
  
  cols = get_sorted_veg_from_excel(file="./Resources/2016_veg_data.xlsx", root="./cache/")
  
  Mos = tif_utils.get_mos_from_dir(dir=moz_dir)
  # cols = size_of_col()
  
  file = open('./output/new_points_KML.csv')
  
  p_tmp = {}
  for line in file.readlines():
    v = line.replace('\n', '').split(',')
    
    ex_p = extract_values_from_point_name(v[0])

    p_tmp.setdefault(ex_p['zone'], {}).setdefault(ex_p['col'], {}).setdefault(ex_p['row'], v_point(v[0], [
      Decimal(v[2]), Decimal(v[1])]))
  
  points = {}
  
  for z, zv in p_tmp.items():
    lastC = list(zv.keys())
    lastC.sort(key=lambda x: (not x.islower(), x))
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
                print('', end="")
              if Moz_contains(m, sv.x, sv.y):
                n_rv = v_point(rv.name, rv.cords)
                n_sv = copy.deepcopy(sv)
                n_sv.name = m.shortened+'_'+n_sv.name
                f_p.setdefault(m.name, {}).setdefault(z, {}).setdefault(c, {}).setdefault(r, n_rv).v_steps.setdefault(d,{}).setdefault(s, n_sv)
                cMs.setdefault(m.name, m)
  
  # Loop over all Mosaics
  for k, cM in cMs.items():
    try:
      print("Loading: " + cM.name)
      img = tiff.TiffFile(cM.root + cM.name)
    except Exception as ex:
      print("Can't open the image. " + cM.name)
      print(ex)
    
    criss_cross_run(f_p[cM.name], img, cols, cM,dir,image_size)
    
    # del img
  file.close()
  global status_imgs
  global status_pretty_names
  
  for k,v in status_imgs.items():
    print(status_pretty_names[k]+str(v))
  
if __name__ == "__main__":
  
  #mass_extract()
  #crisscross(image_size=32,dir='../veg_imgs/32x/train/')
  crisscross(image_size=40, dir='../veg_new/40x/train/')


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