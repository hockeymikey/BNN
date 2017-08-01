import math
import os
import pathlib
import pickle
from collections import OrderedDict
from random import randint

import numpy as np
import tifffile as tiff

from data_prep.lbl_utils import get_sorted_veg_from_excel, get_veg_label_of_img


def get_veg_data(filex="../Veg/2016_veg_data.xlsx", cols = None, dn ='../data_prep/imgs_veg', size=-1, l_c={},mode='tiny'):
  '''
  Get the vegetation data from directory and store in numpy arrays
  
  :param filex: Ezcel file
  :param cols: Sorted excel label data
  :param dn:
  :param size:
  :param l_c: Unique labels with label name -> index in l_s
  :param mode: see lbl_utils.py > clean_label for list of modes.
  :return: tuple ordered: d- numpy array of size with the given tif files
                          l- numpy array of labels for image at given index in d
                          l_s - Unique labels mapped int -> number of images with label
                          l_c - Unique labels with label name -> index and index -> label name in l_s
  '''
  
  if cols == None:
    cols = get_sorted_veg_from_excel(file=filex)
  
  dir = os.listdir(dn)
  # Check type on the array if it should be float or what float32
  d = None
  
  # dir.__len__()
  l = np.array([], dtype=np.int32)
  
  #l_c is used to convert from onehot int to label value
  #l_s is to count number of each label
  l_s = {}
  i = 0
  u = 0
  
  for file in dir:
    if file.endswith('.tif') and (size==-1 or i < size):

      # d[:,i] = np.reshape(ti.asarray(), -1, order='F')
      # d = np.vstack((d, np.reshape(ti.asarray(), -1, order='F')))


      pp = get_veg_label_of_img(file.split('.')[0], cols,mode=mode)

      if isinstance(pp, str):
        # t2 value or ti.asarray can be used.
        if d is None:
          sp = tiff.imread(dn + '/' + file).shape
          d = np.zeros(shape=(0, int(sp[0]), int(sp[1]), int(sp[2])), dtype=np.int32)
  
        d = np.concatenate((d, [tiff.imread(dn + '/' + file)]), axis=0)
  
        l_s.setdefault(pp, 0)
        l_s[pp] += 1
  
        if not (l_c.__contains__(pp)):
          l_c.setdefault(pp, u)
          l_c.setdefault(u, pp)
          u += 1
  
        l = np.append(l, l_c[pp])
      
      i += 1
    elif size != -1 and i > size:
      break
  return (d, l, l_s, l_c)
  
def get_veg_training_data(train="data_prep/imgs_veg/train", test="data_prep/imgs_veg/test",mode='tiny'):
  '''
  Gets the data and organises it to be used in the neural network.
  
  :param train:
  :param test:
  :param mode: see lbl_utils.py > clean_label for list of modes.
  :return:
  '''
  
  file = "../Veg/2016_veg_data.xlsx"
  
  cols = get_sorted_veg_from_excel(file=file)
  
  data = {'X_train': None,
          'X_val': None,
          'y_train': None,
          'y_val': None}
  images, labels, unique, l_c = get_veg_data(cols=cols, dn=train,mode=mode)
  
  y_imgs, y_l, y_unique, l_c_ = get_veg_data(cols=cols, dn=test, l_c=l_c,mode=mode)
  
  data['X_train'] = images
  data['X_val'] = y_imgs
  data['y_train'] = labels
  data['y_val'] = y_l
  
  return data,l_c


def subdivide_data(images, labels, unique, l_c, new_size=150):
  '''
  Old data subdivider into test/training.  Not used anymore.
  
  :param images:
  :param labels:
  :param unique:
  :param l_c:
  :param new_size:
  :return:
  '''
  
  max = images.shape[0]-1
  y_img = np.zeros(shape=(0, 46, 46, 4), dtype=np.float32)
  y_l = np.array([], dtype=np.int32)
  y_l_u = {}
  i = 0
  while i < new_size:
    next = randint(0, max)
    t = int(labels[next])
    ttt = unique[l_c[int(t)]]
    
    if int(unique[str(l_c[t])]) > 3:
      y_img = np.concatenate((y_img, [images[t]]), axis=0)
      y_l = np.concatenate((y_l, [t]))
      
      unique[str(l_c[t])] = int(unique[str(l_c[t])]) - 1
      y_l_u.setdefault(str(l_c[t]), 0)
      y_l_u[str(l_c[t])] = y_l_u[str(l_c[t])] + 1
      
      labels = np.delete(labels, (next), axis=0)
      images = np.delete(images, (next), axis=0)
      i += 1
      max -= 1
  return images, labels, unique, l_c, y_img, y_l, y_l_u

def pickle_veg_data(name='./cache/data_cache',dir='./data_prep/imgs_veg/',mode='tiny'):
  '''
  Grabs the training/testing data and throws into a pixel file.
  Saves time down the road.
  
  :return:
  '''
  
  print('Creating pickle...')
  with open(name+'.pkl','wb') as fi:
    data,l_c = get_veg_training_data(train=dir+"train", test=dir+"test",mode=mode)
    pickle.dump(data, fi)
  with open(name+'_vars.pkl','wb') as fi:
    pickle.dump(l_c, fi)
  with open(name+'_vars.txt','w') as fi:
    for k,v in l_c.items():
      fi.write(str(k)+' -> '+str(v)+'\n')
      fi.flush()

def roll_data_channel_last(datav):
  '''
  Rolls the data from a channel last to channel first
  
  :param datav:
  :return:
  '''
  datav["X_train"] = np.rollaxis(datav["X_train"],3,1)
  datav["X_val"] = np.rollaxis(datav["X_val"], 3, 1)
  return datav

def get_veg_class_size(file="../Veg/2016_veg_data.xlsx", dn='./data_prep/imgs_veg', exclude=['unsafe'],size=-1,name='Classes.txt',clean_mode='tiny'):
  '''
  Get the class size (number of imgs) and write to file "name"
  :param file:
  :param dn:
  :param exclude:
  :param size:
  :param name:
  :param clean_mode: see lbl_utils.py > clean_label for list of modes.
  :return:
  '''
  cols = get_sorted_veg_from_excel(file=file)
  
  imgs = {}
  
  for root, dirs, files in os.walk(dn):
    dirs[:] = [d for d in dirs if d not in exclude]
    
    for f in [file for file in files if file.endswith(('.tif'))]:
      try:
        di = pathlib.Path(root)
        stem = di.stem
        
        tc = f.split('.')
        pp = get_veg_label_of_img(tc[0], cols,clean_mode)
    
        if isinstance(pp, str):
          imgs.setdefault(stem,{})
          imgs[stem].setdefault(pp, []).append(f)
  
      except Exception as exx:
        print(exx)
      

  file = open(name,'w')
  
  u_s = {}
  s = {}
  
  for stem,vv in imgs.items():
    s.setdefault(stem,[])
    u_s.setdefault(stem,{})
    
    l = 0
    for k,v in vv.items():
      vl = v.__len__()
      u_s[stem][k] = vl
      x = 0
  
      while x <= l:
        if l == 0 or x == l:
          s[stem].append(k)
          l += 1
          break
        elif u_s[stem][s[stem][x]] > vl and (x == 0 or u_s[stem][s[stem][x - 1]] <= vl):
          s[stem].insert(x, k)
          l += 1
          break
        else:
          x += 1
          
  
  for st, v in s.items():
    v.reverse()
    file.write(st+"\n---------\n")
    for u in v:
      file.write(u + ': ' + str(u_s[st][u]) + '\n')
    
  file.close()


def scrap_extras(file="../Veg/2016_veg_data.xlsx", dn='./data_prep/imgs_veg', number=2000,clean_mode='tiny'):
  '''
  Moves extra imgs to a folder named extra. So with threshold (number param) of 2000, any class
  below 2000 for imgs are moved and extras in over 2000 classes moved too.
  :param file:
  :param dn:
  :param number:
  :param clean_mode: see lbl_utils.py > clean_label for list of modes.
  :return:
  '''
  
  if not os.path.exists(dn + "/extra"):
    os.makedirs(dn + "/extra")
  
  cols = get_sorted_veg_from_excel(file=file)
  
  dir = os.listdir(dn + "/train")
  
  imgs = {}
  
  for file in dir:
    if file.endswith('.tif'):
      try:
        tc = file.split('.')
        pp = get_veg_label_of_img(tc[0], cols, mode=clean_mode)
        
        if isinstance(pp, str):
          imgs.setdefault(pp, []).append(file)
      except Exception as exx:
        print(exx)
        
  tot = 0
  
  for k, v in imgs.items():
    vl = v.__len__()
    x = 0
    dif = vl - number
    
    if dif > 0:
      while x < dif:
        next = randint(0, vl - 1)
        os.rename(dn + "/train/" + v[next], dn + "/extra/" + v[next])
        print("Moved " + v[next] + " to extras.")
        v.pop(next)
        tot += 1
        vl -= 1
        x += 1
    
      
  print("Moved: " + str(tot))

def sort_veg_imgs_to_test(file="../Veg/2016_veg_data.xlsx", dn='./data_prep/imgs_veg', size=-1,safe=None,rate=0.20,clean_mode='tiny'):
  '''
  Sort tiffs into test folder from training folder
  
  :param file:
  :param dn:
  :param size:
  :param safe:
  :param rate:
  :param clean_mode: see lbl_utils.py > clean_label for list of modes.
  :return:
  '''
  
  if not os.path.exists(dn + "/test"):
    os.makedirs(dn + "/test")
  if not os.path.exists(dn + "/unsafe"):
    os.makedirs(dn + "/unsafe")
  
  cols = get_sorted_veg_from_excel(file=file)
  
  dir = os.listdir(dn + "/train")
  
  imgs = {}
  
  for file in dir:
    
    if file.endswith('.tif'):
      
      try:
        tc = file.split('.')
        pp = get_veg_label_of_img(tc[0], cols,mode=clean_mode)
        
        if isinstance(pp, str):
          imgs.setdefault(pp, []).append(file)
      
      except Exception as exx:
        print(exx)
  tot = 0
  for k, v in imgs.items():
    vl = v.__len__()
    if safe is not None and not safe.__contains__(k):
      x = 0
      while x < vl:
        tot += 1
        os.rename(dn + "/train/" + v[x], dn + "/unsafe/" + v[x])
        print("Moved " + v[x] + " to testing.")
        x+=1
    else:
      tex = math.floor(vl * rate)
  
      if vl > 10 and tex > 0:
        for f in range(tex):
          try:
            next = randint(0, vl - 1)
        
            os.rename(dn + "/train/" + v[next], dn + "/test/" + v[next])
            print("Moved " + v[next] + " to testing.")
            v.pop(next)
            tot += 1
            vl -= 1
          except Exception as ex:
            print(ex)
  print("Moved: " + str(tot))
  
def merge_sorted_imgs(dir,toGrab=['unsafe','test'],dest='/train/'):
  '''
  Takes all specified folders and merges them into dest
  :param dir:
  :param toGrab:
  :param dest:
  :return:
  '''
  for root, dirs, files in os.walk(dir):
    dirs[:] = [d for d in dirs if d in toGrab]
    for f in [file for file in files if file.endswith(('.tif'))]:
      pl = pathlib.Path(root)
      stem = pl.stem
      parent = str(pl.parent)
      os.rename(str(pl)+'/'+f,parent+dest+f)
      print('Moved home: '+f)
      
if __name__ == "__main__":
  #dir = '../veg_imgs/40x/'
  dir = '../veg_new/40x/'
  name = ''
  #merge_sorted_imgs(dir,toGrab=['unsafe','test','extra'])
  pickle_veg_data(name='./cache/new_40x_data_cache_tiny',dir=dir,mode='tiny')
  
  #get_veg_class_size(name='classes_sorted.txt',dn=dir,clean_mode='med')
  #get_veg_class_size(name='classes_sorted_40x.txt',dn=dir)
  #get_veg_class_size(name='classes_so.txt',clean_mode='mini')
  
  #sort_veg_imgs_to_test(safe=['A','C','B','S','Bg','F'])
  #sort_veg_imgs_to_test(safe=['A', 'C', 'B', 'S'],dn='../veg_imgs/32x')
  #sort_veg_imgs_to_test(safe=['A', 'C', 'B', 'S'], dn=dir)
  
  #sort_veg_imgs_to_test(safe=['A','C1','B1-','Sb','C2','B2-','S','C5','Bg','B2+','Sc','F'], dn='../veg_imgs/32x',clean_mode='med')
  #sort_veg_imgs_to_test(safe=['A', 'C', 'B', 'S', 'Bg', 'F'])
  
  #get_sorted_veg_from_excel()
  #scrap_extras(dn=dir)
  
  
  