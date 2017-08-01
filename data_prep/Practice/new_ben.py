import os
import pathlib

import tifffile as tiff

from data_prep.lbl_utils import get_sorted_veg_from_excel, get_veg_label_of_img
filex="../Veg/2016_veg_data.xlsx"

#cols = get_sorted_veg_from_excel(file=filex)

#test = tiff.TiffFile('./data_prep/imgs_veg/test/01_CCBA1_CCBB2-1.tif')
def rename():
  for root, dirs, files in os.walk('./data_prep/imgs_veg'):
    for f in [file for file in files if file.endswith(('.tif'))]:
      
      pp = get_veg_label_of_img(f.split('.')[0], cols)
      i = pathlib.Path(root, f).__str__()
      i = './' + i.replace('\\', '/')
      
      with tiff.TiffFile(i) as ti:
        at = ti.asarray()
      try:
        with tiff.TiffWriter(i) as tif:
          tif.save(at, description=pp)
      except Exception as ex:
        pass


def sort_veg_imgs_to_test(file="../Veg/2016_veg_data.xlsx", dn='./data_prep/imgs_veg', size=-1, safe=None, rate=0.20):
  '''
  Sort tiffs into test folder from training folder

  :param file: Excel file of labels
  :param dn: Directory of imgs
  :param size: How many to move?
  :return:
  '''
  
  cols = get_sorted_veg_from_excel(file=file)
  
  dir = os.listdir(dn + "/train")
  
  imgs = {}
  
  for file in dir:
    
    if file.endswith('.tif'):
      
      try:
        tc = file.split('.')
        pp = get_veg_label_of_img(tc[0], cols)
        
        if isinstance(pp, str):
          imgs.setdefault(pp, []).append(file)
      
      except Exception as exx:
        print(exx)
  tot = 0
  for k, v in imgs.items():
    d_nm = dn+'/'+k
    if not os.path.exists(d_nm):
      os.makedirs(d_nm)
      
    vl = v.__len__()
    
    x = 0
    while x < vl:
      tot += 1
      os.rename(dn + "/train/" + v[x], d_nm + "/" + v[x])
      print("Moved " + v[x] + " to testing.")
      x += 1
      
  print("Moved: " + str(tot))
if __name__ == "__main__":
  sort_veg_imgs_to_test()