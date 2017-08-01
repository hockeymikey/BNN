import re
from decimal import Decimal

from osgeo import gdal, osr


class Mosaic(object):
  '''
  Old Mosaic object.  Why is it still here?
  '''
  def __init__(self, name="None", root="None", cor=None):
    if cor is None:
      self.corners = []
    else:
      self.corners = cor
      self.name = name
      self.root = root

class v_img_set(object):
  '''
  Never used.  Was intended for holding data for training a net?  Something like that
  look at the one hot variable
  '''
  def __init__(self,imgs,labels,totals,hot_con):
    self.imgs = imgs
    self.labels = labels
    self.totals = totals
    self.hot_con = hot_con
    
  def get_hot_name(self, hot):
    name = "Not found"
    try:
      name = self.hot_con[name]
    except:
      pass
    return name

class v_seg(object):
  '''
  Info about a line segment between two points (v_point)
  '''
  def __init__(self, name = "None", steps=None):
    self.name = name
    self.steps = steps

class v_step(object):
  '''
  Holds a step along two points, a location where an image is going to be extracted.
  '''
  
  def __init__(self, name,cords,label,prefix=None,head=False):
    self.head = head
    
    self.name = name
    self.x = cords['x']
    self.y = cords['y']
    self.label = label
    self.prefix = prefix


class v_point(object):
  '''
  Holds a "point" which is a single location on the grid of veg data that is measured from/to
  Holds the data of it's zone,col,row and cords.
  '''
  def __init__(self, name = "None", cords = None):
    name = re.sub('[^0-9a-zA-Z]+', '*', name)
    
    self.name = name
    self.v_steps = {}
    self.cords = cords
    
    if cords != None:
      self.long = Decimal(cords[1])
      self.lat = Decimal(cords[0])
      self.y = Decimal(cords[0])
      self.x = Decimal(cords[1])
    
    if name.__len__() == 5 and not( name[3].isdigit() ):
      self.zone = name[0:3]
      self.col = name[3]
      self.row = name[4]
    else:
      self.zone = name[0:2]
      self.col = name[2]
      self.row = name[3]


class Mosaic_n(object):
  '''
  New mosaic object.  Holds metadata (info) about a particular mosaic like directory and size.
  '''
  def __init__(self, name,short=None):
    tmp = self.tif_ex(name)
    self.corners = tmp[1]
    self.x = tmp[0][0]
    self.y = tmp[0][1]
    self.name = name[name.rfind("/")+1:]
    self.root = name[:name.rfind("/")+1]
    
    self.max_lat = Decimal(self.corners[0][1])
    self.min_lat = Decimal(self.corners[1][1])
    
    self.max_long = Decimal(self.corners[1][0])
    self.min_long = Decimal(self.corners[0][0])
    
    if short == None:
      try:
        self.shortened = re.search('.+?(?=_transparent)', name).group(0)
      except:
        self.shortened = short
    else:
      self.shortened = short
      
    self.address = self.root+self.name
  
  def tif_ex(self, raster):
    '''
    Get lat/long of the tiff of the mosaic
    :param raster:
    :return:
    '''
    
    ds = gdal.Open(raster)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjection())

    new_cs = old_cs.CloneGeogCS()

    transform = osr.CoordinateTransformation(old_cs, new_cs)
    width = Decimal(ds.RasterXSize)
    height = Decimal(ds.RasterYSize)
    gt = ds.GetGeoTransform()
    gtt = []
    for x in gt:
      gtt.append(Decimal(x))
    gt = gtt
    
    maxx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    minx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]
    kk = []
    p1 = transform.TransformPoint(float(minx), float(maxy))
    kk.append([Decimal(p1[0]), Decimal(p1[1])])
    p2 = transform.TransformPoint(float(maxx), float(miny))
    kk.append([Decimal(p2[0]), Decimal(p2[1])])
    return [[ds.RasterXSize,ds.RasterYSize],kk]