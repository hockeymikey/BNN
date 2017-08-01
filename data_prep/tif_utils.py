import os

from osgeo import gdal, osr

# https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
from data_prep.custom_classes import Mosaic_n, Mosaic


def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            #print(x,y)
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def OpenTif(raster):
  '''
  Get latlong of a tiff
  :param raster:
  :return:
  '''
  
  #raster = r'../Mosaics/_2016/20160726_guil_bh_03_120m_transparent_mosaic_group1.tif'
  #raster = r'../Mosaics/june_2016/20150612_PR_BH_01_GC09_075.Ortho_0.tif'
  ds = gdal.Open(raster)
  
  gt = ds.GetGeoTransform()
  cols = ds.RasterXSize
  rows = ds.RasterYSize
  ext = GetExtent(gt, cols, rows)
  
  src_srs = osr.SpatialReference()
  src_srs.ImportFromWkt(ds.GetProjection())
  # tgt_srs=osr.SpatialReference()
  # tgt_srs.ImportFromEPSG(4326)
  tgt_srs = src_srs.CloneGeogCS()
  #print(tgt_srs)
  
  geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)
  return geo_ext


def OpenTif_latlong(raster):
  '''
  Get latlong of a tiff
  
  :param raster:
  :return:
  '''
  # get the existing coordinate system
  ds = gdal.Open(raster)
  old_cs = osr.SpatialReference()
  #old_cs.ImportFromWkt(ds.GetProjectionRef())
  old_cs.ImportFromWkt(ds.GetProjection())
 # create the new coordinate system

  '''wgs84_wkt = """
  GEOGCS["WGS 84",
      DATUM["WGS_1984",
          SPHEROID["WGS 84",6378137,298.257223563,
              AUTHORITY["EPSG","7030"]],
          AUTHORITY["EPSG","6326"]],
      PRIMEM["Greenwich",0,
          AUTHORITY["EPSG","8901"]],
      UNIT["degree",0.01745329251994328,
          AUTHORITY["EPSG","9122"]],
      AUTHORITY["EPSG","4326"]]"""'''
  #wgs84_wkt = old_cs.CloneGeogCS()
  #new_cs = osr.SpatialReference()
  #new_cs.ImportFromWkt(wgs84_wkt)
  new_cs = old_cs.CloneGeogCS()
  
  # create a transform object to convert between coordinate systems
  transform = osr.CoordinateTransformation(old_cs, new_cs)

  # get the point to transform, pixel (0,0) in this case

  width = ds.RasterXSize
  height = ds.RasterYSize
  gt = ds.GetGeoTransform()
  maxx = gt[0]
  miny = gt[3] + width * gt[4] + height * gt[5]
  minx = gt[0] + width * gt[1] + height * gt[2]
  maxy = gt[3]
  
  #print(minx, miny, maxx, maxy)
  
  # get the coordinates in lat long
  kk = []
  p1 = transform.TransformPoint(minx, maxy)
  kk.append([p1[0], p1[1]])
  p2 = transform.TransformPoint(maxx, miny)
  kk.append([p2[0], p2[1]])
  return kk

def MinMax_of_corners(corners):
  '''
  Sorts corners into minx, maxy, maxx, miny list
  :param corners:
  :return:
  '''
  minx = miny = maxx = maxy = 0
  
  for f in corners:
    if f[0] < minx:
      minx = f[0]
    if f[0] > maxx:
      maxx = f[0]
    if f[1] < miny:
      minx = f[1]
    if f[1] > maxy:
      maxx = f[1]
  return [minx, maxy, maxx, miny]

def OldMos_Extract(dir='../../Mosaics'):
  Mos = {}
  for root, dirs, files in os.walk(dir):
    for f in [file for file in files if file.endswith(('.tif'))]:
      #print('\n')
      #print('===========[ ' + f + ' ]==============')
      tmp = OpenTif_latlong(root.replace('\\', '/')+ '/' + f)
      Mos[f] = Mosaic(f, root.replace('\\', '/') + '/', tmp)
      #print(tmp)
  return Mos

def get_mos_from_dir(dir='../../Mosaics'):
  '''
  Walks all tiff files in a partifular directory, returning mosaic objects in dict
  :param dir:
  :return:
  '''
  Mos = {}
  for root, dirs, files in os.walk(dir):
    for f in [file for file in files if file.endswith(('.tif'))]:
      Mos[f] = Mosaic_n(root.replace('\\', '/') + '/'+f,short=f.split("_")[2])
  return Mos

def CSV_Extract():
  '''
  Old CSV extract for tiffs
  :return:
  '''
  
  file = open('output.csv','w')
  for k,v in OldMos_Extract().items():
    file.write(v.name+','+str(v.corners[0][0])+','+str(v.corners[0][1])+','+str(v.corners[1][0])+','+str(v.corners[1][1])+'\n')
  file.close()


def has_pure_alpha(b):
  '''
  Checks an image for alpha. Image must be a numpy array
  '''
  try:
    for i in range(b.shape[0]):
      t = b[i]
      for x in range(t.shape[0]):
        p = t[x]
        if p[0] == 0 and p[1] == 0 and p[2] == 0 and p[3] == 0:
          return True
  except IndexError:
    return True
  
  return False


if __name__ == "__main__":
  CSV_Extract()