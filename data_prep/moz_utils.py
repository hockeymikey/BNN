import math
from decimal import Decimal

from osgeo import gdal, osr

from data_prep.custom_classes import v_point


def cords2pixel(mos, bbox):
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


def get_step_size(a,b,cols,topLeft=None):
  '''
  Given two points and the grids (cols) it returns the step size based on how many points in cols are along that line segment
  '''
  
  if abs(ord(a.col) - ord(b.col)) == 1 and abs(int(a.row) - int(b.row)) == 1:
    try:
      left, right, dir = points_2_direction(a, b)
      path = cols[topLeft.zone][topLeft.col][topLeft.row][dir]
      de = abs(((a.x - b.x) ** 2) + ((b.y - a.y) ** 2).sqrt())
      tmp = de / Decimal(path[1].__len__())
      return tmp
    except KeyError as ex:
      print(ex)
    except Exception as exx:
      print(exx)
  #return 0.0


def points_2_direction(a, b):
  '''
  Give two points, converts to the direction tying the two together.
  '''
  
  #if a.col == 'M':
  #  print('')
  if int(a.row) > int(b.row):
    c = b
    d = a
  else:
    c = a
    d = b
  if c.col < d.col:
    dir = "NW-SE"
  else:
    dir = "NE-SW"
  return c,d,dir


def direction_2_point(dir, c, r):
  '''
  Give a direction as a string on the Excel, converts to the corisponding point in the grid.
  '''
  
  if dir == "NW-SE":
    return (chr(ord(r)+1),chr(ord(c[0])+1))
  elif dir == "NE-SW":
    return (chr(ord(r)-1), chr(ord(c[0]) + 1))


def get_bounding_box(latitude_in_degrees, longitude_in_degrees, half_side_in_m):
  '''
  Given a half side of a square in meters and the center cord,
  returns the cords the top left and bottom left of the bounding box.
  '''
  
  assert half_side_in_m > 0
  assert latitude_in_degrees >= -90.0 and latitude_in_degrees <= 90.0
  assert longitude_in_degrees >= -180.0 and longitude_in_degrees <= 180.0

  y = Decimal('58.71931837720249161712328510')
  if y == latitude_in_degrees:
    print('')
  
  half_side_in_km = half_side_in_m / 1000.0
  lat = math.radians(latitude_in_degrees)
  lon = math.radians(longitude_in_degrees)
  
  radius = 6371
  # Radius of the parallel at given latitude
  parallel_radius = radius * math.cos(lat)
  
  lat_min = lat - half_side_in_km / radius
  lat_max = lat + half_side_in_km / radius
  lon_min = lon - half_side_in_km / parallel_radius
  lon_max = lon + half_side_in_km / parallel_radius
  rad2deg = math.degrees
  
  kj = [[rad2deg(lat_min), rad2deg(lon_min)], [rad2deg(lat_max), rad2deg(lon_max)]]
  return kj


def x_box_convert(dir, p):
  '''
  Given the top-left point of a x-box in the grid and a direction, the gride, returns two points? First being the start
  '''
  
  if dir == "NW-SE":
    return p, v_point(p.zone + chr(ord(p.col) + 1) + str(p.row + 1))
  elif dir == "NE-SW":
    return v_point(p.zone + chr(ord(p.col) + 1) + str(p.row)), v_point(p.zone + p.col + str(p.row - 1))


def get_dist_to_cords(latitude_in_degrees, longitude_in_degrees, dist_m):
  '''
  Returns the cords given a half distance in meters
  '''
  
  assert dist_m > 0
  assert latitude_in_degrees >= -90.0 and latitude_in_degrees <= 90.0
  assert longitude_in_degrees >= -180.0 and longitude_in_degrees <= 180.0
  
  half_side_in_km = dist_m / 1000.0
  lat = math.radians(latitude_in_degrees)
  lon = math.radians(longitude_in_degrees)
  
  radius = 6371
  # Radius of the parallel at given latitude
  parallel_radius = radius * math.cos(lat)
  
  lat_max = lat + half_side_in_km / radius
  lon_max = lon + half_side_in_km / parallel_radius
  rad2deg = math.degrees
  k = rad2deg(lat_max)-latitude_in_degrees
  l = rad2deg(lon_max)
  return [[k, l-longitude_in_degrees]]


def latLonToPixel(geotifAddr, latLonPairs):
  '''
  Given a set of cords and an img location, returns the corresponding pixel pairs
  '''
  
  # Load the image dataset
  ds = gdal.Open(str(geotifAddr))
  
  # Get a geo-transform of the dataset
  gt = ds.GetGeoTransform()
  
  # Create a spatial reference object for the dataset
  srs = osr.SpatialReference()
  srs.ImportFromWkt(ds.GetProjection())
  
  # Set up the coordinate transformation object
  srsLatLong = srs.CloneGeogCS()
  ct = osr.CoordinateTransformation(srsLatLong, srs)
  
  # Go through all the point pairs and translate them to latitude/longitude pairings
  pixelPairs = []
  
  for point in latLonPairs:
    # print(point)
    point[1] = float(point[1])
    point[0] = float(point[0])
    
    # Change the point locations into the GeoTransform space
    (point[1], point[0], holder) = ct.TransformPoint(point[1], point[0])
    
    # Translate the x and y coordinates into pixel values
    x = (point[1] - gt[0]) / gt[1]
    y = (point[0] - gt[3]) / gt[5]
    
    if x < 0:
      x = 0
      print(latLonPairs)
    if y < 0:
      y = 0
      print(latLonPairs)
    
    # Add the point to our return array
    pixelPairs.append([int(x), int(y)])
  return pixelPairs


def topleft_of_points(a,b):
  '''
  Returns topleft of two points so either a or some other point if it's a NE-SW
  :param a:
  :param b:
  :return:
  '''
  if a.col > b.col:
    return v_point(str(a.zone) + str(chr(ord(b.col) - 1)) + str(a.row))
  else:
    return a

def decdeg2dms(dd):
  '''
  Converts a decimal based cord to a degrees, minute seconds based one.
  :param dd:
  :return:
  '''
  # https://stackoverflow.com/questions/2579535/how-to-convert-dd-to-dms-in-python
  is_positive = dd >= 0
  dd = abs(dd)
  minutes, seconds = divmod(dd * 3600, 60)
  degrees, minutes = divmod(minutes, 60)
  degrees = degrees if is_positive else -degrees
  new_min = str(minutes).split('.')[0]
  new_sec = str(seconds).split('.')
  if len(new_sec[0]) == 1:
    new_sec[0] = '0'+new_sec[0]
    
  tmp = str(degrees).split('.')[0] + '.' + new_min + new_sec[0] + new_sec[1]
  return Decimal(tmp)

def Moz_contains_old(m,vX,vY):
  '''
  Old Moz_contains
  :param m:
  :param vX:
  :param vY:
  :return:
  '''
  return ((m.corners[0][0] < vX and m.corners[1][0] > vX) or (m.corners[0][0] > vX and m.corners[1][0] < vX)) and ((m.corners[0][1] < vY and m.corners[1][1] > vY) or (m.corners[0][1] > vY and m.corners[1][1] < vY))

def Moz_contains(m,vX,vY):
  '''
  Given a mosiac_n object and an X (long) and Y (lat) returns if it's contained in the mosaic.
  :param m:
  :param vX:
  :param vY:
  :return:
  '''
  #Could be streamlined but kept ugly for easier debugging just in case.
  tt = (m.min_lat <  vY and m.max_lat > vY)
  kk = (m.min_long > vX and m.max_long < vX)
  
  return tt and kk

  