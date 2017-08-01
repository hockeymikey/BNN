from pykml import parser

importFile = open('../Veg/New 2016 Veg Points.kml')
file = open('output/new_points_KML.csv','w')

doc = parser.parse(importFile).getroot().Document.Folder

for p in doc.Placemark:
  file.write(p.name+','+p.Point.coordinates+"\n")
  print(p.name+','+p.Point.coordinates)
file.close()