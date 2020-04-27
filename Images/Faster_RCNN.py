from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
import copy
import pandas as pd
from optparse import OptionParser
import time
from keras_frcnn2 import config
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.layers import Input
from keras.models import Model
from keras_frcnn2 import roi_helpers
from osgeo import ogr, osr, gdal
import cvGeoReference as georef
import json
import os
import geopandas as gpd
import math
from PIL import Image
import shutil
import gc

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to read the metadata related to the training (generated when training).",
                default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("-i", "--input", dest="input", type = "string", help="Path to tif image.")
parser.add_option("-s", "--shapefile", dest="shapefile", type = "string", help="Path to shapefile.")
parser.add_option("-o","--output",dest="output",type="string", help="Output csv data")
parser.add_option("-m","--model_path",dest="model_path",type="string", help="Model File Path")
parser.add_option("--UAVCapturedDate", dest="UAVCapturedDate", help="UAVCapturedDate")
parser.add_option("--PATType", dest="PATType", help="PATType")
parser.add_option("--InsertedDate", dest="InsertedDate", help="InsertedDate")
parser.add_option("--PetakId", dest="PetakId", help="PetakId")
parser.add_option("--SpeciesName", dest="SpeciesName", help="SpeciesName")                                                                                      
(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')






config_output_filename = options.config_filename

input_ =  options.input 
shapefile_ = options.shapefile 
img_path = options.test_path
output_dir = img_path 
model_filepath = options.model_path

InsertedDate = options.InsertedDate
PATType = options.PATType
PetakId = options.PetakId
SpeciesName = options.SpeciesName
UAVCapturedDate = options.UAVCapturedDate

results = options.output
resultDir = results  + '/' #+ '/Output/'


print("Input: ", input_)
print("Shapefile: ", shapefile_)
print("Test Data Path: ", img_path)
print("Temporary Cut Square Directory: ", output_dir)
print("Output Directory: ", resultDir)
print("Pickle filename : ", config_output_filename)
print("Model filepath : ", model_filepath)
print("UAVCapturedDate : ", UAVCapturedDate)
print("InsertedDate : ", InsertedDate)
print("PATType : ", PATType)
print("PetakId : ", PetakId)

#exit()
if not os.path.exists(resultDir):
    print("Creating result directory:", resultDir)
    os.makedirs(resultDir)

if not os.path.exists(output_dir):
    print("Creating temporary cut square directory:", output_dir)
    os.makedirs(output_dir)
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'resnet50':
    import keras_frcnn2.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn2.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False



class cvNEPos:
    northing = 0.0
    easting = 0.0

    def __init__(self, n, e):
        self.northing = n
        self.easting = e

    def __str__(self):
        return "N: " + str(self.northing) + ", E: " + str(self.easting)


class cvGPSPos:
    lat = 0.0
    lon = 0.0
    alt = 0.0

    def __init__(self, lat, lon, alt=0.0):
        self.lat = lat
        self.lon = lon
        self.alt = alt

    def __str__(self):
        return "Lat: " + str(self.lat) + ", Lon: " + str(self.lon)


class cvGeorefData:
    radPerPixX = 0.0
    radPerPixY = 0.0
    topLeftNE = None
    bottomRightNE = None
    metersPerPixelX = 0.0
    metersPerPixelY = 0.0
    metersPerPixel = 0.0
    topLeftGps = None
    bottomRightGps = None
    coordTransNEToGPS = None
    coordTransGPSToNE = None

    def scale(self, scaling):
        if scaling == 1 or scaling == 0:
            return
        self.metersPerPixelX /= scaling
        self.metersPerPixelY /= scaling
        self.metersPerPixel /= scaling
        self.radPerPixX /= scaling
        self.radPerPixY /= scaling
    
def read_tiff(image_tiff):
    rgb_tiff = gdal.Open(image_tiff)
    img_width = rgb_tiff.RasterXSize
    img_height = rgb_tiff.RasterYSize
    prj_rgb = rgb_tiff.GetProjection()
    jgw_rgb = rgb_tiff.GetGeoTransform()
    rgb_img = tif_to_rgb(rgb_tiff)
    georefData = cvGeorefData()
    georefData = initializeJgw_geotiff(georefData, jgw_rgb, prj_rgb, img_height, img_width)
    return georefData, img_width, img_height, rgb_img, prj_rgb, jgw_rgb

def initializeJgw_geotiff(georefDataNDVI, jgw, prj, width, height):
    spatialRef = osr.SpatialReference()
    #spatialRef.ImportFromEPSG(32647)
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromWkt(prj)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    # cvlog("Input EPSG format:"+str(32647))

    georefDataNDVI.coordTransNEToGPS = osr.CoordinateTransformation(spatialRef, target)
    georefDataNDVI.coordTransGPSToNE = osr.CoordinateTransformation(target, spatialRef)
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(jgw[0], jgw[3])
    WGS84 = ogr.Geometry(ogr.wkbMultiPoint)
    WGS84.AddGeometry(point)
    # Add Bottom-Right point to the point list
    x = 0
    y = 0
    # print("inside condition")
    if (jgw[2] == 0) and (jgw[4] == 0):
        x = jgw[0] + (width - 1) * jgw[1] + (height - 1) * jgw[2]
        y = jgw[3] + (width - 1) * jgw[4] + (height - 1) * jgw[5]
    else:
        return "Error: Affine transformation not implemented."
    georefDataNDVI.topLeftNE = cvNEPos(jgw[3], jgw[0])
    georefDataNDVI.bottomRightNE = cvNEPos(y, x)
    point.AddPoint(x, y)
    WGS84.AddGeometry(point)
    # Make the transformation from input coordinate system to the output (WGS84)
    WGS84.Transform(georefDataNDVI.coordTransNEToGPS)
    points = Geometry2List(WGS84)
    georefDataNDVI.topLeftGps = cvGPSPos(points[0][1], points[0][0])
    georefDataNDVI.bottomRightGps = cvGPSPos(points[1][1], points[1][0])
    georefDataNDVI.radPerPixX = (georefDataNDVI.bottomRightGps.lon - georefDataNDVI.topLeftGps.lon) / (width - 1)
    georefDataNDVI.radPerPixY = (georefDataNDVI.bottomRightGps.lat - georefDataNDVI.topLeftGps.lat) / (height - 1)
    georefDataNDVI.metersPerPixelX = jgw[1]
    georefDataNDVI.metersPerPixelY = jgw[5]
    georefDataNDVI.metersPerPixel = (abs(georefDataNDVI.metersPerPixelX) + abs(georefDataNDVI.metersPerPixelY)) / 2
    return georefDataNDVI

def Geometry2List(input):
    temp = input.ExportToJson()
    jsonPoints = json.loads(temp)
    points = jsonPoints["coordinates"]
    return points
    
def tif_to_rgb(rgb_tiff):
    R = rgb_tiff.GetRasterBand(1)
    G = rgb_tiff.GetRasterBand(2)
    B = rgb_tiff.GetRasterBand(3)
    r_data = R.ReadAsArray()
    g_data = G.ReadAsArray()
    b_data = B.ReadAsArray()

    rgb_img = np.dstack([b_data, g_data, r_data]) #bgr
    
    return rgb_img
    
def pixelToLatLon(georefData, x, y):
    # X-axis assumed to point East and Y-axis to South
    lon = georefData.topLeftGps.lon + x * georefData.radPerPixX
    lat = georefData.topLeftGps.lat + y * georefData.radPerPixY
    return cvGPSPos(lat,lon)

    
def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape
        
    if width <= height:
        ratio = img_min_side/width
        #new_height = int(ratio * height) #mc 21jan
        #new_width = int(img_min_side)
        new_height = int(round(ratio * height))
        new_width = int(round(img_min_side))
    else:
        ratio = img_min_side/height
        #new_width = int(ratio * width) #mc 21jan
        #new_height = int(img_min_side)
        new_width = int(round(ratio * width))
        new_height = int(round(img_min_side))
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    
    return img, ratio    

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    

    dd = int(round(x1 // ratio))
    ee = float(x1 / ratio) 
    
    real_x1 = float(x1 / (ratio-(ratio / 10))) 
    real_y1 = float(y1 / (ratio-(ratio / 10)))
    real_x2 = float(x2 / (ratio-(ratio / 10)))
    real_y2 = float(y2 / (ratio-(ratio / 10)))
    
   
    
    return (real_x1, real_y1, real_x2 ,real_y2)

def get_real_coordinates_cobs(ratio, x1, y1, x2, y2):

    ##mc 17jan
    #real_x1 = int(round(x1 // ratio))
    #real_y1 = int(round(y1 // ratio))
    #real_x2 = int(round(x2 // ratio))
    #real_y2 = int(round(y2 // ratio))
    
    real_x1 = int(round(x1 / ratio))
    real_y1 = int(round(y1 / ratio))
    real_x2 = int(round(x2 / ratio))
    real_y2 = int(round(y2 / ratio))
    
    return (real_x1, real_y1, real_x2 ,real_y2)

    
def read_selected_shpdata(shp_filename):
    shpfile_gdf = gpd.GeoDataFrame.from_file(shp_filename)
    selshpdf = pd.DataFrame(shpfile_gdf)
    print("selshpdf: \n",selshpdf.head())
    selshp = selshpdf['geometry'].astype(str)
    print("selshp :\n",selshp.head())
    if selshp.iloc[0].startswith('POLYGON ((1'):
        selshptype = 'latlon'
    else: 
        selshptype = 'NE'
    print ("selshptype :", selshptype)
    return selshp,selshpdf,selshptype
    
def NEToPixel(georefData, easting, northing):
    x = (easting - georefData.topLeftNE.easting) / georefData.metersPerPixelX
    y = (northing - georefData.topLeftNE.northing) / georefData.metersPerPixelY
    return (int(x), int(y))
    
#############################
#cutting (get x,y)
#############################

ggeorefData, img_width, img_height, rgb_img, prj, jgw = read_tiff(input_)
petaks_name = input_.split("/")[-1]#[7]#[6]
petaks_name_ = petaks_name.split('.')[0]
boundary_ = petaks_name.split("_")[4]
shapefile_ = shapefile_ + boundary_+"/"+boundary_+".shp"

imagelist = [
{'raster': 'input_',
'shape': 'shapefile_',
'label': 'shapefile_'}
]

tiles = {}
Count = 0

RasterFormat = 'GTiff'
PixelRes = 1.0
VectorFormat = 'ESRI Shapefile'
ImageIdx = 0
glb_HnW = 0
degreeShf = 0.25

#mc 25feb
###################Read AOI Shp 
selshp,selshpdf,selshptype = read_selected_shpdata(shapefile_) 

try :
        #read boundary shpdata
    AOLatLonLocations = []
    for i in selshp:
        start_s2 = "MULTIPOLYGON ((("
        end_s2 = ")))"
        start_s = "POLYGON (("
        end_s = "))"
        mid_s = "(("
        items = i.replace(start_s2, '').replace(end_s2, '').replace(start_s, '').replace(end_s, '').replace(mid_s, '').split(',')
        tmp = []
        for j in items:
            b = j.split(' ')
            tmp.append([float(item) for item in b if item != ''])
        AOLatLonLocations.append(tmp)
    print ("shape petak is found")
    #toWrite+="Shape petak is found "+ '\n'
    print ("Completed and start selected shp - 1/2 ")
    #toWrite+="Completed and start selected shp - 1/2 "+ '\n'
except Exception as e:
    print(str(e))
    print("shape petak not found")
    #toWrite+="Shape petak not found "+ '\n'

AOLatLonLocationsNp = []
for i in range(len(AOLatLonLocations)):
    a = np.array(AOLatLonLocations[i])
    AOLatLonLocationsNp.append(a)

selected_trf_actual = []
for j in range (len(AOLatLonLocationsNp)):
    tmp = []
    for item in AOLatLonLocationsNp[j]:
        # some of them have various formats # 
        ### pay attention, some selected shp files come in the format of lat lon and some in NE ### 
        if selshptype == 'latlon':
            items = latLonToPixel(georefData, item[0], item[1])
        if selshptype == 'NE':
            items = NEToPixel(georefData, item[0], item[1])
        #print(items)
        items = [[np.int32(items[0]),np.int32(items[1])]]
        tmp.append(items)
    selected_trf_actual.append(np.array(tmp))

del AOLatLonLocations, AOLatLonLocationsNp
print ("Completed selected shp - 2/2")
#toWrite+="Completed and start selected shp - 2/2 "+ '\n'

img_select_mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1]))
print("len(selected_trf_actual)" + str(len(selected_trf_actual)))
for i in range(len(selected_trf_actual)):
    cv2.fillPoly(img_select_mask, pts=[selected_trf_actual[i]], color=255)

for imagemetadata in imagelist:
    ImageIdx += 1
    Count = 0
    InputImage = input_ #imagemetadata['raster']
    Shapefile = shapefile_ #imagemetadata['shape']
    Labelfile = shapefile_ #imagemetadata['label']
    # Open datasets
    Raster = gdal.Open(InputImage, gdal.GA_ReadOnly)
    Projection = Raster.GetProjectionRef()
    srs=osr.SpatialReference(wkt=Projection)
    if srs.IsProjected:
        print (srs.GetAttrValue('projcs'))
        print (srs.GetAttrValue('geogcs'))

    VectorDriver = ogr.GetDriverByName(VectorFormat)
    VectorDataset = VectorDriver.Open(Shapefile, 0) # 0=Read-only, 1=Read-Write
    layer = VectorDataset.GetLayer()
    FeatureCount = layer.GetFeatureCount()
    print("Feature Count:",FeatureCount)

    # latlng to 32647 / 32748
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)
    target = osr.SpatialReference()
    target.ImportFromEPSG(32748) #target.ImportFromEPSG(32647)
    transform = osr.CoordinateTransformation(source, target)

    def getPixel(geotransform, dx,dy):
      px = geotransform[0]
      py = geotransform[3]
      rx = geotransform[1]
      ry = geotransform[5]
      x = dx*rx + px
      y = dy*ry + py
      return x,y

    def getPixelFromCoord(geotransform, x,y):
      px = geotransform[0]
      py = geotransform[3]
      rx = geotransform[1]
      ry = geotransform[5]
      dx = (x - px)/rx
      dy = (y - py)/ry 
      
      print("DX "+str(dx))
      print("DY "+str(dy))
      #time.sleep(10)
      #return int(dx),int(dy) #mc21jan
      return int(dx),int(dy)
      
    geotransform = Raster.GetGeoTransform()
    cols = Raster.RasterXSize
    rows = Raster.RasterYSize
    
    if(cols>rows):
        glb_HnW = 1
    
    print(['size', [cols,rows]])
    #time.sleep(10)
    tile_size = 350 #350
    stride = 250 #300 24jan #250 #320 #250 30jan
    tile_size_g =250
    stride_g = 250
    print(['topleft',getPixel(geotransform, 0,rows)])
    print(['botright',getPixel(geotransform, cols,0)])

    minX = getPixel(geotransform, 0,rows)[0]
    maxY = getPixel(geotransform, cols, 0)[1]
    maxX = getPixel(geotransform, cols, 0)[0]
    minY = getPixel(geotransform, 0,rows)[1]
    
    print([minX, minY, maxX, maxY])
    
    ##mc 21jan, 23jan comment
    # gridCsv = open("/home/pfiadmin/mc/EPEL_6m_MC/OutputBS/2mar/"+petaks_name+"_250_8_5_3_6_shfxy0.25_resnet_px0.03_BS_grid_0.01.csv", 'w')   #mc
    s = ''  #s = 'Grids\n'
    gridCsv.write(s)

    for x_idx in range(int(math.floor(cols/stride))):  #floor to ceils, kl pke floor pas buffernya g ckup jdi problem, kl pke ceil numpuk di bawah klo buffernya g cukup
          xmin = x_idx * stride
          xmin_g =  x_idx *stride_g
          xmax = x_idx * stride + tile_size
          xmax_g = x_idx * stride_g + tile_size_g
          if xmax > cols:
            xmax = cols
          for y_idx in range(int(math.floor(rows/stride))): #floor to ceils
              ymin = y_idx * stride
              ymin_g = y_idx * stride_g
              ymax = y_idx * stride + tile_size
              ymax_g = y_idx * stride_g + tile_size_g
              if ymax > rows:
                ymax = rows
              [cminX, cminY] = getPixel(geotransform, xmin,ymax)
              [cmaxX, cmaxY] = getPixel(geotransform, xmax,ymin)
              if cmaxX > maxX:
                cmaxX = maxX
              if cmaxY > maxY:
                cmaxY = maxY      
              if cminX < minX:
                cminX = minX
              if cminY < minY:
                cminY = minY           
              if cminX >= minX and cminY >= minY and cmaxX <= maxX and cmaxY <= maxY:
                # Create raster
                Count += 1
                OutTileName = output_dir + str(ImageIdx) + '.' + str(Count)+'.test.tif'
                #if Count % 5 == 1:  #mc 20jan
                #    isTest = 1 
                #else:
                #    isTest = 0
                TempF = "temp.tif"
                OutTile = gdal.Warp(TempF, Raster, format=RasterFormat, outputBounds=[cminX, cminY, cmaxX, cmaxY], dstSRS=Projection)
                OutTile = None # Close dataset
                raster = gdal.Open(TempF, gdal.GA_ReadOnly)
                data = raster.GetRasterBand(2).ReadAsArray()
                #if np.mean(data) > 230 or np.mean(data) < 50: #mc
                    #probably_out = 1
                #else:
                JPEGName = output_dir + str(ImageIdx)+ '.' + str(Count)+'.test.' + str(xmin) +'.'+str(ymin)+'.'+str(xmax)+'.'+str(ymax)+'.jpg'
                #ini koordinat# JPEGName = output_dir + str(ImageIdx)+ '_' + str(Count)+'_test_' + str(cminX) +'_'+str(cminY)+'_'+str(cmaxX)+'_'+str(cmaxY)+'.jpg'
                #JPEGName = output_dir + str(ImageIdx)+ '_' + str(Count)+'_test_' + str(xmin) +'_'+str(ymin)+'_'+str(xmax)+'_'+str(ymax)+'.jpg'
                gdal.Translate(JPEGName, TempF, format='JPEG')
                image = Image.open(JPEGName)
                if image.mode == 'CMYK':
                    image = image.convert('RGB').save(JPEGName)
                #tiles[OutTileName] = {'tile': JPEGName, 'mean': np.mean(data) , 'petak': InputImage, 'tilebound':[cminX, cminY, cmaxX, cmaxY], 'width': (xmax-xmin), 'height': (ymax-ymin), 'isTest': isTest, 'labels': [], 'class': 'epel', 'pixel_topleft': [xmin, ymin],'geotransform': geotransform} 
                tiles[OutTileName] = {'tile': JPEGName, 'mean': np.mean(data) , 'petak': InputImage, 'tilebound':[cminX, cminY, cmaxX, cmaxY], 'width': (xmax-xmin), 'height': (ymax-ymin), 'labels': [], 'class': 'epel', 'pixel_topleft': [xmin, ymin],'geotransform': geotransform} 
                #print(['added', JPEGName]) ##mcmc, change class name

                ##############print grid########## 21jan mc
                
                ########## mc 10feb
                 #(x1r ,y1r,x2r,y2r)= (x1+degreeShf,y1+degreeShf,x2+degreeShf,y2+degreeShf) 
                if(glb_HnW==1):
                    shfx1 = int(xmin_g)-(int(xmin_g)/stride)*degreeShf  #300 ~ 1 px, 600 ~ 0.5px  #6100 pke 0.35
                    shfy1 = int(ymin_g)-(int(ymin_g)/stride)*degreeShf
                    shfx2 = int(xmax_g)-(int(xmax_g)/stride)*degreeShf  #300 ~ 1 px, 600 ~ 0.5px  #6100 pke 0.35
                    shfy2 = int(ymax_g)-(int(ymax_g)/stride)*degreeShf
                #mc 22jan
                else:
                    shfx1 = int(xmin_g)+(int(xmin_g)/stride)*degreeShf  #300 ~ 1 px, 600 ~ 0.5px  #6100 pke 0.35
                    shfy1 = int(ymin_g)+(int(ymin_g)/stride)*degreeShf
                    shfx2 = int(xmax_g)+(int(xmax_g)/stride)*degreeShf  #300 ~ 1 px, 600 ~ 0.5px  #6100 pke 0.35
                    shfy2 = int(ymax_g)+(int(ymax_g)/stride)*degreeShf
                    
                b = img_select_mask[int(shfy1):int(shfy2), int(shfx1):int(shfx2)]
                totalpixels = b.shape[0]*b.shape[1]
                b_sum = np.sum(b == 255)
                if (b_sum/float(totalpixels) < 0.8):  #mc, comment dlu
                    continue

                    
                topLeftGpsPos = pixelToLatLon(georefData, shfx1, shfy1)
                topRightGpsPos = pixelToLatLon(georefData, shfx2, shfy1)
                bottomRightPos = pixelToLatLon(georefData, shfx2, shfy2)
                bottomLeftPos = pixelToLatLon(georefData, shfx1, shfy2)
                
                
                
                #s = 'Grids\n'
                s = s + 'POLYGON (('
                s = s + str(topLeftGpsPos.lon) + ' ' + str(topLeftGpsPos.lat) + ','
                s = s + str(topRightGpsPos.lon) + ' ' + str(topRightGpsPos.lat) + ','
                s = s + str(bottomRightPos.lon) + ' ' + str(bottomRightPos.lat) + ','
                s = s + str(bottomLeftPos.lon) + ' ' + str(bottomLeftPos.lat) + ','
                s = s + str(topLeftGpsPos.lon) + ' ' + str(topLeftGpsPos.lat)
                s = s + '))' + '\n'

    #gridCsv.write(s) #mc 23jan, comment # 10feb
                
                
#############################    
#############################
print("UDA KELAR MOTOOONGGGGGG")
st = time.time()

class_mapping = C.class_mapping


if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

if K.common.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_path_2 = model_filepath
print('Loading weights from {}'.format(model_path_2+ C.model_path))
model_rpn.load_weights(model_path_2+C.model_path, by_name=True)
model_classifier.load_weights(model_path_2+C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True


gridCsv = open(resultDir + "testRCNN.csv", 'w')  #mc


s = ''  #s = 'Grids\n'
ss = ''
gridCsv.write(s)
#mc 25feb# gridCsv2.write(ss)

#mc_nms = []
mc_nms = np.empty(shape=(0,4))
#mc_nms = np.asarray(mc_nms)
mc_nms_probs = []
mc_nms_probs = np.asarray(mc_nms_probs)
latlon_array = np.empty(shape=(0,2))

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    mc = img_name.split(".") #mc = img_name.split(".") ##20jan
    mc_tmp = np.array([mc[3],mc[4]]) #mc 16jan20 #mc_tmp = np.array([mc[3],mc[4]])
    
    
    if(glb_HnW==1):
        shfx1 = int(mc[3])-(int(mc[3])/stride)*degreeShf  #300 ~ 1 px, 600 ~ 0.5px  #6100 pke 0.35
        shfy1 = int(mc[4])-(int(mc[4])/stride)*degreeShf
        shfx2 = int(mc[5])-(int(mc[5])/stride)*degreeShf  #300 ~ 1 px, 600 ~ 0.5px  #6100 pke 0.35
        shfy2 = int(mc[6])-(int(mc[6])/stride)*degreeShf
    #mc 22jan
    else:
        shfx1 = int(mc[3])+(int(mc[3])/stride)*degreeShf   #300 ~ 1 px, 600 ~ 0.5px
        shfy1 = int(mc[4])+(int(mc[4])/stride)*degreeShf
        shfx2 = int(mc[5])+(int(mc[5])/stride)*degreeShf   #300 ~ 1 px, 600 ~ 0.5px
        shfy2 = int(mc[6])+(int(mc[6])/stride)*degreeShf
    
    topLeftGpsPos = pixelToLatLon(georefData, shfx1, shfy1)
    topRightGpsPos = pixelToLatLon(georefData, shfx2, shfy1)
    bottomRightPos = pixelToLatLon(georefData, shfx2, shfy2)
    bottomLeftPos = pixelToLatLon(georefData, shfx1, shfy2)
    
    
    #a = img_select_mask[int(mc[3]):int(mc[4]), int(mc[5]):int(mc[6])]
    a = img_select_mask[int(mc[4]):int(mc[6]), int(mc[3]):int(mc[5])]
    totalpixels = a.shape[0]*a.shape[1]
    a_sum = np.sum(a == 255)
    #print("CAAAAAAAA "+str(a_sum/float(totalpixels)))
    if (a_sum/float(totalpixels) <= 0):
        #cv2.imwrite("/data/u_mc/Coordinate/coba_boundary/uu.png",a) 
        continue
    
    ##mc 4feb
    #s = 'Grids\n'
    ss = ss + 'POLYGON (('
    ss = ss + str(topLeftGpsPos.lon) + ' ' + str(topLeftGpsPos.lat) + ','
    ss = ss + str(topRightGpsPos.lon) + ' ' + str(topRightGpsPos.lat) + ','
    ss = ss + str(bottomRightPos.lon) + ' ' + str(bottomRightPos.lat) + ','
    ss = ss + str(bottomLeftPos.lon) + ' ' + str(bottomLeftPos.lat) + ','
    ss = ss + str(topLeftGpsPos.lon) + ' ' + str(topLeftGpsPos.lat)
    #ss = ss + '))' #+ n_trees +'\n'
    ######################
    
    #########################
    
    ##st = time.time() mc, pindah atas
    filepath = os.path.join(img_path,img_name)

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)

    if K.common.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.8)  #mc, overlap_thresh=0.7 (default 0.7) 

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}
    latlon_a = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []
                latlon_a[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))    
            latlon_a[cls_name].append(mc_tmp)

    all_dets = []

    #mc = img_name.split(".")
    
    for key in bboxes:
        bbox = np.array(bboxes[key])

        #mundurin 1 tab, gajadi
        new_boxes, new_probs, new_latlon_array = roi_helpers.non_max_suppression_fast_inner(bbox, np.array(probs[key]), np.array(latlon_a[key]), overlap_thresh=0.5)  #default:0.5
    
        mc_nms = np.concatenate([mc_nms, new_boxes]) #mc_nms += new_boxes
        mc_nms_probs = np.concatenate([mc_nms_probs,new_probs])
        latlon_array = np.concatenate([latlon_array,new_latlon_array])
            
        ########## mc 20 jan ###########
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates_cobs(ratio, x1, y1, x2, y2)

            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (238,229,99),2)
            #(int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)
            
    
    ss = ss + '))' +"|"+ str(len(new_boxes)) +'\n'
    print('Elapsed time = {}'.format(time.time() - st)) #mc
    

print("DOWO BBOX 3"+str(len(mc_nms)))

#mc 4feb
#gridCsv2.write(ss)

#mc 22jan
shfx = 0.0 #degree shifted
shfy = 0.0 #degree shifted    

for key in range(mc_nms.shape[0]):
    (x1, y1, x2, y2) = mc_nms[key,:]
    
    #maria, sek2 wait, latlon_array ini blm kujadiin latlon
    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates_cobs(ratio, x1, y1, x2, y2) #mc 21jan
    
    pxX = float(latlon_array[key,0])
    pxY = float(latlon_array[key,1])

    #degreeShf = 0.25
    #mc 23jan
    if(glb_HnW==1):
        shfx = pxX-(pxX/stride)*degreeShf  #300 ~ 1 px, 600 ~ 0.5px  #6100 pke 0.35
        shfy = pxY-(pxY/stride)*degreeShf
    #mc 22jan
    else:
        shfx = pxX+(pxX/stride)*degreeShf   #300 ~ 1 px, 600 ~ 0.5px
        shfy = pxY+(pxY/stride)*degreeShf
    
    mc_nms[key,0] = real_x1+ shfx #float(latlon_array[key,0])
    mc_nms[key,1] = real_y1+ shfy #float(latlon_array[key,1])
    mc_nms[key,2] = real_x2+ shfx #float(latlon_array[key,0])
    mc_nms[key,3] = real_y2+ shfy #float(latlon_array[key,1])
    

    

    
new_boxes2, new_probs2, new_latlon_array2 = roi_helpers.non_max_suppression_fast_inner1(mc_nms, mc_nms_probs, latlon_array, overlap_thresh=0.3) #def:0.3
print("DOWO NEW BBOX 4 threshold 0.5 == "+str(len(new_boxes2)))


shf= 0.0 
gridwkt = []
grid_df = pd.DataFrame()          
for jk in range(new_boxes2.shape[0]):
    (x1, y1, x2, y2) = new_boxes2[jk,:]
    
   
    
    
    (x1r ,y1r,x2r,y2r)= (x1+shf,y1+shf,x2+shf,y2+shf) #mc 22 jan
   
    b = img_select_mask[int(y1):int(y2), int(x1):int(x2)]
    totalpixels = b.shape[0]*b.shape[1]
    b_sum = np.sum(b == 255)
    if (b_sum/float(totalpixels) < 0.03):  
        continue
    
    topLeftGpsPos = pixelToLatLon(georefData, x1r, y1r)
    topRightGpsPos = pixelToLatLon(georefData, x2r, y1r)
    bottomRightPos = pixelToLatLon(georefData, x2r, y2r)
    bottomLeftPos = pixelToLatLon(georefData, x1r, y2r)
    
   

    #s = 'Grids\n'
    s = 'POLYGON (('
    s = s + str(topLeftGpsPos.lon) + ' ' + str(topLeftGpsPos.lat) + ','
    s = s + str(topRightGpsPos.lon) + ' ' + str(topRightGpsPos.lat) + ','
    s = s + str(bottomRightPos.lon) + ' ' + str(bottomRightPos.lat) + ','
    s = s + str(bottomLeftPos.lon) + ' ' + str(bottomLeftPos.lat) + ','
    s = s + str(topLeftGpsPos.lon) + ' ' + str(topLeftGpsPos.lat)
    s = s + '))' + ','
    gridwkt.append(s)

grid_df["Grids"] = pd.Series(gridwkt).map(lambda x: x.lstrip('"').rstrip('"'))
grid_df = grid_df.applymap(lambda x: x.replace('"', ''))
grid_df.to_csv(resultDir+"temp.csv", sep = "|", index = False, header = True)


tree_df = pd.read_csv(resultDir+"temp.csv", sep = "|")
tree_df_2 = tree_df.iloc[:,0]
tree_wkts = [ogr.CreateGeometryFromWkt(wkt) for wkt in tree_df_2]
tree_df['resultX'] = pd.Series([i.Centroid().GetX() for i in tree_wkts],index=tree_df.index)
tree_df['resultY'] = pd.Series([i.Centroid().GetY() for i in tree_wkts],index=tree_df.index)


new_df = pd.DataFrame()
new_df["ContourData"] = tree_df['Grids']
new_df["ContourDataOrigin"] = tree_df['Grids']
new_df['TreeId'] = [i for i in range(len(new_df["ContourData"]))]
new_df['TreeContoursPx'] = tree_df['Grids']

tree_lon_gps = tree_df['resultX']
tree_lat_gps = tree_df['resultY']
lonlat = [[x,y] for x,y in zip(tree_lon_gps,tree_lat_gps)]
NE = [[latLonToNE(georefData, float(x[0]), float(x[1])).northing, latLonToNE(georefData, float(x[0]), float(x[1])).easting] for x in lonlat]
Pixel = [NEToPixel(georefData,float(x[0]),float(x[1])) for x in NE]
new_df['TreeCentresLONGps'] = tree_lon_gps
new_df['TreeCentresLATGps'] = tree_lat_gps
new_df['TreeCentresGps'] = ["POINT ("+str(i)+" "+str(j)+")" for i,j in zip(tree_lon_gps,tree_lat_gps)]
new_df['TreeCentresLONPx'] = [x[0] for x in Pixel] #
new_df['TreeCentresLATPx'] = [x[1] for x in Pixel] #
bs_lon_x = [x[0] for x in Pixel]
bs_lat_x = [x[1] for x in Pixel]
new_df['TreeCentresPx'] = ["POINT ("+str(i)+" "+str(j)+")" for i,j in zip(bs_lon_x,bs_lat_x)]
new_df['NDVIScore'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['LowNDVIScore'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['YellowingScore'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['HighYellowingScore'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['CrownRadiusPx'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['CrownAreaPx'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['CrownRadiusM'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['CrownAreaM2'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['LowCrownArea'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['TerrainDTM'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['SurfaceDSM'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['TreeHeight'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['TreeDBH'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['TreeVolumeM3'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['PredictionOfTree'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['TreeFlag'] = ["TRUE" for i in range(len(new_df["ContourData"]))]
new_df['InsertedDate'] = [InsertedDate for i in range(len(new_df["ContourData"]))]
new_df['UAVCapturedDate'] = [UAVCapturedDate for i in range(len(new_df["ContourData"]))]
new_df['PetakId'] = [boundary_ for i in range(len(new_df["ContourData"]))]
new_df['PATType'] = [PATType for i in range(len(new_df["ContourData"]))]
new_df['MasterID'] = ["NULL" for i in range(len(new_df["ContourData"]))]
new_df['TileId'] = ["NULL" for i in range(len(new_df["ContourData"]))]


new_df = new_df[['UAVCapturedDate','PetakId','PATType','MasterID','TileId','TreeId','TreeContoursPx','TreeCentresPx','TreeCentresLONPx','TreeCentresLATPx','ContourData','ContourDataOrigin','TreeCentresGps','TreeCentresLONGps','TreeCentresLATGps','NDVIScore','LowNDVIScore','YellowingScore','HighYellowingScore','CrownRadiusPx','CrownAreaPx','CrownRadiusM','CrownAreaM2','LowCrownArea','TerrainDTM','SurfaceDSM','TreeHeight','TreeDBH','TreeVolumeM3','PredictionOfTree','TreeFlag','InsertedDate']]


                

new_df.to_csv(resultDir + "TREE"+"_"+petaks_name_+"_"+SpeciesName+"_"+PATType+"M"+"_contours_2.csv", sep="|", index=False)   
               

# gridCsv.write(s)

# s = ''
  
print('Elapsed time = {}'.format(time.time() - st))
#os.remove(output_dir+'/.*')

shutil.rmtree(output_dir)
#os.mkdir(output_dir)