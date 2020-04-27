import pandas as pd
import os
import cv2
import os
import sys
import numpy as np
import json
import cv2
from osgeo import ogr, osr, gdal
import cvGeoReference as georef
from common import cvlog, setCompartment, cv_header_log, is_date, perflog
import shapely
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)
from geopandas import GeoDataFrame
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import wkt
from shapely.geometry import Point
from fiona import crs
from pycrs.parser import from_epsg_code
from multiprocessing import Process, Queue, current_process, Pool
from matplotlib import pyplot as plt
#import cvImageLibrary as il
import multiprocessing as mp
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--validated_shp", default="", type=str, help="Path to validated shp file.", required=False)
parser.add_argument("-c", "--validated_csv", default="", type=str, help="Path to validated csv file.", required=False)
parser.add_argument("-g", "--grid_csv", default="", type=str, help="Path to grid csv file.", required=False)
parser.add_argument("-s", "--selected_shp", default="", type=str, help="Path to selected csv shp file.", required=True)
parser.add_argument("-i", "--input_img", default="", type=str, help="Path to input tif file.", required=False)
parser.add_argument("-j", "--input_jpg", default="", type=str, help="Path to input jpg file.", required=False)
parser.add_argument("-w", "--input_jgw", default="", type=str, help="Path to input jgw file.", required=False)
parser.add_argument("-p", "--input_prj", default="", type=str, help="Path to input prj file.", required=False)
parser.add_argument("-t", "--dirPath", default="", type=str, help="Path to outputdir for blindtest.", required=False)
parser.add_argument("-d", "--dirPath1", default="", type=str, help="Path to outputdir.", required=False)
parser.add_argument("-r", "--dirPath2", default="", type=str, help="Path to outputdir.", required=False)
args = parser.parse_args()

validated_shp = args.validated_shp
validated_csv = args.validated_csv
grid_csv = args.grid_csv
selected_shp = args.selected_shp
input_img = args.input_img
input_jpg = args.input_jpg 
input_jgw = args.input_jgw
input_prj = args.input_prj
dirPath = args.dirPath
dirPath1 = args.dirPath1
dirPath2 = args.dirPath2

#########################################################################################################################
## 1. Define helper functions ## 
#########################################################################################################################

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


class cvTree:
    pixelPos = None
    gpsPos = None
    radius = None
    radius_meters = None
    area = None
    area_m2s = None
    contour = None
    enclosingCircle = None
    meanHsv = None
    meanBgHsv = None
    contourIndex = None
    width = None
    height = None
    straightbb = None

    # Nisith ->added to capture the height and width of min bounded
    # rect while processing contours

    minRectW = None
    minRectH = None

    # Alan -> added to capture angle of rotated min bounding box
    minRectA_deg = None
    # Alan -> added to get straight bounding box parameter
    straightbb = None

    # L -> Added for PND
    ndvi_score = None
    isLow_NDVI = None
    yellowing_score = None
    isHigh_Yellowing = None
    isLow_Crown = None

    def __init__(self, x, y, contour, radius, area, radius_meters, area_m2s, gpsPos=None, width=None, height=None,
                 straightbb=None, minRectW=None, minRectH=None, minRectA_deg=None, meanHsv=None, meanBgHsv=None,
                 enclosingCircle=None, ndvi_score=None, isLow_NDVI=None, yellowing_score=None,
                 isHigh_Yellowing=None, isLow_Crown=None):
        self.pixelPos = (x, y)
        self.gpsPos = gpsPos
        self.radius = radius
        self.area = area
        self.radius_meters = radius_meters
        self.area_m2s = area_m2s
        self.straightbb = straightbb
        self.minRectW = minRectW
        self.minRectH = minRectH
        self.minRectA_deg = minRectA_deg
        self.width = width
        self.height = height
        self.contour = contour
        self.enclosingCircle = enclosingCircle
        self.meanHsv = meanHsv
        self.meanBgHsv = meanBgHsv
        ## P&D
        self.ndvi_score = ndvi_score
        self.isLow_NDVI = isLow_NDVI
        self.yellowing_score = yellowing_score
        self.isHigh_Yellowing = isHigh_Yellowing
        self.isLow_Crown = isLow_Crown

    def __str__(self):
        string = str(self.pixelPos[0]) + "," + str(self.pixelPos[1]) + "," + str(self.area_m2s)
        if self.gpsPos is not None:
            string += "," + str(self.gpsPos.lat) + "," + str(self.gpsPos.lon)
        else:
            string += " No GPS position"
        return string


class cvGeohashContent:
    trees = None
    hash = None
    processed = 0

    def __init__(self, hash=None):
        self.hash = hash
        self.trees = []

    def treeCount(self):
        return len(self.trees)


class cvTile:
    id = None
    topLeftGpsPos = None
    bottomRightGpsPos = None
    area = 0
    geohashContents = None
    row = 0
    column = 0

    def __init__(self, topLeftGpsPos, bottomRightGpsPos, geohashes, row, column, area=0):
        self.id = ""
        self.geohashContents = []
        for geohash in geohashes:
            self.id = self.id + geohash.hash
            self.geohashContents.append(geohash)
        if self.id == "":
            self.id = "NULL_TILE"
        self.topLeftGpsPos = topLeftGpsPos
        self.bottomRightGpsPos = bottomRightGpsPos
        self.area = area
        self.row = row
        self.column = column

    def treeCount(self):
        count = 0
        for geohash in self.geohashContents:
            count += geohash.treeCount()
        return count


# Added by Nisith
class cvTilePx:
    id = None
    topLeftPix = None
    bottomRightPix = None
    area = 0
    geohashContents = None
    row = 0
    column = 0

    def __init__(self, id, topLeftPix, bottomRightPix, geohashes, row, column, area=0):
        # self.id = ""
        self.geohashContents = [cvGeohashContent()]
        '''for geohash in geohashes:
            self.id = self.id + geohash.hash
            self.geohashContents.append(geohash)
        if self.id == "":
            self.id = "NULL_TILE"'''
        self.id = id
        self.topLeftPix = topLeftPix
        self.bottomRightPix = bottomRightPix
        self.area = area
        self.row = row
        self.column = column

    def treeCount(self):
        count = 0
        for geohash in self.geohashContents:
            count += geohash.treeCount()
        return count
        return len(self.geohashContents)


class cvDetectionResults:
    tiles = None
    blankAlerts = None
    geohashTileMapping = None
    contours = None
    # Added by Nisith
    tilesPx = None
    # Added by Nisith
    tileWidth = None
    tileHeight = None

    def __init__(self, tiles, geohashTileMapping, tileWidth=None, tileHeight=None):
        self.blankAlerts = []
        self.contours = []
        self.tiles = tiles
        self.geohashTileMapping = geohashTileMapping
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight

    def getTileByTileId(self, tileId):
        if self.tiles is None:
            return None
        return self.tiles.get(tileId)

    def getTileByGeohash(self, hash):
        if self.tiles is None or self.geohashTileMapping is None:
            return None
        t = self.geohashTileMapping.get(hash)
        if t is None:
            return None
        return self.getTileByTileId(t[0])

    def getTileIdByGeohash(self, hash):
        if self.tiles is None or self.geohashTileMapping is None:
            return None
        t = self.geohashTileMapping.get(hash)
        if t is None:
            return None
        return t[0]

    def addTree(self, tree):
        hash = encode.encode(tree.gpsPos.lat, tree.gpsPos.lon)
        t = self.geohashTileMapping.get(hash)
        try:
            self.tiles[t[0]].geohashContents[t[1]].trees.append(tree)
            self.contours.append(tree.contour)

            # Added by Nisith

            return self.tiles[t[0]].geohashContents[t[1]]
        except Exception:
            cvlog("hash: " + str(hash) + " " + str(tree.gpsPos.lat) + " " + str(tree.gpsPos.lon))
            cvlog("t: " + str(t))
            return None

    def hashTilePx(self, val, width):
        return str(int(val / width))

    def addTreePx(self, tree):
        treeHash = ""
        hashedTile = ""
        ##cvlog("tree: "+ str(tree.pixelPos[0])+','+str(tree.pixelPos[1]))
        treeHashW = self.hashTilePx(tree.pixelPos[0], self.tileWidth)
        treeHashH = self.hashTilePx(tree.pixelPos[1], self.tileHeight)
        ##cvlog("Adding tree with hash:"+treeHashW+"+"+treeHashH)
        treeHash = treeHashW + treeHashH
        # geohashContents[0] is used since it's a list containing one item
        ##cvlog("tiles len: "+str(len(self.tiles)))
        if treeHash in self.tiles.keys():
            self.tiles[treeHash].geohashContents[0].trees.append(tree)
        self.contours.append(tree.contour)

    def treeCount(self):
        count = 0
        for tileId, tile in self.tiles.iteritems():
            count += tile.treeCount()
        return count


class cvAlert:
    pixPos = None
    gpsPos = None
    type = None


class cvConfig:
    ### cvCompartmentProcessor.py arguements
    weedEnabled = 0
    treeEnabled = 0
    floodenabled = 0
    mode = ""
    displayEnabled = 0
    writeEnabled = 0
    tree_database = ""
    input_file = ""
    outputfile = ""
    output_dir = None
    fmis_file = ""
    shape_dir = None
    shapefile = ""
    dsmfile = ""  ####Abhishek
    dsm_dir = None  ####Abhishek
    ndvifile = ""  ####L
    ndvi_dir = None  ####L
    dtmfile = ""  ####Alan
    dtm_dir = None  ####Alan
    mulfile = ""  ####Alan
    mul_dir = None  ####Alan
    chmfile = ""  ####Alan
    chm_dir = None  ####Alan
    validationFile = ""
    valid_dir = ""
    debugprefix = ""
    prefix = ""
    compensate_shadow = 0
    weedCSV = ""
    weedSelection = ""
    ### UAV image filenames info
    exeid = ""
    compid = ""
    flight_id = ""
    flight_date = None
    flight_time = None
    plant_date = None
    insert_datetime = None
    ### Info from Database
    species = ""
    compage = ""
    landtype = ""
    plant_date = ""
    comp_district = ""
    comp_region = ""
    img_type = ""
    bri = 0.0
    iri = 0.0
    tot_ha = 0.0
    compensate_shadow = 0
    featid = ""
    scaling = 1
    log_file = ""
    skipFeatureExtraction = False
    skipMaskCreation = False
    start_time = None
    stocking = 0.0  # standard stocking

    def __str__(self):
        content = self.__dict__
        s = "Configuration: \n"
        for item, val in content.iteritems():
            if val is not None:
                s = s + "\t" + item + ": " + str(val) + "\n"
        return s

    def switchmode(self):

        # treeprefix = "TREE_" + prefix + str(config.species) + "_" + str(config.compage.split('m')[0]) + "M"
        # weedprefix = "WEED_" + prefix + str(config.species) + "_" + str(config.compage.split('m')[0]) + "M"
        # floodprefix = "FLOOD_" + prefix + str(config.species) + "_" + str(config.compage.split('m')[0]) + "M"
        # blanksprefix = "BLANKSPOT_" + prefix + str(config.species) + "_" + str(config.compage.split('m')[0]) + "M"

        if self.mode == "":
            self.debugprefix = self.output_dir + '/debug/' + self.prefix
        elif self.mode == "WEED":
            self.debugprefix = self.output_dir + '/debug/WEED/' + "WEED_" + self.prefix + str(self.species) + "_" + str(
                self.compage.split('m')[0]) + "M"
        elif self.mode == "TREE":
            self.debugprefix = self.output_dir + '/debug/TREE/' + "TREE_" + self.prefix + str(self.species) + "_" + str(
                self.compage.split('m')[0]) + "M"
        elif self.mode == "FLOOD":
            self.debugprefix = self.output_dir + '/debug/FLOOD/' + "FLOOD_" + self.prefix + str(
                self.species) + "_" + str(self.compage.split('m')[0]) + "M"
        elif self.mode == "BLANKSPOT":
            self.debugprefix = self.output_dir + '/debug/BLANKSPOT/' + "BLANKSPOT_" + self.prefix + str(
                self.species) + "_" + str(self.compage.split('m')[0]) + "M"


class cvImData:
    img = None
    dsm = None  ####Abhishek
    ndvi_gdal = None  ####L
    dtm = None  ####Abhishek
    chm = None  ####Abhishek
    config = None
    dist = 0.0
    metersPerPixel = 0
    shape = None
    scaling = float(1.0)
    roi = None
    features = None
    mask = None
    processed = None

    # weedCSV = None
    # weedSelection = None

    def __init__(self, config):
        self.config = config

    def showImage(self, name, img, w=1):
        if self.config.displayEnabled:
            cv2.imshow(name, img)
            return cv2.waitKey(w)

    def scale(self, scaling):
        if scaling == 1 or scaling == 0:
            return
        self.img = cv2.resize(self.img, (0, 0), fx=scaling, fy=scaling)
        self.hsv_img = cv2.resize(self.hsv_img, (0, 0), fx=scaling, fy=scaling)
        self.metersPerPixel /= scaling
        self.dist *= scaling
        self.scaling = self.scaling * scaling
        if self.roi is not None:
            self.roi = cv2.resize(self.roi, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)

    def writeDebug(self, filename, img):
        if self.config.writeEnabled == 1:
            cvlog("Writing " + self.config.debugprefix + "_" + filename)
            cv2.imwrite(self.config.debugprefix + "_" + filename, img)

    def width(self):
        if self.img is not None:
            return self.img.shape[1]
        else:
            return 0

    def height(self):
        if self.img is not None:
            return self.img.shape[0]
        else:
            return 0


class cvResult:
    a = None

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


class cvCompartmentProcessor:
    #
    # Internal parameters
    #
    dB = None
    config = None
    results = None
    errorLog = ""
    georefData = None
    im_data = None
    alerts = False
    mode = ""

    # Added by Nisith
    resultsPx = None

    # Added by L
    georefDataNDVI = None  ###L


# end of class
# end of class
# end of class

def Geometry2List(input):
    temp = input.ExportToJson()
    jsonPoints = json.loads(temp)
    points = jsonPoints["coordinates"]
    return points

def __init__(self, config,dB):
    self.filename = config.input_file
    self.config = config
    self.im_data = cvImData(config)
    self.dB = dB
    #self.dB.updateConfig(config) ##Saurabh
    self.alerts = False


def loadImageData(self):
    cvlog ("Loading image...")
    print("################### self.filename: %s " % (self.filename))
    self.im_data.img, self.config.scaling, errorLog = il.load(self.filename)
    #self.im_data.weedCSV = pd.read_csv(self.config.weedCSV) #Wallace
    #self.im_data.weedSelection = self.config.weedSelection #Wallace
    if self.im_data.img is not None:
        # compensate shadow + generate HSV
        # self.im_data.img = il.compensate_shadow(self.im_data) ### Saurabh added compensate shadow
        # self.im_data.writeDebug('ak_comshadow.jpg', self.im_data.img)
        self.im_data.hsv_img = cv2.cvtColor(self.im_data.img, cv2.COLOR_BGR2HSV)
    return errorLog


def initializeJgw(georefData, values, prj, width, height, scaling):
    for i in range(0,4):
        values[i] = float(values[i])/float(scaling)
    if prj is not None:
        prSplit = prj.split("PROJCS[\"UTM Zone ")
        ### to accomodated our prj format with " " as "_", use > 1
        if len(prSplit) > 1:
            correctEPSG = len(prSplit)-1
            format = prSplit[correctEPSG]
            prSplit2 = format.split(",")
            prSplit3 = prSplit2[0].split("N")
            prSplit4 = prSplit3[0].split("S")
            format = int(prSplit2[0])
            prSplit = prj.split(" Hemisphere")
            prSplit2 = prSplit[0].split(",")
            if(prSplit2[1] == " Northern"):
                format = format + 32600
            else:
                format = format + 32700
        else:
            prSplit = prj.split("PROJCS[\"UTM_Zone_")
            if len(prSplit) < 2:
                return "Error: Not supported projection!"
            #cvlog(str(prSplit[1]))
            correctEPSG = len(prSplit)-1
            prSplit2 = prSplit[correctEPSG]
            prSplit3 = prSplit2.split("_")
            format = int(prSplit3[0])
            if(prSplit3[1] == "Northern"):
                format = format + 32600
            else:
                format = format + 32700
    else:
        format = 32647 # Backup option, may not work correct!
    print (format)
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromEPSG(format)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    cvlog("Input EPSG format:"+str(format))

    georefData.coordTransNEToGPS = osr.CoordinateTransformation(spatialRef, target)
    georefData.coordTransGPSToNE = osr.CoordinateTransformation(target, spatialRef)

    values = np.array(map(float,values))
    print values
    if not values is None:
        # Add origin to the point list
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(values[4], values[5])
        WGS84 = ogr.Geometry(ogr.wkbMultiPoint)
        WGS84.AddGeometry(point)
        # Add Bottom-Right point to the point list
        x=0
        y=0
        print ("inside condition")
        if (values[2] == 0) and (values[1] == 0):
            x = values[4] + (width-1) * values[0] + (height-1) * values[1]
            y = values[5] + (width-1) * values[2] + (height-1) * values[3]
        else:
            return "Error: Affine transformation not implemented. "
        georefData.topLeftNE = cvNEPos(values[5], values[4])
        georefData.bottomRightNE = cvNEPos(y, x)
        point.AddPoint(x,y)
        WGS84.AddGeometry(point)
        # Make the transformation from input coordinate system to the output (WGS84)
        WGS84.Transform(georefData.coordTransNEToGPS)
        points = Geometry2List(WGS84)
        georefData.topLeftGps = cvGPSPos(points[0][1], points[0][0])
        georefData.bottomRightGps = cvGPSPos(points[1][1], points[1][0])
        cvlog("Top-Left- and Bottom-Right-coordinates for the image:"+str(georefData.topLeftGps)+", "+str(georefData.bottomRightGps))
        cvlog('georefData Saurabh image is '+ str(georefData))
        georefData.radPerPixX = (georefData.bottomRightGps.lon - georefData.topLeftGps.lon)/(width-1)
        georefData.radPerPixY = (georefData.bottomRightGps.lat - georefData.topLeftGps.lat)/(height-1)
        georefData.metersPerPixelX = values[0]
        georefData.metersPerPixelY = values[3]
        georefData.metersPerPixel = (abs(georefData.metersPerPixelX) + abs(georefData.metersPerPixelY))/2
        cvlog("Pixel size: "+str(georefData.metersPerPixel)+" meters / pixel")
    return ""

def loadGeorefData(self):
    cvlog ("Loading georef data...")
    if os.path.splitext(self.filename)[1]==".jpg":
        errorLog, self.georefData = georef.load(self.filename, self.im_data.width(), self.im_data.height(), self.config.scaling)
    else:
        errorLog, self.georefData = georef.load(self.filename)
    return errorLog

def load(filename, width=0, height=0, scaling=1):
    georefData = cvGeorefData()
    errorLog = ""
    if(os.path.splitext(filename)[1]==".jpg"):
        values = []
        try:
            with open(os.path.splitext(filename)[0]+'.jgw') as worldfile:
                for i in range(0,6):
                    line = worldfile.readline()
                    values.append(line.split('\n')[0])
                prj = None
                with open(os.path.splitext(filename)[0]+'.prj') as prjfile:
                    prj = prjfile.readline()
        except Exception:
            errorLog = "Error: prj or jgw file not found!"
            return errorLog, None
        errorLog = initializeJgw(georefData, values, prj, width, height, scaling)
    else:
        errorLog = "Error: Image format not supported!"
    return errorLog, georefData

def pixelToLatLon(georefData, x, y):
    # X-axis assumed to point East and Y-axis to South
    lon = georefData.topLeftGps.lon + x * georefData.radPerPixX
    lat = georefData.topLeftGps.lat + y * georefData.radPerPixY
    return cvGPSPos(lat,lon)

def latLonToPixel(georefData, lon, lat):
    # X-axis assumed to point East and Y-axis to South
    x = (lon - georefData.topLeftGps.lon) / georefData.radPerPixX
    y = (lat - georefData.topLeftGps.lat) / georefData.radPerPixY
    return (int(x),int(y))

def NEToPixel(georefData, easting, northing):
    x = (easting - georefData.topLeftNE.easting) / georefData.metersPerPixelX
    y = (northing - georefData.topLeftNE.northing) / georefData.metersPerPixelY
    return (int(x), int(y))

def pixelToNE(georefData, x, y):
    easting = georefData.topLeftNE.easting + x * georefData.metersPerPixelX
    northing = georefData.topLeftNE.easting + x * georefData.metersPerPixelY
    return cvNEPos(northing,easting)

def NEToLatLon(georefData, easting, northing):
    nortEast = ogr.Geometry(ogr.wkbPoint)
    nortEast.AddPoint(easting, northing)
    # Make the transformation from input coordinate system to the output (WGS84)
    nortEast.Transform(georefData.coordTransNEToGPS)
    latLon = Geometry2List(nortEast)
    return cvGPSPos(latLon[1], latLon[0])

def latLonToNE(georefData, lat, lon):
    latLon = ogr.Geometry(ogr.wkbPoint)
    latLon.AddPoint(lat, lon)
    # Make the transformation from input coordinate system to the output (WGS84)
    latLon.Transform(georefData.coordTransGPSToNE)
    nortEast = Geometry2List(latLon)
    return cvNEPos(nortEast[0], nortEast[1])
	
	#jadi ltlon, NE, pixel. Ini 3 komponen penting untuk converting imagenya. urutannya 1,2,3 atau sebaliknya. 

def hashTreePx(val,width):
       return str(int(val/width))

def testConversions(georefData, x, y):
    lat,lon = pixelToLatLon(georefData, x, y)
    cvlog("lat,lon : "+str(lat)+","+str(lon))
    x,y = latLonToPixel(georefData, lat, lon)
    cvlog("x,y: "+str(x)+","+str(y))

    e,n = pixelToNE(georefData, x, y)
    cvlog("e,n: "+str(e)+","+str(n))
    x,y = NEToPixel(georefData, e, n)
    cvlog("x,y: "+str(x)+","+str(y))

    lat2,lon2 = NEToLatLon(georefData, e, n)
    cvlog("lat,lon : "+str(lat2)+","+str(lon2))
    e2,n2 = latLonToNE(georefData, lat, lon)
    cvlog("e,n: "+str(e2)+","+str(n2))

    lat2,lon2 = NEToLatLon(georefData, 800000, 25000)
    e2,n2 = latLonToNE(georefData, lat2, lon2)
    cvlog("e,n: "+str(e2)+","+str(n2))
    lat2,lon2 = NEToLatLon(georefData, e2, n2)
    e2,n2 = latLonToNE(georefData, lat2, lon2)
    cvlog("e2,n2: "+str(e2)+","+str(n2))

def gpsToGeohash(gpsPos):
    return geohash.encode(gpsPos.lat, gpsPos.lon,9)

#Cut Square?	
# Create 1M GRIDs
def createTilesPx(georefData, pixWidth, pixHeight, im_data):
    print("Creating Tiles in Px")
    print("pixWidth = " + str(pixWidth))
    print("pixHeight = " + str(pixHeight))
    hashedTiles = {}
    key = ""
    limit = 75  # sck: 3m = 75 sck : original = 100, since 0.04m per pixel, edit = 25
	#3 meter/ resolusi.
    tileCountX = int(pixWidth / limit)
    cvlog("tileCountX = " + str(tileCountX))  # number of tiles in X direction
    tileCountY = int(pixHeight / limit)
    cvlog("tileCountY = " + str(tileCountY))  # number of tiles in Y direction
    widthPx = []
    heightPx = []
    widthTile = limit
    heightTile = limit
    cvlog("widthTile = " + str(widthTile))  # Tile Width
    cvlog("heightTile = " + str(heightTile))  # Tile Height
    for i in range(tileCountX + 1):
        if pixWidth - (i * widthTile) < 10:
            continue
        widthPx.append(i * widthTile)
    # cvlog("widthPx = "+str(widthPx))
    cvlog("widthPxLen = " + str(len(widthPx)))
    for i in range(tileCountY + 1):
        if pixHeight - (i * heightTile) < 10:
            continue
        heightPx.append(i * heightTile)
        # cvlog("heightPx = "+str(heightPx))
    cvlog("heightPxLen = " + str(len(heightPx)))
    tileWidth = widthTile
    tileHeight = heightTile

    # widthPx = np.linspace(0, pixWidth, sampleCountX,dtype=int)
    # widthPx = [int(i) for i in width]
    # print "lons: "+str(lons)
    # heightPx = np.linspace(0, pixHeight, sampleCountY,dtype=int)
    # heightPx = [int(i) for i in height]
    # print "lats: "+str(lats)

    '''tileWidth = math.floor(pixWidth / (sampleCountX-1))
    cvlog("tileWidth"+str(tileWidth))
    tileHeight = math.floor(pixHeight / (sampleCountY-1))
    cvlog("tileHeight"+str(tileHeight))'''
    # hCount = 0
    vis_tiles = im_data.copy()
    plt.imshow(vis_tiles)

    # petak_name = im_data.config.input_file.split('.')[0]
    petak_name = petak
    # outDir = im_data.config.output_dir+'/'+petak_name

    gridCsv = open(petak_name + "_grid_1M.csv", 'w')
    s = 'Grids\n'
    gridCsv.write(s)

    for i in range(len(widthPx)):
        # wCount = 0
        # newline = 1
        # Added by Nisith for visualizing grid
        s = ''

        for j in range(len(heightPx)):
            px1 = (int(widthPx[i]), int(heightPx[j]))
            # cvlog("px1: "+str(px1))
            # print type(lons[i])
            if i == len(widthPx) - 1 and j == len(heightPx) - 1:
                px2 = (pixWidth, pixHeight)
                # cvlog("Bottom-Right Tile")
                # cvlog(str(px1))
                # cvlog(str(px2))
                if px1[1] == px2[1]:
                    continue
            elif i == len(widthPx) - 1 and j != len(heightPx) - 1:
                px2 = (pixWidth, int(heightPx[j + 1]))
                # cvlog("Righmost Tile along row "+str( j))
                # cvlog(str(px1))
                # cvlog(str(px2))
                if px1[0] == px2[0]:
                    continue

            elif i != len(widthPx) - 1 and j == len(heightPx) - 1:
                px2 = (int(widthPx[i + 1]), pixHeight)
                # cvlog("Bottomost Tile along column "+str(i))
                # cvlog(str(px1))
                # cvlog(str(px2 ))
                if px1[1] == px2[1]:
                    continue
            else:
                px2 = (int(widthPx[i + 1]), int(heightPx[j + 1]))
                # cvlog("In between tiles")
                # cvlog(str(px1))
                # cvlog(str(px2))
            # try:
            # cvlog("px2: "+str(px2))
            # cvlog( "len(hashedTiles): "+ str(len(hashedTiles)))

            tilePx = cvTilePx(len(hashedTiles) + 1, px1, px2, [], i + 1, j + 1)
            # except Exception:
            # cvlog("Tile not created")
            whiteArea = np.array([255, 255, 255])
            # print "New"
            # print px1[0]+1,px1[1]+1,im_data.img[px1[0]+1,px1[1]+1]
            # print px1[0]+1,px2[1]-1,im_data.img[px1[0]+1,px2[1]-1]
            # print px2[0]-1,px1[1]+1,im_data.img[px2[0]-1,px1[1]+1]
            # print px2[0]-1,px2[1]-1,im_data.img[px2[0]-1,px2[1]-1]
            # print px1[0]+1,px2[1]-1,im_data.img[int((px2[0]-px1[0])/2), int((px2[1]-px1[1])/2)]
            # print int((px2[0]-px1[0])/2),int((px1[1]-px2[1])/2),im_data.img[int((px2[0]-px1[0])/2), int((px1[1]-px2[1])/2)]
            # if (im_data.img[px1[0]+1,px1[1]+1]==whiteArea).all() and (im_data.img[px2[0]-1,px1[1]+1]==whiteArea).all() and (im_data.img[px2[0]-1,px2[1]-1]==whiteArea).all() and (im_data.img[px1[0]+1,px2[1]-1]==whiteArea).all() and (im_data.img[int((px2[0]-px1[0])/2), int((px2[1]-px1[1])/2)]).all() :

            tileNotValid = 0
            if (im_data[px1[1] + 1, px1[0] + 1] == whiteArea).all():
                tileNotValid += 1
            if (im_data[px1[1] + 1, px2[0] - 1] == whiteArea).all():
                tileNotValid += 1
            if (im_data[px2[1] - 1, px2[0] - 1] == whiteArea).all():
                tileNotValid += 1
            if (im_data[px2[1] - 1, px1[0] + 1] == whiteArea).all():
                tileNotValid += 1
            if (im_data[int((px2[1] - px1[1]) / 2), int((px2[0] - px1[0]) / 2)] == whiteArea).all():
                tileNotValid += 1
            # cvlog("tileNotValidCount="+str(tileNotValid))
            if tileNotValid > 1:
                continue

            topLeftGpsPos = pixelToLatLon(georefData, px1[0], px1[1])

            topRightGpsPos = pixelToLatLon(georefData, px2[0], px1[1])

            bottomRightPos = pixelToLatLon(georefData, px2[0], px2[1])

            bottomLeftPos = pixelToLatLon(georefData, px1[0], px2[1])

            # s= str(px1GPS.lon)+'|'+ str(px1GPS.lat)+'|'+ str(px2GPS.lon)+'|'+ str(px2GPS.lat)+'\n'

            s = s + 'POLYGON (('
            s = s + str(topLeftGpsPos.lon) + ' ' + str(topLeftGpsPos.lat) + ','
            s = s + str(topRightGpsPos.lon) + ' ' + str(topRightGpsPos.lat) + ','
            s = s + str(bottomRightPos.lon) + ' ' + str(bottomRightPos.lat) + ','
            s = s + str(bottomLeftPos.lon) + ' ' + str(bottomLeftPos.lat) + ','
            s = s + str(topLeftGpsPos.lon) + ' ' + str(topLeftGpsPos.lat)
            s = s + '))' + '\n'

            gridCsv.write(s)
            key = hashTreePx(widthPx[i], tileWidth) + hashTreePx(heightPx[j], tileHeight)
            # cvlog("key:" + hashTreePx(widthPx[i] ,tileWidth)+ "+" + hashTreePx(heightPx[j] ,tileHeight) )
            hashedTiles[key] = tilePx
            # cvlog( "key: "+ key)
            cv2.rectangle(vis_tiles, tilePx.topLeftPix, tilePx.bottomRightPix, (0, 0, 255), 2)

    gridCsv.close()
    # im_data.writeDebug('tilesHashed.jpg', vis_tiles)
    return hashedTiles, tileWidth, tileHeight


def save_grid_shape(im_data):
    petak_name = im_data.config.debugprefix
    df = pd.read_csv(petak_name + "_grid.csv", sep="|")
    df["geometry"] = df.Grids.apply(lambda x: shapely.wkt.loads(x))
    crs_proj4 = from_epsg_code(4326).to_proj4()
    gdf = gpd.GeoDataFrame(df, crs=crs_proj4, geometry=df["geometry"])
    gdf.to_file(petak_name + '_grid.shp', driver='ESRI Shapefile')
    return ""
	
def read_weed_shpdata(shp_filename): 
    global weedshp
    global weedshpdf
    shpfile_gdf = gpd.GeoDataFrame.from_file(shp_filename)
    weedshpdf = pd.DataFrame(shpfile_gdf)
    weedshp = weedshpdf['geometry'].astype(str)
    print(weedshp[0:30])

def polygon_area(gridarea):
    polyweed = []
    
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    target = osr.SpatialReference()
    target.ImportFromEPSG(32647)

    transform = osr.CoordinateTransformation(source, target)

    for j in range(len(gridarea)):
        if gridarea[j] is None:
            continue
        else:
            pw = ogr.CreateGeometryFromWkt(gridarea[j])
            pw.Transform(transform)
            wktweed2 = pw.ExportToWkt()
            polyw = ogr.CreateGeometryFromWkt(wktweed2)

            polyw2 = polyw.GetArea()
            polyweed.append(polyw2)
    return polyweed

def read_selected_shpdata(shp_filename):
    global selshp
    global selshpdf 
    shpfile_gdf = gpd.GeoDataFrame.from_file(shp_filename)
    selshpdf = pd.DataFrame(shpfile_gdf)
    selshp = selshpdf['geometry'].astype(str)
    
def read_image(inputimage):
    global img 
    img = cv2.imread(input_img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = img.copy()
    
def imagefiletype(filename,petak_spesific_outdir):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = img.copy()
    if input_img.split('.')[-1] == "tif":
        rgb_gdal = gdal.Open(input_img)
        prj = rgb_gdal.GetProjection()
        jgw = rgb_gdal.GetGeoTransform()    
        isTif = True
        del rgb_gdal
        split = filename.split('/')
        filename = split[-1]
        filename = filename.split('.')[0]
    else: #rgb image
        split = filename.split('/')
        filename = split[-1]
        filepath = input_img[0:len(img_filename)-len(filename)]
        filename = filename.split('.')[0]
        prj_file = open(filepath+filename+".prj")
        jgw_file = open(filepath+filename+".jgw")
        prj = str(prj_file.readlines()[0])
        jgw = jgw_file.readlines()
        isTif = False
    print (filename)
    
def string_to_list_latlon_weed(contour): #make list of list of the coordinate
    start_s2 = "MULTIPOLYGON ((("
    start_s = "POLYGON (("
    end_s2 = ")))"
    mid_s1 = "(("
    end_s = "))"
    split_per_coordinate = contour.replace(start_s2, '').replace(end_s2, '').replace(start_s, '').replace(end_s, '').replace(mid_s1, '').split('), (')
    contour_list = []
    tree_list = []
    if len(split_per_coordinate) > 1:
        weed_list = split_per_coordinate[0]
        tree_list = split_per_coordinate[1: len(split_per_coordinate)-1]
    else:
        weed_list = split_per_coordinate[0]
    split_per_coordinate_weed = weed_list.split(',')
    for coordinate_tuple in split_per_coordinate_weed:
        contour_list.append(coordinate_tuple.split(' '))
    return contour_list

def string_to_list_latlon_tree(contour): #make list of list of the coordinate
    start_s = "POLYGON (("
    end_s = "))"
    split_per_coordinate = contour.replace(start_s, '').replace(end_s, '').split('), (')
    contour_list = []
    tree_contour_list = []
    if len(split_per_coordinate) > 1:
        weed_list = split_per_coordinate[0]
        tree_list = split_per_coordinate[1: len(split_per_coordinate)-1]
    else:
        weed_list = split_per_coordinate[0]
        tree_list = []
    if len(tree_list) > 0:
        for tree in tree_list:
            split_per_coordinate_tree = tree.split(',')
            for coordinate_tuple in split_per_coordinate_tree:
                tree_contour_list.append(coordinate_tuple.split(' '))
    return tree_contour_list 

def string_to_list_latlon(contour): #make list of list of the coordinate
    start_s = "POLYGON (("
    end_s = "))"
    mid_s =")"
    mid_s2 = "("
    split_per_coordinate = contour.replace(start_s, '').replace(end_s, '').replace(mid_s, '').replace(mid_s2, '').split(',')
    contour_list = []
    for coordinate_tuple in split_per_coordinate:
        contour_list.append(coordinate_tuple.split(' '))
    return contour_list

#convert lat-lon contour list to pixel X-Y contour list
# input: contour list in lat-lon (list-of-list format)
# return: list of list of the vertices in pixel X-Y format
def list_latlon_to_list_pixelXY(contour_list_latlon,isTif,prj,jgw): 
    xy_contour_list = []
    for contour in contour_list_latlon:
        if len(contour)>2:
            longitude = 2
            latitude = 1
        else:
            longitude = 1
            latitude = 0
        xy_contour_list.append(convert_latlon_to_pixelXY(float(contour[longitude]),float(contour[latitude]),isTif,prj,jgw))
    return xy_contour_list

def list_latlon_to_list_pixelXY_tree(contour_list_latlon,isTif,prj,jgw): 
    xy_contour_list = []
    for contour in contour_list_latlon:
        b = convert_latlon_to_pixelXY(float(contour[1]),float(contour[0]),isTif,prj,jgw)
        xy_contour_list.append(b)
    return xy_contour_list

def convert_latlon_to_pixelXY(lat,lon,isTif,prj="",jgw=[]):
    dst = osr.SpatialReference(prj)
    transform = jgw
    if isTif:
        x_NorthWest = transform[0]
        y_NorthWest = transform[3]
        meter_perPixel_X = transform[1]
        meter_perPixel_Y = transform[5]
    else:
        x_NorthWest = float(transform[4])
        y_NorthWest = float(transform[5])
        meter_perPixel_X = float(transform[0])
        meter_perPixel_Y = float(transform[3])
        
    src = osr.SpatialReference()
    src.SetWellKnownGeogCS('WGS84')
    ct = osr.CoordinateTransformation(src,dst)
    xy = ct.TransformPoint(lon, lat)
    
    #print ("x = {0} , y = {1}".format(xy[0],xy[1]))
    x = np.int32(((xy[0] - x_NorthWest) / meter_perPixel_X))
    y = np.int32(((xy[1] - y_NorthWest) / meter_perPixel_Y))
    return [x,y]

#########################################################################################################################

## 0. Set Directory 

os.chdir(os.getcwd())
print(os.getcwd())
 
#1. Read Img 
read_image(input_img) #ada function-nya

#2. Read Weed Data
read_weed_shpdata(validated_shp) #ngeluarin hasil berupa polygon

#3. Read Selected Shp 
read_selected_shpdata(selected_shp) #ada di s input berupa data.shp

#4. Set petak name 
filename_parts = input_img.split('_')  #kenapa ini cuman RI doang ya????? Karena yang ada cuman Riau doang.
for i in range(len(filename_parts)):
    if filename_parts[i]=="RI":
        petak = filename_parts[i] + "_" + filename_parts[i+1] + "_" + filename_parts[i+2]
print ("Petak: "+petak)

#imagefiletype(input_img,dirPath)
    
im_data = img
config = im_data	

load(input_img)
georefData = cvGeorefData() #ada kelasnya dan didalam function
print ("georefData" + str(georefData))
width = img.shape[1]
print ("Width:" + str(width))
height = img.shape[0]
print ("Height:" + str(height))

values = []
scaling = 1
with open(input_jgw) as worldfile:
    for i in range(0, 6):
        line = worldfile.readline()
        values.append(line.split('\n')[0])
        prj = None
with open(input_prj) as prjfile:
    prj = prjfile.readline()
with open (input_jgw) as worldfile:
    jgw = worldfile.readlines()        

initializeJgw(georefData, values, prj, width, height, scaling)

createTilesPx(georefData, width, height, im_data) #hasil ini cek di csv (output)
print ("creating grid")
grid_csv = petak + "_grid_1M.csv"
grid_data = pd.read_csv(grid_csv, sep ="|")
print ("no of ori grids:" +str(len(grid_data)))
grid_data = grid_data.drop_duplicates()
print ("no of actual grids:" +str(len(grid_data))) 
#grid_data.to_csv(grid_csv, sep ="|")

#5. Create grid 

#if grid_csv is None:
#    print ("creating grid")
#    createTilesPx(georefData, width, height, im_data)
#    grid_csv = petak + "_grid_1M.csv"
#    grid_data = pd.read_csv(grid_csv, sep ="|")
#    grid_data = grid_data.drop_duplicates()
#    grid_data.to_csv(gridcsv, sep ="|")
#else:
#    grid_data = pd.read_csv(grid_csv, sep ="|")


#6. Process selected shp file 

#selected_list = []
#for i in range (len(selshp)):
#    start_s = "POLYGON (("
#    end_s = "))"
#    items = selshp[i].replace(start_s,'').replace(end_s,'').split(',')
#    selected_list.append(items)

selected_trf = []
for i in selshp:
    item = i[10: len(i)-2]
    mid_s = ")"
    mid_s2 = "("
    items = item.replace(mid_s, '').replace(mid_s2, '').split(',')
    tmp = []
    for j in items:
        b = j.split(' ')
        tmp.append([float(item) for item in b if item != ''])
    selected_trf.append(tmp)

print ("type of selected shp input (can be NE/latlon):" +str(selected_trf[0]))

selected_trf_new = []
for i in range(len(selected_trf)):
    a = np.array(selected_trf[i])
    selected_trf_new.append(a)

selected_trf_actual = []
for j in range (len(selected_trf_new)):
    tmp = []
    for item in selected_trf_new[j]:
        items = latLonToPixel(georefData, item[0], item[1]) # USE FOR LATLON
        #items = NEToPixel(georefData, item[0], item[1]) # USE FOR NE
        items = [[np.int32(items[0]),np.int32(items[1])]]
        tmp.append(items)
    selected_trf_actual.append(np.array(tmp))

del selected_trf, selected_trf_new

#7a. Process weed shp 
print ("total no of weed contours:"+str(len(weedshp)))
print ("type of weed contours:"+str(weedshp[0]))

############# Added by Julian #############
#7a.1 - remove small contours
weedshparea = polygon_area(weedshp)
print("### Weed Shape Area ###", weedshparea[0:3])
print("### Length Weed Shape Area ###:"+str(len(weedshparea)))

ind = np.where(np.array(weedshparea) > np.float(0.5)) #banyaknya label weed dalam gambar.
print("List of index: ",ind)

weedshp = weedshp.iloc[ind]
print("### total no of weed contours after removing small contours: ###"+ str(len(weedshp)))
#print ("type of weed contours:"+str(weedshp))

### Check to ensure small contours are removed
# test = []
# for i in range (len(weedshp)):
#     if weedshparea[i] > np.float(0.5):
#         test.append(weedshp[i])
# print("### total no of weed contours after removing small contours: ###"+ str(len(test)))

# df = pd.DataFrame(np.array(test))
# df.to_csv("C:/Users/AIIC_02/Desktop/Julian/EPEL6M/test.csv", sep='|')

############################################

weed_trf_actual = []
for i in weedshp: #range (len(weedshp)):
    #print("This is i:", i, " Type: ", type(weedshp[i]))
    contour = i #weedshp[i]
    x = list_latlon_to_list_pixelXY(string_to_list_latlon_weed(contour), False, prj, jgw)
    weed_trf_actual.append(np.array(x))
        
#7b. Process trees within weed shp 
tmp = []
for i in weedshp: #in range (len(weedshp)):
    contour = i #weedshp[i]
    x = list_latlon_to_list_pixelXY(string_to_list_latlon_weed(contour), False, prj, jgw)
    tmp.append(np.array(x))

tmp2 = []
for i in weedshp: #range (len(weedshp)):
    contour = i
    x = string_to_list_latlon_tree(contour)
    if len(x) >0:
        tmp2.append(x)

tmp3 = []
for i in range(len(tmp2)):
    tmp4 = []
    for item in tmp2[i]:
        w = filter(None, item)
        tmp4.append(w)
    tmp3.append(tmp4)
        
tree_trf_actual =[]
for item in tmp3:
    x = list_latlon_to_list_pixelXY(item, False, prj, jgw)
    tree_trf_actual.append(np.array(x))     

#8. Plot weed msg 
weed_img = np.zeros((img.shape[0], img.shape[1]))
print ("oriweedimg:" +str(np.sum(weed_img)))
for i in range(len(weed_trf_actual)):
    cv2.fillPoly(weed_img, pts=[weed_trf_actual[i]], color=255)
print ("actualweedimg:" +str(np.sum(weed_img)))
for i in range(len(tree_trf_actual)):
    cv2.fillPoly(weed_img, pts = [tree_trf_actual[i]], color = 0)
print ("weedimgwtree:" +str(np.sum(weed_img)))
cv2.imwrite(petak + "_weedmask.jpg", weed_img)

#9. Process Grid Data 

gridwkt = grid_data['Grids']

print ("totalnogrids:"+str(len(gridwkt)))

grid_trf_actual = []
for i in range (len(gridwkt)):
    contour = gridwkt.iloc[i]
    x = list_latlon_to_list_pixelXY(string_to_list_latlon(contour), False, prj, jgw)
    grid_trf_actual.append(np.array(x))

grid_XY_list = []
for i in range(len(grid_trf_actual)):
    start_X = grid_trf_actual[i][0][0]
    end_X = grid_trf_actual[i][1][0]
    start_Y = grid_trf_actual[i][0][1]
    end_Y = grid_trf_actual[i][2][1]
    XY = [np.int32(start_Y), np.int32(end_Y), np.int32(start_X), np.int32(end_X)]
    grid_XY_list.append(XY)

#grid_XY_set = set(tuple (x) for x in grid_XY)
#grid_XY_list = [list(x) for x in grid_XY_set]


#10. Plot selected areas
#Masking
img_select_mask = np.zeros((img.shape[0], img.shape[1]))
for i in range(len(selected_trf_actual)):
    cv2.fillPoly(img_select_mask, pts=[selected_trf_actual[i]], color=255)


#10A.  Set Threshold 
cropped_img_weed_eg = weed_img[grid_XY_list[1][0]:grid_XY_list[1][1],grid_XY_list[1][2]:grid_XY_list[1][3]]
totalpixels = cropped_img_weed_eg.shape[0]*cropped_img_weed_eg.shape[1]
print ("totalnopixels:" +str(totalpixels))


#11. Filter grid only in selected areas

grid_actual = []
for i in range(len(grid_XY_list)):
    a = img_select_mask[grid_XY_list[i][0]:grid_XY_list[i][1], grid_XY_list[i][2]:grid_XY_list[i][3]]
    a_sum = np.sum(a == 255)
    #print ("a_sum:" +str(a_sum))
    if (a_sum/float(totalpixels) >= 0.9) == True:
        grid_actual.append(grid_XY_list[i])
        
grid_trf_actual_sel =[]
for i in range(len(grid_trf_actual)):
    b = img_select_mask[grid_trf_actual[i][0][1]:grid_trf_actual[i][2][1], grid_trf_actual[i][0][0]:grid_trf_actual[i][1][0]]
    b_sum = np.sum(b == 255)
    #print ("b_sum:" +str(b_sum))
    if (b_sum/float(totalpixels) >= 0.9) == True:
        grid_trf_actual_sel.append(grid_trf_actual[i])

#12. Aggregate no of grids weed and notweed 
percent_weed_th = 0.3 #Masing2 grid kalo min 0.3 masuk ke weed, 0,1 - 0,3 ke tree weed, sisanya ke not weed
gridweed = []
gridnotweed = []
for i in range(len(grid_actual)):
    tmp = weed_img[grid_actual[i][0]:grid_actual[i][1], grid_actual[i][2]:grid_actual[i][3]]
    n_white_pix = np.sum(tmp ==255)
    if (n_white_pix / float(totalpixels) >= percent_weed_th) == True:
        gridweed.append(i)
    else:
        gridnotweed.append(i)
        
print ("totalgridsinselectedarea:"+str(len(grid_actual)))
print ("sample_totalgrids:"+str(grid_actual[0]))
print ("totalgridswktinselectedarea:"+str(len(grid_trf_actual_sel)))
print ("sample_gridwkt:"+str(grid_trf_actual_sel[0]))
print ("noofgridsweed:"+str(len(gridweed)))
print ("noofgridsnotweed:"+str(len(gridnotweed)))

dirPath1 = dirPath+ petak+"/"+"weed"
dirPath2 = dirPath+ petak+"/"+"notweed"
dirPath3 = dirPath+ petak+"/"+"treeweed"
if not os.path.exists(dirPath1):
    os.makedirs(dirPath1)
if not os.path.exists(dirPath2):
    os.makedirs(dirPath2)
if not os.path.exists(dirPath3):
    os.makedirs(dirPath3)

#13. Cut squares 
for i in range(len(grid_actual)):
    tmp = weed_img[grid_actual[i][0]:grid_actual[i][1], grid_actual[i][2]:grid_actual[i][3]]
    n_white_pix = np.sum(tmp == 255)
    percent_weed = n_white_pix / float(totalpixels) 
    if (percent_weed >= percent_weed_th) == True:
        cropped_img_weed = img[grid_actual[i][0]:grid_actual[i][1], grid_actual[i][2]:grid_actual[i][3]]
        cv2.imwrite(dirPath1+"/"+ petak+"_"+str(i)+ "_" +str(grid_actual[i][0])+","+str(grid_actual[i][1])+","+str(grid_actual[i][2]) +"," + str(grid_actual[i][3])+"_weed.jpg", cropped_img_weed)
    elif(percent_weed > 0.1 and percent_weed < percent_weed_th):
       cropped_img_weed = img[grid_actual[i][0]:grid_actual[i][1], grid_actual[i][2]:grid_actual[i][3]]
       cv2.imwrite(dirPath3+"/"+ petak+"_"+str(i)+ "_" +str(grid_actual[i][0])+","+str(grid_actual[i][1])+","+str(grid_actual[i][2]) +"," + str(grid_actual[i][3])+"_treeweed.jpg", cropped_img_weed)
    else:
        cropped_img_notweed = img[grid_actual[i][0]:grid_actual[i][1], grid_actual[i][2]:grid_actual[i][3]]
        cv2.imwrite(dirPath2+"/"+ petak+"_"+str(i)+"_"+ str(grid_actual[i][0])+"," + str(grid_actual[i][1])+"," +str(grid_actual[i][2])+","+str(grid_actual[i][3])+"_notweed.jpg", cropped_img_notweed)
    

