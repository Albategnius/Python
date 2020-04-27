# -*- coding: utf-8 -*-
"""
"""

from osgeo import gdal, osr
import time
import os
import os.path
import sys
import glob
# import sys
import platform

# append current abspath of geotiff2jpg_jgw_prj.py
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="Input Dir containing geotiff images for conversion to jpg.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output Dir of converted jpg, prj, jgw")
    parser.add_argument('-s', "--scale", action='store_true', help="True to resize jpg.")
    return parser


def create_jpg_prj_jgw(filename, output_dir, scaling_bool):
    ## for Linux use this: split_result = filename.split('/')
    ## for Windows use this: split_result = filename.split('\\')
    print ("Filename: {0}".format(filename))
    if platform.system() == "Linux":
        print (platform.system())
        split_result = filename.split('/')
    elif platform.system() == "Windows":
        print (platform.system())
        split_result = filename.split('\\')
    ## Create output directory if not exists
    print("Output Directory")
    print(os.path.dirname(output_dir))
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir),0777)
    ## Debugging
    print (split_result)
    fn = split_result[len(split_result) - 1]
    output_fn = fn.split('.')[0]
    print ('Converting: ' + output_fn)

    src_img = gdal.Open(filename)

    ## Generate .prj file

    spatialref = src_img.GetProjection()
    src_spatialref = osr.SpatialReference()
    ## Import projection from spatial reference of geotiff image
    src_spatialref.ImportFromWkt(spatialref)
    src_spatialref.MorphToESRI()
    prj_text = src_spatialref.ExportToWkt()
    ## Convert to ComputerVision code required format
    if prj_text.split(',')[0] == 'PROJCS[\"WGS_1984_UTM_Zone_47N\"':
        prj_text = prj_text.replace('PROJCS[\"WGS_1984_UTM_Zone_47N\"', 'PROJCS[\"UTM_Zone_47_Northern_Hemisphere\"')
    elif prj_text.split(',')[0] == 'PROJCS[\"WGS_1984_UTM_Zone_48S\"':
        prj_text = prj_text.replace('PROJCS[\"WGS_1984_UTM_Zone_48S\"', 'PROJCS[\"UTM_Zone_48_Southern_Hemisphere\"')
    else:
        print("New type of prj detected, please check PROJCS.")
        print("Please check for new type prj and add check condition in geotiff2jpg_jgw_prj.py")

    prj_file = open(output_dir + output_fn + ".prj", 'w+')
    prj_file.write(prj_text)
    prj_file.close()

    ## Generate .jpg file

    band_no = src_img.RasterCount
    print ('Number of Raster Band in GeoTiff Image = %d' % band_no)
    ## Check for number of bands in geotiff. RGB = 3, RGBA = 4, MUL = 1. 
    ## A = Alpha, usually mask band
    ## Jpeg quality > 75
    ## equivalent in command line: gdal_translate -of JPEG -B 1 -B 2 -B 3 -co worldfile=yes -co quality=75 inputfile.tif outputfile.jpg
    ##
    ## 2 files produced for gdal.Translate:
    ## filename.aux.xml --- contains prj + jgw info
    ## filename.wld --- wld file (worldfile) == jgw file
    ##
    src_img_height = src_img.RasterXSize
    src_img_width = src_img.RasterYSize
    scale_h, scale_w = 0, 0
    image_size = float(src_img_height * src_img_width * 3.0)
    #scaling_bool = False
    if image_size >= 2**31 and scaling_bool:
        print("Image Size larger than 2**31, resize is done.", "Image size is : ", image_size)
        scale_h = scale_w = float(2.0**31)/image_size*100.0        # 0-100%
        print("Scale for resize : %f , new_h : %f , new_w : %f " % (scale_h, scale_h * src_img_height, scale_w * src_img_width))
        # scaled_h, scaled_w = int(src_img_height * scale), int(src_img_width * scale)

    if band_no == 3 or band_no == 4:
        band_list = [1,2,3]
    elif band_no == 1:
        band_list = [1]

    gdal.Translate(output_dir + output_fn + ".jpg", src_img, outputType=gdal.GDT_Byte, format='JPEG',
                   bandList=band_list, widthPct=scale_w, heightPct=scale_h, resampleAlg="bilinear",
                   creationOptions=['worldfile=yes', 'quality=90'])
    ## Generate .jgw file
    ## check if .wld exist, if not throw an error. If wld file is not create, gdal_translate did not work as expected. (DEBUG) 
    fn = output_dir + output_fn + '.wld'
    if os.path.isfile(fn):
        print ("wld file present")
        os.rename(output_dir + output_fn + '.wld', output_dir + output_fn + '.jgw')  ## generate .jgw from .wld
    else:
        print ("wld file not generated")


def main(args):

    inputdir = args.input_dir
    outputdir = args.output_dir
    scaling_bool = args.scale

    print ("scale : " + str(scaling_bool))

    if platform.system() == "Windows" and inputdir[-1] != '\\':
        inputdir = inputdir + '\\'
    elif platform.system() == "Linux" and inputdir[-1] != '/':
        inputdir = inputdir + '/'
    if platform.system() == "Windows" and outputdir[-1] != '\\':
        outputdir = outputdir + '\\'
    elif platform.system() == "Linux" and outputdir[-1] != '/':
        outputdir = outputdir + '/'
    print ('Start of Conversion')
    print ('--------------------------------------')
    print ("input dir = " + inputdir)
    print ('--------------------------------------')
    print ('output dir = ' + outputdir)
    print ('--------------------------------------')
    start_time = time.time()
    ## Search for all .tif files in inputdir    
    geotiff_files = glob.glob(inputdir + "*RGB.tif")

    if geotiff_files is None:
        print ('No tif files found')
    else:
        ## make directory if is not already exist.
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        for f in geotiff_files:
            create_jpg_prj_jgw(f, outputdir, scaling_bool)

        print ('--------------------------------------')
        print ('Conversion Finish!!')
        end_time = time.time() - start_time
        print ('Conversion time = %d s' % end_time)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
