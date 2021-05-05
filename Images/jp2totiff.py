from osgeo import gdal
in_ds = gdal.OpenEx('/home/pfiadmin/Robby/TreeCount/ACRA_1M_Req/ACRA_1M_Images/S02_202104241312_JB_DA_TPH0002501_RGB.jp2')
out_ds = gdal.Translate('/home/pfiadmin/Robby/TreeCount/ACRA_1M_Req/ACRA_1M_Images/S02_202104241312_JB_DA_TPH0002501_RGB.tif', in_ds)
