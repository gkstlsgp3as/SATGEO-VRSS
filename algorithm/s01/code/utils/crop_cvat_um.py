import os,sys
import glob
sys.path.append('/mnt/d/yolov5/utils')
from gdal_preprocess import *
from osgeo import gdal
import numpy as np
np.random.seed(1004)
#E:\snu\2024\민군
image_list = glob.glob('/mnt/d/dataset/umbra_for_split_2/*.tif')

#image_list = [os.path.basename(f).replace('txt','tif') for f in files]#[os.path.basename(f) for f in files]

origin_image_folder = "/mnt/d/umbra_for_split_2" #D:\umbra_open_data
# Set the patch size
patch_size = 6000
mk_cvat_d(image_list, origin_image_folder, img_size=7000)