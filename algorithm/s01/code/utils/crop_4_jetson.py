import os,sys
import glob
from osgeo import gdal
import numpy as np
np.random.seed(1004)
import cv2
def mk_cvat_d(image_list, origin_image_folder, img_size=4000):
    for i in image_list:
        # last_name = i.split('_')[-1]
        first_name = os.path.basename(i)#os.path.splitext(i)[0]
        first_name = os.path.splitext(os.path.basename(i))[0]
        image_path = os.path.join(origin_image_folder, i)
        raster = gdal.Open(image_path)
        rgb_image = np.array(raster.GetRasterBand(1).ReadAsArray(),np.uint8)

        # 분할 구간 설정
        h, w = rgb_image.shape[:2]

        hd = [x for x in range(0, h, img_size-100)]
        wd = [x for x in range(0, w, img_size-100)]
        hd[-1] = h - img_size; wd[-1] = w - img_size
        
        for h_id, div_h in enumerate(hd[:-1]):
            for w_id, div_w in enumerate(wd[:-1]):
                # 분할된 이미지의 좌표
                x1, y1 = div_w, div_h
                x2, y2 = div_w+img_size, div_h+img_size

                # 이미지 크롭
                crop = rgb_image[y1:y2, x1:x2]
                
                
                div_boxes = []
                save_name = str(first_name) + '_' + str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2) +'.tif'
                line = save_name

                print(x1,y1,x2,y2)
                save_img_path = '/mnt/d/yolov5/test/chile_20230729'
 
                print(save_img_path)
                print(os.path.join('/mnt/d/yolov5/test/chile_20230729', save_name))#/mnt/d/dataset/for_split
                cv2.imwrite(os.path.join(save_img_path, save_name), crop)

image_list = glob.glob('/mnt/d/yolov5/test/2023-07-29-13-37-17_UMBRA-05/chile_test.tif')

#image_list = [os.path.basename(f).replace('txt','tif') for f in files]#[os.path.basename(f) for f in files]

origin_image_folder = "/mnt/d/yolov5/ksas" #D:\umbra_open_data

mk_cvat_d(image_list, origin_image_folder, img_size=640)

