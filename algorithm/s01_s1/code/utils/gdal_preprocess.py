import numpy as np
from osgeo import gdal
import os
import cv2
import pandas as pd
import json

# labelme로 레이블링한 어노테이션 정보를 ai factory 양식에 맞게 변환
def get_labelme_annotation():
    final_df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
    json_list = [x for x in os.listdir('./dataset/train_dataset/') if x.endswith('json')]

    for json_path in json_list:
        image_name = json_path.split('.')[0] + '.tif'
        
        with open('./train_rgb_dataset/'+json_path) as j:
            json_data = json.load(j)
        bboxes = []
        for i in json_data['shapes']:
            box_info = i['points']
            if len(box_info) != 2:
                continue
            
            x1, y1 = box_info[0]
            x2, y2 = box_info[1]

            if x1 >= x2:
                tmp_x = x1
                x1 = x2
                x2 = tmp_x

            box = [int(x) for x in [x1, y1, x2, y2]]

            bboxes.append(box)

        # to csv
        arr_bboxes = np.array(bboxes)
        df = pd.DataFrame(columns=['image', 'X', 'Y', 'W', 'H'])
        df['image'] = [image_name for x in range(len(arr_bboxes))]
        df['X'] = arr_bboxes[:, 0]
        df['Y'] = arr_bboxes[:, 1]
        df['W'] = arr_bboxes[:, 2] - arr_bboxes[:, 0]
        df['H'] = arr_bboxes[:, 3] - arr_bboxes[:, 1]

        final_df = final_df.append(df, ignore_index=True)
        # print(final_df.shape)


# 학습자료를 분할(SAR image와 TrainingDataset 전부)
def division_testset(input_band=None, img_size=640):
    img_list, div_coord = [], []

    h, w = input_band.shape[:2]
    
    hd = [x for x in range(0, h, img_size-60)]
    wd = [x for x in range(0, w, img_size-60)]
    
    hd[-1] = h - img_size; wd[-1] = w - img_size
    hd.sort(); wd.sort()
    
    for h_id, div_h in enumerate(hd[:-1]):
        for w_id, div_w in enumerate(wd[:-1]):
            # Div position
            x1, y1 = div_w, div_h
            x2, y2 = div_w+img_size, div_h+img_size
            # Crop
            crop = input_band[y1:y2, x1:x2]
            img_list.append(crop)
            div_coord.append([div_w, div_h])

    return img_list, div_coord
    
    
# band 1 ~ 3을 0 ~ 255 값을 갖는 rgb로 변환
def band_to_rgb(tif_path,bandnumber,partest=False):
    from utils.cfg import Cfg
    import numpy as np
    from sklearn import preprocessing

    raster = gdal.Open(tif_path)

    # transformation of 3-banded SAR image
    if bandnumber==3:
        bands = []
        for i in range(raster.RasterCount):
            band = raster.GetRasterBand(i+1)
            meta = band.GetMetadata()
            if band.GetMinimum() is None or band.GetMaximum()is None:
                band.ComputeStatistics(0)

            band_data = np.array(raster.GetRasterBand(i+1).ReadAsArray())

            if i == 0:
                # band 1의 최대값을 0.1로
                max_num = Cfg.max1
                min_num = Cfg.min1

            elif i == 1:
                # band 2의 최대값을 0.5로
                max_num = Cfg.max2
                min_num = Cfg.min2

            else:
                # band 3의 최대값을 1.0으로
                max_num = Cfg.max3
                min_num = Cfg.min3

                # GRDH DPIVD measurement
                if partest == True:

                    DoP=np.divide(Svh,Svv)
                    p1=np.divide(Svv,Svh+Svv)
                    SEiN=10*np.multiply(Svh,Svv)
                    #SEiN2=preprocessing.normalize(np.power(SEiN,2))

                    #band_data=np.multiply(DoP,np.multiply(p1,SEiN2))
                    band_data = np.multiply(DoP, np.multiply(p1, SEiN))
                    #band_data = band_data + 30

                    # Remove the data as 0 where infinite, NaN or negative
                    band_data[np.isinf(band_data)]=0
                    band_data[np.where(band_data<0)]=0
                    band_data[np.isnan(band_data)] = 0


                    #band_data=10*np.log10(band_data)

                    max_num = np.quantile(band_data,0.98)
                    min_num = 0
                    #min_num = np.quantile(band_data,0.05)

                    #print(np.max(band_data))


            # 0 ~ 255 이미지로 변환
            if partest == True:
                if i == 0:
                    Svh = band_data
                elif i == 1:
                    Svv = band_data

            band_data[band_data > max_num] = max_num
            band_data[band_data < min_num] = min_num
            band_data = band_data * (255./max_num)
            #band_data = band_data * ((255 - min_num)/ (max_num - min_num))

            bands.append(band_data)

         # band 1, 2, 3을 RGB로 변환
        rgb = np.dstack((bands[2], bands[1], bands[0]))
        #rgb = np.array(rgb)
        rgb = np.array(rgb, np.uint8)

    # transformation of single-banded SAR image
    elif bandnumber==1:
        max_num = Cfg.max1
        min_num = Cfg.min1

        band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data1, 0.9, axis=None)
        # band_data1=band_data1/0.8191
        band_data1[band_data1 > max_num] = max_num
        band_data1[band_data1 < min_num] = min_num
        band_data1 = band_data1 * (255. / max_num)

        rgb = np.zeros((band_data1.shape[0], band_data1.shape[1], 3))
        rgb[:, :, 0] = band_data1

        # For Band2(Min/Max)
        max_num = Cfg.max2
        min_num = Cfg.min2

        band_data2 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data2, 0.9, axis=None)
        # band_data2 = band_data2 / 0.8191
        band_data2[band_data2 > max_num] = max_num
        band_data2[band_data2 < min_num] = min_num
        band_data2 = band_data2 * (255. / max_num)

        rgb[:, :, 1] = band_data2

        # For Band3(Min/Max)
        max_num = Cfg.max3
        min_num = Cfg.min3

        band_data3 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data3, 0.9, axis=None)
        # band_data3 = band_data3 / 0.8191
        band_data3[band_data3 > max_num] = max_num
        band_data3[band_data3 < min_num] = min_num
        band_data3 = band_data3 * (255. / max_num)

        rgb[:, :, 2] = band_data3

        #rgb = np.dstack((rgb[:,:,2], rgb[:,:,1], rgb[:,:,0]))
        #rgb = np.array(rgb)
        rgb = np.array(rgb, np.uint8)


    # transformation of double-banded SAR image(B1,B2,B2)
    elif bandnumber==2:
        max_num = Cfg.max1
        min_num = Cfg.min1

        band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # max_num = np.quantile(band_data1, 0.9, axis=None)
        # band_data1=band_data1/0.8191
        band_data1[band_data1 > max_num] = max_num
        band_data1[band_data1 < min_num] = min_num
        band_data1 = band_data1 * ((255 - min_num) / (max_num - min_num))

        rgb = np.zeros((band_data1.shape[0], band_data1.shape[1], 3))
        rgb[:, :, 0] = band_data1

        # For Band2(Min/Max)
        max_num = Cfg.max2
        min_num = Cfg.min2

        band_data2 = np.array(raster.GetRasterBand(2).ReadAsArray())
        # max_num = np.quantile(band_data2, 0.9, axis=None)
        # band_data2 = band_data2 / 0.8191
        band_data2[band_data2 > max_num] = max_num
        band_data2[band_data2 < min_num] = min_num
        band_data2 = band_data2 * ((255 - min_num) / (max_num - min_num))

        rgb[:, :, 1] = band_data2

        # For Band3(Min/Max)
        max_num = Cfg.max3
        min_num = Cfg.min3

        band_data3 = np.array(raster.GetRasterBand(2).ReadAsArray())
        # max_num = np.quantile(band_data3, 0.9, axis=None)
        # band_data3 = band_data3 / 0.8191
        band_data3[band_data3 > max_num] = max_num
        band_data3[band_data3 < min_num] = min_num
        band_data3 = band_data3 * ((255 - min_num)/(max_num - min_num))


        rgb[:, :, 2] = band_data3

        rgb = np.array(rgb, np.uint8)

    return rgb


# 외각 라인 검출(육지를 제거하기 위해)
def line_detection(input_array):
    input_image = np.array(input_array, np.uint8)
    
    # 비교적 잡음이 적은 band 1 영상에 대해 수행
    gray_image = input_image[:,:,2]
    blur_image = cv2.medianBlur(gray_image, 5) 

    # band 1 침식과정을 통해 흰색 노이즈 제거
    erode_image = cv2.erode(blur_image, (3,3), iterations=5)

    # threshhold
    thr = 35
    ret, thresh = cv2.threshold(erode_image, thr, 255, 0)

    # 육지 정보를 저장할 이미지
    line_filter = np.zeros(input_array.shape[:2], np.uint8)
    
    # 외각 라인 검출
    try:
        ext_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _, ext_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in ext_contours:
        # 각 라인들의 면적
        area = cv2.contourArea(c)
        # 면적이 600 이상일 경우 육지로 판단하고 해당 위치의 픽셀값을 1로
        # 600 미만일 경우 0
        if area >= 600:
            line_filter = cv2.drawContours(line_filter, [c], -1, 1, -1)
    
    return line_filter


# Corresponding function to geotiffread of MATLAB
def geotiffread(tif_name, num_band):
    import gdal
    import numpy as np

    gdal.AllRegister()

    ds = gdal.Open(tif_name)

    if num_band == 3:
        band1 = ds.GetRasterBand(1)
        arr1 = band1.ReadAsArray()
        band2 = ds.GetRasterBand(2)
        arr2 = band2.ReadAsArray()
        band3 = ds.GetRasterBand(3)
        arr3 = band3.ReadAsArray()

        cols, rows = arr1.shape

        arr = np.zeros((cols, rows, 3))
        arr[:, :, 0] = arr1
        arr[:, :, 1] = arr2
        arr[:, :, 2] = arr3

    elif num_band == 1:
        band1 = ds.GetRasterBand(1)
        arr = band1.ReadAsArray()

        cols, rows = arr.shape


    else:
        print('cannot open except number of band is 1 or 3')

    gt = ds.GetGeoTransform()
    gt = np.array(gt)
    gt[2] = gt[0] + gt[1] * (rows - 1)
    gt[4] = gt[3] + gt[5] * (cols - 1)

    gt.astype(np.double)

    return arr, gt


# Median filtering on Oversampled image(Especially K5 0.3m)
def median_filter(img, filter_size=(5,5), stride=1):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    img_shape = np.shape
    result_shape=tuple(np.int64(np.array(image_shape)-np.array(filter_size))/stride+1)

    result=np.zeros(result_shape)
    for h in range(0,result_shape[0],stride):
        for w in range(0,result_shape[1],stride):
            tmp=img[h:h+filter_size[0],w:w+filter_size[1]]
            tmp=np.sort(tmp.ravel())
            result[h,w]=tmp[int(filter_size[0]*filter_size[1]/2)]

    return result


if __name__ == '__main__':
    # get_labelme_annotation()
    division_trainset(div_num=40)
   


    

