import numpy as np
from osgeo import gdal
import os
import cv2
import pandas as pd
import json
import datatable as dt
np.random.seed(1004)

def contrast_stretching(array, newMin=0, newMax=0.3):

    array_2 = np.nan_to_num(array, nan=0)
    array_2[array_2<=newMin] = newMin
    array_2[array_2>=newMax] = newMax

    result = (array_2 - newMin) / (newMax - newMin) * 255

    return result

def mk_cvat_d(image_list, origin_image_folder, img_size=4000):
    for i in image_list:
        # last_name = i.split('_')[-1]
        print
        first_name = os.path.basename(i)#os.path.splitext(i)[0]
        image_path = os.path.join(origin_image_folder, i)
        print(image_path)
        raster = gdal.Open(image_path)
        rgb_image = np.array(raster.GetRasterBand(1).ReadAsArray(),np.uint8)

        # 분할 구간 설정
        h, w = rgb_image.shape[:2]

        hd = [x for x in range(0, h, img_size-500)]
        wd = [x for x in range(0, w, img_size-500)]
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
                save_img_path = '/mnt/d/dataset/for_split2'
 
                print(save_img_path)
                # f = open(os.path.join(save_txt_path, save_name.replace("tif","txt")), 'w')
                
                # '''
                # from PIL import Image
                # img = Image.fromarray(crop_noline)
                # img.show()
                # '''
                
                # for b in bboxes:
                #     # 현재 분할된 이미지의 x, y 구간
                #     if (x1 <= b[0] <= x2) and (y1 <= b[1] <= y2) and (x1 <= b[2] <= x2) and (y1 <= b[3] <= y2):
                #         # 원본 bbox 좌표를 분할된 이미지 좌표로 변환; b = [x1 y1 x2 y2 class w h]
                #         dx1 = b[0] - x1
                #         dy1 = b[1] - y1
                #         dx2 = b[2] - x1
                #         dy2 = b[3] - y1
                #         centx = dx1+(dx2-dx1)//2; centy = dy1+(dy2-dy1)//2 
                        
                #         bbox = [labels2[b[4]], centx/img_size, centy/img_size, (dx2-dx1)/img_size, (dy2-dy1)/img_size] #cls, center_x, center_y, width, height
                #         #print([b[4], centx, centy, (dx2-dx1), (dy2-dy1)])

                #         div_boxes.append(bbox)
                #                 # if (np.array(bbox[1:])>1).any():
                #                 #     print(bbox)
                print(os.path.join('/mnt/d/dataset/for_split2', save_name))#/mnt/d/dataset/for_split
                cv2.imwrite(os.path.join(save_img_path, save_name), crop)


def division_set(image_list, origin_image_folder, div_set, datatype='kompsat', img_size=640):    
    print("Start dividing images\n\n")
    for i in image_list:
        # last_name = i.split('_')[-1]
        first_name = os.path.splitext(i)[0]
        image_path = os.path.join(origin_image_folder, i)
        print(image_path)
        raster = gdal.Open(image_path)

        image_array = np.array(raster.GetRasterBand(1).ReadAsArray(),np.uint8)
        #rgb_image = cv2.imread(image_path)
        rgb_image = image_array.astype(np.uint8)
    
        # max_num = 0.5
        # min_num = 0.0

        # band_data1= np.array(raster.GetRasterBand(1).ReadAsArray())

        # band_data1[band_data1 >= max_num] = max_num
        # band_data1[band_data1 <= min_num] = min_num
        # band_data1 = band_data1 * (255. / max_num)
        # band_data1 = np.array(band_data1, np.uint8)
        # rgb_image = np.dstack((band_data1,band_data1,band_data1))
        # max_num = Cfg.max1
        # min_num = Cfg.min1
        # raster = gdal.Open(image_path)
        
        # band_data1 = np.array(raster.GetRasterBand(1).ReadAsArray())
        # # max_num = np.quantile(band_data1, 0.9, axis=None)
        # # band_data1=band_data1/0.8191
        # band_data1[band_data1 >= max_num] = max_num
        # band_data1[band_data1 <= min_num] = min_num
        # rgb_image = band_data1 * (255. / max_num)
        

        # x, y, w, h를 x1, y1, x2, y2로 변환 class_id, x, y, width, height
        html_colors = ["#3d3df5", "#46e026", "#fa3253", "#ffcc33", "#ff00cc", "#aaf0d1"] # the color of label
        # aircraft_list = ["A220", "A320/321", "A330", "ARJ21", "Boeing737", "other", "Boeing787"]

        # def replace_with_aircraft2(aircraft): ## this for open aircraft dataset
        #     if aircraft not in aircraft_list:
        #         return "Aircraft2"
        #     else:
        #         return aircraft
            
        blue = []
        green = []
        red = []

        keys = [0, 1, 2, 3, 4, 5, 6,7]
        values = ['ship', 'Transportplane', 'Fighterairplane', 'Airplane1', 'Airplane2', 'Ambiguous', 'movingairplane','Warship']#['Helicopter', 'Transportplane', 'Fighterairplane', 'Airplane1', 'Airplane2', 'Ambiguous', 'movingairplane','Warship']
        labels = dict(zip(keys, values))
        #labels2 = dict(zip(values,keys))
        for color_code in html_colors:
            # Convert the color code to an RGB tuple of integers
            r, g, b = tuple(int(color_code[i:i+2], 16) for i in (1, 3, 5))
            blue.append(b)
            green.append(g)
            red.append(r)
            
        blue = dict(zip(keys, blue))
        green = dict(zip(keys, green))
        red = dict(zip(keys, red))
        print(image_path.replace("tif", "txt"))#jpg
        d = dt.fread(image_path.replace(".tif", ".txt"),encoding='utf-8')#.jpg
        image_df = d.to_pandas()
        image_df['C0'].astype('int')
        # raster = gdal.Open(image_path)
        # band = raster.GetRasterBand(1)
        # array = band.ReadAsArray()
        # image_width = array.shape[1]
        # image_height = array.shape[0]
        image_width = rgb_image.shape[1]
        image_height = rgb_image.shape[0]
        image_df.columns = ['class','X','Y','W','H']
        #image_df['class'] = image_df['class'].apply(replace_with_aircraft2)
        
        image_df['blue'] = image_df['class'].copy()
        image_df['green'] = image_df['class'].copy()
        image_df['red'] = image_df['class'].copy()
        
        image_df = image_df.replace({"nclass":labels, "blue":blue, "green":green, "red":red})
        image_df['class'] = image_df['class'].astype('int')
        image_df['X'] = ((image_df['X']- (image_df['W']/2)) * image_width).astype('int')
        image_df['Y'] = ((image_df['Y']- (image_df['H']/2)) * image_height).astype('int')
        image_df['W'] = (image_df['W'] * image_width).astype('int')
        image_df['H'] = (image_df['H'] * image_height).astype('int')
        #print(image_df['class'])
        #image_df['class'] = [col.lower().strip() for col in image_df['class'].to_numpy() if isinstance(col, str)]
        image_df['X2'] = image_df['X'] + image_df['W']
        image_df['Y2'] = image_df['Y'] + image_df['H']
        bboxes = image_df[['X', 'Y', 'X2', 'Y2', 'class', 'W', 'H','blue','green','red']].to_numpy()
        # print(bboxes)
        print(bboxes)
        # 분할 구간 설정
        h, w = rgb_image.shape[:2]

        hd = [x for x in range(0, h, img_size-50)]
        wd = [x for x in range(0, w, img_size-50)]
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
                save_txt_path = '/mnt/z/umbra_ship_train/patch/'#'/mnt/e/detect_target/data/umbra_split/'#'/mnt/e/detect_target/data/for_split/' #'/mnt/d/dataset/for_split'/mnt/e/detect_target/data/for_split
                save_img_path = '/mnt/z/umbra_ship_train/patch/'#'/mnt/e/detect_target/data/umbra_split/'#'/mnt/e/detect_target/data/for_split/' #'/mnt/d/dataset/for_split'
                print(save_txt_path)
                print(save_img_path)
                f = open(os.path.join(save_txt_path, save_name.replace("tif","txt")), 'w')
                
                '''
                from PIL import Image
                img = Image.fromarray(crop_noline)
                img.show()
                '''
                
                for b in bboxes:
                    # 현재 분할된 이미지의 x, y 구간
                    if (x1 < b[2] and b[0] < x2) and (y1 < b[3] and b[1] < y2):
                    #if (x1 <= b[0] <= x2) and (y1 <= b[1] <= y2) and (x1 <= b[2] <= x2) and (y1 <= b[3] <= y2):
                        # 원본 bbox 좌표를 분할된 이미지 좌표로 변환; b = [x1 y1 x2 y2 class w h]
                        # Adjust the bbox coordinates to the crop area
                        original_width = b[2] - b[0]
                        original_height = b[3] - b[1]
                        dx1 = max(b[0] - x1, 0)
                        dy1 = max(b[1] - y1, 0)
                        dx2 = min(b[2] - x1, img_size)
                        dy2 = min(b[3] - y1, img_size)
                        centx = dx1 + (dx2 - dx1) / 2
                        centy = dy1 + (dy2 - dy1) / 2
                        width = dx2 - dx1
                        height = dy2 - dy1
                        # dx1 = b[0] - x1
                        # dy1 = b[1] - y1
                        # dx2 = b[2] - x1
                        # dy2 = b[3] - y1
                        #centx = dx1+(dx2-dx1)//2; centy = dy1+(dy2-dy1)//2 
                        #labels2[b[4]]
                        # Filter out small bounding boxes
                        # if width >= original_width *0.6 and height >= original_height *0.6:
                        #     bbox = [b[4], centx/img_size, centy/img_size, (dx2-dx1)/img_size, (dy2-dy1)/img_size] #cls, center_x, center_y, width, height
                        # #print([b[4], centx, centy, (dx2-dx1), (dy2-dy1)])

                        #     div_boxes.append(bbox)
                        # if b[4] == "movingairplane" and width > 70 and height > 70:
                        #     bbox = [b[4], centx/img_size, centy/img_size, (dx2-dx1)/img_size, (dy2-dy1)/img_size] #cls, center_x, center_y, width, height
                        #     div_boxes.append(bbox)
                        bbox = [b[4], centx/img_size, centy/img_size, (dx2-dx1)/img_size, (dy2-dy1)/img_size] #cls, center_x, center_y, width, height
                        div_boxes.append(bbox)
                                # if (np.array(bbox[1:])>1).any():
                                #     print(bbox)

                cv2.imwrite(os.path.join(save_img_path, save_name), crop)
                # for d in div_boxes:
                #         #class_name = 'ship' if strd[4]==0 else 'other'
                #     f.write('%s %.6f %.6f %.6f %.6f\n' % (d[0], d[1], d[2], d[3], d[4]))
                # f.close()

                if len(div_boxes) > 0:
                    
                    for d in div_boxes:
                        #class_name = 'ship' if strd[4]==0 else 'other'
                        # if d[0] == 'movingairplane':
                        #     d[0] = 'Airplane2'
                            
                        #if d[0] == 'movingairplane': #'Airplane2' or d[0] == 'movingairplane':'Fighterairplane'
                        f.write('%s %.6f %.6f %.6f %.6f\n' % (d[0], d[1], d[2], d[3], d[4]))
                        # print(d[0])
                        # if d[0] == 'Fighterairplane' or d[0] == 'Helicopter' or d[0] == 'False':
                        #     f.write('%s %.6f %.6f %.6f %.6f\n' % (d[0], d[1], d[2], d[3], d[4]))
                    f.close()
                    # for d in div_boxes:
                    #     #class_name = 'ship' if strd[4]==0 else 'other'
                    #     f.write('%s %.6f %.6f %.6f %.6f\n' % (d[0], d[1], d[2], d[3], d[4]))
                    # f.close()
                # else:
                #     os.remove(os.path.join(save_img_path, save_name))
                #     os.remove(os.path.join(save_txt_path, save_name.replace("tif","txt")))           
                # elif d[0] != 'Fighterairplane' or d[0] != 'Helicopter':
                    if np.random.rand() > 0.5:#0.01:
                    #if d[0] != 'Fighterairplane' or d[0] != 'Helicopter':
                        os.remove(os.path.join(save_img_path, save_name))
                        os.remove(os.path.join(save_txt_path, save_name.replace("tif","txt")))

def band_to_rgb(tif_path):

    print(tif_path)
    raster = gdal.Open(tif_path)

    image_array = np.array(raster.GetRasterBand(1).ReadAsArray(),np.uint8)
    rgb_image = image_array.astype(np.uint8)

    return np.dstack((rgb_image,rgb_image,rgb_image))


def division_testset(input_band=None, img_size=640):
    img_list, div_coord = [], []
    # 분할 구간 설정
    h, w = input_band.shape[:2]

    hd = [x for x in range(0, h, img_size-100)]
    wd = [x for x in range(0, w, img_size-100)]
    hd[-1] = h - img_size; wd[-1] = w - img_size
        
    for h_id, div_h in enumerate(hd[:-1]):
        for w_id, div_w in enumerate(wd[:-1]):
            # 분할된 이미지의 좌표
            x1, y1 = div_w, div_h
            x2, y2 = div_w+img_size, div_h+img_size

            dw = x2-x1; dh = y2-y1
            # Crop
            crop = input_band[y1:y2, x1:x2]
            img_list.append(crop)
            div_coord.append([dw, dh, div_w, div_h])

    return img_list, div_coord