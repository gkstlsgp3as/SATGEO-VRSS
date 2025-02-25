# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --model_weight_file yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --model_weight_file yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadSAR,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_xywh_box
from utils.torch_utils import select_device, time_synchronized
from sqlalchemy.orm import Session
from app.config.settings import settings
import argparse
import gc
import csv
import os
import platform
import sys
from pathlib import Path
import numpy as np
import cv2
from osgeo import gdal
import time
import torch
import random
import logging
from utils.recognition import sub_ap_um,cos_sim,contrast,amplitude
from utils.cfg import Cfg

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from torchvision.ops import nms
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    model_weight_file=ROOT / "yolov5s.pt",  # model path or triton URL
    input_dir=ROOT / "data/images",  # nitf file for detection
    output_dir=ROOT / "runs/detect",  # save results path
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    ):
    
    source = str(input_dir)
    for_total_nms = []
    # Directories
    #output_dir = os.makedirs(output_dir + 'output/', exist_ok=True)

    save_dir = Path(output_dir)  # increment run
    # if save_dir.name != "output":
    #     save_dir = save_dir / "output"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(model_weight_file, device=device, dnn=False, data=None, fp16=Cfg.half)
    stride, names, pt = model.stride, model.names, model.pt
    
    img_size = check_img_size(img_size, s=stride)  # check image size

    # Dataloader
    old_img_w = old_img_h = img_size
    old_img_b = 3
    t0 = time.time()

    dataset = LoadSAR(source, img_size=img_size, stride=stride, save_dir=save_dir)
    image_name = os.path.basename(dataset.files[0]).split('.')[0]
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[153,97,98]] #unkwon bessel color

    for path, rgb_band, div_img_list, div_coord in dataset:  
    # 테스트 이미지를 1/div_num 만큼 width, height를 분할하고, 크롭된 이미지와 위치좌표를 반환
        for d_id, img0 in enumerate(div_img_list):
            
            # 원본 이미지 좌표로 변환하기 위해 분할 좌표를 저장
            div_x, div_y = div_coord[d_id][0], div_coord[d_id][1]
            img = letterbox(img0, img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416; already applied
            im = img.copy()  # inference image
            im = np.ascontiguousarray(im)# 연속된 배열로 변환
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = im.shape[0]
                old_img_h = im.shape[2]
                old_img_w = im.shape[3]
                for i in range(3):
                    model(im)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(im)[0] # [x1, y1, x2, y2, object_confidence, class_confidence]
            t2 = time_synchronized()
            pred = non_max_suppression(pred, Cfg.conf_thres, Cfg.iou_thres, max_det=Cfg.max_det)#, classes
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', img0
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)#.replace('.tif','_{}.tif'.format(d_id)))  # img.jpg
                txt_path = str(save_dir / p.stem) # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh


                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh = [el*div_coord[d_id][0]  if i//2==0 else el*div_coord[d_id][1]  for i, el in enumerate(xywh)]
                        xywh = [int(el) for el in xywh]
                        
                        # (cx, cy, w, h) -> (x1, y1, x2, y2) 변환
                        cx, cy, w, h = xywh
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2

                        # 변환된 좌표
                        xyxy_converted = [x1, y1, x2, y2]

                        # 원본 이미지 좌표로 변환하기 위해 분할 좌표를 저장
                        xyxy_converted[0] += div_coord[d_id][2]
                        xyxy_converted[1] += div_coord[d_id][3]
                        xyxy_converted[2] += div_coord[d_id][2]
                        xyxy_converted[3] += div_coord[d_id][3]

                        line = (cls, *xyxy_converted, conf) #if opt.save_conf else (cls, *xywh)  # label format
                        # line[1] = line[1] - line[3] / 2  # top left x
                        # line[2] = line[2] - line[4] / 2  # top left y
                        # line[3] = line[1] + line[3] / 2 # bottom right x
                        # line[4] = line[2] + line[4] / 2 # bottom right y
                        tensor_data = [torch.tensor(item, device=device) if not isinstance(item, torch.Tensor) else item for item in line]
                        # Concatenate into one tensor
                        tensor_data = torch.stack(tensor_data)
                        for_total_nms.append(tensor_data) # [class, x, y, w, h, conf]
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
    

    bfnms_tensor = torch.tensor([item for sublist in for_total_nms for item in sublist]).reshape(-1,6)
    scores = bfnms_tensor[:,5]
    labels = bfnms_tensor[:,0]
    bboxes = bfnms_tensor[:,1:5]; #bboxes[:,2] = bboxes[:,2] + bboxes[:,0]; bboxes[:,3] = bboxes[:,3] + bboxes[:,1]
    after_indices = nms(bboxes, scores, 0.1)
    bboxes = np.array(bboxes[after_indices])
    scores = np.array(scores[after_indices])
    labels = np.array(labels[after_indices])


# Open the CSV file in append mode
    if Cfg.output_format == 1:
        save_label = txt_path + '.txt'
        with open(save_label, 'a', newline='') as txtfile:
            # Create a CSV writer object
            csv_writer = csv.writer(txtfile, delimiter=',')
            
            # Write the header row if the file is empty
            if txtfile.tell() == 0:
                csv_writer.writerow(['ImageName', 'Lon', 'Lat', 'Color', 'Size', 'X', 'Y', 'W', 'H'])
            
            # Write the data rows
            for annotation, b, s in zip(bboxes, labels, scores):
                left, top, right, bottom = annotation
                cx = (left + right) / 2
                cy = (top + bottom) / 2
                w = right - left
                h = bottom - top
                
                # Write a single row to the CSV
                csv_writer.writerow([image_name, cx, cy, '153097098', s, left, top, w, h])
                cv2.rectangle(rgb_band, (int(left), int(top)), (int(right), int(bottom)), colors[0],8)
                
    elif Cfg.output_format == 2:
        save_label = txt_path + '.csv'
        with open(save_label, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)
            
            # Write the header row if the file is empty
            if csvfile.tell() == 0:
                csv_writer.writerow(['ImageName', 'Lon', 'Lat', 'Color', 'Size', 'X', 'Y', 'W', 'H'])
            
            # Write the data rows
            for annotation, b, s in zip(bboxes, labels, scores):
                left, top, right, bottom = annotation
                cx = (left + right) / 2
                cy = (top + bottom) / 2
                w = right - left
                h = bottom - top
                
                # Write a single row to the CSV
                csv_writer.writerow([image_name, cx, cy, '153097098', s, left, top, w, h])
                cv2.rectangle(rgb_band, (int(left), int(top)), (int(right), int(bottom)), colors[0],8)
            
    # Save results (image with detections)
    if Cfg.save_img:
        save_path = str(save_dir / Path(path).name)
    
        print(f" The image with the result is saved in: {save_path.replace('.nitf','.tif')}")
        cv2.imwrite(save_path.replace('.nitf','.tif'), rgb_band)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        print(f'Done. ({time.time() - t0:.3f}s)')



def get_args():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weight_file", 
        nargs="+", 
        type=str, 
        default="../weights/Umbra_ship.pt", 
        help="model path or triton URL"
    )
    parser.add_argument(
        '-I', "--input_dir", 
        type=str, 
        default="../data/input/", 
        help="file of SLC data(nitf, tif, h5)"
    )
    parser.add_argument(
        '-O', "--output_dir", 
        type=str, 
        default="./output/", 
        help="Path to update a single detection csv/txt file"
    )
    
    parser.add_argument(
        "--device", 
        default="", 
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    
    args = parser.parse_args()
    print_args(vars(args))
    return args


def process(db: Session, satellite_sar_image_id: str):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    #check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    model_weight_file = settings.S01_MODEL_PATH
    input_dir = settings.S01_INPUT_PATH
    output_dir = settings.S01_OUTPUT_PATH
    device = ""
    
    run(model_weight_file, input_dir, output_dir, device)


if __name__ == "__main__":
    start_time = time.time()

    args = get_args()
    #img_size = Cfg.img_size * 2 if len(Cfg.img_size) == 1 else 1  # expand
    
    run(**vars(args))

    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")