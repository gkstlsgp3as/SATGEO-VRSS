from scipy.signal import get_window
from scipy import signal
import scipy.stats as stats
import json
from scipy.constants import c, milli
import numpy as np
import pandas as pd
import os
import math
import scipy
from osgeo import gdal
from scipy import ndimage
from skimage import filters
from scipy.ndimage import zoom
import gc
from scipy.constants import c, milli
from skimage.feature import peak_local_max
from skimage.transform import radon, iradon_sart
from extract_parameter_ver4 import Umbra
import os
import glob


def get_sicd_file(directory):
    # searching .nitf 
    nitf_files = glob.glob(os.path.join(directory, '*.nitf'))
    if not nitf_files:
        raise FileNotFoundError("No .nitf files found in the directory.")
    
    # filtering for named SICD file
    sicd_files = [f for f in nitf_files if 'SICD' in os.path.basename(f)]
    
    if not sicd_files:
        raise FileNotFoundError("No .nitf files with 'SICD' in the filename found in the directory.")
    
    return sicd_files[0]

def get_json_file(directory):
    # .json 파일 검색
    json_files = glob.glob(os.path.join(directory, '*.json'))
    
    if not json_files:
        raise FileNotFoundError("No .json files found in the directory.")
    
    # 파일명에 "METADATA"가 포함된 파일 필터링
    metadata_files = [f for f in json_files if 'METADATA' in os.path.basename(f)]
    
    if not metadata_files:
        raise FileNotFoundError("No .json files with 'METADATA' in the filename found in the directory.")
    
    return metadata_files[0]

def make_complex(data):#data is file pth of h5
    dataset = gdal.Open(data, gdal.GA_ReadOnly)
    X = dataset.ReadAsArray().astype(np.float16)
    band1 = X[0,:,:]
    band2 = X[1,:,:]

    SLC_data = np.array(band1) + 1j * np.array(band2)
    return SLC_data

def load_json(file_path):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
    
    return json_data

def shanon(arr): # this function for calculating entropy
        Ez = np.nansum(np.abs(arr)**2)
        Entcomp = np.nansum((np.abs(arr)**2) * (np.log(np.abs(arr)**2)))
        ent = np.log(Ez) - (Entcomp / Ez)
        return ent

def convert_coordinates(label, img_w, img_h):

    class_name, x_center, y_center, width, height= label
    x_center, y_center, width, height = (
        float(x_center) * img_w, 
        float(y_center) * img_h, 
        float(width) * img_w, 
        float(height) * img_h
    )
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)

    return [int(x1), int(y1), int(x2), int(y2),class_name]

def read_label(label_path, img_w, img_h):
    yolo_labels = []

# Open the file and read its contents
    with open(label_path, 'r') as file:
        for line in file:
            # Split the line by spaces (assuming it's space-separated)
            parts = line.strip().split()
            
            # Extract class name and coordinates
            class_name = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            label = convert_coordinates([class_name, x_center, y_center, width, height], img_w, img_h)
            # Append the parsed data as a list
            yolo_labels.append(label)
    yolo_labels = np.array(yolo_labels)
    return yolo_labels;print(yolo_labels)


def average_two_closest(a, b, c):
    # Put the values in a list and sort them
  values = sorted([a, b, c])
  
  # Calculate the differences between adjacent sorted values
  diff1 = abs(values[0] - values[1])
  diff2 = abs(values[1] - values[2])
  # Remove the value showing the largest difference with the other two
  if diff1 > diff2:
      # Remove the smallest value
      remaining_values = values[1:]
  else:
      # Remove the largest value
      remaining_values = values[:2]
  
  # Calculate the average of the remaining two values
  return sum(remaining_values) / 2

def findCOGdifference(angles, threshold):
    n = len(angles)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(angles[i] - angles[j]) > threshold:
                return True, (angles[i], angles[j])
    return False, ()

class Velocticy_est_UMBRA():
    ## firstly, load aux files and SLC file for ready to making 
    def __init__(self, file_path, start_Azimuthvel, end_Azimuthvel, spacing, bbox,lam=0.031): #file_path is file path of slc and bbox is path of bounding boxes
        self.SLC_data = make_complex(get_sicd_file(file_path))
        self.Aux = load_json(get_json_file(file_path)) #file_path.replace("_SICD_MM.nitf", "_MM_METADATA.json")) 
        self.lam = lam
        self.c0 = c
        self.vsar = 7600
        self.bandwidth = 1200e6
        self.az_resolution = float(self.Aux['derivedProducts']['SICD'][0]['slantResolution']['azimuthMeters'])
        self.Prf_ac = self.vsar/self.az_resolution
        self.mid_slr = float(self.Aux['collects'][0]['slantRangeMeters'])

        self.slantspacing = float(self.Aux['derivedProducts']['SICD'][0]['slantResolution']['rangeMeters'])
        self.slr = np.tile(np.linspace(self.mid_slr - self.slantspacing * round(np.shape(self.SLC_data)[1] / 2),
                            self.mid_slr + self.slantspacing * round(np.shape(self.SLC_data)[1] / 2),
                            np.shape(self.SLC_data)[1]),(self.SLC_data.shape[0],1))
        self.inc_angle = float(self.Aux['collects'][0]['angleIncidenceDegrees'])
        self.fc = self.c0 / self.lam
        self.bbox_array = read_label(bbox, self.SLC_data.shape[1], self.SLC_data.shape[0]) # x1,y1,x2,y2,class_name
        self.Azimuthvel_array = np.arange(start_Azimuthvel, end_Azimuthvel, spacing)



    def patchmaker(self):
        '''
        this method is for making patch data of SLC.
        data_SLC_2DBox is nested array of SLC data.
        SlantRangeBox is nested array of Slant Range data.
        f_azimuth is nested array of azimuth frequency data.
        f_range is nested array of range frequency data.
        '''
        data_SLC_2DBox = [None] * len(self.bbox_array)
        SlantRangeBox = [None] * len(self.bbox_array)
        f_azimuth = [None] * len(self.bbox_array)
        f_range = [None] * len(self.bbox_array)        
        oversamplePar=1
        for num in range(len(self.bbox_array)):
            tempbbox = self.bbox_array[num].astype(int)
            SLC_datatemp = self.SLC_data[tempbbox[1]:tempbbox[3], tempbbox[0]:tempbbox[2]].T #[int(top):int(bottom), int(left):int(right)]
            data_SLC_2DBox[num] = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.fft.fft(SLC_datatemp, int(SLC_datatemp.shape[1]*oversamplePar), axis=1), 1), axis=0), 0)

            # Slant Range chip
            zoom_factor_y = SLC_datatemp.shape[0] * oversamplePar / self.slr[tempbbox[1]:tempbbox[3], tempbbox[0]:tempbbox[2]].shape[0]
            zoom_factor_x = SLC_datatemp.shape[1] / self.slr[tempbbox[1]:tempbbox[3], tempbbox[0]:tempbbox[2]].shape[1]
            SlantRangeBox[num] = zoom(self.slr[tempbbox[1]:tempbbox[3], tempbbox[0]:tempbbox[2]], 
                                    (zoom_factor_y, zoom_factor_x))

            f_rangetemp = np.linspace(self.fc-self.bandwidth/2, self.fc+self.bandwidth/2, len(SLC_datatemp[0]))
            f_range[num] = np.tile(f_rangetemp, (len(SLC_datatemp)*oversamplePar, 1))

            f_azimuthtemp = np.linspace(-(self.Prf_ac/2), (self.Prf_ac/2), len(SLC_datatemp)*oversamplePar)
            
            f_azimuth[num] = np.tile(f_azimuthtemp.reshape(-1, 1), (1, len(SLC_datatemp[1])))

        return data_SLC_2DBox,SlantRangeBox,f_azimuth,f_range
    
    def velocity_estimation(self,data_SLC_2DBox,SlantRangeBox,f_azimuth,f_range):
        '''
        in_velocity_target method for calculating phase information and comp phase in data_SLC_2DBox
        returned nested_array is contained the target chip of different velocity
        '''
        dim1 = len(self.Azimuthvel_array)
        dim2 = len(self.bbox_array)
        nested_array = np.array([[None]*dim1 for _ in range(dim2)])
        for i in range(self.bbox_array.shape[0]): 
            for j in range(len(self.Azimuthvel_array)):
                Phase_2 = np.pi * SlantRangeBox[i] * (1 / ((2 / self.lam) * 
                                                        (1 + (f_range[i] * self.lam / c)))) * ((1 / self.vsar**2) - (1 / (self.vsar - self.Azimuthvel_array[j])**2))
                      
                data_ship_comp = np.fft.ifft2(data_SLC_2DBox[i]* np.exp(-2 * 1j*(Phase_2 * f_azimuth[i]**2))).astype(np.complex64)
                nested_array[i][j] = data_ship_comp
        return nested_array

    def calculate_entropy(self,nested_array):
        '''
        this part is for calculating entropy of each target chip
        min_entropy_nes_array is the array contained the minimum entropy of each target chip
        min_entropy_ind is the index of the minimum entropy of each target chip
        '''

        dim1 = len(self.Azimuthvel_array)
        dim2 = len(self.bbox_array)
        entropy_nes_array = np.array([[None]*dim1 for _ in range(dim2)])

        for i in range(self.bbox_array.shape[0]): # i is target bbox j is velocity (90~-90, spacing: 0.5)
            for j in range(len(self.Azimuthvel_array)):
                #ent = contrast(np.abs(nested_array[i][j]))
                ent = shanon(np.abs(nested_array[i][j]))
                entropy_nes_array[i][j] = ent

        return entropy_nes_array
    

    def calculate_velocity(self,entropy_nes_array):
        # this is the part which return target azimuth velocity
        vc_azimuth = []
        for i in range(self.bbox_array.shape[0]):
            vc_azimuth.append(self.Azimuthvel_array[np.argmin(entropy_nes_array[i,:])])

        
        return vc_azimuth
    
    def extract_rftarget(self, nested_array, entropy_nes_array):
        # this method is for making refocused target chip patches.
        '''
        nested_array is the array contained the target chip of different velocity
        entropy_nes_array 0-axis is len of targets and 1-axis is azimuth velocity
        '''
        refocused_targetchip = []
        for i in range(self.bbox_array.shape[0]):
            idx = np.argmin(entropy_nes_array[i,:])
            refocused_targetchip.append(nested_array[i][idx])

        return refocused_targetchip

    def COGEstimation_vessel(self, refocused_targetchip, vc_azimuth,heading_angle,maxThres=1):

        '''
        refocused_targetchip is the array contained the refocused target chip
        vc_azimuth is the array contained the azimuth velocity of each target chip
        heading_angle could be acquried in swlee code.
        '''
        targetBBox = self.bbox_array
        SOGEstim = np.zeros((len(vc_azimuth),1))
        COGEstim = np.zeros((len(vc_azimuth),1))
        targetBBoxExp=np.zeros((len(vc_azimuth),6))

        for num0 in range(len(refocused_targetchip)):
            image=refocused_targetchip[num0];image = np.abs(image)**2
            
            max_value = np.max(image)
            image = np.where(image > max_value/maxThres, max_value/maxThres, image)
            
            # Resize image w.r.t. azimuth and range resolution
            #azimuthNum,rangeNum=np.shape(image)    
            #image = cv2.resize(image, (round(rangeNum*rngresolution),round(azimuthNum*azresolution)), interpolation=cv2.INTER_LINEAR)
          
            # Perform radon transform and select 3 candidates
            sinogram = radon(image)
            dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]    
            axis_candidate = peak_local_max(sinogram, min_distance=3, num_peaks=3, threshold_rel=0.3)
            
            the_sino = np.zeros_like(sinogram)
            the_sino[axis_candidate[:, 0], axis_candidate[:, 1]] = sinogram[axis_candidate[:, 0], axis_candidate[:, 1]]
            
            i_img = iradon_sart(the_sino)
        
            # Among 3 selected axis, confirm whether sidelobe exists
            COGcandidate=axis_candidate[:,1]
            resultTF, SidelobeAngles =findCOGdifference(COGcandidate,20)
            
            if resultTF is False:
                COG=np.mean(COGcandidate)
            else:
                try:
                    COG=average_two_closest(COGcandidate[0],COGcandidate[1],COGcandidate[2])
                except:
                    COG=np.mean(COGcandidate)

            #COGcandidate is for extracting canditate three peak COG

            # Transform vx into SOG using estimated heading
            COGnew=round(COG)-heading_angle
            
            if COGnew<0:
                COGnew=COGnew+180
            elif COGnew>180:
                COGnew=COGnew-180
                
            # use value of vx to detail COG from 0 to 360 
            if vc_azimuth[num0]<0:
                COGnew=COGnew+180
                
            
            SOGEstim[num0][0]=round(abs(vc_azimuth[num0]/math.cos(COG*3.1415/180)),2)
            COGEstim[num0][0]=COGnew
                        
            if SOGEstim[num0][0]>10:
                SOGEstim[num0][0]=10
        
        targetBBoxExp[:,0:4]=targetBBox[:,0:4]
        targetBBoxExp[:,4]=SOGEstim.ravel() 
        targetBBoxExp[:,5]=COGEstim.ravel()         

        return COG
