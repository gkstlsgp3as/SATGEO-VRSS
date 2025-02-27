# -*- coding: utf-8 -*-
"""
@Time          : 2024/12/18 00:00
@Author        : Shinhye Han
@File          : dataset.py
@Noice         : 
@Description   : Dataset definitions for ship classification tasks, including platform information.
@How to use    : Import the `ShipClassificationDataset` class to load and preprocess datasets.

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch
from torchvision.transforms import Compose

FUSAR_MEAN = 2.677

def read_platform(meta_file: str) -> List[str]:
    """
    Reads the meta file and returns the platform information.

    Args:
        meta_file (str): Path to the metadata file.

    Returns:
        str: name of the platform mentioned in the metadata.
    """
    with open(meta_file, 'r') as meta:
        meta_data = json.load(meta)
    return meta_data['Platform']

class ShipClassificationDataset(Dataset):
    def __init__(self, path: str, transform: Compose, classes: List[str]) -> None:
        """
        Initializes the ShipClassificationDataset with image paths, transformations, and class labels.

        Args:
            path (str): Directory path where images are stored.
            transform (Compose): Transformations to be applied to the images.
            classes (List[str]): List of class labels.
        """
        self.classes = classes
        self.path = path
        self.img_files = os.listdir(self.path); 
        #self.img_files = [im for im in self.img_files if im.split('_')[0] in self.classes]
        #print(self.img_files)
        self.imgs = []
        self.img_scales = []
        for im in self.img_files:
            img = cv2.imread(str(self.path)+'/'+im, cv2.IMREAD_UNCHANGED)
            if img.max != 0:
                img_norm = np.array(255 * ((img - img.min()) / (img.max() - img.min())), np.uint8)
            t, t_otsu = cv2.threshold(img_norm, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
            thres_m = (t_otsu/255) * img_norm
            self.img_scales.append(thres_m.mean())
            self.imgs.append(img)
        
        DATA_MEAN = np.array(self.img_scales).mean() 
        for i, img in enumerate(self.imgs):
            img = img / (DATA_MEAN / FUSAR_MEAN)
            
            self.imgs[i] = Image.fromarray(img)
            

        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Any:
        """
        Returns the image and label at the specified index.

        Args:
            idx (int): Index of the image and label to retrieve.

        Returns:
            Any: Transformed image and corresponding label as a tuple.
        """
        x = self.transform(self.imgs[idx])
        y = self.img_files[idx].split('_')[0]
        
        return x