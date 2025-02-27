# -*- coding: utf-8 -*-
'''
@Time          : 2024/12/18 00:00
@Author        : Shinhye Han
@File          : s05_classify_unidentified_ships.py
@Noice         : 
@Description   : Perform multi-class classification on unidentified ship chips from SAR.
@How to use    : python s05_classify_unidentified_ships.py --input_dir {image path} --meta_file {metafile path}

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import argparse
import time
from pathlib import Path
from typing import List
import logging

import pandas as pd
import torch
from torchvision import transforms

from utils.datasets import ShipClassificationDataset
from models.models import select_model
from sqlalchemy.orm import Session

## TODO: 관련 DB 모델 및 서비스 import 
from app.config.settings import settings
from app.service import sar_ship_unidentification_service
from utils.cfg import Cfg


def classify_unidentified_ships(chip_dir: str, meta_file: str, ) -> None:
    """
    Perform multi-class classification on ship images.

    Args:
        db (Session): db session to connect tables
        satellite_image_id (str): id to get satellite images path 
    """
    

    # Image Preprocessing
    img_transforms = transforms.Compose([
        transforms.Pad(padding=(Cfg.img_size, Cfg.img_size), fill=0),
        transforms.Resize(Cfg.img_size),
        transforms.CenterCrop(Cfg.img_size),
        transforms.ToTensor(),
        lambda x: (x > 1000) * 1000 + (x < 1000) * x,
        lambda x: 255 * (x - x.min()) / (x.max() - x.min()),
        lambda x: x / 255,
        lambda x: x.repeat(3, 1, 1),
    ])

    # Load Model and Dataset
    model = select_model(Cfg.classes, meta_file)
    dataset = ShipClassificationDataset(chip_dir, transform=img_transforms, classes=Cfg.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    init_time = time.time()

    # Perform Classification
    predictions = []
    labels = []

    for img, label in iter(dataset):
        labels.append(label)
        img = img.to(device).unsqueeze(0)
        y_pred, _ = model(img)

        _, top_pred = y_pred.topk(2, 1)
        predictions.append(top_pred[0][0].detach().cpu())
        
    return predictions


def process(db: Session, satellite_sar_image_id: str):
    """
    Processes SAR satellite images to identify and classify unidentified ships.
    
    This function queries the database for satellite SAR image data, classifies any unidentified ships
    in the image, and updates the database with the classification results.

    Parameters:
    - db: The SQLAlchemy session object for database interaction.
    - satellite_sar_image_id: The unique identifier for the SAR satellite image.

    The function performs the following steps:
    1. Retrieves SAR image data from the database.
    2. Converts the retrieved ORM objects into a dictionary.
    3. Classifies unidentified ships using a machine learning model.
    4. Updates the classification results back into the database.
    """
    # Define directories and file paths based on settings
    chip_dir = settings.S02_CHIP_PATH
    meta_file = settings.S05_META_FILE
    
    # Record start time for processing
    start_time = time.time()

    # Retrieve SAR ship identification data from the database
    data = sar_ship_unidentification_service.get_sar_ship_unidentification(db, satellite_sar_image_id)
    # Convert ORM objects to dictionary for easier manipulation
    data_dict = [record.__dict__ for record in data]

    # Classify unidentified ships using the specified directory and metadata file
    ship_class_predictions = classify_unidentified_ships(chip_dir, meta_file)
    
    # Update the prediction results in the dictionary for each record
    #for record in data_dict:
    #    record['prediction_ship_type'] = [Cfg.classes[pred] for pred in ship_class_predictions]
    data_dict['prediction_ship_type'] = [Cfg.classes[pred] for pred in ship_class_predictions]

    # Convert the list of dictionaries to a DataFrame for bulk database update
    df = pd.DataFrame(data_dict)

    # Update the database with the new ship classification predictions
    sar_ship_unidentification_service.bulk_upsert_sar_ship_unidentification_type(db, df)

    # Calculate and log the processing time
    processed_time = time.time() - start_time
    logging.info(f"Processed SAR image classification in {processed_time:.2f} seconds")
    