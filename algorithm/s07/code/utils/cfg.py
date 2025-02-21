# -*- coding: utf-8 -*-
"""
@Time          : 2025/01/09 00:00
@Author        : Hyunsun Lee
@File          : cfg.py
@Noice         : 
@Description   : Configuration file for ship classification tasks.
@How to use    : Import algorithm_info, training_params, and output_params from cfg.py.

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
"""

import os
from easydict import EasyDict
import torch
import numpy as np

Cfg = EasyDict()
Cfg.MODEL_PATH = "C:/Users/user/Documents/vesseltrack/VRSS_git/SATGEO-python-template_hyunsun/app/algorithm/s07/models/weights"
Cfg.MODEL_NAME = 'model_SAIS_ALL_3857_epoch_62.pt'
Cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Cfg.ENCODER_LENGTH = 12
Cfg.DECODER_LENGTH = 6
Cfg.OUTPUT_SIZE = 5
Cfg.INPUT_SIZE = 5
Cfg.HIDDEN_SIZE = 128
Cfg.MAX_VALUES = np.array([6.0000000e+05, 1.5892023e+05, 2.1707131e+05, 5.1589398e+02, 5.2627698e+02])
Cfg.MIN_VALUES = np.array([ 0, -1.4327106e+05, -1.2994346e+05, -3.8387001e+01, -1.0402200e+02])
Cfg.PREVIOUS_TIME = 2 # hours
