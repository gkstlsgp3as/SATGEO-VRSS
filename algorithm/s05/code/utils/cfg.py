# -*- coding: utf-8 -*-
"""
@Time          : 2024/12/18 00:00
@Author        : Shinhye Han
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


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.epsg = "4326"             # EPSG code for geospatial outputs
Cfg.color_map = {            # Color mapping for visualizations
        "Cargo": (255, 0, 0),    # Red
        "Fishing": (0, 255, 0),  # Green
        "Tanker": (0, 0, 255),   # Blue
        "Sailing": (255, 255, 0),  # Yellow
        "TugTow": (255, 0, 255)   # Magenta
    }

Cfg.classes = ['Cargo', 'Fishing', 'Sailing', 'Tanker', 'TugTow'] 
Cfg.img_size = 224
