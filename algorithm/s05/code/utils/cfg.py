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

Cfg.classes = ['Cargo', 'Fishing', 'Passenger', 'Tanker', 'TugTow', 'DiveVessel', 'Dredger', 'PortTender', 'Bulk', 'Container'] 
Cfg.img_size = 224
