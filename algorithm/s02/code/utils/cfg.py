import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.start_az = -30
Cfg.end_az = 30
Cfg.spacing = 0.1