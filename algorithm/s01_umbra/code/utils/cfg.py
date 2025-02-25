import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.start_az = -30
Cfg.end_az = 30
Cfg.spacing = 0.1

Cfg.img_size = [640]

Cfg.conf_thres = 0.25
Cfg.iou_thres = 0.45
Cfg.max_det = 1000

Cfg.half = True
Cfg.save_img = True
Cfg.output_format = 1 # txt