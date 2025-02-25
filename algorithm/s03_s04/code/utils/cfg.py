
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

# input data path
Cfg.SARname='W:/ship_ais/MFMS/remove_known_facilities/input/S1A_IW_GRDH_1SDV_20231209T092333_20231209T092402_051576_0639F8_652A.tif'
Cfg.SARmetaname='W:/ship_ais/MFMS/remove_known_facilities/input/s1a-iw-grd-vv-20231209t092333-20231209t092402-051576-0639f8-001.xml'
Cfg.AISname='W:/ship_ais/MFMS/remove_known_facilities/input/P_AIS_20231209_084009__20231209_094009.csv'
Cfg.SARvesselname='W:/ship_ais/MFMS/remove_known_facilities/input/MarineFacilityDet.csv'
