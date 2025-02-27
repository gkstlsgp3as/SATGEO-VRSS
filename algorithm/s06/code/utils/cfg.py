
import os
from easydict import EasyDict

# Define base directory (optional, useful for dynamic path management)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration dictionary
Cfg = EasyDict()

# Input data paths
Cfg.time_interval = 10.0  # Time interval for prediction (minutes)
Cfg.time_end = 60.0  # End of time for prediction (minutes)