from models.Umbra_velocity_estimation import *
import pandas as pd
import json
import argparse
from tifffile import imwrite
from pathlib import Path
from utils.cfg import Cfg

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_slc_dir', 
        type=str, 
        default='../data/input/', 
        help='the file path that SICD and METADATA are stored'
    )
    parser.add_argument(
        '--start_az', 
        type=int, 
        default='start_Azimuthvel', 
        help='start azimuth velocity'
    )
    parser.add_argument(
        '--end_az', 
        type=int, 
        default='end_Azimuthvel', 
        help='end azimuth velocity'
    )
    parser.add_argument(
        '--spacing', 
        type=float, 
        default='0.5', 
        help='spacing for start to end velocity'
    )
    parser.add_argument(
        '--bbox', 
        type=str, 
        default="../data/input/bbox.txt", 
        help='bounding box file path'
    )
    parser.add_argument(
        '--output_json_file', 
        type=str, 
        default='../data/output/output.json', 
        help='save json path'
    )
    parser.add_argument(
        '-C', "--chip_dir", 
        type=str, 
        default='../output/chips/', 
        help="Path to chip images"
    )

    # Parse arguments
    args = parser.parse_args()
    
    return args

def process(): 
    pass

if __name__ == '__main__':
    
    args = get_args()
    print(args);print("start estimation velocity!")

    Velocticy_est_UMBRA_instance = Velocticy_est_UMBRA(args.input_slc_dir, start_Azimuthvel=Cfg.start_az, end_Azimuthvel=Cfg.end_az, spacing=Cfg.spacing, bbox=args.bbox)
    data_SLC_2DBox, SlantRangeBox, f_azimuth, f_range = Velocticy_est_UMBRA_instance.patchmaker() ## extracting RoI in SLC data
    nested_array = Velocticy_est_UMBRA_instance.velocity_estimation(data_SLC_2DBox, SlantRangeBox, f_azimuth, f_range) # adding velocity phase to SLC
    
    entropy_nes_array = Velocticy_est_UMBRA_instance.calculate_entropy(nested_array) # calculating entropy of each target chip
    vc_azimuth = Velocticy_est_UMBRA_instance.calculate_velocity(entropy_nes_array) # find minumum entropy for each target chip
    refocused_targetchip = Velocticy_est_UMBRA_instance.extract_rftarget(nested_array, entropy_nes_array)# refocusing target chip has refocused target array
    # make total array from bbox_array and azimuth velocity
    total_array = Velocticy_est_UMBRA_instance.bbox_array;total_array = total_array.astype(int) 
    total_array = np.column_stack((total_array, vc_azimuth))

    #check the chip_dir and save the refocused target chips to the chip_dir
    chip_dir = Path(args.chip_dir)  # increment run
    chip_dir.mkdir(parents=True, exist_ok=True)

    [imwrite(chip_dir / f'image_{i}.tif', np.abs(img)) for i, img in enumerate(refocused_targetchip)]


    # Initialize an empty list to store the JSON-friendly dictionary data
    json_data = []

    # Loop through each row of the array
    for row in total_array:
        bbox = row[0:4]  # First four elements are for bbox
        class_id = row[4]  # Fifth element is the class
        velocity = row[5]  # Last element is the velocity
        
        # Create a dictionary for each row
        item = {
            "bbox": {
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3])
            },
            "class": int(class_id),
            "velocity": float(velocity)
        }
        
        # Append the dictionary to the json_data list
        json_data.append(item)

    # Convert the list of dictionaries to JSON format
    json_output = json.dumps(json_data, indent=4)

    with open(args.output_json_file, 'w') as json_file:
        json_file.write(json_output)
    
    print('end')
    