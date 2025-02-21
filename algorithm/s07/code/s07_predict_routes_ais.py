import argparse
from utils.cfg import Cfg
import torch
from pyproj import Proj, Transformer
import pyproj

from app.service.ship_prediction_route_service  import get_ship_data, save_ship_prediction_route
from sqlalchemy.orm import Session
from models.models import Encoder, Decoder, Seq2Seq
import numpy as np

from utils.data_preprocessing import TestLoader
import datetime as dt
import pandas as pd
import time
import logging
## TODO: 관련 DB 모델 및 서비스 import 

class ConvertAxis:
    def __init__(self):
        self.projection_4326 = pyproj.CRS("EPSG:4326")
        self.projection_3857 = pyproj.CRS("EPSG:3857")
        self.transformer_4326_to_3857 = pyproj.Transformer.from_crs(
            self.projection_4326, self.projection_3857, always_xy=True
        )
        self.transformer_3857_to_4326 = pyproj.Transformer.from_crs(
            self.projection_3857, self.projection_4326, always_xy=True
        )

    def epsg4326_to_3857(self, lon_data, lat_data):
        transformed_coords = [self.transformer_4326_to_3857.transform(lng, lat) for lng, lat in zip(lon_data, lat_data)]
        lat_3857, lon_3857 = zip(*transformed_coords)
        return list(lon_3857), list(lat_3857)

    def epsg3857_to_4326(self, lon_data, lat_data):
        transformed_coords = [self.transformer_3857_to_4326.transform(lng, lat) for lng, lat in zip(lon_data, lat_data)]
        lat_4326, lon_4326 = zip(*transformed_coords)
        return list(lon_4326), list(lat_4326)


def load_model() -> torch.nn.Module:
    torch.cuda.empty_cache()

    checkpoint_path = f'{Cfg.MODEL_PATH}/{Cfg.MODEL_NAME}'
    output_size = Cfg.OUTPUT_SIZE

    model = Seq2Seq(
        input_size=Cfg.INPUT_SIZE,
        hidden_size=Cfg.HIDDEN_SIZE,
        output_size=output_size
    ).to(Cfg.DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=Cfg.DEVICE)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(state_dict)
    return model


def inference(model: torch.nn.Module, source_seq: np.ndarray) -> np.ndarray:
    """
    Performs inference on a given sequence using the loaded model.

    Parameters:
    - model: The loaded PyTorch model for inference.
    - source_seq: A numpy array of input sequences for model prediction.

    Returns:
    - A numpy array containing the model's predictions.
    """
    
    print('Inference')
    device = next(model.parameters()).device
    source_seq = torch.tensor(source_seq, dtype=torch.float32).to(device)
    print('source_seq: ', source_seq.shape)
    model.eval()

    with torch.no_grad():
        encoder_inputs = source_seq[:, :Cfg.ENCODER_LENGTH, :]
        decoder_input = source_seq[:, Cfg.ENCODER_LENGTH - 1:Cfg.ENCODER_LENGTH, :]
        predictions = model.inference(
            encoder_inputs, decoder_input, Cfg.DECODER_LENGTH, Cfg.OUTPUT_SIZE
        )
        print('predictions: ', predictions.shape)

    return predictions.cpu().numpy()


class Normalize:
    def __init__(self):
        self.min_values = Cfg.MIN_VALUES
        self.max_values = Cfg.MAX_VALUES
        self.date_value = self.max_values[0]
        self.denominator = self.max_values - self.min_values

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        normalized_data = (data[:, :, 1:5] - self.min_values[1:]) / self.denominator[1:]
        data[:, :, 1:5] = normalized_data
        data[:, :, 0] /= self.date_value
        return data

    def rescale_data(self, data: np.ndarray) -> np.ndarray:
        data[:, 0] *= self.date_value
        rescaled_data = data[:, 1:3] * self.denominator[1:3] + self.min_values[1:3]
        data[:, 1:] = rescaled_data
        return data


def set_dataset(data: pd.DataFrame):
    seq_length = Cfg.ENCODER_LENGTH + Cfg.DECODER_LENGTH
    test_loader = TestLoader()
    test_loader.load_test_trajectory(data, seq_length)
    
    if len(test_loader.trajectory) < 1:
        print('Trajectory length is too short. Less than 1.')
        del test_loader
        return None
    
    norm = Normalize()
    test_loader.trajectory = norm.normalize_data(test_loader.trajectory)

    seq_encoder = test_loader.trajectory
    source_seq = seq_encoder[:, :Cfg.ENCODER_LENGTH, :5]
    source_values = seq_encoder[:, :Cfg.ENCODER_LENGTH, 5:]

    return seq_encoder, source_values


def save_predictions(pred_values: np.ndarray, source_values: np.ndarray):
    time_source = source_values[:, 0].tolist()
    lng_source = source_values[:, 1].tolist()
    lat_source = source_values[:, 2].tolist()

    delta_time, delta_lng, delta_lat = pred_values[:, 0], pred_values[:, 1], pred_values[:, 2]
    time_pred, lng_pred, lat_pred = [], [], []

    lng, lat = lng_source[-1], lat_source[-1]
    time_i = time_source[-1]

    for d_t, dlng, dlat in zip(delta_time, delta_lng, delta_lat):
        time_i += d_t
        lng += dlng
        lat += dlat
        time_pred.append(time_i)
        lng_pred.append(lng)
        lat_pred.append(lat)

    time_source_utc = [dt.datetime.fromtimestamp(ts / 1000.0) for ts in time_source]
    time_pred_utc = [dt.datetime.fromtimestamp(ts / 1000.0) for ts in time_pred]

    axis = ConvertAxis()
    lat_pred, lng_pred = axis.epsg3857_to_4326(lng_pred, lat_pred)
    lat_source, lng_source = axis.epsg3857_to_4326(lng_source, lat_source)

    source_data = {'datetime': time_source_utc, 'lon': lng_source, 'lat': lat_source}
    pred_data = {'datetime': time_pred_utc, 'lon': lng_pred, 'lat': lat_pred}

    return source_data, pred_data

def process(db: Session, current_time:pd.Timestamp, target_time: pd.datetime, mmsi: str):
    start_time = time.time()
    
    data = get_ship_data(db, mmsi, target_time)


    source_length = Cfg.ENCODER_LENGTH + Cfg.DECODER_LENGTH
    model = load_model()

    seq_encoder, source_values = set_dataset(data)
    # pred_seq = inference(model, seq_encoder[:, :source_length, :5])
    pred_seq = inference(model, seq_encoder[0:1, :source_length, :5])
    pred_value = Normalize().rescale_data(pred_seq[0])

    source_data, predict_data = save_predictions(pred_value, source_values[0])
    print('source_data: ', source_data)
    print('predict_data: ', predict_data)
    finish_time = time.time()
    if not (len(predict_data['lon']) == len(predict_data['lat']) == len(predict_data['datetime'])):
        raise ValueError("lon, lat, datetime 리스트의 길이가 같아야 합니다.")

    predict_route = [
        {"lon": predict_data['lon'][i], "lat": predict_data['lat'][i], "datetime": predict_data['datetime'][i]}
        for i in range(len(predict_data['lon']))
    ]
    print('predict_route: ', predict_route)
    save_data = {
        'mmsi': mmsi,
        'request_time': current_time,
        'target_time': target_time,
        'target_lon': source_data['lon'][-1], # 추후에 시간 맞춰 수정
        'target_lat': source_data['lat'][-1], # 추후에 시간 맞춰 수정
        'arrive_lon': predict_data['lon'][-1], 
        'arrive_lat': predict_data['lat'][-1],
        'require_time': start_time - finish_time, # 추후에 시간 맞춰 수정
        'distance': 0.0, # 추후에 산정하여 추가
        'predict_route': predict_route,
        'predict_circle': 3.14  # 예시 값, 확률반경 알고리즘 추후 추가

    }

    save_ship_prediction_route(db, save_data)
    
    # Calculate and log the processing time
    processed_time = time.time() - start_time
    logging.info(f"Processed SAR image classification in {processed_time:.2f} seconds")

