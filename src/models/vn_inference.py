"""Tải mô hình giá nhà TP.HCM và dự đoán từ dữ liệu người dùng nhập."""
import json

import joblib
import numpy as np
import pandas as pd

from src.utils.paths import ROOT

MODEL_PATH = ROOT / 'artifacts/vn_house_price_model.joblib'
META_PATH = ROOT / 'artifacts/vn_model_metadata.json'


def load_vn_model():
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    return joblib.load(MODEL_PATH), meta


def build_row(values, meta):
    """Tạo một dòng đặc trưng; toạ độ lấy theo tâm quận vì người dùng không biết toạ độ."""
    if values['district'] not in meta['districts']:
        raise ValueError('Quận/huyện không có trong dữ liệu huấn luyện.')
    if not values['size'] or values['size'] <= 0:
        raise ValueError('Diện tích phải lớn hơn 0.')
    center = meta['district_centers'].get(values['district'], {})
    row = {**values, 'latitude': center.get('latitude', np.nan), 'longitude': center.get('longitude', np.nan)}
    return pd.DataFrame([row])[meta['features']]


def out_of_range(values, meta):
    """Các thuộc tính nằm ngoài khoảng 1%–99% của tập Train."""
    return [c for c, (lo, hi) in meta['input_ranges'].items()
            if values.get(c) is not None and not lo <= values[c] <= hi]


def predict_vn(values, model, meta):
    pred = float(model.predict(build_row(values, meta))[0])
    if not np.isfinite(pred) or pred <= 0:
        raise ValueError('Mô hình trả về kết quả không hợp lệ.')
    return pred
