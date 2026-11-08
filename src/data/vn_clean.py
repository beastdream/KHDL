"""Làm sạch dữ liệu nhà ở TP.HCM và chia Train / Validation / Test.

Mỗi bước lọc được ghi lại số dòng bị loại vào "phễu dữ liệu" (reports/vn_data_funnel.md).
Các ngưỡng lọc là quy tắc nghiệp vụ cố định (không học từ dữ liệu), nên áp dụng
cho toàn bộ dữ liệu trước khi chia mà không gây rò rỉ thông tin từ tập Test.

Chạy: python -m src.data.vn_clean
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.paths import ROOT, REPORTS, PROCESSED, write_json

RAW = ROOT / 'data/raw/vn_hcm_listings.csv'
SEED = 42

TARGET = 'price_billion'          # Giá bán, đơn vị tỷ VNĐ
NUMERIC = ['size', 'floors', 'rooms', 'toilets', 'width', 'length', 'latitude', 'longitude']
CATEGORICAL = ['district', 'house_type', 'legal', 'direction', 'furnishing']
# is_main_street của nguồn không khớp với loại nhà "mặt phố" nên không dùng; house_type đã chứa thông tin này.
BINARY = ['car_alley', 'expanding_back']
FEATURES = NUMERIC + CATEGORICAL + BINARY

# Khung toạ độ thô của TP.HCM (cũ) để bỏ toạ độ nhập sai.
LAT_RANGE, LON_RANGE = (10.3, 11.2), (106.3, 107.1)


def _step(funnel, name, before, after):
    funnel.append({'Bước': name, 'Còn lại': int(after), 'Bị loại': int(before - after)})


def clean(raw):
    funnel = []
    df = raw.copy()
    _step(funnel, 'Dữ liệu thu thập (tin không trùng mã)', len(df), len(df))

    n = len(df); df = df[(df['type'] == 's') & (df['region_name'].astype(str).str.contains('Hồ Chí Minh'))]
    _step(funnel, 'Chỉ giữ tin BÁN tại TP.HCM', n, len(df))

    n = len(df); df = df[df['price'].notna() & df['size'].notna() & df['area_name'].notna()]
    _step(funnel, 'Bỏ tin thiếu giá, diện tích hoặc quận', n, len(df))

    # Quy tắc nghiệp vụ: nhà ở TP.HCM giá 0,3–300 tỷ, diện tích đất 10–1.000 m².
    n = len(df); df = df[df['price'].between(3e8, 3e11) & df['size'].between(10, 1000)]
    _step(funnel, 'Bỏ giá ngoài 0,3–300 tỷ hoặc diện tích ngoài 10–1.000 m²', n, len(df))

    # Giá/m² ngoài 15–1.500 triệu gần như chắc chắn là nhập sai đơn vị hoặc sai diện tích.
    ppm2 = df['price'] / df['size'] / 1e6
    n = len(df); df = df[ppm2.between(15, 1500)]
    _step(funnel, 'Bỏ giá/m² ngoài 15–1.500 triệu (nhập sai đơn vị)', n, len(df))

    df = df.copy()
    # Mặt tiền phải ngắn hơn chiều dài; nếu bị nhập ngược thì đổi chỗ.
    swap = df['width'].notna() & df['length'].notna() & (df['width'] > df['length'])
    df.loc[swap, ['width', 'length']] = df.loc[swap, ['length', 'width']].values
    swapped = int(swap.sum())

    # Giá trị vô lý của thuộc tính phụ được coi là thiếu (để mô hình tự điền), không bỏ cả dòng.
    bounds = {'floors': (1, 15), 'rooms': (1, 30), 'toilets': (1, 30), 'width': (1, 50), 'length': (2, 200)}
    nulled = {}
    for col, (lo, hi) in bounds.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        nulled[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    bad_geo = ~(df['latitude'].between(*LAT_RANGE) & df['longitude'].between(*LON_RANGE))
    nulled['latitude/longitude'] = int((bad_geo & df['latitude'].notna()).sum())
    df.loc[bad_geo, ['latitude', 'longitude']] = np.nan

    # Một căn có thể bị nhiều môi giới đăng lại: cùng phường, diện tích, giá, số tầng, phòng ngủ.
    key = ['area_name', 'ward_name', 'size', 'price', 'floors', 'rooms']
    n = len(df); df = df.drop_duplicates(subset=key, keep='first')
    _step(funnel, 'Bỏ tin đăng trùng (cùng phường, diện tích, giá, tầng, phòng)', n, len(df))

    traits = df.get('characteristics_label', pd.Series(index=df.index, dtype=object)).fillna('').astype(str)
    out = pd.DataFrame({
        'ad_id': df['ad_id'].astype('int64'),
        'list_date': pd.to_datetime(df['list_time'], unit='ms').dt.date.astype(str),
        'size': df['size'].astype(float),
        'floors': df['floors'], 'rooms': df['rooms'], 'toilets': df['toilets'],
        'width': df['width'], 'length': df['length'],
        'latitude': df['latitude'], 'longitude': df['longitude'],
        'district': df['area_name'].str.strip(),
        'house_type': df.get('house_type_label'),
        'legal': df.get('legal_label'),
        'direction': df.get('direction_label'),
        'furnishing': df.get('furnishing_label'),
        'car_alley': traits.str.contains('xe hơi', case=False).astype(int),
        'expanding_back': traits.str.contains('nở hậu', case=False).astype(int),
        TARGET: df['price'] / 1e9,
    }).reset_index(drop=True)
    notes = {'width_length_swapped': swapped, 'set_to_missing': nulled}
    return out, funnel, notes


def split(df):
    train, rest = train_test_split(df, test_size=0.30, random_state=SEED)
    valid, test = train_test_split(rest, test_size=0.50, random_state=SEED)
    return {'train': train, 'valid': valid, 'test': test}


def write_report(df, funnel, notes, parts):
    missing = (df[FEATURES].isna().mean() * 100).round(1)
    lines = ['# Phễu dữ liệu nhà ở TP.HCM (Nhà Tốt)', '',
             'Nguồn: tin bán nhà ở công khai trên nhatot.com, thu thập cho mục đích học tập '
             '(xem reports/vn_collection.json). Đã bỏ thông tin người đăng.', '',
             '| Bước | Còn lại | Bị loại |', '|---|---:|---:|']
    lines += [f"| {r['Bước']} | {r['Còn lại']:,} | {r['Bị loại']:,} |" for r in funnel]
    lines += ['', f"Đổi chỗ mặt tiền/chiều dài bị nhập ngược: {notes['width_length_swapped']} tin.",
              'Giá trị vô lý được chuyển thành "thiếu" (không bỏ dòng): ' +
              ', '.join(f'{k}: {v}' for k, v in notes['set_to_missing'].items()) + '.', '',
              '## Tỷ lệ thiếu theo cột (%)', '', '| Cột | Thiếu % |', '|---|---:|']
    lines += [f'| {c} | {v} |' for c, v in missing.items()]
    lines += ['', '## Chia dữ liệu (seed 42)', '', '| Tập | Số dòng |', '|---|---:|']
    lines += [f'| {k} | {len(v):,} |' for k, v in parts.items()]
    p = df[TARGET]
    lines += ['', '## Giá bán (tỷ VNĐ)', '',
              f'Trung vị {p.median():.2f}; trung bình {p.mean():.2f}; nhỏ nhất {p.min():.2f}; '
              f'lớn nhất {p.max():.2f}; độ lệch (skew) {p.skew():.2f}.']
    (REPORTS / 'vn_data_funnel.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    raw = pd.read_csv(RAW, encoding='utf-8-sig')
    df, funnel, notes = clean(raw)
    parts = split(df)
    for name, part in parts.items():
        part.to_csv(PROCESSED / f'vn_{name}.csv', index=False, encoding='utf-8')
    write_report(df, funnel, notes, parts)
    write_json(REPORTS / 'vn_data_split.json', {
        'seed': SEED, 'rows': len(df), 'features': FEATURES, 'target': TARGET,
        'funnel': funnel, 'notes': notes, 'counts': {k: len(v) for k, v in parts.items()}})
    print(json.dumps(funnel, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
