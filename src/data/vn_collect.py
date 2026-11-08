"""Thu thập tin bán nhà ở TP.HCM từ Nhà Tốt (dữ liệu công khai) cho mục đích học tập.

Nguyên tắc:
- Chỉ lấy tin BÁN (st=s), mục "Nhà ở" (cg=1020), khu vực TP.HCM (region_v2=13000).
- Nghỉ giữa các request để không gây tải cho máy chủ.
- Chỉ giữ thông tin về căn nhà. Không lưu tên, mã tài khoản, ảnh, số nhà
  hay phần mô tả (thường chứa số điện thoại) của người đăng.

Chạy: python -m src.data.vn_collect --max-pages 200
Kết quả: data/raw/vn_hcm_listings.csv và reports/vn_collection.json
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.utils.paths import ROOT, REPORTS

API = 'https://gateway.chotot.com/v1/public/ad-listing'
PARAMS = {'cg': 1020, 'region_v2': 13000, 'st': 's', 'limit': 50}
HEADERS = {'User-Agent': 'Mozilla/5.0 (student data science project; non-commercial)'}
OUTPUT = ROOT / 'data/raw/vn_hcm_listings.csv'

# Các trường về căn nhà được giữ lại. Mọi trường khác (account_*, full_name,
# seller_info, avatar, images, body, street_number...) bị bỏ ngay khi tải về.
KEEP = ['ad_id', 'list_time', 'subject', 'price', 'size', 'living_size', 'width', 'length',
        'area_name', 'ward_name', 'street_name', 'floors', 'rooms', 'toilets',
        'house_type', 'direction', 'property_legal_document', 'furnishing_sell',
        'is_main_street', 'latitude', 'longitude', 'region_name', 'type']

# Nhãn chữ của các trường mã hoá, lấy từ feature_params của chính tin đăng.
LABELS = {'house_type': 'house_type_label', 'direction': 'direction_label',
          'property_legal_document': 'legal_label', 'furnishing_sell': 'furnishing_label',
          'pty_characteristics': 'characteristics_label'}

PHONE = re.compile(r'(?:\+?84|0)(?:[\s.\-]?\d){8,10}')


def to_row(ad):
    row = {k: ad.get(k) for k in KEEP}
    # Tiêu đề đôi khi có số điện thoại: che lại.
    if isinstance(row['subject'], str):
        row['subject'] = PHONE.sub('[SĐT]', row['subject'])
    # Toạ độ làm tròn ~100 m: đủ cho phân tích theo khu vực, không chỉ đúng một căn.
    for k in ('latitude', 'longitude'):
        if isinstance(row[k], (int, float)):
            row[k] = round(row[k], 3)
    items = (ad.get('feature_params') or {}).get('compare', {}).get('items', [])
    for item in items:
        if item.get('id') in LABELS:
            row[LABELS[item['id']]] = item.get('value')
    return row


def collect(max_pages=200, delay=1.5):
    session = requests.Session()
    rows, seen, errors = [], set(), []
    total = None
    for page in range(max_pages):
        offset = page * PARAMS['limit']
        try:
            resp = session.get(API, params={**PARAMS, 'o': offset}, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # thử lại một lần sau khi nghỉ lâu hơn
            errors.append({'offset': offset, 'error': repr(exc)})
            time.sleep(delay * 5)
            continue
        total = data.get('total', total)
        ads = data.get('ads', [])
        if not ads:
            break
        for ad in ads:
            if ad.get('ad_id') not in seen:
                seen.add(ad.get('ad_id'))
                rows.append(to_row(ad))
        if (page + 1) % 20 == 0:
            print(f'Trang {page + 1}/{max_pages}: {len(rows)} tin', flush=True)
            pd.DataFrame(rows).to_csv(OUTPUT, index=False, encoding='utf-8-sig')
        time.sleep(delay)
    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    info = {'source': 'Nhà Tốt (nhatot.com), API công khai ad-listing',
            'query': PARAMS, 'collected_at': datetime.now(timezone.utc).isoformat(),
            'pages_requested': max_pages, 'reported_total': total,
            'rows_unique': len(df), 'errors': errors,
            'privacy': 'Đã bỏ thông tin người đăng; che số điện thoại trong tiêu đề; toạ độ làm tròn 3 chữ số.',
            'purpose': 'Đồ án môn Khoa học Dữ liệu, phi thương mại, không phát tán dữ liệu.'}
    (REPORTS / 'vn_collection.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pages', type=int, default=200)
    parser.add_argument('--delay', type=float, default=1.5)
    args = parser.parse_args()
    result = collect(args.max_pages, args.delay)
    print(f'Đã lưu {len(result)} tin vào {OUTPUT}')
