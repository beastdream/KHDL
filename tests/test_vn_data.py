"""Kiểm tra các quy tắc làm sạch dữ liệu TP.HCM trên dữ liệu giả lập."""
import pandas as pd
import pytest

from src.data.vn_clean import clean, FEATURES, TARGET
from src.data.vn_collect import to_row


def listing(**overrides):
    row = {'ad_id': 1, 'list_time': 1794146566911, 'subject': 'Bán nhà', 'price': 6e9, 'size': 50,
           'living_size': None, 'width': 4, 'length': 12.5, 'area_name': 'Quận 7', 'ward_name': 'Phường Tân Phú',
           'street_name': None, 'floors': 3, 'rooms': 3, 'toilets': 3, 'house_type': 1, 'direction': 2,
           'property_legal_document': 1, 'furnishing_sell': 3, 'is_main_street': True,
           'latitude': 10.73, 'longitude': 106.72, 'region_name': 'Tp Hồ Chí Minh', 'type': 's',
           'house_type_label': 'Nhà ngõ, hẻm', 'legal_label': 'Đã có sổ', 'direction_label': 'Hướng Tây',
           'furnishing_label': 'Hoàn thiện cơ bản', 'characteristics_label': 'Hẻm xe hơi, Nhà nở hậu'}
    row.update(overrides)
    return row


def test_clean_rules():
    raw = pd.DataFrame([
        listing(ad_id=1),
        listing(ad_id=2, type='u', price=12e6),          # tin cho thuê
        listing(ad_id=3, price=None),                    # thiếu giá
        listing(ad_id=4, price=60e9, size=20),           # 3 tỷ/m²: nhập sai
        listing(ad_id=5),                                # trùng nội dung với tin 1
        listing(ad_id=6, size=80, width=20, length=4),   # mặt tiền/chiều dài bị đảo
        listing(ad_id=7, size=70, floors=99),            # số tầng vô lý -> thiếu
    ])
    df, funnel, notes = clean(raw)
    assert sorted(df.ad_id) == [1, 6, 7]
    assert funnel[-1]['Còn lại'] == 3
    row6 = df.set_index('ad_id').loc[6]
    assert (row6.width, row6.length) == (4, 20)
    assert pd.isna(df.set_index('ad_id').loc[7, 'floors'])
    assert df.set_index('ad_id').loc[1, TARGET] == pytest.approx(6.0)
    assert df.loc[0, 'car_alley'] == 1 and df.loc[0, 'expanding_back'] == 1
    assert set(FEATURES) <= set(df.columns)


def test_collector_drops_personal_data():
    ad = {**listing(), 'account_name': 'Nguyễn Văn A', 'account_id': 99, 'seller_info': {'phone': 'x'},
          'body': 'Gọi 0909123456', 'street_number': '12', 'latitude': 10.123456,
          'subject': 'Bán nhà gọi 0909 123 456'}
    row = to_row(ad)
    for key in ('account_name', 'account_id', 'seller_info', 'body', 'street_number'):
        assert key not in row
    assert '0909' not in row['subject']
    assert row['latitude'] == 10.123
