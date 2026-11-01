import numpy as np
import json
import tomllib
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.data.app_views import train_data, performance_data, eda_charts, filter_train
from src.data.week3_inputs import load_split
from src.models.evaluate_validation import metrics
from src.utils.paths import ROOT
from src.models.inference import load_model, predict
from src.data.schema import FEATURES
from src.ui.presentation import style_chart


def test_views_data_sources_and_filters():
    train = train_data()
    assert train.equals(load_split('train'))
    assert len(eda_charts(train)) == 7
    assert filter_train(train, [], (1,10), (0,1e9), (0,1e9)).empty
    subset = filter_train(train, ['NAmes'], (4,6), (100000,200000), (1000,2000))
    assert subset.Neighborhood.eq('NAmes').all()
    assert subset.OverallQual.between(4,6).all()
    assert subset.SalePrice.between(100000,200000).all()
    assert subset.GrLivArea.between(1000,2000).all()
    _, detail, meta = performance_data()
    np.testing.assert_array_equal(detail['Thực tế'], load_split('valid').SalePrice)
    assert np.isclose(metrics(detail['Thực tế'], detail['Dự đoán'])['MAE'], meta['validation_metrics']['MAE'])


def test_all_pages_and_prediction_form():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].value == f'{len(load_split("train")):,}'
    for page in ['Khám phá dữ liệu', 'Dự đoán giá nhà', 'Hiệu năng mô hình', 'Về dự án']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        if page == 'Khám phá dữ liệu':
            assert len(app.get('plotly_chart')) == 7
            app.multiselect[0].set_value([]).run()
            assert not app.exception and len(app.info) == 1
        elif page == 'Dự đoán giá nhà':
            app.button[0].click().run()
            assert not app.exception and 'USD' in app.metric[0].value
            assert len(app.success) == 1
        elif page == 'Hiệu năng mô hình':
            assert len(app.get('plotly_chart')) == 4


def test_form_values_reach_unchanged_model_and_result_card():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    app.sidebar.radio[0].set_value('Dự đoán giá nhà').run()
    assert len(app.number_input) == 6 and len(app.selectbox) == 1 and len(app.slider) == 1
    values = {'GrLivArea': 1800, 'Neighborhood': 'CollgCr', 'BedroomAbvGr': 4,
              'FullBath': 2, 'GarageCars': 2, 'YearBuilt': 2005, 'OverallQual': 7, 'TotalBsmtSF': 1100}
    labels = {'GrLivArea': 'Diện tích sinh hoạt (ft²)', 'BedroomAbvGr': 'Phòng ngủ trên mặt đất',
              'FullBath': 'Phòng tắm đầy đủ', 'GarageCars': 'Sức chứa garage (xe)',
              'YearBuilt': 'Năm xây dựng', 'TotalBsmtSF': 'Diện tích tầng hầm (ft²)'}
    for column, label in labels.items():
        next(w for w in app.number_input if w.label == label).set_value(values[column])
    app.selectbox[0].set_value(values['Neighborhood'])
    app.slider[0].set_value(values['OverallQual'])
    app.button[0].click().run()
    assert not app.exception and not app.error
    model, meta = load_model()
    expected = predict(pd.DataFrame([values], columns=FEATURES), model, meta)[0]
    assert app.metric[0].label == 'Giá nhà ước tính'
    assert app.metric[0].value == f'${expected:,.0f} USD'
    assert meta['inference_device'].upper() in app.success[0].value
    # Widget boundaries continue to be enforced; no schema or numerical bounds changed.
    area = next(w for w in app.number_input if w.label == labels['GrLivArea'])
    assert area.min == 1 and area.max == 20000


def test_chart_presentation_preserves_values():
    for figure in eda_charts(train_data()):
        data_before = figure.to_json()
        before = json.loads(data_before)['data']
        styled = style_chart(figure)
        # Presentation may recolor traces, but all data/encodings remain unchanged.
        after = json.loads(styled.to_json())['data']
        for traces in (before, after):
            for trace in traces:
                for property_name in ('marker', 'line'):
                    properties = trace.get(property_name, {})
                    if isinstance(properties.get('color'), str):
                        properties.pop('color')
                    if not properties:
                        trace.pop(property_name, None)
        assert after == before
        assert styled.layout.paper_bgcolor == '#FFFFFF'
        assert styled.layout.font.color == '#172033'
        assert styled.layout.autosize


def test_theme_config_and_palette_contrast():
    from streamlit import config
    settings = tomllib.loads((ROOT / '.streamlit/config.toml').read_text(encoding='utf-8'))
    theme = settings['theme']
    assert theme['base'] == 'light'
    assert theme['sidebar']['textColor'] == '#FFFFFF'
    assert theme['sidebar']['backgroundColor'] == '#152238'
    known = config.get_config_options()
    for name in theme:
        if name != 'sidebar':
            assert 'theme.' + name in known
    for name in theme['sidebar']:
        assert 'theme.sidebar.' + name in known
    def luminance(color):
        channels = [int(color[i:i+2], 16) / 255 for i in (1,3,5)]
        linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in channels]
        return sum(a*b for a,b in zip(linear, [.2126,.7152,.0722]))
    def contrast(a,b):
        light,dark = sorted([luminance(a),luminance(b)], reverse=True)
        return (light+.05)/(dark+.05)
    for foreground, background in [('#172033','#F5F7FB'),('#172033','#FFFFFF'),
                                    ('#64748B','#FFFFFF'),
                                    ('#FFFFFF','#152238'),('#FFFFFF','#243650'),
                                    ('#166534','#E4F5EA')]:
        assert contrast(foreground, background) >= 4.5
    assert contrast('#FFFFFF','#FF5252') >= 3  # 20px bold primary button text.
    assert contrast('#FFFFFF','#E63E45') >= 3
