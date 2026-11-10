"""HousePrice AI — local Streamlit application."""
from datetime import datetime
import json
from zoneinfo import ZoneInfo
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data.week3_inputs import install_data_guard
from src.data.app_views import train_data, filter_train, eda_charts, performance_data
from src.data.schema import FEATURES
from src.models.inference import load_model, predict
from src.models.vn_inference import load_vn_model, predict_vn, out_of_range
from src.ui.presentation import load_custom_css, show_chart
from src.utils.paths import REPORTS

st.set_page_config(page_title='HousePrice AI', page_icon='🏠', layout='wide', initial_sidebar_state='auto')


@st.cache_resource
def resources():
    install_data_guard()
    return load_model()


@st.cache_resource
def vn_resources():
    return load_vn_model()


@st.cache_data
def training_data():
    return train_data()


load_custom_css()
st.sidebar.title('🏠 HousePrice AI')
st.sidebar.caption('DỮ LIỆU · MÔ HÌNH · DỰ ĐOÁN')
page = st.sidebar.radio('Điều hướng', ['Tổng quan', 'Khám phá dữ liệu', 'Dự đoán giá nhà', 'Hiệu năng mô hình', 'Giá nhà TP.HCM', 'Về dự án'])
st.sidebar.divider()
st.sidebar.caption('Ames, Iowa (USD) · 8 thuộc tính\n\nTP.HCM (tỷ VNĐ) · 15 thuộc tính')
st.markdown('<div class="hp-intro"><h1>HousePrice AI</h1><p>Phân tích bất động sản · Ames, Iowa (USD) và TP.HCM (tỷ VNĐ)</p></div>', unsafe_allow_html=True)
try:
    model, metadata = resources()
    train = training_data()
except (OSError, ValueError, RuntimeError) as exc:
    st.error(f'Không thể tải mô hình hoặc dữ liệu: {exc}')
    st.info('Cần hoàn tất huấn luyện và kiểm tra artifact trước khi sử dụng ứng dụng.')
    st.stop()

if page == 'Tổng quan':
    st.header('Tổng quan thị trường trong tập Train')
    st.write('Khám phá giá bán lịch sử và ước tính giá nhà từ 8 đặc điểm bằng mô hình XGBoost.')
    columns = st.columns(3)
    for column, label, value in zip(columns, ['Bản ghi Train', 'Thuộc tính đầu vào', 'Giá trung bình'],
                                     [f'{len(train):,}', len(FEATURES), f'${train.SalePrice.mean():,.0f}']):
        column.metric(label, value)
    columns = st.columns(3)
    for column, label, value in zip(columns, ['Giá trung vị', 'Giá thấp nhất', 'Giá cao nhất'],
                                     [train.SalePrice.median(), train.SalePrice.min(), train.SalePrice.max()]):
        column.metric(label, f'${value:,.0f}')
    st.subheader('Từ đặc điểm căn nhà đến giá ước tính')
    st.write('XGBoost kết hợp nhiều cây quyết định để học quan hệ giữa diện tích, vị trí, chất lượng và giá bán. Các thống kê và biểu đồ dưới đây chỉ dùng Train.')
    st.info(f"Mô hình: {metadata['selected_model']} · Training: {metadata['training_device']} · Inference: {metadata['inference_device'].upper()}")
    for chart in eda_charts(train)[:2]:
        show_chart(chart)

elif page == 'Khám phá dữ liệu':
    st.header('Khám phá dữ liệu Train')
    st.caption('Lọc để tìm nhóm nhà tương đồng. Biểu đồ hỗ trợ hover, zoom và tải ảnh.')
    with st.expander('Bộ lọc dữ liệu', expanded=True):
        left, right = st.columns(2)
        neighborhoods = left.multiselect('Khu vực', metadata['neighborhoods'], default=metadata['neighborhoods'])
        quality = right.slider('Chất lượng', 1, 10, (1, 10))
        price = left.slider('Khoảng giá (USD)', int(train.SalePrice.min()), int(train.SalePrice.max()), (int(train.SalePrice.min()), int(train.SalePrice.max())))
        area = right.slider('Khoảng diện tích (ft²)', int(train.GrLivArea.min()), int(train.GrLivArea.max()), (int(train.GrLivArea.min()), int(train.GrLivArea.max())))
    subset = filter_train(train, neighborhoods, quality, price, area)
    if subset.empty:
        st.info('Không có căn nhà phù hợp. Hãy mở rộng bộ lọc.')
    else:
        a, b, c = st.columns(3)
        a.metric('Số nhà phù hợp', len(subset))
        b.metric('Giá trung vị', f'${subset.SalePrice.median():,.0f}')
        c.metric('Diện tích trung bình', f'{subset.GrLivArea.mean():,.0f} ft²')
        for chart in eda_charts(subset):
            show_chart(chart)
        st.caption('Tương quan không chứng minh quan hệ nhân quả. Nhóm ít mẫu có thống kê kém ổn định.')

elif page == 'Dự đoán giá nhà':
    st.header('Dự đoán giá bất động sản')
    st.write('Nhập các đặc điểm của căn nhà để nhận giá bán ước tính từ mô hình XGBoost.')
    st.caption('Áp dụng cho dữ liệu lịch sử tại Ames · Diện tích: ft² · Giá ước tính: USD')
    with st.form('prediction'):
        left, right = st.columns(2)
        values = {}
        values['GrLivArea'] = left.number_input('Diện tích sinh hoạt (ft²)', 1, 20000, 1500)
        values['Neighborhood'] = right.selectbox('Khu vực', metadata['neighborhoods'], index=metadata['neighborhoods'].index('NAmes') if 'NAmes' in metadata['neighborhoods'] else 0)
        values['BedroomAbvGr'] = left.number_input('Phòng ngủ trên mặt đất', 0, 50, 3)
        values['FullBath'] = right.number_input('Phòng tắm đầy đủ', 0, 50, 2)
        values['GarageCars'] = left.number_input('Sức chứa garage (xe)', 0, 50, 2)
        values['YearBuilt'] = right.number_input('Năm xây dựng', 1800, datetime.now(ZoneInfo('Asia/Bangkok')).year, 2000)
        values['OverallQual'] = left.slider('Chất lượng tổng thể', 1, 10, 6)
        values['TotalBsmtSF'] = right.number_input('Diện tích tầng hầm (ft²)', 0, 20000, 900)
        submitted = st.form_submit_button('Dự đoán giá nhà', type='primary', width='stretch')
    if submitted:
        try:
            frame = pd.DataFrame([values], columns=FEATURES)
            with st.spinner('Đang ước tính giá nhà…'):
                value = float(predict(frame, model, metadata)[0])
            with st.container(key='prediction_result'):
                st.metric('Giá nhà ước tính', f'${value:,.0f} USD')
                st.success(f"Mô hình: {metadata['selected_model']} · {len(FEATURES)} thuộc tính · Inference: {metadata['inference_device'].upper()}")
            if any(values[c] < train[c].min() or values[c] > train[c].max() for c in FEATURES if c != 'Neighborhood'):
                st.warning('Một số thuộc tính nằm ngoài khoảng quan sát của Train; độ tin cậy có thể thấp hơn.')
            st.caption('Đây là giá ước tính từ dữ liệu lịch sử tại Ames, không phải định giá giao dịch hiện tại. MAE Validation là sai số trung bình, không phải khoảng bảo đảm cho từng căn nhà.')
        except ValueError as exc:
            st.error(str(exc))

elif page == 'Hiệu năng mô hình':
    st.header('Hiệu năng trên Validation')
    comparison, detail, metadata = performance_data()
    st.caption(f"{len(detail)} mẫu Validation · MAE / RMSE bằng USD · Kết quả đọc từ báo cáo đã xuất")
    st.subheader('Mô hình được chọn: ' + metadata['selected_model'])
    cols = st.columns(3)
    for col, metric in zip(cols, ['MAE', 'RMSE', 'R2']):
        value = metadata['validation_metrics'][metric]
        col.metric('R²' if metric == 'R2' else metric + ' (USD)', f'{value:.4f}' if metric == 'R2' else f'${value:,.2f}')
    st.dataframe(comparison, hide_index=True, width='stretch')
    show_chart(px.bar(comparison, y='Model', x=['MAE', 'RMSE'], orientation='h', barmode='group',
                      title='So sánh sai số trên Validation', labels={'Model': 'Mô hình', 'value': 'Sai số (USD)', 'variable': 'Chỉ số'}))
    actual = px.scatter(detail, x='Thực tế', y='Dự đoán', title='Giá thực tế và dự đoán', labels={'Thực tế': 'Giá thực tế (USD)', 'Dự đoán': 'Giá dự đoán (USD)'})
    low, high = float(detail['Thực tế'].min()), float(detail['Thực tế'].max())
    actual.add_scatter(x=[low, high], y=[low, high], mode='lines', name='Dự đoán hoàn hảo')
    show_chart(actual)
    residual = px.histogram(detail, x='Sai số', title='Phân bố sai số: thực tế − dự đoán', labels={'Sai số': 'Sai số (USD)'})
    residual.update_yaxes(title='Số căn nhà')
    show_chart(residual)
    show_chart(px.scatter(detail, x='Thực tế', y='Sai số tuyệt đối', title='Sai số tuyệt đối theo giá', labels={'Thực tế': 'Giá thực tế (USD)', 'Sai số tuyệt đối': 'Sai số tuyệt đối (USD)'}))
    st.subheader('Thông tin mô hình')
    left, right = st.columns(2)
    left.metric('Training device', metadata['training_device'])
    right.metric('Inference device', metadata['inference_device'].upper())
    st.caption(f"Thời gian fit: {metadata['training_seconds']:.2f} giây · CV: {metadata['cv_seconds']:.2f} giây. GPU dùng khi huấn luyện; CPU inference là cấu hình chủ động.")
    with st.expander('Hyperparameters của mô hình đã lưu'):
        st.json(metadata['hyperparameters'])
    st.info('Validation đã được dùng để chọn mô hình, nên kết quả này có thể lạc quan. Test được đánh giá riêng sau khi đóng băng. Dữ liệu ít, outlier và khác biệt thị trường hạn chế khả năng khái quát.')
    final_path = REPORTS / 'test_metrics.json'
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding='utf-8'))
        st.subheader('Đánh giá cuối trên Test')
        st.caption(f"{final['rows']} mẫu Test nội bộ · Model đã đóng băng · Không tính lại metrics khi mở trang")
        cols = st.columns(3)
        for col, metric in zip(cols, ['MAE', 'RMSE', 'R2']):
            value = final['models']['XGBoost'][metric]
            col.metric('Test R²' if metric == 'R2' else 'Test ' + metric + ' (USD)', f'{value:.4f}' if metric == 'R2' else f'${value:,.2f}')
        st.dataframe(pd.DataFrame(final['models']).T.rename_axis('Model').reset_index(), hide_index=True, width='stretch')
        for filename, caption in [('actual_vs_predicted', 'Test: giá thực tế và dự đoán'),
                                  ('residual_distribution', 'Test: phân bố phần dư'),
                                  ('residual_vs_predicted', 'Test: phần dư theo dự đoán'),
                                  ('absolute_error_distribution', 'Test: phân bố sai số tuyệt đối'),
                                  ('error_by_price_range', 'Test: MAE theo khoảng giá'),
                                  ('percentage_error_by_price_range', 'Test: phần trăm sai số theo khoảng giá'),
                                  ('feature_importance', 'Permutation Importance trên Validation')]:
            image = REPORTS / f'figures/final/{filename}.png'
            if image.exists():
                st.image(str(image), caption=caption, width='stretch')
        importance = REPORTS / 'feature_importance.json'
        if importance.exists():
            ranking = json.loads(importance.read_text(encoding='utf-8'))['ranking']
            st.dataframe(pd.DataFrame(ranking).drop(columns='repeats'), hide_index=True, width='stretch')
            st.caption('Importance đo mức tăng MAE khi hoán vị thuộc tính gốc trên Validation; không chứng minh nhân quả.')
        benchmark_path = REPORTS / 'gpu_benchmark.json'
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text(encoding='utf-8'))
            st.subheader('Benchmark CPU / GPU (Train)')
            rows = [{'Device': name, 'Status': item['status'],
                     'Fit median (s)': item.get('training_median_seconds'),
                     '1 sample (ms)': item.get('latency_1', {}).get('median_ms'),
                     '100 samples (ms)': item.get('latency_100', {}).get('median_ms')}
                    for name, item in benchmark['devices'].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.caption('3 lần fit sau warmup, cùng cấu hình; benchmark không thay model chính. Peak VRAM chưa đo được.')
    else:
        st.info('Chưa có báo cáo đánh giá Test đã xuất.')

elif page == 'Giá nhà TP.HCM':
    st.header('Dự đoán giá nhà ở TP.HCM')
    st.caption('Dữ liệu tự thu thập từ tin bán nhà công khai trên Nhà Tốt · Giá: tỷ VNĐ · Diện tích: m²')
    try:
        vn_model, vn_meta = vn_resources()
    except (OSError, ValueError) as exc:
        st.error(f'Chưa có mô hình TP.HCM: {exc}')
        st.info('Chạy lần lượt: python -m src.data.vn_collect → python -m src.data.vn_clean → python -m src.models.vn_train')
        st.stop()
    test = vn_meta['test']
    cols = st.columns(4)
    cols[0].metric('Tin dùng để huấn luyện', f"{vn_meta['train_rows']:,}")
    cols[1].metric('MAE Test', f"{test['MAE']:.2f} tỷ")
    cols[2].metric('Sai số % trung vị', f"{test['MedAPE_%']:.1f}%")
    cols[3].metric('R² Test', f"{test['R2']:.3f}")
    choices = vn_meta['choices']

    def default(options, value):
        return options.index(value) if value in options else 0

    with st.form('vn_prediction'):
        left, right = st.columns(2)
        v = {}
        v['district'] = left.selectbox('Quận / huyện', vn_meta['districts'], index=default(vn_meta['districts'], 'Quận Gò Vấp'))
        v['house_type'] = right.selectbox('Loại nhà', choices['house_type'], index=default(choices['house_type'], 'Nhà ngõ, hẻm'))
        v['size'] = left.number_input('Diện tích đất (m²)', 10.0, 1000.0, 60.0, step=5.0)
        v['floors'] = right.number_input('Số tầng', 1, 15, 3)
        v['rooms'] = left.number_input('Số phòng ngủ', 1, 30, 3)
        v['toilets'] = right.number_input('Số WC', 1, 30, 3)
        v['width'] = left.number_input('Mặt tiền (m)', 1.0, 50.0, 4.0, step=0.5)
        v['length'] = right.number_input('Chiều dài (m)', 2.0, 200.0, 15.0, step=0.5)
        v['legal'] = left.selectbox('Pháp lý', choices['legal'], index=default(choices['legal'], 'Đã có sổ'))
        v['furnishing'] = right.selectbox('Nội thất', choices['furnishing'], index=default(choices['furnishing'], 'Hoàn thiện cơ bản'))
        v['direction'] = left.selectbox('Hướng cửa chính', choices['direction'], index=default(choices['direction'], 'Hướng Đông Nam'))
        v['car_alley'] = int(right.checkbox('Hẻm xe hơi'))
        v['expanding_back'] = int(right.checkbox('Nhà nở hậu'))
        submitted = st.form_submit_button('Dự đoán giá', type='primary', width='stretch')
    if submitted:
        try:
            price = predict_vn(v, vn_model, vn_meta)
            a, b = st.columns(2)
            a.metric('Giá ước tính', f'{price:,.2f} tỷ VNĐ')
            b.metric('Đơn giá', f"{price * 1000 / v['size']:,.0f} triệu/m²")
            outside = out_of_range(v, vn_meta)
            if outside:
                st.warning('Các thuộc tính nằm ngoài khoảng thường gặp trong dữ liệu (1%–99%): '
                           + ', '.join(outside) + '. Kết quả kém tin cậy hơn.')
            st.caption(f"Mô hình: {vn_meta['selected_model']}. Một nửa số căn trong tập Test có sai số dưới "
                       f"{test['MedAPE_%']:.0f}%. Đây là ước tính theo giá rao bán, không phải định giá chính thức.")
        except ValueError as exc:
            st.error(str(exc))
    st.subheader('So sánh mô hình trên Validation')
    st.dataframe(pd.DataFrame(vn_meta['validation']).T.rename_axis('Mô hình').reset_index().round(3),
                 hide_index=True, width='stretch')
    st.caption('Chi tiết: reports/vn_data_funnel.md (làm sạch dữ liệu) và reports/vn_model_report.md (mô hình).')

else:
    st.header('Về dự án')
    st.subheader('House Price Prediction — XGBoost Regression')
    st.write('HousePrice AI nghiên cứu dự đoán giá bất động sản bằng XGBoost với GPU Acceleration. Dataset Ames Housing gồm dữ liệu lịch sử tại Ames, Iowa, Hoa Kỳ.')
    st.markdown('[Nguồn: Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)')
    st.subheader('Quy trình nghiên cứu')
    st.write('Dữ liệu được chia cố định: Train 70%, Validation 15%, Test 15%, seed 42. Tiền xử lý được fit riêng trong từng fold Train; Validation dùng để so sánh và lựa chọn. Test chỉ dành cho đánh giá cuối ở Tuần 4.')
    st.subheader('Công nghệ và GPU Acceleration')
    st.write('Công nghệ: Python, pandas, scikit-learn, XGBoost, NVIDIA CUDA, Plotly và Streamlit. GPU tăng tốc việc xây cây; preprocessing và inference của ứng dụng chạy trên CPU.')
    st.subheader('Giới hạn áp dụng')
    st.warning('Không áp dụng mô hình này để tuyên bố giá nhà tại Việt Nam hoặc thị trường hiện tại. Mô hình chỉ dùng 8 đặc điểm và chưa đo độ bất định của từng dự đoán.')
    st.subheader('Hướng phát triển')
    st.write('Thu thập dữ liệu Việt Nam có quyền sử dụng, chuẩn hóa vị trí và thời gian, đánh giá theo thời gian/địa lý, kiểm tra drift và xây dựng khoảng dự đoán trước ứng dụng thực tế.')

st.divider()
st.caption('HousePrice AI · Ames Housing · Nghiên cứu dự đoán giá bất động sản')
