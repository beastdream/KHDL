"""Train-only charts and exported model performance data."""
import json
import pandas as pd
import plotly.express as px
from src.data.schema import NUMERIC_FEATURES, TARGET
from src.data.week3_inputs import load_split
from src.utils.paths import REPORTS

LABELS = {'SalePrice': 'Giá bán (USD)', 'GrLivArea': 'Diện tích sinh hoạt (ft²)',
          'OverallQual': 'Chất lượng (1–10)', 'Neighborhood': 'Khu vực',
          'YearBuilt': 'Năm xây dựng', 'BedroomAbvGr': 'Số phòng ngủ'}


def train_data():
    return load_split('train')


def filter_train(frame, neighborhoods, quality, price, area):
    return frame.loc[frame.Neighborhood.isin(neighborhoods)
        & frame.OverallQual.between(*quality) & frame.SalePrice.between(*price)
        & frame.GrLivArea.between(*area)].copy()


def eda_charts(frame):
    if frame.empty:
        return []
    grouped = frame.groupby('Neighborhood', as_index=False).SalePrice.median().sort_values('SalePrice')
    figures = [
        px.histogram(frame, x=TARGET, nbins=40, marginal='box', title='Phân bố giá nhà', labels=LABELS),
        px.scatter(frame, x='GrLivArea', y=TARGET, color='OverallQual', hover_data=['Neighborhood', 'YearBuilt'], title='Diện tích và giá bán', labels=LABELS),
        px.bar(grouped, x=TARGET, y='Neighborhood', orientation='h', title='Giá trung vị theo khu vực', labels=LABELS),
        px.box(frame, x='OverallQual', y=TARGET, title='Chất lượng và giá bán', labels=LABELS),
        px.imshow(frame[NUMERIC_FEATURES + [TARGET]].corr(), zmin=-1, zmax=1, color_continuous_scale='RdBu_r', text_auto='.2f', title='Tương quan Pearson'),
        px.scatter(frame, x='YearBuilt', y=TARGET, color='OverallQual', hover_data=['Neighborhood'], title='Năm xây dựng và giá bán', labels=LABELS),
        px.box(frame, x='BedroomAbvGr', y=TARGET, title='Số phòng ngủ và giá bán', labels=LABELS)]
    for figure in figures:
        figure.update_layout(template='plotly_white', colorway=['#0d9488', '#2563eb', '#f59e0b'], height=460,
                              margin=dict(l=25, r=25, t=65, b=30))
    return figures


def performance_data():
    metadata = json.loads((REPORTS.parent / 'artifacts/model_metadata.json').read_text(encoding='utf-8'))
    exported = pd.read_csv(REPORTS / 'validation_predictions.csv')
    detail = exported.rename(columns={'Actual': 'Thực tế', 'Predicted': 'Dự đoán'})[['Thực tế', 'Dự đoán']].copy()
    detail['Sai số'] = detail['Thực tế'] - detail['Dự đoán']
    detail['Sai số tuyệt đối'] = detail['Sai số'].abs()
    return pd.read_csv(REPORTS / 'model_comparison.csv'), detail, metadata

