"""Analyze exported final predictions; never reopen Test or refit the model."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.final_inputs import install_data_guard
from src.models.evaluate_test import verify_freeze
from src.utils.paths import REPORTS, sha256, write_json


def analyze():
    install_data_guard()
    verify_freeze()
    results = json.loads((REPORTS / 'test_metrics.json').read_text())
    path = REPORTS / 'test_predictions.json'
    assert sha256(path) == results['predictions_sha256']
    data = pd.DataFrame(json.loads(path.read_text())['records'])
    data['Residual'] = data.Actual - data.Predicted
    data['AbsoluteError'] = data.Residual.abs()
    data['PercentageError'] = data.AbsoluteError / data.Actual.abs().replace(0, np.nan) * 100
    bins = [0, 100000, 200000, 300000, 400000, np.inf]
    labels = ['<100k', '100–200k', '200–300k', '300–400k', '>=400k']
    data['PriceRange'] = pd.cut(data.Actual, bins, labels=labels, right=False, include_lowest=True)
    grouped = data.groupby('PriceRange', observed=True).agg(
        count=('Actual', 'size'), MAE=('AbsoluteError', 'mean'),
        mean_residual=('Residual', 'mean'), MAPE=('PercentageError', 'mean'))
    folder = REPORTS / 'figures/final'
    folder.mkdir(parents=True, exist_ok=True)
    def save(name, xlab, ylab):
        plt.xlabel(xlab); plt.ylabel(ylab); plt.tight_layout()
        plt.savefig(folder / f'{name}.png', dpi=160); plt.close()
    plt.figure(figsize=(7, 5)); plt.scatter(data.Actual, data.Predicted, alpha=.65, color='#152238')
    low, high = data.Actual.min(), data.Actual.max()
    plt.plot([low, high], [low, high], '--', color='#FF5252')
    save('actual_vs_predicted', 'Actual price (USD)', 'Predicted price (USD)')
    plt.figure(figsize=(7, 5)); plt.hist(data.Residual, bins=30, color='#152238'); plt.axvline(0, color='#FF5252')
    save('residual_distribution', 'Actual minus predicted (USD)', 'Count')
    plt.figure(figsize=(7, 5)); plt.scatter(data.Predicted, data.Residual, alpha=.65, color='#152238'); plt.axhline(0, color='#FF5252')
    save('residual_vs_predicted', 'Predicted price (USD)', 'Residual (USD)')
    plt.figure(figsize=(7, 5)); plt.hist(data.AbsoluteError, bins=30, color='#152238')
    save('absolute_error_distribution', 'Absolute error (USD)', 'Count')
    for column, filename, ylabel in [('MAE', 'error_by_price_range', 'Mean absolute error (USD)'),
                                      ('MAPE', 'percentage_error_by_price_range', 'Mean absolute percentage error (%)')]:
        plt.figure(figsize=(7, 5)); plt.bar(grouped.index.astype(str), grouped[column], color='#152238')
        save(filename, 'Actual price range (USD)', ylabel)
    summary = {'mean_residual': float(data.Residual.mean()), 'median_residual': float(data.Residual.median()),
               'residual_skew': float(data.Residual.skew()), 'underprediction_fraction': float((data.Residual > 0).mean()),
               'zero_actual_count': int(data.Actual.eq(0).sum()), 'max_absolute_error': float(data.AbsoluteError.max()),
               'groups': grouped.reset_index().to_dict(orient='records')}
    write_json(REPORTS / 'error_analysis.json', summary)
    test, valid = results['models']['XGBoost'], results['validation_selected_model']
    rows = ['# Phân tích sai số Test', '', 'Phần dư = giá thực tế − dự đoán; dương là dự đoán thấp. Chỉ đọc test_predictions.json đã xuất.', '',
            f'Trung bình phần dư {summary["mean_residual"]:,.2f} USD; trung vị {summary["median_residual"]:,.2f} USD; '
            f'tỷ lệ dự đoán thấp {summary["underprediction_fraction"]:.1%}; skew {summary["residual_skew"]:.3f}. '
            'Trung bình phản ánh thiên lệch tổng thể, không có nghĩa mọi căn nhà bị sai cùng hướng.', '',
            '| Khoảng giá USD | n | MAE USD | MAPE % | Phần dư TB USD |', '|---|---:|---:|---:|---:|']
    for r in summary['groups']:
        rows.append(f'| {r["PriceRange"]} | {r["count"]} | {r["MAE"]:.2f} | {r["MAPE"]:.2f} | {r["mean_residual"]:.2f} |')
    worst = grouped.MAE.idxmax()
    rows += ['', f'Nhóm MAE cao nhất: {worst}, n={int(grouped.loc[worst,"count"])}. '
             'Đây là phân nhóm theo giá thực tế sau đánh giá, không phải tiêu chí chọn lại model. '
             'So sánh cả MAPE để tránh kết luận chỉ do đơn vị giá; nhóm ít mẫu có độ bất định lớn.', '',
             f'MAE Test {test["MAE"]:,.2f} USD, RMSE {test["RMSE"]:,.2f} USD. '
             f'RMSE/MAE = {test["RMSE"]/test["MAE"]:.2f}: sai số lớn ảnh hưởng RMSE mạnh hơn. '
             f'Sai số tuyệt đối lớn nhất {summary["max_absolute_error"]:,.2f} USD.', '',
             f'So với Validation, MAE đổi {(test["MAE"]/valid["MAE"]-1)*100:+.2f}%, '
             f'RMSE đổi {(test["RMSE"]/valid["RMSE"]-1)*100:+.2f}%, R² đổi {test["R2"]-valid["R2"]:+.5f}. '
             'MAE gần nhau trong khi RMSE Test tăng, phù hợp ảnh hưởng một số sai số lớn. '
             'Không có suy giảm đồng loạt cho thấy overfitting nghiêm trọng; cũng không đủ cơ sở loại trừ overfitting '
             'hoặc gọi khác biệt có ý nghĩa thống kê khi chưa định lượng độ bất định. Validation đã qua lựa chọn model.', '',
             '219 mẫu holdout là nhỏ; vài nhà đắt có thể làm thay đổi RMSE đáng kể. Một split ngẫu nhiên không đại diện '
             'cho thay đổi thời gian, địa lý hoặc thị trường Việt Nam. Không tối ưu sau Test. '
             f'Có {summary["zero_actual_count"]} giá thực tế bằng 0; MAPE loại mẫu giá 0 thay vì chia cho 0.', '',
             'Biểu đồ: figures/final/actual_vs_predicted.png, residual_distribution.png, residual_vs_predicted.png, '
             'absolute_error_distribution.png, error_by_price_range.png, percentage_error_by_price_range.png.']
    (REPORTS / 'error_analysis.md').write_text('\n'.join(rows), encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
