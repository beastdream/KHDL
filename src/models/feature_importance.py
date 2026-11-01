"""Permutation of original columns on Validation through the frozen Pipeline."""
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from src.data.final_inputs import install_data_guard
from src.data.week2_inputs import load_split
from src.data.schema import FEATURES
from src.models.evaluate_test import verify_freeze
from src.models.inference import load_model
from src.utils.paths import REPORTS, write_json


def explain():
    install_data_guard(); verify_freeze()
    model, _ = load_model()
    valid = load_split('valid')
    before = joblib.hash(model)
    result = permutation_importance(model, valid[FEATURES], valid.SalePrice,
        scoring='neg_mean_absolute_error', n_repeats=20, random_state=42, n_jobs=1)
    assert joblib.hash(model) == before
    ranking = sorted([{'feature': name, 'mean_mae_increase': float(mean), 'std': float(std),
                       'repeats': values.tolist()} for name, mean, std, values in
                      zip(FEATURES, result.importances_mean, result.importances_std, result.importances)],
                     key=lambda r: r['mean_mae_increase'], reverse=True)
    write_json(REPORTS / 'feature_importance.json', {'split': 'valid', 'scoring': 'MAE increase (USD)',
               'n_repeats': 20, 'seed': 42, 'ranking': ranking})
    plt.figure(figsize=(8, 5))
    items = ranking[::-1]
    plt.barh([r['feature'] for r in items], [r['mean_mae_increase'] for r in items],
             xerr=[r['std'] for r in items], color='#152238')
    plt.xlabel('Validation MAE increase after permutation (USD)'); plt.tight_layout()
    (REPORTS / 'figures/final').mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS / 'figures/final/feature_importance.png', dpi=160); plt.close()
    text = ['# Permutation Importance trên Validation', '',
            'Pipeline đã đóng băng, scoring neg_mean_absolute_error, 20 lần hoán vị, seed 42. '
            'Giá trị là mức tăng MAE (USD), error bar là độ lệch chuẩn qua các hoán vị, không phải khoảng tin cậy.', '',
            '| Hạng | Thuộc tính | Tăng MAE USD | Std USD |', '|---:|---|---:|---:|']
    for i, r in enumerate(ranking, 1):
        text.append(f'| {i} | {r["feature"]} | {r["mean_mae_increase"]:.2f} | {r["std"]:.2f} |')
    text += ['', f'Quan trọng nhất theo phép đo này: {ranking[0]["feature"]}; thấp nhất: {ranking[-1]["feature"]}. '
             'Hoán vị Neighborhood ở cột gốc trước preprocessing, nên toàn bộ nhóm dummy one-hot thay đổi cùng nhau; '
             'không cộng tùy tiện importance của từng dummy.', '',
             'EDA Train cho thấy quan hệ giá với chất lượng, diện tích, khu vực và năm xây dựng '
             '(eda_analysis.md). Xếp hạng ở đây đo đóng góp dự đoán có điều kiện trong model, '
             'không bằng thứ tự tương quan Pearson. Các biến như diện tích, tầng hầm, garage và chất lượng có thể '
             'chia sẻ thông tin; biến tương quan có thể bù nhau và làm giảm importance riêng lẻ. '
             'Hoán vị cũng có thể tạo tổ hợp ít gặp. Importance thấp không chứng minh biến vô ích, '
             'importance cao không chứng minh nhân quả. Không loại feature, tuning hoặc chọn lại model từ kết quả này; không dùng Test.']
    (REPORTS / 'feature_importance.md').write_text('\n'.join(text), encoding='utf-8')
    verify_freeze()
    return ranking


if __name__ == '__main__':
    print(json.dumps(explain(), ensure_ascii=False, indent=2))
