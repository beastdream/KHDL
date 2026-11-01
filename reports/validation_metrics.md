# Validation metrics

Chỉ đánh giá Validation; không dùng Test.

| Model | MAE | RMSE | R2 | Device |
|---|---:|---:|---:|---|
| Dummy Baseline | 58741.92 | 81912.06 | -0.0682 | CPU |
| XGBoost | 17792.26 | 25752.38 | 0.8944 | cuda:0 |

RMSE giảm 68.56% so với giá trung vị Train. MAE/RMSE tính bằng USD.