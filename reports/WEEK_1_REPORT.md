# Báo cáo Tuần 1 — House Price Prediction

Ngày chạy: 2026-10-17T16:31:05.271590+07:00

## 1. Tổng quan dataset

Nguồn: Kaggle House Prices, data/raw/train.csv. Số dòng: 1460; số cột: 81. Target: SalePrice.
Giữ nguyên file gốc. test.csv ở thư mục gốc là dữ liệu thi Kaggle không có nhãn, không phải holdout nội bộ.

## 2. Kết quả audit

Trạng thái: PASS. Không tự xóa bản ghi/outlier.
Duplicate: 0; duplicate nội dung: 0; lỗi schema: {}.
Kiểm tra chất lượng trên toàn bộ nguồn. Thống kê min/max/mean/median/std, target và biểu đồ chỉ dùng TRAIN; không EDA trên test.
Chi tiết: [dataset_audit.md](dataset_audit.md), [dataset_audit.json](dataset_audit.json).

## 3. Bảng Train/Valid/Test

| Tập | Số mẫu | Tỷ lệ |
|---|---:|---:|
| train | 1022 | 70.00% |
| valid | 219 | 15.00% |
| test | 219 | 15.00% |

Seed: 42; hai lần train_test_split; không fit preprocessing. Giữ Id và index nguồn trong manifest.
SHA-256 test.csv: `7383f69e2cf6e315087e4a4c6836c3499a26f249cde4f6da1e08e2a19bcc979a`.
Test được khóa đến đánh giá cuối Tuần 4; chạy lại kiểm tra hash trước khi ghi.

## 4. Kiểm tra GPU

GPU READY

```text
name, memory.total [MiB], driver_version
NVIDIA GeForce GTX 1650, 4096 MiB, 566.07

```
Python: 3.11.9; XGBoost: 3.2.0; driver CUDA compatibility: 12.7.
Kiểm tra fit trên dữ liệu smoke độc lập, device=cuda/tree_method=hist, xác nhận device và GPU updater trong Booster.
Chi tiết đầy đủ: [gpu_audit.md](gpu_audit.md).

## 5. Các lỗi còn tồn tại

Không có lỗi chặn dữ liệu.
Missing predictors được giữ nguyên; chỉ fit imputer/encoder trên train trong tuần tiếp theo.
GPU: Không có lỗi chặn smoke test.

## 6. Kết quả pytest

```text
.....................                                                    [100%]
21 passed in 4.36s

```

## 7. Kết luận

PASS

GPU READY

Chỉ thực hiện Tuần 1. Không chuyển Tuần 2 nếu kiểm tra dữ liệu chưa PASS.