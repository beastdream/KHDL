# Kết quả kiểm thử cuối

{'PASS': 73, 'FAIL': 0, 'SKIPPED': 14}

Ba process riêng giữ nguyên policy Tuần 2, Tuần 3; policy Final chỉ cấp quyền đọc holdout trong evaluator. E2E đối chiếu kết quả đã xuất, không đánh giá lại Test và không huấn luyện lại artifact cuối.

## SKIPPED: từng trường hợp

- `tests.test_week1::test_csv_schema_and_target`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_split_counts_disjoint_reproducible[20]`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_split_counts_disjoint_reproducible[101]`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_split_counts_disjoint_reproducible[1460]`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_invalid_data_rejected[target]`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_invalid_data_rejected[id]`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_invalid_data_rejected[area]`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_invalid_data_rejected[missing_column]`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_invalid_data_rejected[duplicate_content]`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_missing_predictors_preserved`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_freeze_and_tamper_detection`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_persisted_artifacts`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_untracked_holdout_never_overwritten`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.
- `tests.test_week1::test_audit_scope_and_no_removal`: Week 2 prohibits raw/holdout access; original Week 1 PASS retained, not re-run.

SKIP không phải PASS. 14 bài cũ cần raw/holdout vẫn giữ policy cũ; E2E mới kiểm tra split theo metadata/seed và provenance Test qua báo cáo chính thức. Không chạy lại pipeline audit/split trên raw; bằng chứng Tuần 1 được giữ nguyên.

Lệnh thực tế: `.\.venv\Scripts\python.exe -m src.utils.validate_final`. JUnit tổng hợp: final_pytest.xml; từng suite: final_legacy_pytest.xml, final_week3_pytest.xml, final_e2e_pytest.xml.