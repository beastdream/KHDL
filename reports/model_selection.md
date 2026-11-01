# Model selection

XGBoost tuned CV#3

Lowest Validation MAE among baseline and top three CV candidates; RMSE tie-break.

Validation is used for selection, so these metrics are not an unbiased final evaluation. Test remains locked until Week 4. No refit on Train+Validation.

{
  "Model": "XGBoost tuned CV#3",
  "MAE": 17431.275390625,
  "RMSE": 24782.905075878414,
  "R2": 0.9022146463394165,
  "CV_MAE": 20260.7140625,
  "Device": "cuda:0",
  "TrainingSeconds": 1.5885708999994677
}