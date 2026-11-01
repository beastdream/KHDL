# GPU environment audit

GPU READY

```json
{
  "python_version": "3.11.9",
  "nvidia_smi": {
    "returncode": 0,
    "stdout": "Sat Oct 17 16:32:03 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 566.07                 Driver Version: 566.07         CUDA Version: 12.7     |\n|-----------------------------------------+------------------------+----------------------+\n| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce GTX 1650      WDDM  |   00000000:01:00.0 Off |                  N/A |\n| N/A   53C    P8              2W /   51W |     151MiB /   4096MiB |      0%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n                                                                                         \n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A     15708    C+G   ...Brave-Browser\\Application\\brave.exe      N/A      |\n|    0   N/A  N/A     19564    C+G   ...Brave-Browser\\Application\\brave.exe      N/A      |\n+-----------------------------------------------------------------------------------------+\n",
    "stderr": ""
  },
  "gpu_inventory": {
    "returncode": 0,
    "stdout": "name, memory.total [MiB], driver_version\nNVIDIA GeForce GTX 1650, 4096 MiB, 566.07\n",
    "stderr": ""
  },
  "driver_cuda_compatibility": "12.7",
  "cuda_note": "Driver-reported compatibility is not proof of installed CUDA toolkit or successful training.",
  "gpu_smoke": {
    "requested_device": "cuda",
    "success": true,
    "warnings": [],
    "xgboost_version": "3.2.0",
    "booster_config": {
      "learner": {
        "generic_param": {
          "device": "cuda:0",
          "fail_on_invalid_gpu_id": "0",
          "n_jobs": "0",
          "nthread": "0",
          "random_state": "42",
          "seed": "42",
          "seed_per_iteration": "0",
          "validate_parameters": "1"
        },
        "gradient_booster": {
          "gbtree_model_param": {
            "num_parallel_tree": "1",
            "num_trees": "4"
          },
          "gbtree_train_param": {
            "process_type": "default",
            "tree_method": "hist",
            "updater": "grow_gpu_hist",
            "updater_seq": "grow_gpu_hist"
          },
          "name": "gbtree",
          "specified_updater": false,
          "tree_train_param": {
            "alpha": "0",
            "colsample_bylevel": "1",
            "colsample_bynode": "1",
            "colsample_bytree": "1",
            "eta": "0.300000012",
            "gamma": "0",
            "grow_policy": "depthwise",
            "interaction_constraints": "",
            "lambda": "1",
            "learning_rate": "0.300000012",
            "max_bin": "256",
            "max_cat_threshold": "64",
            "max_cat_to_onehot": "4",
            "max_delta_step": "0",
            "max_depth": "2",
            "max_leaves": "0",
            "min_child_weight": "1",
            "min_split_loss": "0",
            "monotone_constraints": "()",
            "refresh_leaf": "1",
            "reg_alpha": "0",
            "reg_lambda": "1",
            "sampling_method": "uniform",
            "sparse_threshold": "0.20000000000000001",
            "subsample": "1"
          },
          "updater": [
            {
              "hist_train_param": {
                "debug_synchronize": "0",
                "max_cached_hist_node": "18446744073709551615"
              },
              "name": "grow_gpu_hist"
            }
          ]
        },
        "learner_model_param": {
          "base_score": "[-2.8539598E-2]",
          "boost_from_average": "1",
          "num_class": "0",
          "num_feature": "4",
          "num_target": "1"
        },
        "learner_train_param": {
          "booster": "gbtree",
          "disable_default_eval_metric": "0",
          "multi_strategy": "one_output_per_tree",
          "objective": "reg:squarederror"
        },
        "metrics": [
          {
            "name": "rmse"
          }
        ],
        "objective": {
          "name": "reg:squarederror",
          "reg_loss_param": {
            "scale_pos_weight": "1"
          }
        }
      },
      "version": [
        3,
        2,
        0
      ]
    },
    "actual_device": "cuda:0"
  },
  "status": "GPU READY",
  "cpu_fallback": null
}
```

Reference: https://xgboost.readthedocs.io/en/stable/gpu/index.html
