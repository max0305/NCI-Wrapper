# Baseline Runner 使用說明

## 1. 簡介
本框架提供統一的 baseline 執行流程：
- SparseRetrieval: TF-IDF 斷詞→計數→TF-IDF 編碼→餘弦相似度檢索
- NCI: 有監督 Dual-Encoder 檢索，透過 train.sh / infer.sh 完成訓練和推論
- GR-as-MVDR

整體流程： (split_raw) → do_preprocess → setup → do_train → do_eval → save_results → 彙總

## 2. 安裝需求
抓取專案
```
  git clone https://github.com/max0305/NCI-Wrapper
```
建立並啟用 conda 環境
```
  conda env create -f environment.yml
  conda activate baseline
```
安裝 GR-as-MVDR 依賴項 SEAL
```
  cd schemes/GR_as_MVDR
  conda install swig
  git clone --recursive https://github.com/facebookresearch/SEAL.git
  env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
  pip install -e .
  cp beam_search.py /path/to/SEAL/seal/
```

## 3. 基本指令
(1) 完整流程（資料切分、前處理、全部 baseline 訓練 + 評估）：
```
  python main.py --baseline all --split_raw --do_preprocess --do_train --do_eval
```
(2) 只評估 NCI（假設資料已處理、模型已訓練好）：
```
  python main.py --baseline nci --do_eval
```
(3) 只跑 SparseRetrieval（無訓練）：
```
  python main.py --baseline sparse --do_eval
```
(4) 只跑 MVDR：
```
  python main.py --baseline mvdr  --do_preprocess --do_train --do_eval
```
(4) 只跑 GR：
```
  python main.py --baseline gr  --do_preprocess --do_train --do_eval
```
## 4. 參數說明（main.py）

| 參數             | 類型      | 預設                                | 說明                                                         |
| ---------------- | --------- | ----------------------------------- | ------------------------------------------------------------ |
| `--split_raw`    | Boolean   | `False`                             | 隨機切分原始 JSON 成 train/eval (80/20)                      |
| `--do_preprocess`| Boolean   | `False`                             | 執行資料前處理，產生 `processed/` 與 `splits/`               |
| `--do_train`     | Boolean   | `False`                             | 執行各 baseline 的 `train()` 方法                            |
| `--do_eval`      | Boolean   | `False`                             | 執行 `evaluate()`，並輸出評估指標                            |
| `--baseline`     | String    | `"all"`                             | 指定要執行哪些 baseline；可用 `all`、`nci`、`sparse`，或多者逗號分隔 |
| `--dataset_path` | String    | `"./dataset/raw/item_profile_sample.json"` | 原始資料路徑                                                 |
| `--output_dir`   | String    | `"./outputs"`                       | 所有結果與 log 的根目錄                                      |
| `--n_gpu`        | Integer   | `1`                                 | 使用的 GPU 數量                            |
| `--seed`         | Integer   | `42`                                | 隨機種子，確保切分與訓練過程可重現                            |
| `--device`       | String    | `"cuda"`                            | PyTorch 裝置；無 GPU 時可設為 `"cpu"`                       |
| `--log_level`    | String    | `"INFO"`                            | Logger 等級：`DEBUG` / `INFO` / `WARNING` / `ERROR`         |

**NCI 專用參數：**


| 參數             | 類型    | 範例／預設   | 說明                        |
| ---------------- | ------- | ------------ | --------------------------- |
| `--model_info`   | String  | `"small"`    | 子腳本使用的模型尺寸，可選 `small`／`base`／`large` |
| `--epoch`        | Integer | `3`          | 訓練迭代輪數                 |
| `--qg_num`       | Integer | *見程式註解* | 資料增強時的 query generation 數量 |
| `--class_num`    | Integer | *見程式註解* | 資料增強時的類別標籤數量     |


5. 輸出結果與指標
執行後會在 outputs/ 下依 baseline 及 timestamp 建立資料夾，如：
```
outputs/
├─ sparse/2025-07-24_23-15-42/
│   ├─ metrics.json             # MRR, Recall@10...
│   └─ vocab.json               # TF-IDF 詞彙表索引
└─ nci/2025-07-24_23-18-10/
    └─ metrics.json
```

> NCI 輸出權重及結果請至 `NCL/NCI_model/logs` 底下查看
### 全域 log
logs/run_YYYYMMDD_HHMMSS.log
