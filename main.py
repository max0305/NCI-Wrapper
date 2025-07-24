# main.py
"""
執行範例：
1. 完整跑資料前處理 + 所有 baseline 的訓練與評估
   python main.py --baseline all --do_preprocess --do_train --do_eval

2. 只對 NCI 做評估（資料已處理完）
   python main.py --baseline nci --do_eval
"""
import argparse
import importlib
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

# --------------------------------------------------
# 1. 解析命令列參數
# --------------------------------------------------
def parse_args():
    """定義並解析 command-line 參數 (args)"""
    p = argparse.ArgumentParser(description="Unified Baseline Runner")
    # —— Pipeline flag ——
    p.add_argument("--split_raw", action="store_true",
                   help="原始資料分割")
    p.add_argument("--do_preprocess", action="store_true",
                   help="執行資料前處理")
    p.add_argument("--do_train", action="store_true",
                   help="執行訓練")
    p.add_argument("--do_eval", action="store_true",
                   help="執行驗證 / 測試")

    # —— Baseline 選擇 ——
    p.add_argument("--baseline", type=str, default="all",
                   help="要執行的 baseline 名稱；可用 'all' 或 'nci,bert' 多選")

    # —— 資料與輸出路徑 ——
    p.add_argument("--dataset_path", type=str, default="./dataset/raw/item_profile_sample.json",
                   help="資料目錄 (raw/ processed/ splits/)")
    p.add_argument("--output_dir", type=str, default="./outputs",
                   help="所有 baseline 結果輸出根目錄")
    p.add_argument("--config", type=str, default=None,
                   help="可選的 YAML/JSON 設定檔路徑")

    # —— 其他雜項 ——
    p.add_argument("--n_gpu", type=str, default="1",
                   help="GPU 數量")
    p.add_argument("--device", type=str, default="cuda",
                   help="訓練 / 推論所使用裝置")
    p.add_argument("--seed", type=int, default=42,
                   help="固定隨機種子以重現結果")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # —— preprocess 參數 ——
    p.add_argument("--model_info", type=str, default="base", 
                    choices=["small", "base", "large"])
    p.add_argument("--qg_num", type=int, default=15) 
    p.add_argument("--class_num", type=int, default=30) 

    # —— train 參數 ——
    p.add_argument("--epoch", type=str, default="3")
    
    return p.parse_args()


# --------------------------------------------------
# 2. 工具函式：logger、隨機種子、動態 import
# --------------------------------------------------
def init_logger(level="INFO", log_dir: str = "./logs") -> logging.Logger:
    """初始化全域 logger（console + file）"""
    os.makedirs(log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"run_{time_str}.log"

    # 讓 root logger 只有一條 FileHandler
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")]
    )
    
    # 取得子 logger，額外加一條 Console handler
    logger = logging.getLogger(__name__)
    if not logger.handlers:                      # 防止重複 addHandler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(console)

    #logger.propagate = False                     # 阻斷往 root

    return logger


def set_seed(seed: int = 42):
    """固定 Python / NumPy / PyTorch 隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_baseline_class(module_path: str, class_name: str):
    """動態匯入 baseline 類別，回傳 class obj"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# --------------------------------------------------
# 3. Baseline Registry：維護可用的 baseline 清單
#    key = 指令列名稱；value = (module_path, class_name)
# --------------------------------------------------
BASELINE_REGISTRY = {
    "nci":  ("schemes.nci",  "NCI"),        # 例：schemes/nci.py → class NCI
    "sparse": ("schemes.sparse", "SparseRetrieval"),
}


# --------------------------------------------------
# 4. 資料前處理流程
# --------------------------------------------------
#def run_preprocess(dataset_path: str, logger: logging.Logger, model_info: str, qg_num: int, class_num: int):
def run_preprocess(dataset_path: str, logger: logging.Logger, seed: int):
    """
    呼叫 dataset.prepare_dataset.prepare_all
    - raw/ → processed/ + splits/
    """
    try:
        #from dataset.baseline import convert_to_nq_splits
        #from dataset.data_process import data_process
        from dataset.split import split_and_save
    except ModuleNotFoundError as e:
        logger.error("找不到 dataset 模組，請確認路徑")
        raise e

    # output_dir = "./schemes/Neural-Corpus-Indexer-NCI/Data_process/NQ_dataset"
    split_and_save(input_path=dataset_path ,seed=seed)

    '''
    logger.info("開始轉換為 nq 格式 …")
    convert_to_nq_splits(dataset_path, output_dir, 0.2, 42)
    logger.info("開始資料前處理 …")
    data_process(model_info, qg_num, class_num)
    logger.info("資料處理完成！")
    '''


# --------------------------------------------------
# 5. 執行指定 / 全部 baseline 的流程
# --------------------------------------------------
def build_baselines(args) -> List:
    """
    根據 args.baseline 回傳 baseline instance list
    - args.baseline 可為 'all' 或 'nci,bert'
    """
    targets = []
    if args.baseline.lower() == "all":
        targets = list(BASELINE_REGISTRY.keys())
    else:
        # 去空白、分割、轉小寫
        targets = [x.strip().lower() for x in args.baseline.split(",")]

    baselines = []
    for name in targets:
        if name not in BASELINE_REGISTRY:
            raise ValueError(f"未知 baseline 名稱：{name}")
        module_path, class_name = BASELINE_REGISTRY[name]
        cls = import_baseline_class(module_path, class_name)
        baselines.append(cls(args))          # 建立 instance，pass 全套 args
    return baselines


# --------------------------------------------------
# 6. 主程式：組合所有步驟
# --------------------------------------------------
def main():
    args = parse_args()
    logger = init_logger(args.log_level)
    set_seed(args.seed)

    # ——— 6.1 資料前處理 ———
    if args.split_raw:
        #run_preprocess(args.dataset_path, logger, args.model_info, args.qg_num, args.class_num)
        run_preprocess(args.dataset_path, logger, args.seed)

    
    # ——— 6.2 構建 baseline instance 列表 ———
    baselines = build_baselines(args)
    logger.info(f"將執行的 baseline：{', '.join([b.__class__.__name__ for b in baselines])}")

    
    # ——— 6.3 逐一跑每個 baseline 的 pipeline ———
    summary_rows = []
    for baseline in baselines:
        name = baseline.__class__.__name__
        logger.info(f"=== [{name}] setup()    ===")
        baseline.setup(args)

        # — train —
        if args.do_train:
            logger.info(f"=== [{name}] train()    ===")
            baseline.train(args)

        # — eval —
        if args.do_eval:
            logger.info(f"=== [{name}] evaluate() ===")
            metrics = baseline.evaluate(args)
            baseline.save_results(metrics, output_dir=args.output_dir)
            logger.info(f"{name} 結果：{metrics}")
            summary_rows.append((name, metrics))

    # ——— 6.4 彙總結果 ———
    if summary_rows:
        logger.info("===== 完整結果彙總 =====")
        for name, metrics in summary_rows:
            metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(f"{name:<15s} : {metrics_str}")



if __name__ == "__main__":
    main()
