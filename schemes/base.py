# schemes/base.py
# ------------------------------------------------------
# 依賴最小：只用到標準庫 + torch / numpy
# ------------------------------------------------------
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import json
import logging
import os
import random
import time

import numpy as np
import torch


class BaseBaseline(ABC):
    """
    所有 baseline wrapper 的共同基底。
    - 不直接實例化，僅被繼承。
    - 強制子類實作四大抽象方法：
      setup(), train(), evaluate(), save_results()
    """

    # ---------- 共用初始化 --------------------------------------------------
    def __init__(self, args):
        self.args = args                    # CLI 旗標物件
        self.seed = args.seed
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # 建立主要輸出資料夾：outputs/<method>/<timestamp>/
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(args.output_dir) / self.__class__.__name__.lower() / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Logger：console + file
        self.logger = self._init_logger(self.run_dir / "run.log", args.log_level)

        # 固定隨機種子
        self._set_seed(self.seed)

        # 共用屬性（子類可覆寫）
        self.processed_dir = Path(args.dataset_path) / "processed"
        self.splits_dir    = Path(args.dataset_path) / "splits"

    # ---------- 四大抽象方法（子類必須實作） -------------------------------
    @abstractmethod
    def setup(self) -> None:
        """準備資料、模型、optimizer…（不執行 heavy 訓練）"""
        ...

    @abstractmethod
    def train(self) -> None:
        """執行訓練迴圈；結束後確保 checkpoint 已存檔"""
        ...

    @abstractmethod
    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """
        執行驗證 / 測試，回傳指標 dict
        例如：{"MRR": 0.412, "Recall@10": 0.673}
        """
        ...

    @abstractmethod
    def save_results(self, metrics: Dict[str, float], output_dir: Path) -> None:
        """將指標與重要檔案複製 / 寫入 output_dir"""
        ...

    # ---------- 通用工具 ----------------------------------------------------
    # 1) Logger
    def _init_logger(self, log_path: Path, level: str) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(level.upper())
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    # 2) 固定隨機種子
    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 3) 資料載入範例（可被子類覆寫）
    def get_dataloader(self, split: str):
        """
        預設：讀 jsonl -> Dataset -> DataLoader
        若各 baseline 需要特殊處理，可以在子類重新實作
        """
        # TODO: 真正 dataset 實作
        raise NotImplementedError("請在子類中實作 get_dataloader() 或覆寫此方法")

    # 4) Checkpoint 讀寫
    def save_checkpoint(self, state: Dict[str, Any], name: str = "last.pt"):
        ckpt_path = self.run_dir / name
        torch.save(state, ckpt_path)
        self.logger.info(f"✓ 已儲存 checkpoint 至 {ckpt_path}")

    def load_checkpoint(self, ckpt_path: Path):
        if ckpt_path.exists():
            self.logger.info(f"⟳ 讀取 checkpoint：{ckpt_path}")
            return torch.load(ckpt_path, map_location=self.device)
        else:
            self.logger.warning(f"找不到 checkpoint：{ckpt_path}")
            return None

    # 5) 測速用 decorator
    def measure_time(fn):
        def wrapper(self, *args, **kwargs):
            t0 = time.time()
            result = fn(self, *args, **kwargs)
            self.logger.info(f"{fn.__name__} 花費 {time.time()-t0:.2f}s")
            return result
        return wrapper


if __name__ == "__main__":
    BaseBaseline()