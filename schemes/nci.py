from __future__ import annotations
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict
import os, shutil
import json
import re

from .base import BaseBaseline

class NCI(BaseBaseline): 
    """
    讓 NCI 專案可以被 main.py 以統一流程呼叫：
        setup()  -> train()  -> evaluate()  -> save_results()
    """

    def __init__(self, args):
        super().__init__(args)


    # ----------------------------------------------------------------------

    def setup(self, args) -> None:
        """
        1) 檢查 NCI 原始碼是否存在於 schemes/Neural-Corpus-Indexer-NCI
        2) 設定工作路徑、環境變數
        3) 做資料格式轉換
        """
        self.repo_dir: Path = Path(__file__).parent / "NCI"
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"NCI 原始碼不存在：{self.repo_dir}\n"
                "請先 `git clone https://github.com/solidsea98/Neural-Corpus-Indexer-NCI"
            )
        
        if args.do_preprocess:
            from schemes.NCI.Data_process.NQ_dataset.baseline import convert_to_nq_splits
            from schemes.NCI.Data_process.NQ_dataset.data_process import data_process

            input_train_path = "./dataset/train.json"
            input_dev_path = "./dataset/dev.json"
            output_dir = "./schemes/NCI/Data_process/NQ_dataset/"

            self.logger.info("開始轉換為 nq 格式 …")
            convert_to_nq_splits(input_train_path=input_train_path, input_dev_path=input_dev_path, output_dir=output_dir)
            self.logger.info("開始資料前處理 …")
            data_process(args.model_info, args.qg_num, args.class_num, output_dir)
            self.logger.info("資料處理完成！")
                

        # 建立工作子資料夾儲存 ckpt / log
        self.work_dir: Path = self.run_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 將共用 processed 資料夾 export 成 NCI 能認得的環境變數
        self.env = {**os.environ, "NCI_FINAL": os.path.join(os.getcwd(), str(self.run_dir))}

        # TODO: 如 NCI 需要把 processed 轉成 partition 檔，可在此呼叫
        # self._prepare_partition_files()

        self.logger.info(f"NCI setup 完成；repo={self.repo_dir}")


    # ----------------------------------------------------------------------

    def train(self, args):
        """
        呼叫 NCI 的 train.sh
        確保此函式在子程序 *完成訓練* 前不會返回
        """
        print(os.path.join(self.repo_dir, "NCI_model", "train.sh"))
        cmd = ["bash", "train.sh",
               args.model_info,
               args.n_gpu,
               args.epoch,
        ]       

        self._run_subprocess(cmd, os.path.join(self.repo_dir, "NCI_model"), "訓練")

    # ----------------------------------------------------------------------

    def evaluate(self, args):
        """
        1) 找到最佳 checkpoint
        2) 呼叫 infer.sh
        3) 解析輸出，回傳指標 dict
        """

        cmd = ["bash", "infer.sh",
               args.model_info,
        ]

        self._run_subprocess(cmd, os.path.join(self.repo_dir, "NCI_model"), "推論")

        metrics = self._parse_metrics(result_path=self.run_dir / "metrics.json")



        # TODO: 依 NCI 實際輸出格式萃取 MRR / Recall@K
        return metrics

    # ----------------------------------------------------------------------

    def save_results(self, metrics: Dict[str, float], output_dir: Path) -> None:
        """
        將指標與關鍵檔案寫到 output_dir/nci/…
        """
        # 已經在 nci 的 main.py 中存了
        '''
        nci_out = Path(output_dir) / "nci"
        nci_out.mkdir(parents=True, exist_ok=True)

        # 1) dump 指標
        (nci_out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

        # 2) 複製 checkpoint / log
        shutil.copytree(self.work_dir, nci_out / "work", dirs_exist_ok=True)

        self.logger.info(f"✓ 已儲存 NCI 結果至 {nci_out}")
        '''


    # ---------- 私用輔助 ---------------------------------------------------

    def _run_subprocess(self, cmd_list, cwd, stage: str) -> None:
        """
        執行 bash 指令；回傳 stdout 字串。 任何非零退出碼立即 raise。
        """
        self.logger.info(f"NCI {stage} 指令：{' '.join(str(x) for x in cmd_list)}")
        subprocess.run(
            cmd_list,
            cwd=cwd,
            env=self.env,
            text=True,
        )

        self.logger.info(f"NCI {stage} 完成")

    def _parse_metrics(self, result_path) -> Dict[str, float]:
        """
        從 stdout 或 infer_dir/metrics.txt 萃取結果，轉成統一格式
        這裡示範抓 'MRR@10 = 0.512' 與 'Recall@100 = 0.842'
        """
        if not result_path.exists():
            raise FileNotFoundError(f"找不到 {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        assert "Recall" in metrics and "MRR" in metrics, "缺少必要欄位"

        return metrics

    # ----------------------------------------------------------------------