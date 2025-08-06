from __future__ import annotations
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict
import os, shutil
import json
import re
import glob

from .base import BaseBaseline

class GR(BaseBaseline): 
    """
    讓 GR 專案可以被 main.py 以統一流程呼叫：
        setup()  -> train()  -> evaluate()  -> save_results()
    """

    def __init__(self, args):
        super().__init__(args)


    # ----------------------------------------------------------------------

    def setup(self, args) -> None:
        """
        1) 檢查 GR 原始碼是否存在於 schemes/GR-as-MVDR
        2) 設定工作路徑、環境變數
        3) 做資料格式轉換
        """
        self.repo_dir: Path = Path(__file__).parent / "GR_as_MVDR"
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"GR 原始碼不存在：{self.repo_dir}\n"
            )
        
        if args.do_preprocess:
            from schemes.GR_as_MVDR.datasets.yelp.data_process import GR_as_MVDR_data_process

            input_train_path = "./dataset/train.json"
            input_dev_path = "./dataset/dev.json"

            self.logger.info("開始 GR_as_MVDR 資料前處理 …")
            GR_as_MVDR_data_process(input_train_path=input_train_path, input_dev_path=input_dev_path)
            self.logger.info("資料處理完成！")
                

        # 建立工作子資料夾儲存 ckpt / log
        self.work_dir: Path = self.run_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 將共用 processed 資料夾 export 成 GR 能認得的環境變數
        self.env = {**os.environ, "GR_FINAL": os.path.join(os.getcwd(), str(self.run_dir))}

        self.logger.info(f"GR setup 完成；repo={self.repo_dir}")


    # ----------------------------------------------------------------------

    def train(self, args):
        """
        呼叫 GR 的 train
        確保此函式在子程序 *完成訓練* 前不會返回
        """
        print(self.run_dir)
        cmd = [f". common_settings/common.params.nq320k.sh && bash gr_run_scripts/run.nq.sh train {self.run_dir}"]       

        self._run_subprocess(cmd, self.repo_dir, "訓練")

    # ----------------------------------------------------------------------

    def evaluate(self, args):
        """
        1) 找到最佳 checkpoint
        2) 呼叫 infer.sh
        3) 解析輸出，回傳指標 dict
        """
        pattern = os.path.join(self.run_dir, "..", "**", "*.pt")
        pt_files = glob.glob(pattern, recursive=True)
        if not pt_files:
            raise 
        latest = max(pt_files, key=os.path.getmtime)
        print(latest)

        cmd = [f". common_settings/common.params.nq320k.sh && bash gr_run_scripts/run.nq.sh index {self.run_dir} {latest}"]

        self._run_subprocess(cmd, self.repo_dir, "推論")

        metrics = self._parse_metrics(result_path=self.run_dir / "metrics.json")



        # TODO: 依 GR 實際輸出格式萃取 MRR / Recall@K
        return metrics

    # ----------------------------------------------------------------------

    def save_results(self, metrics: Dict[str, float], output_dir: Path) -> None:
        """
        將指標與關鍵檔案寫到 output_dir/GR/…
        """
        # 已經在 GR 中存了
        pass


    # ---------- 私用輔助 ---------------------------------------------------

    def _run_subprocess(self, cmd_list, cwd, stage: str) -> None:
        """
        執行 bash 指令；回傳 stdout 字串。 任何非零退出碼立即 raise。
        """
        self.logger.info(f"GR {stage} 指令：{' '.join(str(x) for x in cmd_list)}")
        subprocess.run(
            cmd_list,
            shell=True, 
            executable="/bin/bash",
            cwd=cwd,
            env=self.env,
            text=True,
        )

        self.logger.info(f"GR {stage} 完成")

    def _parse_metrics(self, result_path) -> Dict[str, float]:
        """
        從 stdout 或 infer_dir/metrics.txt 萃取結果，轉成統一格式
        這裡示範抓 'MRR@10 = 0.512' 與 'Recall@100 = 0.842'
        """
        if not result_path.exists():
            raise FileNotFoundError(f"找不到 {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        assert "Recall@10" in metrics and "MRR@10" in metrics, "缺少必要欄位"

        return metrics

    # ----------------------------------------------------------------------