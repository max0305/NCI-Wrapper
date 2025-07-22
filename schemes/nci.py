from __future__ import annotations
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict
import os, shutil
import json

from .base import BaseBaseline

class NCI(BaseBaseline): 
    """
    讓 NCI 專案可以被 main.py 以統一流程呼叫：
        setup()  -> train()  -> evaluate()  -> save_results()
    """

    def __init__(self, args):
        super().__init__(args)


    # ----------------------------------------------------------------------

    def setup(self) -> None:
        """
        1) 檢查 NCI 原始碼是否存在於 schemes/Neural-Corpus-Indexer-NCI
        2) 設定工作路徑、環境變數
        3) 做資料格式轉換
        """
        self.repo_dir: Path = Path(__file__).parent / "Neural-Corpus-Indexer-NCI"
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"NCI 原始碼不存在：{self.repo_dir}\n"
                "請先 `git clone https://github.com/solidsea98/Neural-Corpus-Indexer-NCI"
            )

        # 建立工作子資料夾儲存 ckpt / log
        self.work_dir: Path = self.run_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 將共用 processed 資料夾 export 成 NCI 能認得的環境變數
        self.env = {**os.environ, "NCI_DATA": str(self.processed_dir)}

        # TODO: 如 NCI 需要把 processed 轉成 partition 檔，可在此呼叫
        # self._prepare_partition_files()

        self.logger.info(f"NCI setup 完成；repo={self.repo_dir}, work_dir={self.work_dir}")


    # ----------------------------------------------------------------------

    def train(self):
        """
        呼叫 NCI 的 train.sh
        確保此函式在子程序 *完成訓練* 前不會返回
        """
        cmd = ["bash", "train.sh",
               "--data", "$NCI_DATA",          # train.sh 裡要能展開此 env
               "--output", str(self.work_dir),
               "--cuda", self.args.device]     # 也可以改用 CUDA_VISIBLE_DEVICES

        self._run_subprocess(cmd, "訓練")

    # ----------------------------------------------------------------------

    def evaluate(self, split):
        """
        1) 找到最佳 checkpoint
        2) 呼叫 infer.sh
        3) 解析輸出，回傳指標 dict
        """
        ckpt = self.work_dir / "checkpoints" / "best.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"找不到最佳模型權重：{ckpt}")

        infer_dir = self.work_dir / "infer"
        infer_dir.mkdir(exist_ok=True)

        cmd = ["bash", "infer.sh",
               "--data", "$NCI_DATA",
               "--ckpt", str(ckpt),
               "--output", str(infer_dir),
               "--split", split]

        stdout = self._run_subprocess(cmd, "推論")

        # TODO: 依 NCI 實際輸出格式萃取 MRR / Recall@K
        metrics = self._parse_metrics(stdout, infer_dir)
        return metrics

    # ----------------------------------------------------------------------

    def save_results(self, metrics: Dict[str, float], output_dir: Path) -> None:
        """
        將指標與關鍵檔案寫到 output_dir/nci/…
        """
        nci_out = Path(output_dir) / "nci"
        nci_out.mkdir(parents=True, exist_ok=True)

        # 1) dump 指標
        (nci_out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

        # 2) 複製 checkpoint / log
        shutil.copytree(self.work_dir, nci_out / "work", dirs_exist_ok=True)

        self.logger.info(f"✓ 已儲存 NCI 結果至 {nci_out}")


    # ---------- 私用輔助 ---------------------------------------------------

    def _run_subprocess(self, cmd_list, stage: str) -> str:
        """
        執行 bash 指令；回傳 stdout 字串。 任何非零退出碼立即 raise。
        """
        self.logger.info(f"NCI {stage} 指令：{' '.join(cmd_list)}")
        res = subprocess.run(
            cmd_list,
            cwd=self.repo_dir,
            env=self.env,
            text=True,
            capture_output=True
        )
        # 印 log
        self.logger.debug(f"stdout:\n{res.stdout}")
        self.logger.debug(f"stderr:\n{res.stderr}")

        if res.returncode != 0:
            self.logger.error(f"NCI {stage} 失敗，exit={res.returncode}")
            raise subprocess.CalledProcessError(res.returncode, cmd_list, res.stdout, res.stderr)

        self.logger.info(f"NCI {stage} 完成")
        return res.stdout

    def _parse_metrics(self, stdout: str, infer_dir: Path) -> Dict[str, float]:
        """
        從 stdout 或 infer_dir/metrics.txt 萃取結果，轉成統一格式
        這裡示範抓 'MRR@10 = 0.512' 與 'Recall@100 = 0.842'
        """
        mrr, r100 = 0.0, 0.0
        for line in stdout.splitlines():
            if "MRR@" in line:
                mrr = float(line.strip().split()[-1])
            if "Recall@100" in line:
                r100 = float(line.strip().split()[-1])

        metrics = {"MRR@10": mrr, "Recall@100": r100}
        self.logger.info(f"指標：{metrics}")
        return metrics

    # ----------------------------------------------------------------------