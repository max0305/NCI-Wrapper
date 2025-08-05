# schemes/sparse.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import logging
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schemes.base import BaseBaseline   

class SparseRetrieval(BaseBaseline):
    """
    一個最簡版的 Sparse Retrieval baseline：
    - 以 TF-IDF 向量化 text_description
    - 召回 top-k，計算 MRR / Recall@k
    """
    def __init__(self, args):
        super().__init__(args)
        self.logger: logging.Logger 

        # 之後會在 setup() 產生
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix = None        # 稀疏矩陣
        self.doc_ids: list[str] = []  # 與矩陣 row 對齊

    def _load_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------
    # 1) setup：向量化文件
    # -------------------------------------------------
    def setup(self, args):
        # 讀取原始未切分檔案
        corpus_path = "./dataset/raw/item_profile_sample.json"
        data = self._load_json(corpus_path)

        # 建立 corpus 與 doc_ids
        corpus  = [item["text_description"] for item in data]
        self.doc_ids = [item["item_id"] for item in data]

        # TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=20000
        )
        self.doc_matrix = self.vectorizer.fit_transform(corpus)
        self.logger.info(f"向量化完成，文件數 = {len(corpus)}")

    # -------------------------------------------------
    # 2) train：TF-IDF 不用訓練 → 空實作
    # -------------------------------------------------
    def train(self, args):
        self.logger.info("SparseRetrieval 無需額外訓練，跳過。")

    # -------------------------------------------------
    # 3) evaluate：對 eval split 跑簡易搜尋
    # -------------------------------------------------
    def evaluate(self, args) -> Dict[str, float]:
        # 讀 dev（或 eval）查詢
        split_path = "./dataset/dev.json"
        data = self._load_json(split_path)
        queries  = [item["text_description"] for item in data]
        q_ids    = [item["item_id"] for item in data]

        # 查詢向量
        q_matrix = self.vectorizer.transform(queries)
        sims = cosine_similarity(q_matrix, self.doc_matrix, dense_output=False)

        recall_k = 10
        hits, mrr = 0, 0.0
        for qi, qid in enumerate(q_ids):
            # 取 Top-k，先把自己當 ground truth
            sim_row = sims[qi].toarray().ravel()
            topk_idx = sim_row.argsort()[::-1][:recall_k]
            topk_doc_ids = [self.doc_ids[i] for i in topk_idx]

            if qid in topk_doc_ids:
                hits += 1
                rank = topk_doc_ids.index(qid) + 1
                mrr += 1 / rank

        mrr   /= len(q_ids)
        recall =  hits / len(q_ids)
        return {f"Recall@{recall_k}": recall, "MRR": mrr}

    # -------------------------------------------------
    # 4) save_results：metrics + index
    # -------------------------------------------------
    def save_results(self, metrics: dict, output_dir: Path):
        out_dir = self.run_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. 存指標
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False))

        # 2. 存 vocab —— 先把 numpy.int64 轉成 int
        safe_vocab = {k: int(v) for k, v in self.vectorizer.vocabulary_.items()}
        (out_dir / "vocab.json").write_text(
            json.dumps(safe_vocab, indent=2, ensure_ascii=False))

        self.logger.info(f"指標與 vocab 已輸出至 {out_dir}")

