import json
import gzip
import os
import random

import papermill as pm
from pathlib import Path

'''
# ------------ 可調參數 ------------
INPUT_PATH  = "./raw/item_profile_sample.json"
OUTPUT_DIR  = "../Neural-Corpus-Indexer-NCI/Data_process/NQ_dataset"
DEV_RATIO   = 0.2            # 20% 當作 dev，其餘當 train
SEED        = 42             # 為了可重現的隨機切分
# ----------------------------------
'''

# dummy annotation
dummy_annotation = {
    "long_answer":   {"start_token": -1, "end_token": -1},
    "short_answers": [],
    "yes_no_answer": "NONE"
}

def convert_to_nq_splits(input_path, output_dir, dev_ratio, seed):
    # 讀入所有 items
    with open(input_path, 'r', encoding='utf-8') as rf:
        items = json.load(rf)

    # 隨機打散並切分
    random.seed(seed)
    random.shuffle(items)
    split_idx  = int(len(items) * dev_ratio)
    dev_items   = items[:split_idx]
    train_items = items[split_idx:]

    # 輸出檔名
    splits = {
        "dev":   f"v1.0-simplified_nq-dev-all.jsonl.gz",
        "train": f"v1.0-simplified_simplified-nq-train.jsonl.gz"
    }

    # 將 items 寫成 NQ style
    def write_split(split_name, split_items):
        out_path = os.path.join(output_dir, splits[split_name])
        with gzip.open(out_path, 'wt', encoding='utf-8') as wf:
            for item in split_items:
                iid    = item["iid"]
                biz_id = item["item_id"]
                title   = f"res {iid}"
                abs_txt = "|".join(item.get("features", []))
                content = item.get("text_description", "")

                # 三段：H1 標題 + P 摘要 + 正文
                document_text = f"<H1>{title}</H1> <P>{abs_txt}</P> {content}"

                # whitespace tokenize
                tokens = document_text.split()
                document_tokens = [{"token": t} for t in tokens]

                example = {
                    "example_id":     biz_id,
                    "question_text":  f"Tell me about {title}",
                    "document_title": title,
                    "document_text":  document_text,
                    "document_tokens": document_tokens,
                    "annotations":    [dummy_annotation]
                }
                wf.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"產生 {split_name} 檔：{out_path}")

    # 執行寫檔
    write_split("dev",   dev_items)
    write_split("train", train_items)


'''
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    convert_to_nq_splits(INPUT_PATH, OUTPUT_DIR, DEV_RATIO, SEED)
    print("Done")
'''
