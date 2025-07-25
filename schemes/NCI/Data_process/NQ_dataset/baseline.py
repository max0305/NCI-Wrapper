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

def convert_to_nq_splits(input_train_path, input_dev_path, output_dir="./", seed=42):
    # 讀入所有 items
    with open(input_train_path, 'r', encoding='utf-8') as rf:
        train_items = json.load(rf)

    with open(input_dev_path, 'r', encoding='utf-8') as rf:
        dev_items = json.load(rf)

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
                title   = f"Restaurant #{iid}"

                price_val = float(item.get("price", 0.0))

                # 將 feature + importance 組成 "food: 0.40 | cleaness: 0.20" 
                rating_pairs = [f"price: {price_val:.2f}"] + [
                    f"{feat}: {rating:.2f}"
                    for feat, rating in zip(
                        item.get("features", []),
                        item.get("importance", [])
                    )
                ]
                # 加上前綴，並以 "|" 分隔
                abs_txt = "Attributes and ratings — " + " | ".join(rating_pairs)

                content = item.get("text_description", "")

                # 三段：H1 標題 + P 摘要 + 正文
                document_text = f"<H1>{title}</H1> <P>{abs_txt}</P> {content}"

                # whitespace tokenize
                tokens = document_text.split()
                document_tokens = [{"token": t} for t in tokens]

                # query ground truth
                question_text = (
                    f"What are the main attributes and price level of {title}?"
                )

                example = {
                    "example_id":     biz_id,
                    "question_text":  question_text, 
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
