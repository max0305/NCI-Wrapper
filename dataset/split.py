import json
import random
from pathlib import Path
from typing import List, Tuple, Union

def split_data(
    items: List[dict],
    train_ratio: float,
    seed: int
) -> Tuple[List[dict], List[dict]]:
    """
    依 seed 隨機打亂 items，並依 train_ratio 切分。
    回傳 (train_items, eval_items)。
    """
    random.seed(seed)
    items_copy = items[:]          # 複製一份，避免修改原 list
    random.shuffle(items_copy)
    n_train = int(len(items_copy) * train_ratio)
    return items_copy[:n_train], items_copy[n_train:]


def load_json(path: Union[str, Path]) -> List[dict]:
    """
    讀取 JSON 檔，並檢查它必須是一個 list。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"輸入檔案 {path} 必須是一個 JSON list")
    return data


def save_json(data: List[dict], path: Union[str, Path]) -> None:
    """
    將 list of dict 寫入 JSON 檔，並自動建立必要的資料夾。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_and_save(
    input_path: Union[str, Path],
    train_ratio: float = 0.8,
    seed: int = 42,
    train_filename: str = "train.json",
    eval_filename: str = "dev.json"
) -> None:
    """
    一次完成讀取、切分與儲存：
      1. 讀取 input_path
      2. 隨機切分（train_ratio, seed）
      3. 分別寫入 output_dir/train_filename 與 output_dir/eval_filename
    """
    items = load_json(input_path)
    train_items, dev_items = split_data(items, train_ratio, seed)

    out_dir = "./dataset/"
    save_json(train_items, out_dir + train_filename)
    save_json(dev_items,  out_dir + eval_filename)
