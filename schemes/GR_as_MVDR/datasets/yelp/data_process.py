# data_process.py
def GR_as_MVDR_data_process(input_train_path, input_dev_path):
    import json
    import csv
    import os

    BASE_DIR = os.path.dirname(__file__) + '/'

    ## define
    TRAIN_RAW       = input_train_path
    DEV_RAW         = input_dev_path
    QRELS           = BASE_DIR + "qrel.dev.small10.tsv"
    TRIPLES         = BASE_DIR + "triples.train.small10.tsv"
    COLLECTIONS     = BASE_DIR + "collection.small10.tsv"
    QUERIES         = BASE_DIR + "queries.all.small10.tsv"
    EVAL_QUERIES    = BASE_DIR + "queries.dev.small10.tsv"
    qrel_data = []
    pids_all = set()

    with open(TRAIN_RAW, 'r', encoding='utf-8') as rf:
        train_items = json.load(rf)

    with open(DEV_RAW, 'r', encoding='utf-8') as rf:
        dev_items = json.load(rf)

    def query_generation(qid, text):
        return f"What are the main attributes and price level of Restaurant #{qid}?"

    ## collection.tsv
    with open(COLLECTIONS, 'w', newline="") as wf:
        for item in train_items + dev_items:
            pid = item["iid"]
            biz_id = item["item_id"]
            title   = f"Restaurant #{pid}"

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
            abs_txt = "Attributes and ratings — " + " & ".join(rating_pairs)

            content = item.get("text_description", "")

            # 正文
            document_text = f"{title} | {abs_txt}. {content}"

            pids_all.add(pid)

            tsv_writer = csv.writer(wf, delimiter="\t")
            tsv_writer.writerows([[pid, document_text]])


    ## queries.all.tsv
    ## queries.dev.tsv
    with open(QUERIES, 'w', newline="") as all_file:
        with open(EVAL_QUERIES, 'w', newline="") as dev_file:
            for index, item in enumerate(train_items + dev_items):
                pid = item["iid"]

                query = query_generation(pid, " ")

                qrel_data.append([index, [pid]])

                train_writer = csv.writer(all_file, delimiter="\t")
                train_writer.writerows([[index, query]])

                # dev start
                if index >= len(train_items):
                        dev_writer = csv.writer(dev_file, delimiter="\t")
                        dev_writer.writerows([[index, query]])



    # aug_ground_truth() in here


    ## qrel.dev.tsv
    with open(QRELS, 'w', newline="") as wf:
        for index, pids in qrel_data:
            if index >= len(train_items):
                for pid in pids:
                    tsv_writer = csv.writer(wf, delimiter="\t")
                    tsv_writer.writerows([[index, 0, pid, 1]])


    # triples.train.small10.tsv
    with open(TRIPLES, 'w', newline="") as wf:
        for index, pids in qrel_data:
            if index < len(train_items):
                positive_pid = pids[0]
                negative_pids = [x for x in pids_all if x not in pids]
                tsv_writer = csv.writer(wf, delimiter="\t")
                tsv_writer.writerows([[index, positive_pid] + negative_pids])





