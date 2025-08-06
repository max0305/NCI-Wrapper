import csv

# 收集所有在 triples 裡出現的 docid
docids = set()
with open("triples.train.small10.tsv") as f:
    for line in f:
        parts = line.strip().split("\t")
        # parts[0] 是 qid，parts[1:] 全是 docid
        docids.update(parts[1:])

# 寫入新的 collection.tsv
with open("collection.small10.tsv", "w", encoding="utf-8") as outf:
    writer = csv.writer(outf, delimiter="\t")
    for docid in sorted(docids, key=int):
        # 你可以把 DUMMY 換成 docid 自己，或其他任何 placeholders
        writer.writerow([docid, "DUMMY_TEXT_FOR_DOC_" + docid])

print("生成 collection.tsv，包含", len(docids), "個 docid")