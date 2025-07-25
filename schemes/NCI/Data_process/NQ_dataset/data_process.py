def data_process(model_info, qg_num, class_num, output_dir):
    import pandas as pd
    import pickle
    import torch
    import os
    import re
    import random
    import csv
    import jsonlines
    import numpy as np
    import pickle
    import time
    import gzip
    from tqdm import tqdm, trange
    from sklearn.cluster import KMeans
    from typing import Any, List, Sequence, Callable
    from itertools import islice, zip_longest
    from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSeq2SeqLM
    from sklearn.cluster import MiniBatchKMeans
    import subprocess
    from pathlib import Path

    import textwrap, sys

    base_dir = output_dir

###################################################################################################

    nq_dev = []

    with gzip.open(os.path.join(base_dir, "v1.0-simplified_nq-dev-all.jsonl.gz"), "r+") as f:
        for item in jsonlines.Reader(f):
            
            arr = []
            ## question_text
            question_text = item['question_text']
            arr.append(question_text)

            tokens = []
            for i in item['document_tokens']:
                tokens.append(i['token'])
            document_text = ' '.join(tokens)
            
            ## example_id
            example_id = str(item['example_id'])
            arr.append(example_id)

            # document_text = item['document_text']
            ## long_answer
            annotation = item['annotations'][0]
            has_long_answer = annotation['long_answer']['start_token'] >= 0

            long_answers = [
                a['long_answer']
                for a in item['annotations']
                if a['long_answer']['start_token'] >= 0 and has_long_answer
            ]
            if has_long_answer:
                start_token = long_answers[0]['start_token']
                end_token = long_answers[0]['end_token']
                x = document_text.split(' ')
                long_answer = ' '.join(x[start_token:end_token])
                long_answer = re.sub('<[^<]+?>', '', long_answer).replace('\n', '').strip()
            arr.append(long_answer) if has_long_answer else arr.append('')

            # short_answer
            has_short_answer = annotation['short_answers'] or annotation['yes_no_answer'] != 'NONE'
            short_answers = [
                a['short_answers']
                for a in item['annotations']
                if a['short_answers'] and has_short_answer
            ]
            if has_short_answer and len(annotation['short_answers']) != 0:
                sa = []
                for i in short_answers[0]:
                    start_token_s = i['start_token']
                    end_token_s = i['end_token']
                    shorta = ' '.join(x[start_token_s:end_token_s])
                    sa.append(shorta)
                short_answer = '|'.join(sa)
                short_answer = re.sub('<[^<]+?>', '', short_answer).replace('\n', '').strip()
            arr.append(short_answer) if has_short_answer else arr.append('')

            ## title
            arr.append(item['document_title'])

            ## abs
            if document_text.find('<P>') != -1:
                abs_start = document_text.index('<P>')
                abs_end = document_text.index('</P>')
                abs = document_text[abs_start+3:abs_end]
            else:
                abs = ''
            arr.append(abs)

            ## content
            if document_text.rfind('</Ul>') != -1:
                final = document_text.rindex('</Ul>')
                document_text = document_text[:final]
                if document_text.rfind('</Ul>') != -1:
                    final = document_text.rindex('</Ul>')
                    content = document_text[abs_end+4:final]
                    content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                    content = re.sub(' +', ' ', content)
                    arr.append(content)
                else:
                    content = document_text[abs_end+4:final]
                    content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                    content = re.sub(' +', ' ', content)
                    arr.append(content)
            else:
                content = document_text[abs_end+4:]
                content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                content = re.sub(' +', ' ', content)
                arr.append(content)
            doc_tac = item['document_title'] + ": " + abs + ". " + content
            arr.append(doc_tac)
            language = 'en'
            arr.append(language)
            nq_dev.append(arr)

    nq_dev_df = pd.DataFrame(nq_dev)
    nq_dev_df.to_csv(os.path.join(base_dir, "nq_dev.tsv"), sep="\t", mode = 'w', header=None, index =False)

###################################################################################################

    in_path  = os.path.join(base_dir, "v1.0-simplified_simplified-nq-train.jsonl.gz")
    out_path = os.path.join(base_dir, "nq_train.tsv")
    batch_size = 20000

    if os.path.exists(out_path):
        # 用 'w' 打開再關閉，就會把檔案內容全部清空
        open(out_path, 'w', encoding='utf-8').close()

    buffer = []
    with gzip.open(in_path, "r+") as f:
        reader = jsonlines.Reader(f)
        for enum_index, item in enumerate(reader, start=1):
            ## question_text
            arr = []
            question_text = item['question_text']
            arr.append(question_text)

            ## example_id
            example_id = str(item['example_id'])
            arr.append(example_id)
            
            document_text = item['document_text']
            
            ## long_answer
            annotation = item['annotations'][0]
            has_long_answer = annotation['long_answer']['start_token'] >= 0

            long_answers = [
                a['long_answer']
                for a in item['annotations']
                if a['long_answer']['start_token'] >= 0 and has_long_answer
            ]
            if has_long_answer:
                start_token = long_answers[0]['start_token']
                end_token = long_answers[0]['end_token']
                x = document_text.split(' ')
                long_answer = ' '.join(x[start_token:end_token])
                long_answer = re.sub('<[^<]+?>', '', long_answer).replace('\n', '').strip()
            arr.append(long_answer) if has_long_answer else arr.append('')

            # short_answer
            has_short_answer = annotation['short_answers'] or annotation['yes_no_answer'] != 'NONE'
            short_answers = [
                a['short_answers']
                for a in item['annotations']
                if a['short_answers'] and has_short_answer
            ]
            if has_short_answer and len(annotation['short_answers']) != 0:
                sa = []
                for i in short_answers[0]:
                    start_token_s = i['start_token']
                    end_token_s = i['end_token']
                    shorta = ' '.join(x[start_token_s:end_token_s])
                    sa.append(shorta)
                short_answer = '|'.join(sa)
                short_answer = re.sub('<[^<]+?>', '', short_answer).replace('\n', '').strip()
            arr.append(short_answer) if has_short_answer else arr.append('')

            ## title
            if document_text.find('<H1>') != -1:
                title_start = document_text.index('<H1>')
                title_end = document_text.index('</H1>')
                title = document_text[title_start+4:title_end]
            else:
                title = ''
            arr.append(title)

            ## abs
            if document_text.find('<P>') != -1:
                abs_start = document_text.index('<P>')
                abs_end = document_text.index('</P>')
                abs = document_text[abs_start+3:abs_end]
            else:
                abs = ''
            arr.append(abs)

            ## content
            if document_text.rfind('</Ul>') != -1:
                final = document_text.rindex('</Ul>')
                document_text = document_text[:final]
                if document_text.rfind('</Ul>') != -1:
                    final = document_text.rindex('</Ul>')
                    content = document_text[abs_end+4:final]
                    content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                    content = re.sub(' +', ' ', content)
                    arr.append(content)
                else:
                    content = document_text[abs_end+4:final]
                    content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                    content = re.sub(' +', ' ', content)
                    arr.append(content)
            else:
                content = document_text[abs_end+4:]
                content = re.sub('<[^<]+?>', '', content).replace('\n', '').strip()
                content = re.sub(' +', ' ', content)
                arr.append(content)

            doc_tac = title + ": " + abs + ". " + content
            arr.append(doc_tac)

            language = 'en'
            arr.append(language)
            buffer.append(arr)

            # 每累積 batch_size 筆就寫一次、清空 buffer
            if enum_index % batch_size == 0:
                df = pd.DataFrame(buffer)
                df.to_csv(out_path, sep="\t", mode="a", header=False, index=False)
                buffer.clear()

        # 寫入最後不足 batch_size 的餘數
        if buffer:
            df = pd.DataFrame(buffer)
            df.to_csv(out_path, sep="\t", mode="a", header=False, index=False)

###################################################################################################

    ## Mapping tool

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def lower(x):
        # 用 BERT 分詞器把原始文字拆成 sub‑word tokens
        text = tokenizer.tokenize(x)
        # 再把 tokens 轉成對應的 vocabulary id
        id_ = tokenizer.convert_tokens_to_ids(text)
        # 然後再 decode 回字串，得到 BERT 規範的 "小寫＋sub‑word 斷詞" 格式，如 "New York City" 轉成 “new york city”
        return tokenizer.decode(id_)

###################################################################################################

    ## doc_tac denotes the concatenation of title, abstract and content

    # 結構化到 DataFrame
    nq_dev = pd.read_csv(os.path.join(base_dir, 'nq_dev.tsv'), \
                        names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'doc_tac', 'language'],\
                        header=None, sep='\t', nrows=100)

    nq_train = pd.read_csv(os.path.join(base_dir, 'nq_train.tsv'), \
                        names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'doc_tac', 'language'],\
                        header=None, sep='\t', nrows=20)

    #
    nq_dev['title'] = nq_dev['title'].map(lower)
    nq_train['title'] = nq_train['title'].map(lower)

###################################################################################################

    ## Concat train doc and validation doc to obtain full document collection

    nq_all_doc = nq_train.append(nq_dev)
    nq_all_doc.reset_index(inplace = True)

###################################################################################################

    ## Remove duplicated documents based on titles

    nq_all_doc.drop_duplicates('title', inplace = True)
    nq_all_doc.reset_index(inplace = True)

###################################################################################################

    ## The total amount of documents : 109739

    print("data lengh: ", len(nq_all_doc))

###################################################################################################

    ## Construct mapping relation

    title_doc = {}
    title_doc_id = {}
    id_doc = {}
    ran_id_old_id = {}
    idx = 0
    for i in range(len(nq_all_doc)):
        # 標題 → 文件文字
        title_doc[nq_all_doc['title'][i]] =  nq_all_doc['doc_tac'][i]

        # 標題 → 新整數 id (idx)
        title_doc_id[nq_all_doc['title'][i]] = idx

        # 新整數 id (idx)→ 文件文字
        id_doc[idx] = nq_all_doc['doc_tac'][i]

        # 新整數 id (idx)→ 原始 id
        ran_id_old_id[idx] = nq_all_doc['id'][i]

        # 新整數 id
        idx += 1

###################################################################################################

    ## Construct Document Content File

    train_file = open(os.path.join(base_dir, "NQ_doc_content.tsv"), 'w') 

    for docid in id_doc.keys():
        train_file.write('\t'.join([str(docid), '', '', id_doc[docid], '', '', 'en']) + '\n')
        train_file.flush()

###################################################################################################

##############################################
# Generate BERT embeddings for each document #
##############################################

###################################################################################################

    subprocess.run(
        ["bash", "bert/bert_NQ.sh", "1"],
        cwd=base_dir,
        text=True,
        check=True
    )


###################################################################################################

    ## Concat bert embedding
    output_bert_base_tensor_nq_qg = []
    output_bert_base_id_tensor_nq_qg = []
    for num in trange(1):
        with open(os.path.join(base_dir, f'bert/pkl/NQ_output_tensor_512_content_{num}.pkl'), 'rb') as f:
            data = pickle.load(f)
        f.close()
        output_bert_base_tensor_nq_qg.extend(data)

        with open(os.path.join(base_dir,f'bert/pkl/NQ_output_tensor_512_content_{num}_id.pkl'), 'rb') as f:
            data = pickle.load(f)
        f.close()
        output_bert_base_id_tensor_nq_qg.extend(data)

    train_file = open(os.path.join(base_dir,f"bert/NQ_doc_content_embedding_bert_512.tsv"), 'w') 

    for idx, doc_tensor in enumerate(output_bert_base_tensor_nq_qg):
        embedding = '|'.join([str(elem) for elem in doc_tensor])
        train_file.write('\t'.join([str(output_bert_base_id_tensor_nq_qg[idx]), '', '', '', '', '', 'en', embedding]) + '\n')
        train_file.flush()

###################################################################################################

#############################################################
# Apply Hierarchical K-Means on it to generate semantic IDs #
#############################################################

###################################################################################################

    subprocess.run(
        ["bash", "kmeans/kmeans_NQ.sh", f"{class_num}", f"{model_info}"],
        cwd=base_dir,
        text=True,
        check=True
    )


###################################################################################################

    with open(os.path.join(base_dir, f'kmeans/IDMapping_NQ_bert_512_k{class_num}_c{class_num}_seed_7.pkl'), 'rb') as f:
        kmeans_nq_doc_dict = pickle.load(f)
    ## random id : newid
    new_kmeans_nq_doc_dict_512 = {}
    for old_docid in kmeans_nq_doc_dict.keys():
        new_kmeans_nq_doc_dict_512[str(old_docid)] = '-'.join(str(elem) for elem in kmeans_nq_doc_dict[old_docid])
        
    new_kmeans_nq_doc_dict_512_int_key = {}
    for key in new_kmeans_nq_doc_dict_512:
        new_kmeans_nq_doc_dict_512_int_key[int(key)] = new_kmeans_nq_doc_dict_512[key]


###################################################################################################

####################
# Query Generation #
####################

###################################################################################################

    subprocess.run(
        ["bash", "qg/NQ_qg.sh", "1", f"{qg_num}", f"{model_info}"],
        cwd=base_dir,
        text=True,
        check=True
    )

###################################################################################################

    ## merge parallel results
    output_bert_base_tensor_nq_qg = []
    output_bert_base_id_tensor_nq_qg = []
    for num in trange(1):
        with open(os.path.join(base_dir, f'qg/pkl/NQ_output_tensor_512_content_64_{qg_num}_{num}.pkl'), 'rb') as f:
            data = pickle.load(f)
        f.close()
        output_bert_base_tensor_nq_qg.extend(data)

        with open(os.path.join(base_dir, f'qg/pkl/NQ_output_tensor_512_content_64_{qg_num}_{num}_id.pkl'), 'rb') as f:
            data = pickle.load(f)
        f.close()
        output_bert_base_id_tensor_nq_qg.extend(data)

###################################################################################################

    import pickle
    from collections import defaultdict

    # 1. load
    with open(os.path.join(base_dir, f'qg/pkl/NQ_output_tensor_512_content_64_{qg_num}_0.pkl'), 'rb') as f:
        queries = pickle.load(f)
    with open(os.path.join(base_dir, f'qg/pkl/NQ_output_tensor_512_content_64_{qg_num}_0_id.pkl'), 'rb') as f:
        docids = pickle.load(f)

    # 2. 简单统计
    print(f'Total generated queries: {len(queries)}')
    print(f'Unique documents     : {len(set(docids))}')

    # 3. 看前 5 条
    print('\n=== 前 5 條合成查詢 ===')
    for i in range(5):
        print(f'{i:2d}. doc {docids[i]}: {queries[i]}')

    # 4. 按 docid 分组，查看某个文档的所有合成查询
    grp = defaultdict(list)
    for did, q in zip(docids, queries):
        grp[did].append(q)

    # 随机选一个 doc
    sample_doc = next(iter(grp))
    print(f'\n=== 文檔 {sample_doc} 的 {len(grp[sample_doc])} 條合成查詢 ===')
    for q in grp[sample_doc]:
        print('-', q)

###################################################################################################

    qg_dict = {}
    for i in trange(len(output_bert_base_tensor_nq_qg)):
        if(output_bert_base_id_tensor_nq_qg[i] not in qg_dict):
            qg_dict[output_bert_base_id_tensor_nq_qg[i]] = [output_bert_base_tensor_nq_qg[i]]
        else:
            qg_dict[output_bert_base_id_tensor_nq_qg[i]].append(output_bert_base_tensor_nq_qg[i])

###################################################################################################

    ## nq_512_qg20.tsv
    QG_NUM = qg_num

###################################################################################################

    qg_file = open(os.path.join(base_dir, "NQ_512_qg.tsv"), 'w') 

    for queryid in tqdm(qg_dict):
        for query in qg_dict[queryid][:QG_NUM]:
            qg_file.write('\t'.join([query, str(ran_id_old_id[int(queryid)]), queryid, new_kmeans_nq_doc_dict_512[queryid]]) + '\n')
            qg_file.flush()

###################################################################################################

    new_kmeans_nq_doc_dict_512_int_key = {}
    for key in new_kmeans_nq_doc_dict_512:
        new_kmeans_nq_doc_dict_512_int_key[int(key)] = new_kmeans_nq_doc_dict_512[key]

###################################################################################################

    nq_train['randomid'] = nq_train['title'].map(title_doc_id)
    nq_train['id_512'] = nq_train['randomid'].map(new_kmeans_nq_doc_dict_512_int_key)

    nq_train_ = nq_train.loc[:, ['query', 'id', 'randomid', 'id_512']]  
    nq_train_.to_csv(os.path.join(base_dir, 'nq_train_doc_newid.tsv'), sep='\t', header=None, index=False, encoding='utf-8')

###################################################################################################

    nq_dev['randomid'] = nq_dev['title'].map(title_doc_id)
    nq_dev['id_512'] = nq_dev['randomid'].map(new_kmeans_nq_doc_dict_512_int_key)


    nq_dev_ = nq_dev.loc[:, ['query', 'id', 'randomid', 'id_512']]  
    nq_dev_.to_csv(os.path.join(base_dir, 'nq_dev_doc_newid.tsv'), sep='\t', header=None, index=False, encoding='utf-8')

###################################################################################################

    nq_all_doc_non_duplicate = nq_train.append(nq_dev)
    nq_all_doc_non_duplicate.reset_index(inplace = True)

    nq_all_doc_non_duplicate['id_512'] = nq_all_doc_non_duplicate['randomid'].map(new_kmeans_nq_doc_dict_512_int_key)

    nq_all_doc_non_duplicate['ta'] = nq_all_doc_non_duplicate['title'] + ' ' + nq_all_doc_non_duplicate['abstract']

    nq_all_doc_non_duplicate = nq_all_doc_non_duplicate.loc[:, ['ta', 'id', 'randomid','id_512']]  
    nq_all_doc_non_duplicate.to_csv(os.path.join(base_dir, 'nq_title_abs.tsv'), sep='\t', header=None, index=False, encoding='utf-8')

###################################################################################################

    queryid_oldid_dict = {}
    bertid_oldid_dict = {}
    map_file = os.path.join(base_dir, "./nq_title_abs.tsv")
    with open(map_file, 'r') as f:
        for line in f.readlines():
            query, queryid, oldid, bert_k30_c30 = line.split("\t")
            queryid_oldid_dict[oldid] = queryid
            bertid_oldid_dict[oldid] = bert_k30_c30

    train_file = os.path.join(base_dir, "./NQ_doc_content.tsv")
    doc_aug_file = open(os.path.join(base_dir, f"./NQ_doc_aug.tsv"), 'w') 
    with open(train_file, 'r') as f:
        for line in f.readlines():
            docid, _, _, content, _, _, _ = line.split("\t")
            content = content.split(' ')
            add_num = max(0, len(content)-3000) / 3000
            for i in range(10+int(add_num)):
                begin = random.randrange(0, len(content))
                # if begin >= (len(content)-64):
                #     begin = max(0, len(content)-64)
                end = begin + 64 if len(content) > begin + 64 else len(content)
                doc_aug = content[begin:end]
                doc_aug = ' '.join(doc_aug)
                queryid = queryid_oldid_dict[docid]
                bert_k30_c30 = bertid_oldid_dict[docid]
                # doc_aug_file.write('\t'.join([doc_aug, str(queryid), str(docid), str(bert_k30_c30)]) + '\n')
                doc_aug_file.write('\t'.join([doc_aug, str(queryid), str(docid), str(bert_k30_c30)]))
                doc_aug_file.flush()

###################################################################################################