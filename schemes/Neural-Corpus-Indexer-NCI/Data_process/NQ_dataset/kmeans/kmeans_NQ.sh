class_num=$1
cd kmeans
nohup python -u kmeans.py --bert_size 512 --k $class_num --c $class_num > nq_kmeans_512.log 2>&1 
