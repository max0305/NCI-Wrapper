# +
cd qg

model_info=$3

conda install -c conda-forge git-lfs -y
git lfs install

# 判斷模型名稱
if [ "$model_info" = "small" ]; then
    # 改用正確的小模型名稱
    model_name="doc2query/msmarco-t5-small-v1"
    echo "使用模型：$model_name"

    # 使用 huggingface-cli 或 transformers 下載模型，避免 git 錯誤
    git clone https://huggingface.co/$model_name doc2query-t5-${model_info}-msmarco
else
    # 組合完整模型名稱
    model_name="castorini/doc2query-t5-$model_info-msmarco"
    echo "使用模型：$model_name"

    git clone https://huggingface.co/$model_name doc2query-t5-${model_info}-msmarco
fi

mkdir pkl
mkdir log
ITER_NUM=`expr $1 - 1`
PARTITION_NUM=$1
qg_num=$2

for ITER in $(seq 0 $ITER_NUM)
do
nohup python -u qg.py --idx $ITER --cuda_device $ITER --partition_num ${PARTITION_NUM} --model_info $model_info --return_num ${qg_num} --max_len 64 --dataset NQ > log/NQ_$ITER.log 2>&1 
done
