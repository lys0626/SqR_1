#!/bin/bash
# --- [新增] 强制使用 Conda 环境的动态库 ---
export LD_PRELOAD=/home/dsj/anaconda3/envs/lys2/lib/libstdc++.so.6
# --- [新增] 设置分布式训练的主节点信息 (单机训练必须配置) ---
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR='localhost'
export MASTER_PORT='12345'
# --- 配置区域 ---
# 数据集名称: mimic 或 nih
DATANAME="nih"
DATA_DIR="/data/nih-chest-xrays"
# 输出目录
OUTPUT_DIR="./output/${DATANAME}_resnet50_rolt_splicemix_sgd"
# 类别数 (MIMIC-CXR通常是13或14，NIH是14，请根据实际情况修改)
NUM_CLASS=14
# 显卡ID


# --- 启动命令 ---
# 如果需要多卡训练 (DDP)，请使用: python -m torch.distributed.launch --nproc_per_node=N ...
# 下面是单机单卡示例:

python main_mlc.py \
  --dataname ${DATANAME} \
  --dataset_dir ${DATA_DIR} \
  --img_size 224 \
  --num_class ${NUM_CLASS} \
  --backbone resnet50 \
  --output ${OUTPUT_DIR} \
  --batch-size 64 \
  --workers 8 \
  --epochs 80 \
  --start-epoch 0 \
  --pretrained \
  --optim SGD \
  --lr 0.1 \
  --momentum 0.9 \
  --wd 1e-4 \
  --amp \
  --enable_splicemix \
  --splicemix_prob 1 \
  --keep_input_proj \
  --hidden_dim 1024 \
  --output exp1_1024