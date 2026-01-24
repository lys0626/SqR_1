#!/bin/bash
# --- [新增] 强制使用 Conda 环境的动态库 ---
export LD_PRELOAD=/home/dsj/anaconda3/envs/lys2/lib/libstdc++.so.6
# --- [新增] 设置分布式训练的主节点信息 (单机训练必须配置) ---
export CUDA_VISIBLE_DEVICES=5
export MASTER_ADDR='localhost'
export MASTER_PORT='12351'
# --- 配置区域 ---
# 数据集名称: mimic 或 nih
DATANAME="nih"
DATA_DIR="/data/nih-chest-xrays"
HIDDEN_DIM=1024
DIM_FEEDFORWARD=4096
OPTIM="AdamW"   # 注意：main_mlc.py 中要求是 'AdamW' (区分大小写)，不要写成 'Adamw'
LR="5e-5"
DEC_LAYERS=4
NHEADS=16
BATCH_SIZE=128
# 输出目录
OUTPUT_DIR="./experiment/${DATANAME}_${HIDDEN_DIM}_${DIM_FEEDFORWARD}_${OPTIM}_${LR}_${DEC_LAYERS}_2_1_${NHEADS}_${BATCH_SIZE}"
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
    --output ${OUTPUT_DIR} \
  --backbone resnet50 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --epochs 100 \
  --start-epoch 0 \
  --pretrained \
  --optim ${OPTIM} \
  --lr ${LR} \
  --momentum 0.9 \
  --wd 1e-2 \
  --amp \
  --enable_splicemix \
  --splicemix_prob 1 \
  --keep_input_proj \
  --hidden_dim ${HIDDEN_DIM} \
  --dim_feedforward ${DIM_FEEDFORWARD} \
  --dec_layers ${DEC_LAYERS} \
  --enc_layers 1 \
  --nheads ${NHEADS} \
  --scheduler OneCycle \
