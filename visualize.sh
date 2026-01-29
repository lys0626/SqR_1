#!/bin/bash
# --- [新增] 强制使用 Conda 环境的动态库 ---
export LD_PRELOAD=/home/dsj/anaconda3/envs/lys2/lib/libstdc++.so.6
# --- [新增] 设置分布式训练的主节点信息 (单机训练必须配置) ---
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR='localhost'
export MASTER_PORT='12345'

# 3. 运行 Python 脚本
# 注意：下面的参数必须和训练时的配置对齐！
python visualize_noise.py \
  --dataname nih \
  --dataset_dir /data/nih-chest-xrays \
  --img_size 224 \
  --num_class 14 \
  --output ./visualize/analysis_output \
  --resume /data/dsj/lys/SqR-main/experiment/nih_512_2048_AdamW_AdamW_5e-5_5e-5_2_32_OneCycle_StepLR_20_-2_1_NO/checkpoint_stage1_final.pth.tar \
  --backbone resnet50 \
  --batch-size 64 \
  --hidden_dim 512 \
  --dim_feedforward 2048 \
  --keep_input_proj \
  --enc_layers 1 \
  --dec_layers 2 \
  --nheads 4