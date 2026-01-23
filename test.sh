#!/bin/bash

# ================= 配置区域 =================
# 1. 指定使用的显卡 (例如使用第0号卡)
export LD_PRELOAD=/home/dsj/anaconda3/envs/lys2/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=4
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'
# 2. 数据集设置
DATASET="nih"               # 选项: nih, mimic, coco14
DATA_DIR="/data/nih-chest-xrays" # [请修改] 你的数据集实际路径
NUM_CLASS=14                # NIH=14, MIMIC=13, COCO=80

# 3. 模型与图片设置
BACKBONE="resnet50"
IMG_SIZE=224                # 必须与训练时一致
BATCH_SIZE=128              # 测试时显存占用小，可以适当调大

# 4. 权重文件路径 (必须指定)
# [请修改] 指向你训练好的 checkpoint.pth.tar 文件
RESUME="/data/dsj/lys/SqR-main/experiment/nih_Splicemix__1024_2_2_-2_128/model_best.pth.tar"

# ================= 运行命令 =================
echo "----------------------------------------------------------------"
echo "Starting Evaluation on ${DATASET}..."
echo "Using Checkpoint: ${RESUME}"
echo "----------------------------------------------------------------"

python main_mlc.py \
    --dataname ${DATASET} \
    --dataset_dir ${DATA_DIR} \
    --num_class ${NUM_CLASS} \
    --img_size ${IMG_SIZE} \
    --backbone ${BACKBONE} \
    --batch-size ${BATCH_SIZE} \
    --resume ${RESUME} \
    -e \
    --output ./test/test_result_nih \
    --workers 8 \
    --keep_input_proj \
    --hidden_dim=1024
# 注释说明:
# -e : 开启评估模式 (evaluate)
# --resume : 加载权重