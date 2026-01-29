import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# --- [必须] 路径初始化 ---
import _init_paths 

# 引入项目依赖
from lib.models.query2label import build_q2l
from lib.dataset.get_dataset import get_datasets
from utilities.nih import nihchest  # 直接引入 Dataset 类以便手动实例化
from rolt_handler import RoLT_Handler
from main_mlc import parser_args

# NIH 类别名称
CLASS_NAMES = [
    'Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax',
    'Consolidation', 'Pleural_Thickening', 'Cardiomegaly', 'Emphysema', 'Edema',
    'Fibrosis', 'Pneumonia', 'Hernia'
]

def main():
    # 1. 获取参数
    args = parser_args()
    args.world_size = 1
    args.rank = 0
    args.gpu = 0
    # 强制单卡
    torch.cuda.set_device(args.gpu)
    
    print(f"Loading checkpoint from: {args.resume}")
    
    # 2. [核心修改] 准备两套数据
    # Set A: 用于 RoLT 特征提取 (必须带 Norm，和训练一致)
    train_dataset_for_model, _ = get_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset_for_model, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # Set B: 用于可视化 (无 Augment, 无 Norm, 只有 Resize)
    # 这样你看到的就是清晰的原图
    vis_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    # 重新实例化一个 Dataset，专门用于“看图”
    # 注意：只要 mode='train' 且 root 相同，读取图片的顺序和 train_dataset_for_model 是一模一样的
    vis_dataset = nihchest(root=args.dataset_dir, mode='train', transform=vis_transform)
    print("Visualization dataset initialized (No Augmentation).")
    
    # 3. 加载模型
    model = build_q2l(args).cuda()
    
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.resume}")

    # 4. 初始化 RoLT Handler (用带 Norm 的 loader)
    rolt = RoLT_Handler(args, model, train_loader, args.num_class, args.hidden_dim)
    
    # 5. 运行 RoLT 逻辑
    print("Running RoLT feature extraction and GMM cleaning...")
    all_feats, all_logits, all_targets, all_indices = rolt.extract_features_and_mask()
    rolt.update_prototypes(all_feats, all_targets)
    label_clean_matrix = rolt.gmm_cleaning(all_feats, all_targets)
    
    # 6. 筛选样本
    # 找出: 原始标签=1 但 GMM认为=False 的样本
    noisy_mask = (all_targets == 1) & (~label_clean_matrix)
    samples_with_noise_indices = torch.where(noisy_mask.sum(dim=1) > 0)[0]
    
    print(f"Total samples: {len(all_targets)}")
    print(f"Samples containing at least one GMM-rejected label: {len(samples_with_noise_indices)}")
    
    if len(samples_with_noise_indices) == 0:
        print("No noisy labels found!")
        return

    # 7. 可视化 (从 vis_dataset 取图)
    vis_count = min(9, len(samples_with_noise_indices))
    # 固定随机种子以便复现
    np.random.seed(42) 
    selected_indices = np.random.choice(samples_with_noise_indices.cpu().numpy(), vis_count, replace=False)
    
    plt.figure(figsize=(15, 15))
    
    for i, idx in enumerate(selected_indices):
        # [修改] 从 vis_dataset 读取干净的图
        # vis_dataset 返回 (image, label, index)
        img_tensor, _, _ = vis_dataset[idx] 
        
        # 因为没有 Normalize，直接 permute 即可显示
        # Clip 以防万一，但通常 ToTensor 后就是 [0,1]
        img_vis = img_tensor.permute(1, 2, 0).numpy()
        img_vis = np.clip(img_vis, 0, 1)
        
        # 获取标签信息 (逻辑同前)
        target_vec = all_targets[idx].cpu()
        target_indices = torch.where(target_vec == 1)[0]
        original_labels = [CLASS_NAMES[k] for k in target_indices]
        
        is_clean_vec = label_clean_matrix[idx].cpu()
        rejected_indices = torch.where((target_vec == 1) & (~is_clean_vec))[0]
        rejected_labels = [CLASS_NAMES[k] for k in rejected_indices]
        
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img_vis, cmap='gray') # X光片通常用灰度显示更清晰
        
        title_text = f"ID: {all_indices[idx].item()}\n"
        title_text += f"Orig: {', '.join(original_labels)}\n"
        title_text += f"Rejected: {', '.join(rejected_labels)}"
        
        ax.set_title(title_text, fontsize=10, loc='left', color='blue')
        ax.axis('off')
        
        # 标注
        ax.text(5, 20, "NOISY DETECTED", color='red', weight='bold', bbox=dict(facecolor='white', alpha=0.8))

    # 8. 保存
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    save_path = os.path.join(args.output, 'noise_analysis_vis_clean.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nVisualization saved to: {save_path}")

if __name__ == '__main__':
    main()