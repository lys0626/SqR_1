import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy
import distutils.version
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter

import _init_paths
from lib.dataset.get_dataset import get_datasets

from lib.utils.logger import setup_logger
import lib.models as models
import lib.models.aslloss
from lib.models.query2label import build_q2l
# 修改引用路径：使用新的 Metric 工具
from lib.utils.metric import AveragePrecisionMeter
from lib.utils.misc import clean_state_dict
from lib.utils.slconfig import get_raw_dict

# --- [新增] 引入自定义模块 ---
# 假设这些文件都在根目录下
from rolt_handler import RoLT_Handler
from SpliceMix import SpliceMix
# --- [新增] 时间格式化辅助函数 ---
def sec_to_str(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"
def parser_args():
    parser = argparse.ArgumentParser(description='Query2Label MSCOCO Training')
    parser.add_argument('--dataname', help='dataname', default='coco14', choices=['coco14', 'mimic', 'nih'])
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/comp_robot')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')

    parser.add_argument('--output', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='SGD', type=str, choices=['AdamW', 'Adam_twd', 'SGD'],
                        help='which optim to use')
    # --- [新增] 学习率调度器参数 ---
    parser.add_argument('--scheduler', default='StepLR', type=str, choices=['OneCycle', 'StepLR'],
                        help='Which scheduler to use: OneCycle (default) or StepLR')
    parser.add_argument('--step_size', default=40, type=int,
                        help='Period of learning rate decay (epochs) for StepLR')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Multiplicative factor of learning rate decay for StepLR')
    # -----------------------------
    # --- [新增] Momentum 参数 (SGD用) ---
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # ASL loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False, 
                        help='disable_torch_grad_focal_loss in asl')              
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                                            help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                                            help='scale factor for clip')  

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')


    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=95, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')


    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')              
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ') 

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')


    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * training
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')

    # --- [新增] ResNet/SpliceMix 相关参数 ---
    parser.add_argument('--enable_splicemix', action='store_true', default=False,
                        help='Whether to enable SpliceMix augmentation')
    parser.add_argument('--splicemix_prob', default=1, type=float,
                        help='Probability of applying SpliceMix')
    parser.add_argument('--splicemix_mode', default='SpliceMix', type=str,
                        choices=['SpliceMix', 'SpliceMix-CL'],
                        help='Mode of SpliceMix: Standard SpliceMix or SpliceMix-CL settings')

    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args

best_mAUC = 0  # 修改变量名

def main():
    args = get_args()
    
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)

def main_worker(args, logger):

    global best_mAUC

    # build model
    model = build_q2l(args)
    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay) # 0.9997 实现模型参数的指数平均移动，不参与反向传播也不会被优化器直接更新，
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                       device_ids=[args.local_rank],
                                                         broadcast_buffers=False,
                                                         find_unused_parameters=True  # <--- 必须加，单卡也得加
                                                         )

    # --- [修改] Criterion ---
    # 使用 BCEWithLogitsLoss 替代 ASL
    criterion = torch.nn.BCEWithLogitsLoss()

    # optimizer
    # --- [修改] Optimizer: 支持参数分组和 SGD ---
    args.lr_mult = args.batch_size / 256
    # base_lr = args.lr_mult * args.lr
    base_lr = args.lr
    # 1. 参数分组：Backbone与Splicemix分支的分类器 学习率x1，其他部分x0.1
    backbone_params = []
    other_params = []
    for name, param in model.module.named_parameters():
        if not param.requires_grad:
            continue
        # 根据 Query2Label 的定义，Backbone 属性名为 'backbone'
        if 'backbone' in name or 'fc_splicemix' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    # 打印分组信息确认
    if args.rank == 0:
        logger.info(f"Optimizer Grouping: {len(backbone_params)} backbone params (lr1), {len(other_params)} other params (lr*0.1)")

    param_dicts = [
        {
            "params": backbone_params, 
            "lr": base_lr    # Backbone与fc_Splicemix使用的学习率
        },
        {
            "params": other_params, 
            "lr": base_lr/2      # 其他部分 (Transformer, FC, Heads) 使用基础学习率
        },
    ]
    #初始化优化器
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=base_lr, # 这里的 lr 仅作为 fallback，实际生效的是 param_dicts 里的
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'SGD': # [新增] SGD 逻辑
        optimizer = torch.optim.SGD(
            param_dicts,
            lr=base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    elif args.optim == 'Adam_twd':
        # Adam_twd 逻辑保持原样，如果需要分层学习率，这里的 parameters 需要重新构建
        # 鉴于 add_weight_decay 比较特殊，这里暂时维持原样
        logger.warning("Adam_twd does not support backbone lr split currently in this script.")
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            base_lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )          
    else:
        raise NotImplementedError

    # tensorboard
    #控制日志记录的权限
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None
    '''
        #args.resume参数 指向权重文件
        训练时：用于在一个较好的基础上开始训练（加载权重），省去从零收敛的时间。
        测试时：必须使用，用于加载训练好的模型进行评估。
    '''
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))
            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dict Found!!!")
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_dataset, val_dataset = get_datasets(args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    # 注意：确保 Dataset 已经修改为返回 (image, target, index)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        # [修改] 接收元组，提取 mAUC
        metrics_res, _ = validate(val_loader, model, criterion, args, logger)
        mAUC = metrics_res['mAUC']
        logger.info(' * mAUC {mAUC:.5f}'.format(mAUC=mAUC))
        return

    # --- [新增] 初始化 SpliceMix ---
    #冗余代码
    #n_grids = [0]  代表2x2拼接方式没有进行图像拼接
    if args.splicemix_mode == 'SpliceMix-CL':
        splicemix_obj = SpliceMix(mode='SpliceMix', grids=['2x2'], n_grids=[0], mix_prob=args.splicemix_prob)
    else:
        splicemix_obj = SpliceMix(mode='SpliceMix', grids=['2x2'], n_grids=[0], mix_prob=args.splicemix_prob)
    splicemix_augmentor = splicemix_obj.mixer
    # if args.splicemix_mode == 'SpliceMix-CL':
    #     splicemix_obj = SpliceMix(mode='SpliceMix', grids=['1x2','2x2'], n_grids=[0], mix_prob=args.splicemix_prob)
    # else:
    #     splicemix_obj = SpliceMix(mode='SpliceMix', grids=['1x2','2x2'], n_grids=[0], mix_prob=args.splicemix_prob)
    # splicemix_augmentor = splicemix_obj.mixer
    # --- [新增] 初始化 RoLT Handler ---
    # args.num_class 和 args.hidden_dim 必须正确设置
    rolt_handler = RoLT_Handler(args, model, train_loader, args.num_class, args.hidden_dim)

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    mAUCs = AverageMeter('mAUC', ':5.5f', val_only=True)
    
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, mAUCs],
        prefix='=> Test Epoch: ')

    # -------------------------------------------------------------------------
    # [修改] 动态选择学习率调度器 (Scheduler)
    # -------------------------------------------------------------------------
    # args.step_per_batch 是一个我们动态添加的属性 (Flag)，
    # 用于告诉后面的 train 函数：这个调度器是该每个 Batch 更新 (如 OneCycle)，还是每个 Epoch 更新 (如 StepLR)。
    
    if args.scheduler == 'OneCycle':
        # [原有逻辑] OneCycleLR: 需要在每个 Batch 结束后更新 (step)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.lr, 
            steps_per_epoch=len(train_loader), 
            epochs=args.epochs, 
            pct_start=0.2
        )
        args.step_per_batch = True # 标记：Batch 级更新

    elif args.scheduler == 'StepLR':
        # [新增逻辑] StepLR: 只需要在每个 Epoch 结束后更新 (step)
        # step_size: 多少个 epoch 衰减一次
        # gamma: 衰减倍率 (例如 0.1 表示变为原来的 10%)
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=args.step_size, 
            gamma=args.gamma
        )
        args.step_per_batch = False # 标记：Epoch 级更新

    else:
        # 防御性编程：如果传入了未实现的调度器名称，抛出错误
        raise NotImplementedError("Scheduler {} not implemented".format(args.scheduler))
    
    # 记录一些统计变量
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    mAUCs = AverageMeter('mAUC', ':5.5f', val_only=True)
    best_epoch = -1
    end = time.time()
    best_epoch = -1
    
    # ================= [替换] 整个训练循环 =================
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()

        # --- 1. RoLT 数据清洗与统计 ---
        clean_mask_dict, soft_label_dict = rolt_handler.step(epoch)
        
        n_total = len(train_dataset)
        if len(clean_mask_dict) == 0:
            n_clean = n_total
        else:
            # 统计 False (Noisy) 的数量，剩下的就是 Clean
            n_noisy_count = sum(1 for v in clean_mask_dict.values() if v is False)
            n_clean = n_total - n_noisy_count

        # 确保模型可训练
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        # --- 2. 训练阶段 ---
        train_start = time.time()
        
        # 调用 train 函数
        loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger,
                     splicemix_augmentor, clean_mask_dict, soft_label_dict)
        
        train_duration = sec_to_str(time.time() - train_start)

        # [关键] StepLR 必须在 Epoch 结束时更新
        # args.step_per_batch 是你在调度器部分定义的变量
        if not getattr(args, 'step_per_batch', True):
            scheduler.step()

        # 获取学习率 (用于打印)
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        lr_str = " ".join([f"{lr:.5g}" for lr in current_lrs])
        
        # [日志格式 1] Train Log
        # 样例: [Epoch 17, lr[0.005 0.05 ]] [Train] time:02:34s, loss: 0.1927 .
        if args.rank == 0:
            logger.info(f"[Epoch {epoch}, lr[{lr_str} ]] [Train] time:{train_duration}s, loss: {loss:.4f} .")
        
        if summary_writer:
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', current_lrs[0], epoch)

        # --- 3. 验证阶段 ---
        if epoch % args.val_interval == 0:
            val_start = time.time()
            
            # validate 返回 (metrics_res, val_loss) -> 需要下一步修改 validate 函数
            metrics_res, val_loss = validate(val_loader, model, criterion, args, logger)
            
            # EMA 模型验证 (可选)
            metrics_res_ema, _ = validate(val_loader, ema_m.module, criterion, args, logger)
            
            val_duration = sec_to_str(time.time() - val_start)

            # 提取指标
            mAUC = metrics_res['mAUC']
            mi_f1, ma_f1 = metrics_res['micro_F1'], metrics_res['macro_F1']
            mi_p, ma_p = metrics_res['micro_P'], metrics_res['macro_P']
            mi_r, ma_r = metrics_res['micro_R'], metrics_res['macro_R']

            losses.update(loss)
            mAUCs.update(mAUC)

            is_best = mAUC > best_mAUC
            if is_best:
                best_epoch = epoch
            best_mAUC = max(mAUC, best_mAUC)

            # [日志格式 2] Test Log
            timestamp = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
            log_prefix = f"[{args.rank}|{timestamp}]"
            
            if args.rank == 0:
                logger.info(
                    f"{log_prefix}: [Test] time: {val_duration}s, loss: {val_loss:.4f}, "
                    f"mAUC: {mAUC:.4f}, miF1: {mi_f1:.4f}, maF1: {ma_f1:.4f}, "
                    f"miP: {mi_p:.4f}, maP: {ma_p:.4f} ."
                )

            # [日志格式 3] Best Result Log
            if args.rank == 0:
                logger.info(f"{log_prefix}: --[Test-best] (E{best_epoch}), mAUC: {best_mAUC:.4f}")

            if summary_writer:
                summary_writer.add_scalar('val_mAUC', mAUC, epoch)
                summary_writer.add_scalar('val_loss', val_loss, epoch)

            # --- 4. 写入 log.txt ---
            if args.rank == 0:
                log_txt_path = os.path.join(args.output, 'log.txt')
                
                lr_backbone = current_lrs[0]
                lr_head = current_lrs[1] if len(current_lrs) > 1 else current_lrs[0]
                
                with open(log_txt_path, 'a') as f:
                    # [新增] 每次写数据前，都先写一遍表头
                    # 建议在表头前加个换行符 \n，让视觉上更清晰
                    header = (
                        "\nEpoch\tTrain_Loss\tVal_Loss\tVal_mAUC\t"
                        "mi_F1\tma_F1\tmi_R\tma_R\tmi_P\tma_P\t"
                        "Clean_S\tTotal_S\tLR_Backbone\tLR_Head\n"
                    )
                    f.write(header)
                    
                    # 写数据
                    log_line = (
                        f"{epoch}\t{loss:.5f}\t{val_loss:.5f}\t{mAUC:.5f}\t"
                        f"{mi_f1:.5f}\t{ma_f1:.5f}\t{mi_r:.5f}\t{ma_r:.5f}\t{mi_p:.5f}\t{ma_p:.5f}\t"
                        f"{n_clean}\t{n_total}\t{lr_backbone:.8f}\t{lr_head:.8f}\n"
                    )
                    f.write(log_line)

            # 保存 Checkpoint
            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.backbone,
                    'state_dict': model.state_dict(),
                    'best_mAUC': best_mAUC,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))

            # Early Stop
            if args.early_stop:
                if best_epoch >= 0 and epoch - best_epoch > 8:
                    break

def compute_consistency_target(preds_original, flag, device):
    """
    [修改说明] 性能优化版本：使用列表收集结果，避免在循环中频繁调用 torch.cat
    """
    mix_dict = flag['mix_dict']
    
    # 初始化为列表
    preds_m_r_list = []
    
    for i, (rand_ind, g_row, g_col, n_drop, drop_ind) in enumerate(zip(
        mix_dict['rand_inds'], mix_dict['rows'], mix_dict['cols'], 
        mix_dict['n_drops'], mix_dict['drop_inds']
    )):
        current_preds = preds_original[rand_ind]
        
        if n_drop > 0:
            if drop_ind.dim() == 1:
                mask = (drop_ind[:, None] == 1)
            else:
                mask = (drop_ind == 1)
            current_preds = torch.masked_fill(current_preds, mask, -1e3)
            
        # [修改] 存入列表，而不是直接 cat
        preds_m_r_list.append(current_preds)
        
    # [修改] 最后一次性拼接
    if len(preds_m_r_list) > 0:
        preds_m_r = torch.cat(preds_m_r_list, dim=0)
    else:
        preds_m_r = torch.tensor([], device=device)
        
    return preds_m_r

def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger, 
          splicemix_augmentor, clean_mask_dict, soft_label_dict):
    
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    losses = AverageMeter('Loss', ':5.3f')
    model.train()
    end = time.time()
    # 定义一致性损失函数 (BCE)
    # 混合图片的预测值 (logits) vs 重构的预测值 (sigmoid概率)
    # 参考 SpliceMix_CL.py: loss_cl = bce(preds_m, preds_m_r.sigmoid())
    consistency_criterion = torch.nn.BCEWithLogitsLoss()
    # 这里的 train_loader 必须返回 indices
    for i, (images, target, indices) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        indices = indices.cuda(non_blocking=True)

        # --- A. 区分样本 ---
        if len(clean_mask_dict) == 0:
            # 默认全 Clean
            batch_clean_mask = torch.ones(images.size(0), dtype=torch.bool).cuda()
        else:
            batch_clean_mask = torch.tensor([clean_mask_dict.get(idx.item(), True) for idx in indices]).bool().cuda()
        
        batch_noisy_mask = ~batch_clean_mask
        clean_idxs = torch.where(batch_clean_mask)[0]
        noisy_idxs = torch.where(batch_noisy_mask)[0]

        # ------------------------------------------------------------------
        # B. 分支 1: SpliceMix 增强分支 (仅处理 Clean 样本，训练 GAP Head)
        # ------------------------------------------------------------------
        loss_splicemix_branch = torch.tensor(0.0).cuda()
        #必须有4张干净图才能组成一个2x2的网格   
        if args.enable_splicemix and len(clean_idxs) > 4:
            images_clean = images[clean_idxs]
            targets_clean = target[clean_idxs]
            
            # 1. 执行混合
            # mixed_images_all 包含 [原始图片, 混合图片]
            # flag 包含 mix_ind
            mixed_images_all, mixed_targets_all, flag = splicemix_augmentor(images_clean, targets_clean)
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                # 2. 前向传播 (双分类器模式)
                # 返回值: (Transformer_Logits, Transformer_Features, GAP_Logits)
                # [关键] 我们只关心第3个: GAP Head 的输出
                _, _, out_gap_mixed = model(mixed_images_all)
                
                # 3. 计算基础 BCE Loss
                # 策略: 对原始图片和混合图片都进行监督 ("既看又练")
                # 此时更新的是 Backbone + FC_SpliceMix
                loss_splicemix_basic = criterion(out_gap_mixed, mixed_targets_all)
                loss_splicemix_branch = loss_splicemix_basic

                # 4. [CL逻辑] SpliceMix-CL 一致性损失
                if args.splicemix_mode == 'SpliceMix-CL':
                    # 解析 mix_ind
                    mix_ind = flag['mix_ind'] # 0: 原始图片, 1: 混合图片
                    
                    # 分离 Logits (全部来自 GAP Head)
                    logits_original_gap = out_gap_mixed[mix_ind == 0] # 原始图在 GAP Head 的预测
                    logits_mixed_gap = out_gap_mixed[mix_ind == 1]    # 混合图在 GAP Head 的预测
                    
                    # 构造目标 (利用原始图的 GAP 预测值来指导混合图)
                    target_consistency_logits = compute_consistency_target(logits_original_gap, flag, images.device)
                    
                    # 计算一致性损失
                    loss_consistency = consistency_criterion(logits_mixed_gap, torch.sigmoid(target_consistency_logits).detach())
                    
                    # 累加 Loss
                    loss_splicemix_branch += loss_consistency

        # ------------------------------------------------------------------
        # C. 分支 2: Q2L 原生分支 (处理所有样本，训练 Transformer Head)
        # ------------------------------------------------------------------
        loss_q2l_branch = 0.0
        with torch.cuda.amp.autocast(enabled=args.amp):
            # 前向传播所有原始图片
            # [关键] 我们只关心第1个: Transformer Head 的输出
            # 第2个 features_clean 虽然返回了，但此处 Loss 计算不需要，仅用于 RoLT 外部调用
            # 第3个 out_gap_clean 也不需要，因为 GAP Head 已经在上面专门训练过了
            out_trans_all, features_clean, _ = model(images)
            valid_parts = 0
            # C.1 干净样本 -> 原始硬标签 Loss
            if len(clean_idxs) > 0:
                loss_clean = criterion(out_trans_all[clean_idxs], target[clean_idxs])
                loss_q2l_branch += loss_clean
                valid_parts += 1
            # C.2 噪声样本 -> 软伪标签 Loss (RoLT 生成)
            if len(noisy_idxs) > 0:
                soft_targets_list = []
                # [优化前]
                # for idx in indices[noisy_idxs]:
                #     s_label = soft_label_dict.get(idx.item(), target[torch.where(indices==idx)[0][0]])
                #     soft_targets_list.append(s_label)
                
                # [优化后] 直接遍历 noisy_idxs (它是 batch 内的下标，如 0, 3, 5...)
                for k in noisy_idxs:
                    global_idx = indices[k].item() # 获取该样本在整个数据集中的全局ID
                    # 尝试从字典取软标签，取不到则用原始 target[k]
                    s_label = soft_label_dict.get(global_idx, target[k])
                    soft_targets_list.append(s_label)
                soft_targets = torch.stack(soft_targets_list).to(images.device)
                loss_noisy = criterion(out_trans_all[noisy_idxs], soft_targets)
                loss_q2l_branch += loss_noisy
                valid_parts += 1
            
            # 平均 Q2L 分支内的 Loss
            if valid_parts > 0:
                loss_q2l_branch /= valid_parts

        # --- D. 总 Loss ---
        final_loss = 0.0
        branch_count = 0
        if args.enable_splicemix and len(clean_idxs) > 4:
            final_loss += loss_splicemix_branch
            branch_count += 1
        if valid_parts > 0:
            final_loss += loss_q2l_branch
            branch_count += 1
        
        if branch_count > 0:
            final_loss /= branch_count

        # Backprop
        optimizer.zero_grad()
        scaler.scale(final_loss).backward()
        # --- [新增] 梯度裁剪 (关键修复) ---
        # 在 step 之前，必须先 unscale 梯度
        # scaler.unscale_(optimizer)
        # 对 Transformer 结构，max_norm 通常设为 0.1 或 1.0
        #如果Loss下降的很慢，可以将max_norm稍微调大例如1或5，接近NaN问题0.1是最安全的选项
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        # -------------------------------
        scaler.step(optimizer)
        scaler.update()
        # --- [修改] 仅当使用 OneCycleLR 等需要 per-batch 更新的调度器时才执行 ---
        if args.step_per_batch:
            scheduler.step()
        # -------------------------------------------------------------------

        if epoch >= args.ema_epoch:
            ema_m.update(model)

        losses.update(final_loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    #AveragePrecisionMeter自定义类，用于累积预测结果并计算mAP/mAUC等指标，False代表不考虑数据集中标记为困难的样本，在mimic和nih数据集中无效
    meter = AveragePrecisionMeter(difficult_examples=False)
    losses = AverageMeter('Loss', ':5.3f') # 用于记录验证集 Loss
    
    model.eval()
    
    for i, (images, target, _) in enumerate(val_loader): 
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        #混合精度上下文
        with torch.cuda.amp.autocast(enabled=args.amp):
            #在这里是取Splicemix分支fc的预测结果，如果想去Q2L分支的预测结果应该采用output, _, _ = model(images)
            _, _, output = model(images)
            # [新增] 计算 Loss
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            
            meter.add(output, target, filename=[]) 

    # 获取所有指标
    metrics_res = meter.compute_all_metrics()
    
    # [修改] 返回 (指标字典, 平均Loss)
    return metrics_res, losses.avg

# --- 工具类保持不变 ---
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')

class AverageMeter(object):
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AverageMeterHMS(AverageMeter):
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name, 
                             val=str(datetime.timedelta(seconds=int(self.val))), 
                             sum=str(datetime.timedelta(seconds=int(self.sum))))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()