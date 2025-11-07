import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn # 导入损失函数
from datetime import datetime
from tqdm import tqdm
# import torch.distributed as dist # (分布式训练暂不启用)
import matplotlib.pyplot as plt
import importlib
import random
from torch.utils.data import Dataset

# --- 1. 设备设置 ---
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
print(f"正在使用设备: {device}")

# --- 2. 路径设置 (来自你的代码) ---
# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
# --- 3. 导入自定义模块 ---
from utils.commn import save_config, __mkdir__
from main.model import build_optimizer, build_model
# (新导入)
from utils.data_loader import get_train_valid_dataloaders
from utils.data_normalization import create_normalizer, save_normalizer, load_normalizer
from utils.loss_func import custom_loss

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块1：参数配置 (来自你的代码)
def __init__():
    parser = argparse.ArgumentParser(description='torch_version enhanced s2s model for 14s wind forecast')

    parser.add_argument('--configs', type=str, default='enhanced_s2s', help='configs of model')
    parser.add_argument('--local rank', type=int, default=0, help='local rank for distributed training') # (暂不启用)
    parser.add_argument('--project', type=str, default='14s', help='project name, 14s, 60min or ....')
    args = parser.parse_args()
    
    # 动态加载 config
    module_name = 'main.configs.train.' + args.project + '.' + args.configs
    try:
        xconfig = importlib.import_module(module_name)
    except ImportError:
        print(f"错误: 无法加载配置文件: {module_name}")
        print(f"请确保 config文件{module_name}存在")
        sys.exit(1)
        
    # (你的原始逻辑)
    # (注意: 你的 config 文件现在会自动创建 work_dir)
    config_save_path = os.path.join(xconfig.work_dir, args.project, args.configs)
    save_config(module_name, config_save_path)
    # __mkdir__(os.path.join(xconfig.work_dir, 'model', 'o.pth')) # (config文件已创建 model 目录)
    
    xconfig.logger.info(f'[Work Dir]: {xconfig.work_dir}')
    xconfig.logger.info(f'[Configs]: {module_name}')
    xconfig.logger.info(f'[Configs save path]: {config_save_path}')
    print(f'[Configs]: {module_name}')
    
    # 设置随机种子
    torch.manual_seed(xconfig.rand_seed)
    np.random.seed(xconfig.rand_seed)
    random.seed(xconfig.rand_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(xconfig.rand_seed)

    return xconfig

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块2：标准化
def normalizedata(base_dir, folder_list, x_suffix, y_suffix, device=None):

    print(f"正在从基础目录 '{base_dir}' 加载数据文件夹...")
    
    all_x_data = []
    all_y_data = []
    
    for folder in folder_list:
        # --- (新逻辑) ---
        # 根据文件夹名构建文件名
        x_filename = folder + x_suffix
        y_filename = folder + y_suffix
        # -----------------
        
        x_file_path = os.path.join(base_dir, folder, x_filename)
        y_file_path = os.path.join(base_dir, folder, y_filename)
        
        print(f"  -> 正在加载: {folder}/{x_filename} 和 {folder}/{y_filename}")
        try:
            x_data = np.load(x_file_path)
            y_data = np.load(y_file_path)
            
            # 验证样本数
            if x_data.shape[0] != y_data.shape[0]:
                print(f"  -> 警告: 文件夹 {folder} 中的 X 和 Y 样本数不匹配! 跳过此文件夹。")
                continue
                
            all_x_data.append(x_data)
            all_y_data.append(y_data)
            
        except FileNotFoundError as e:
            print(f"  -> 错误: 找不到文件! {e}。跳过此文件夹。")
        except Exception as e:
            print(f"  -> 加载 {folder} 时出错: {e}。跳过此文件夹。")

    if not all_x_data or not all_y_data:
        raise RuntimeError("未能成功加载任何数据! 请检查 config 中的 'data_config' 路径。")

    # 将所有加载的数据连接成一个大数组
    combined_x = np.concatenate(all_x_data, axis=0)
    combined_y = np.concatenate(all_y_data, axis=0)
    
    return combined_x, combined_y
def normalization(cfg):
    x, y = normalizedata(
        base_dir=cfg.data_config['base_dir'],
        folder_list=cfg.data_config['train_folders'],
        x_suffix=cfg.data_config['x_suffix'], # (修改)
        y_suffix=cfg.data_config['y_suffix'], # (修改)
        device=device
    ) 


    all_x_reshaped = x.reshape(-1, x.shape[-1])
    
    normalizer_all, _ = create_normalizer(
        data=all_x_reshaped,
        method=cfg.normalize_config['method'],
        feature_range=(cfg.normalize_config['feature_range_min'], cfg.normalize_config['feature_range_max'])
    )
    
    # 使用所有目标变量数据创建标准化器
    all_y_reshaped = y.reshape(-1, 1)
    
    target_normalizer, _ = create_normalizer(
        data=all_y_reshaped,
        method=cfg.normalize_config['method'],
        feature_range=(cfg.normalize_config['feature_range_min'], cfg.normalize_config['feature_range_max']))
    

    normalizer_path = os.path.join(cfg.normalize_config['normalizer_save_dir'], f"normalizer_x_{cfg.normalize_config['method']}.pkl")
    save_normalizer(normalizer_all, normalizer_path)
    print(f"全局标准化器已保存至: {normalizer_path}")
    
    target_normalizer_path = os.path.join(cfg.normalize_config['normalizer_save_dir'], f"normalizer_y_{cfg.normalize_config['method']}.pkl")
    save_normalizer(target_normalizer, target_normalizer_path)
    print(f"目标变量标准化器已保存至: {target_normalizer_path}")
    
    # 保存标准化器信息到args中，供后续使用
    cfg.x_normalizer = normalizer_all
    cfg.y_normalizer = target_normalizer





# ---------------------------------------------------------------------------------------------------------------------------------
# 模块3：训练函数
def train_epoch(model, dataloader, optimizer, criterion_func, cfg, device):
    """
    执行一个训练周期 (使用 Teacher Forcing)
    """
    model.train() # 切换到训练模式
    total_loss = 0.0

    # 使用 tqdm 显示进度条
    pbar = tqdm(dataloader, desc=f"Training Epoch {cfg.current_epoch+1}/{cfg.num_epochs}", leave=False)
    
    for x_batch, y_batch in pbar:
        # 1. 数据移动到设备 (Dataloader 可能已经做了, 但双重保险)
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if y_batch.dim() == 2:
            y_batch = y_batch.unsqueeze(-1) # -> (batch_size, seq_len, 1)
        
        # 2. 梯度清零
        optimizer.zero_grad()
        
        # 3. 前向传播 (Teacher Forcing)
        #    模型 `forward` 接收 (encoder_input, decoder_input)
        #    在训练时, decoder_input 就是我们的目标 y_batch
        output = model(x_batch, y_batch) 
        
        # 4. 计算损失
        #    比较模型输出和真实目标
        loss = criterion_func(output, y_batch)
        
        # 5. 反向传播
        loss.backward()
        
        # (可选) 梯度裁剪 (防止RNN梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 6. 更新权重
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")
        
    return total_loss / len(dataloader)

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块4：验证函数
def validate_epoch(model, dataloader, criterion, cfg, device):
    """
    执行一个验证周期 (使用自回归)
    """
    model.eval() # 切换到评估模式
    total_loss = 0.0

    
    
    with torch.no_grad(): # 在评估时关闭梯度计算
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            if y_batch.dim() == 2:
                y_batch = y_batch.unsqueeze(-1) # -> (batch_size, seq_len, 1)
            
            # 1. 前向传播 (自回归)
            #    使用 `predict_autoregressive` 方法
            predictions = model.predict_autoregressive(x_batch)

            if torch.isnan(predictions).any():
                print("!!! 验证集预测中出现 NaN !!!")
            print(f"验证集预测 (shape: {predictions.shape}):")
            print(f"  - 均值: {predictions.mean().item():.4f}")
            print(f"  - 最大值: {predictions.max().item():.4f}")
            print(f"  - 最小值: {predictions.min().item():.4f}")

            
            # 2. 计算损失
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    

    
    return total_loss / len(dataloader)

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块4：主函数 (补全)
def main():
    # 1. 导入参数
    if_normalized = True
    cfg = __init__()
    logger = cfg.logger # 获取日志记录器

    # 2. 标准化
    if not if_normalized:
        normalization(cfg)
        exit(0)
    else:
        cfg.x_normalizer = load_normalizer(cfg.normalizer['x'])
        cfg.y_normalizer = load_normalizer(cfg.normalizer['y'])
        


    # --- (!! 关键修正 !!) ---
    # 2. (修正顺序) *首先* 创建 Dataloaders
    #    这会在任何 CUDA 初始化之前创建子进程 (如果 num_workers > 0)
    try:
        logger.info("正在创建 Dataloaders...")####NOTE: dataloader还未加normalization部分
        train_loader, val_loader = get_train_valid_dataloaders(
            data_config=cfg.data_config, # (使用 data_config)
            batch_size=cfg.batch_size,
            x_normalizer= cfg.x_normalizer,
            y_normalizer= cfg.y_normalizer,
            num_workers=cfg.num_workers
        )
        logger.info(f"Dataloaders 创建成功。")
        logger.info(f"训练集批次数: {len(train_loader)}, 验证集批次数: {len(val_loader)}")
    except Exception as e:
        logger.error(f"创建 Dataloader 失败: {e}", exc_info=True)
        return

    # --- (!! 关键修正 !!) ---
    # 3. (修正顺序) *现在* 安全地设置设备和初始化 CUDA
    if use_gpu:
        try:
            # (修正) 设置 local_rank，如果提供了的话。
            # DDP 通常需要这个。对于单GPU，设为 0。
            gpu_id = 0 
            if cfg.local_rank != -1:
                gpu_id = cfg.local_rank
            
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
            logger.info(f"已设置使用 GPU: {gpu_id}")
        except Exception as e:
            logger.error(f"设置 GPU 失败: {e}. 回退到 CPU。", exc_info=True)
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("未检测到 GPU，使用 CPU。")
    
    # 4. 导入 main.model 中模型配置，并构建模型
    try:
        logger.info(f"正在构建模型: {cfg.model_cfg.model_name}...")
        # (修正) 假设我们只构建第一个模型 (索引 0)
        model = build_model(cfg.model_cfg)
        model.to(device) # (修正) 将模型移动到设备
        logger.info("模型构建成功并已移动到设备。")
    except Exception as e:
        logger.error(f"构建模型失败: {e}", exc_info=True)
        return
    
    # 5. 构建优化器和损失函数
    optimizer, scheduler = build_optimizer(cfg, model)
    criterion = custom_loss(r_rmse=cfg.loss_config.r_rmse, r_mae=cfg.loss_config.r_mae, r_mask1=cfg.loss_config.r_mask1,
                            r_mask2=cfg.loss_config.r_mask2, r_smooth=cfg.loss_config.r_smooth,
                            range_min=cfg.loss_config.range_min, range_max=cfg.loss_config.range_max)
    logger.info(f"优化器: {cfg.opt_type}, 调度器: {cfg.scheduler}, 损失函数: MSELoss")

    # 6. (可选) 加载预训练模型
    if cfg.eval_model is not None:
        try:
            logger.info(f"正在加载预训练模型: {cfg.eval_model}")
            model.load_state_dict(torch.load(cfg.eval_model, map_location=device))
        except Exception as e:
            logger.warning(f"加载预训练模型失败: {e}. 从头开始训练。")
    
    # 7. 开始训练循环
    logger.info(f"--- 开始训练, 从 Epoch {cfg.start_epoch+1} 到 {cfg.num_epochs} ---")
    best_val_loss = float('inf')

    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        cfg.current_epoch = epoch # 更新 config 中的 epoch 计数
        
        # 7.1 训练
        try:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg, device)
        except Exception as e:
            logger.error(f"Epoch {epoch+1} 训练期间发生错误: {e}", exc_info=True)
            break # 发生严重错误，停止训练

        # 7.2 验证
        try:
            val_loss = validate_epoch(model, val_loader, criterion, cfg, device)
        except Exception as e:
            logger.error(f"Epoch {epoch+1} 验证期间发生错误: {e}", exc_info=True)
            val_loss = float('inf') # 标记为失败

        # 7.3 日志记录
        log_msg = f"Epoch {epoch+1}/{cfg.num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        logger.info(log_msg)

        # 7.4 学习率调度器步进
        if scheduler:
            scheduler.step()
            # (可选) 记录当前学习率
            # current_lr = optimizer.param_groups[0]['lr']
            # logger.debug(f"Epoch {epoch+1} LR scheduler step. New LR: {current_lr:.8f}")


        # 7.5 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(cfg.model_save_dir, "best_model.pth")
            try:
                torch.save(model.state_dict(), save_path)
                logger.info(f"*** 新的最佳模型已保存 (Val Loss: {best_val_loss:.6f}) ***")
            except Exception as e:
                logger.warning(f"保存最佳模型失败: {e}")

        # 7.6 定期保存检查点
        if (epoch + 1) % cfg.save_interval == 0:
            save_path = os.path.join(cfg.model_save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            try:
                torch.save(model.state_dict(), save_path)
                logger.info(f"已保存检查点: {save_path}")
            except Exception as e:
                logger.warning(f"保存检查点失败: {e}")

    logger.info("--- 训练完成 ---")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")

if __name__ == '__main__':
    main()

