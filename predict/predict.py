import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
from datetime import datetime
from tqdm import tqdm
import importlib
import random
from collections import OrderedDict
import pickle # (!! 新增 !!)

# --- 1. 路径设置 ---
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
# --- 2. 导入自定义模块 ---
from utils.commn import save_config, __mkdir__
from main.model import build_model
from utils.data_loader import get_test_dataloaders
from utils.data_normalization import load_normalizer

# --- 3. 设备设置 ---
use_gpu = torch.cuda.is_available()
print(f"检测到 GPU: {use_gpu}")

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块1：Config 初始化
# ---------------------------------------------------------------------------------------------------------------------------------
def __init__():
    parser = argparse.ArgumentParser(description='S2S Model Prediction')
    parser.add_argument('--configs', type=str, default='enhanced_s2s', help='configs of model')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training (DDP)')
    parser.add_argument('--project', type=str, default='14s', help='project name, e.g., 14s, 60min')
    args = parser.parse_args()
    
    module_name = f'main.configs.test.{args.project}.{args.configs}'
    try:
        xconfig = importlib.import_module(module_name)
    except ImportError:
        print(f"!! 错误: 无法加载配置文件: {module_name} !!")
        print(f"   请确保 'main/configs/test/{args.project}/{args.configs}.py' 文件存在。")
        sys.exit(1)
        
    config_save_path = os.path.join(xconfig.work_dir, args.project, args.configs)
    try:
        save_config(module_name, config_save_path + "_predict_snapshot.py")
    except Exception as e:
        print(f"警告: 无法保存 config 快照: {e}")
    
    xconfig.logger.info(f'[Work Dir]: {xconfig.work_dir}')
    xconfig.logger.info(f'[Configs]: {module_name}')
    
    torch.manual_seed(xconfig.rand_seed)
    np.random.seed(xconfig.rand_seed)
    random.seed(xconfig.rand_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(xconfig.rand_seed)

    xconfig.local_rank = args.local_rank
    return xconfig

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块2：加载模型
# ---------------------------------------------------------------------------------------------------------------------------------
def load_checkpoint(checkpoint_path, model):
    """
    (修正版)
    正确加载一个 model.state_dict() 保存的 .pth 文件。
    """
    print(f"--- 正在加载模型检查点: {checkpoint_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"!! 加载 checkpoint 文件失败: {e} !!")
        return model

    new_state_dict = OrderedDict()
    is_ddp = False
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除 'module.'
            is_ddp = True
        else:
            new_state_dict[k] = v
    
    if is_ddp:
        print("检测到 'module.' 前缀 (DDP 模型), 已自动移除。")

    try:
        model.load_state_dict(new_state_dict)
        model.to(device) 
        model.eval()     
        print("模型参数加载成功并已切换到 .eval() 模式。")
    except Exception as e:
        print(f"!! 加载 state_dict 到模型时失败: {e} !!")
        
    return model

# ---------------------------------------------------------------------------------------------------------------------------------
# (!! 新增 !!) 模块3: 加载时间戳 (适配 .pkl 文件)
# ---------------------------------------------------------------------------------------------------------------------------------
def load_all_time_info(data_config, logger):
    """
    独立于 Dataloader，加载所有 Y 时间戳。
    这依赖于 test_loader 的 shuffle=False，以保证顺序匹配。
    """
    logger.info("正在独立加载时间戳 (从 .pkl 文件)...")
    base_dir = data_config['base_dir']
    folder_list = data_config['test_folders']
    
    if 'time_info_suffix' not in data_config or 'time_info_key' not in data_config:
        logger.error("!! 错误: 'data_config' 中未找到 'time_info_suffix' 或 'time_info_key' !!")
        logger.error("   请在 test config 文件中添加这两个键。")
        return None
        
    time_suffix = data_config['time_info_suffix'] # e.g., '_time_info.pkl'
    time_key = data_config['time_info_key']       # e.g., 'y_time_model1'
    
    all_times = []
    
    for folder in folder_list:
        # e.g., DATA-16-20240601-add-feature_time_info.pkl
        time_filename = folder + time_suffix 
        time_file_path = os.path.join(base_dir, folder, time_filename)
        
        try:
            with open(time_file_path, 'rb') as f:
                data_pkl = pickle.load(f)
            
            # (!! 关键 !!) 从字典中提取你需要的时间戳
            times_data = data_pkl[time_key]
            all_times.append(times_data)
            
        except FileNotFoundError:
            logger.error(f"!! 严重错误: 找不到时间文件: {time_file_path} !!")
            return None
        except KeyError:
            logger.error(f"!! 严重错误: 在 {time_file_path} 中找不到键: '{time_key}' !!")
            return None
        except Exception as e:
            logger.error(f"加载 {time_file_path} 时出错: {e}")
            return None

    if not all_times:
        logger.error("未能加载任何时间戳数据。")
        return None

    # 拼接所有时间戳
    try:
        # 假设时间戳已经是 numpy 数组
        combined_times = np.concatenate(all_times, axis=0)
        combined_times = np.expand_dims(combined_times, axis=-1)
        logger.info(f"时间戳加载成功。形状: {combined_times.shape}")
        return combined_times
    except Exception as e:
        logger.error(f"拼接时间戳时出错: {e}")
        return None

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块4：测试函数 (与之前相同)
# ---------------------------------------------------------------------------------------------------------------------------------
def test_epoch(model, dataloader, device):
    """
    执行一个完整的测试周期 (使用 Autoregressive 预测)
    """
    model.eval() 
    all_predictions = []
    all_ground_truth = []

    desc = "Testing Model (Autoregressive)"
    pbar = tqdm(dataloader, desc=desc, leave=False, total=len(dataloader))

    with torch.no_grad(): 
        for x_batch, y_batch in pbar:
            
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            
            if y_batch.dim() == 2:
                y_batch = y_batch.unsqueeze(-1) 

            predictions = model.predict_autoregressive(x_batch)

            all_predictions.append(predictions.cpu().numpy())
            all_ground_truth.append(y_batch.cpu().numpy())

    try:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_ground_truth = np.concatenate(all_ground_truth, axis=0)
    except ValueError as e:
        print(f"!! 错误：无法拼接批次结果: {e}")
        return None, None
        
    return all_predictions, all_ground_truth

# ---------------------------------------------------------------------------------------------------------------------------------
# (!! 新增 !!) 模块5: 评估函数 (来自你的 TF 代码)
# ---------------------------------------------------------------------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred, logger):
    """评估预测结果 (PyTorch 版本)"""
    logger.info("正在评估预测结果...")
    
    try:
        # (N, 14, 1) -> (N, 14)
        if y_true.ndim == 3: y_true = y_true.squeeze(-1)
        if y_pred.ndim == 3: y_pred = y_pred.squeeze(-1)

        # 1. 计算 RMSE
        # (y_pred - y_true)**2 -> (N, 14)
        # np.mean(..., axis=-1) -> (N,) -> 每个样本的 MSE
        # np.sqrt(...) -> (N,) -> 每个样本的 RMSE
        # np.mean(...) -> 最终的平均 RMSE
        mse = np.mean((y_pred - y_true) ** 2, axis=-1)
        rmse = np.sqrt(mse)
        rmse_score = np.mean(rmse)
        
        # 2. 计算 MAE
        mae = np.mean(np.abs(y_pred - y_true), axis=-1)
        mae_score = np.mean(mae)
        
        logger.info(f"均方根误差 (RMSE): {rmse_score:.4f}")
        logger.info(f"平均绝对误差 (MAE): {mae_score:.4f}")
        
        return rmse_score, mae_score
        
    except Exception as e:
        logger.error(f"评估指标时出错: {e}")
        return -1.0, -1.0

# ---------------------------------------------------------------------------------------------------------------------------------
# (!! 新增 !!) 模块6: 保存结果 (来自你的 TF 代码)
# ---------------------------------------------------------------------------------------------------------------------------------
def save_results(y_true_unscaled, y_pred_unscaled, all_times, output_path, output_len, model_version, logger):
    """
    按照你的 TF 格式保存所有结果
    y_true/y_pred 形状: (N, 14, 1)
    all_times 形状: (N, 14)
    """
    logger.info("正在保存预测结果...")
    
    # 0. 确保输出路径存在
    __mkdir__(output_path)
    
    # 1. 准备 DataFrame 数据 (DFs)
    logger.info("准备 DataFrames...")
    try:
        # (N, 14, 1) -> (N, 14)
        y_true_flat = y_true_unscaled.squeeze(-1)
        y_pred_flat = y_pred_unscaled.squeeze(-1)
        
        # (N, 14) -> 使用第一列 (N,) 作为索引
        index_list = all_times[:, 0]
        
        # (!! 关键 !!) 你的项目是14s, 5秒间隔, 不是小时
        # (我们假设 output_len=14)
        col_names = [f't{j+1}' for j in range(output_len)]
        
        true_df = pd.DataFrame(y_true_flat, index=index_list, columns=[f'true_{c}' for c in col_names])
        pred_df = pd.DataFrame(y_pred_flat, index=index_list, columns=[f'pred_{c}' for c in col_names])
        
        # 2. 保存为 Pickle 格式
        true_df.to_pickle(os.path.join(output_path, f"y_true_df_{model_version}.pkl"))
        pred_df.to_pickle(os.path.join(output_path, f"y_pred_df_{model_version}.pkl"))
        
        # 3. 保存 100 行样本
        sample_size = min(100, len(true_df))
        true_df.head(sample_size).round(2).to_csv(os.path.join(output_path, f"y_true_df_{model_version}_sample.csv"))
        pred_df.head(sample_size).round(2).to_csv(os.path.join(output_path, f"y_pred_df_{model_version}_sample.csv"))
        logger.info("DataFrame (PKL 和 CSV 样本) 保存成功。")

    except Exception as e:
        logger.error(f"保存 DataFrame 时出错: {e}", exc_info=True)

    # 4. 准备并保存 Summary (汇总)
    logger.info("准备 Summary DataFrame...")
    try:
        summary_data = []
        num_samples = y_true_unscaled.shape[0]
        
        for i in range(num_samples):
            base_time = all_times[i, 0] # (N, 14) -> 第 i 个样本的第 0 个时间
            for j in range(output_len):
                true_val = y_true_unscaled[i, j, 0]
                pred_val = y_pred_unscaled[i, j, 0]
                pred_time = base_time + pd.Timedelta(seconds=j+1) # 第 i 个样本的第 j 个时间
                
                summary_item = {
                    'sample_id': i,
                    'base_time': base_time,
                    'time': pred_time,
                    'true_value': true_val,
                    'predicted_value': pred_val,
                    'error': pred_val - true_val,
                    'abs_error': abs(pred_val - true_val)
                }
                summary_data.append(summary_item)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_path, f"summary_{model_version}.csv"), index=False)
        logger.info("Summary (CSV) 保存成功。")
        
    except Exception as e:
        logger.error(f"保存 Summary 时出错: {e}", exc_info=True)

    # 5. 保存评估指标
    logger.info("保存评估指标...")
    try:
        rmse_score, mae_score = evaluate_predictions(y_true_unscaled, y_pred_unscaled, logger)
        metrics_df = pd.DataFrame({
            'metric': ['RMSE', 'MAE'],
            'value': [rmse_score, mae_score]
        })
        metrics_df.to_csv(os.path.join(output_path, f"metrics_{model_version}.csv"), index=False)
        logger.info("Metrics (CSV) 保存成功。")
    
    except Exception as e:
        logger.error(f"保存 Metrics 时出错: {e}", exc_info=True)
        
    logger.info(f"--- 所有结果已保存到: {output_path} ---")


# ---------------------------------------------------------------------------------------------------------------------------------
# 模块7：主函数 (已补全)
# ---------------------------------------------------------------------------------------------------------------------------------
def main():
    # 1. 初始化参数
    cfg = __init__()
    logger = cfg.logger
    
    # 2. 数据准备
    cfg.x_normalizer = load_normalizer(cfg.normalizer['x'])
    cfg.y_normalizer = load_normalizer(cfg.normalizer['y'])
    try:
        logger.info("正在创建 Dataloaders (shuffle=False)...")
        test_loader = get_test_dataloaders(
            data_config=cfg.data_config, 
            batch_size=cfg.batch_size,
            x_normalizer= cfg.x_normalizer,
            y_normalizer= cfg.y_normalizer,
            num_workers=cfg.num_workers
        )
        logger.info(f"Dataloaders 创建成功。测试集批次数: {len(test_loader)}")
    except Exception as e:
        logger.error(f"创建 Dataloader 失败: {e}", exc_info=True)
        return
    
    # 3. (!! 新增 !!) 独立加载时间戳
    all_times = load_all_time_info(cfg.data_config, logger)
    if all_times is None:
        logger.error("无法加载时间戳，预测中止。")
        return

    # 4. 验证数据一致性
    if len(test_loader.dataset) != len(all_times):
        logger.error("!! 严重错误: Dataloader 样本数与时间戳样本数不匹配 !!")
        logger.error(f"   Dataloader 样本数: {len(test_loader.dataset)}")
        logger.error(f"   时间戳样本数 (来自 .pkl): {len(all_times)}")
        return
    
    # 5. 设置环境
    if use_gpu:
        try:
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

    # 6. 构建模型
    model = build_model(cfg.model_cfg)
    
    # 7. 加载模型权重
    checkpoint_path = cfg.eval_model
    model = load_checkpoint(checkpoint_path, model).to(device)
    logger.info("模型构建成功并已加载权重。")
    
    # 8. 进行预测
    logger.info("--- 开始执行模型预测 ---")
    try:
        scaled_preds, scaled_true = test_epoch(model, test_loader, device)
        if scaled_preds is None:
             logger.error("预测失败，未返回任何结果。")
             return
        logger.info(f"预测完成。Scaled Preds 形状: {scaled_preds.shape}, Scaled True 形状: {scaled_true.shape}")
        
        # 9. 反归一化
        logger.info("正在反归一化预测结果...")
        unscaled_preds = cfg.y_normalizer.inverse_transform(scaled_preds)
        unscaled_true = cfg.y_normalizer.inverse_transform(scaled_true)
        logger.info("反归一化完成。")
        
        # 10. (!! 新增 !!) 保存所有结果
        # (自动创建 'predict' 目录)
        output_dir = checkpoint_path.replace('/model/best_model.pth', '/predict/')
        
        save_results(
            y_true_unscaled=unscaled_true,
            y_pred_unscaled=unscaled_preds,
            all_times=all_times,
            output_path=output_dir,
            output_len=cfg.model_cfg.output_len, # e.g., 14
            model_version=cfg.model_cfg.model_name, # e.g., 'enhanced_s2s'
            logger=logger
        )
        
    except Exception as e:
        logger.error(f"预测或保存过程中发生严重错误: {e}", exc_info=True)
        return

if __name__ == '__main__':
    # 你的 main() 函数现在会处理所有事情
    # (不再需要在这里硬编码路径)
    main()