# -*- coding: utf-8 -*-
"""
使用预训练的基础模型 (Base Model) 和残差扩散模型 (DDPM/DiffusionNet)
进行风速预测。
"""
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
import pickle
import math # <--- 引入 math
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from main.model import build_model
# --- 1. 路径设置 ---
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. 导入自定义模块 ---
from utils.commn import save_config, __mkdir__
# (!!) 注意: build_model 需要能同时构建 Base Model 和 DiffusionNet1D
#     你可能需要修改 build_model 或在这里复制/导入模型定义
# from main.model import build_model # 假设 build_model 能处理两种模型
from utils.data_loader import get_test_dataloaders # <--- 使用你的 get_test_dataloaders
from utils.data_normalization import load_normalizer

# --- (!!) 复制/导入模型定义 ---
#     (这是最安全的方式，确保代码独立性)
# TODO: 将你的 BaseModel (enhanced_s2s.py) 和 DiffusionNet1D 相关类复制到这里
#     或者确保它们能被正确导入


# --- DiffusionNet1D 相关类 (从训练脚本复制) ---
class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels: int, endpoint: bool = False, amp_mode: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.endpoint = endpoint
        self.amp_mode = amp_mode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.float32
        if self.endpoint:
            freqs = torch.arange(self.num_channels // 2, device=x.device, dtype=dtype)
            freqs = freqs / (self.num_channels // 2 - 1)
        else:
            freqs = torch.arange(self.num_channels // 2, device=x.device, dtype=dtype)
            freqs = freqs / (self.num_channels // 2)
        freqs = (10000. ** -freqs).to(dtype=dtype)
        args = x.to(dtype=dtype).ger(freqs)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        return emb.to(dtype=x.dtype)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, dilation: int):
        super().__init__()
        self.emb_proj = nn.Linear(emb_channels, out_channels * 2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        emb_out = self.emb_proj(emb)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = h * scale.unsqueeze(-1) + shift.unsqueeze(-1)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)

class DiffusionNet1D(nn.Module):
    # --- (!!) 使用 config 中的维度 ---
    def __init__(self, cfg):
        super().__init__()
        self.c_in = cfg.C_IN
        self.l_in = cfg.L_IN
        self.c_out = cfg.C_OUT
        self.l_out = cfg.L_OUT
        self.model_channels = cfg.MODEL_CHANNELS
        self.emb_channels = cfg.EMB_CHANNELS

        self.condition_encoder = nn.Sequential(
            nn.Conv1d(self.c_in, self.model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.model_channels), nn.SiLU(),
            nn.Conv1d(self.model_channels, self.model_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(self.model_channels, self.emb_channels), nn.SiLU()
        )
        self.map_noise = PositionalEmbedding(num_channels=self.model_channels)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.model_channels, self.emb_channels), nn.SiLU(),
            nn.Linear(self.emb_channels, self.emb_channels), nn.SiLU()
        )
        self.in_conv = nn.Conv1d(self.c_out, self.model_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock1D(self.model_channels, self.model_channels, self.emb_channels, dilation=1),
            ResidualBlock1D(self.model_channels, self.model_channels, self.emb_channels, dilation=2),
            ResidualBlock1D(self.model_channels, self.model_channels, self.emb_channels, dilation=4),
        ])
        self.out_conv = nn.Conv1d(self.model_channels, self.c_out, kernel_size=3, padding=1)

    def forward(self, y_noisy: torch.Tensor, sigma: torch.Tensor, x_past: torch.Tensor) -> torch.Tensor:
        # y_noisy 形状: [B, C_out, L_out]
        # sigma 形状: [B]
        # x_past 形状: [B, C_in, L_in]
        c = self.condition_encoder(x_past)
        emb = self.map_noise(sigma)
        emb = self.time_embedding(emb)
        total_emb = c + emb
        h = self.in_conv(y_noisy)
        for block in self.blocks:
            h = block(h, total_emb)
        y_pred_residual = self.out_conv(h)
        return y_pred_residual


# --- 3. 设备设置 ---
use_gpu = torch.cuda.is_available()
print(f"检测到 GPU: {use_gpu}")

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块1：Config 初始化
# ---------------------------------------------------------------------------------------------------------------------------------
def __init__():
    parser = argparse.ArgumentParser(description='DDPM Model Prediction')
    # (!!) 修改为加载 ddpm.py
    parser.add_argument('--configs', type=str, default='ddpm', help='configs of model (ddpm.py)')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank (unused here)')
    # (!!) 修改 project 名称
    parser.add_argument('--project', type=str, default='14s', help='project name')
    args = parser.parse_args()

    # (!!) 修改为加载 ddpm.py config
    module_name = f'main.configs.test.{args.project}.{args.configs}'
    try:
        xconfig = importlib.import_module(module_name)
    except ImportError:
        print(f"!! 错误: 无法加载配置文件: {module_name} !!")
        print(f"   请确保 'main/configs/test/{args.project}/{args.configs}.py' 文件存在。")
        sys.exit(1)

    # (!!) 修改 config 快照保存路径
    config_save_path = os.path.join(xconfig.project_path, args.configs) # 使用 config 中的 project_path
    try:
        save_config(module_name, config_save_path)
    except Exception as e:
        print(f"警告: 无法保存 config 快照: {e}")

    xconfig.logger.info(f'[Work Dir]: {xconfig.work_dir}')
    xconfig.logger.info(f'[Project Dir]: {xconfig.project_path}')
    xconfig.logger.info(f'[Configs]: {module_name}')

    torch.manual_seed(xconfig.rand_seed)
    np.random.seed(xconfig.rand_seed)
    random.seed(xconfig.rand_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(xconfig.rand_seed)

    xconfig.local_rank = args.local_rank
    return xconfig

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块2：加载模型检查点
# ---------------------------------------------------------------------------------------------------------------------------------
def load_checkpoint(checkpoint_path, model, logger):
    """
    加载模型检查点 (state_dict)。
    """
    logger.info(f"--- 正在加载模型检查点: {checkpoint_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    try:
        # (!!) 修改: 直接加载 state_dict
        state_dict = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        logger.error(f"!! 加载 checkpoint 文件失败: {e} !!", exc_info=True)
        return model # 返回未加载的模型

    # 处理 DDP ('module.' 前缀)
    new_state_dict = OrderedDict()
    is_ddp = False
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
            is_ddp = True
        else:
            new_state_dict[k] = v

    if is_ddp:
        logger.info("检测到 'module.' 前缀 (DDP 模型), 已自动移除。")

    try:
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        logger.info("模型参数加载成功并已切换到 .eval() 模式。")
    except Exception as e:
        logger.error(f"!! 加载 state_dict 到模型时失败: {e} !!", exc_info=True)
        # 不返回，让主程序知道失败了

    return model

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块3: 加载时间戳 (与 base model predict 脚本相同)
# ---------------------------------------------------------------------------------------------------------------------------------
def load_all_time_info(data_config, logger):
    """独立于 Dataloader 加载所有 Y 时间戳。"""
    logger.info("正在独立加载时间戳 (从 .pkl 文件)...")
    base_dir = data_config['base_dir']
    folder_list = data_config['test_folders']

    if 'time_info_suffix' not in data_config or 'time_info_key' not in data_config:
        logger.error("!! 错误: 'data_config' 中未找到 'time_info_suffix' 或 'time_info_key' !!")
        return None

    time_suffix = data_config['time_info_suffix']
    time_key = data_config['time_info_key']
    all_times = []

    for folder in folder_list:
        time_filename = folder + time_suffix
        time_file_path = os.path.join(base_dir, folder, time_filename)
        try:
            with open(time_file_path, 'rb') as f: data_pkl = pickle.load(f)
            times_data = data_pkl[time_key]
            # (!!) 确保时间戳是 numpy 数组
            if isinstance(times_data, list): times_data = np.array(times_data)
            all_times.append(times_data)
        except FileNotFoundError:
            logger.error(f"!! 严重错误: 找不到时间文件: {time_file_path} !!")
            return None
        except KeyError:
            logger.error(f"!! 严重错误: 在 {time_file_path} 中找不到键: '{time_key}' !!")
            return None
        except Exception as e:
            logger.error(f"加载 {time_file_path} 时出错: {e}", exc_info=True)
            return None

    if not all_times:
        logger.error("未能加载任何时间戳数据。")
        return None

    try:
        # (!!) 调整拼接和维度扩展
        combined_times = np.concatenate(all_times, axis=0)
        # 假设 pkl 中的时间戳已经是 (N, 14) 或类似的形状
        if combined_times.ndim == 1: # 如果是 (N,)
             # 这可能意味着 pkl 只存了开始时间，需要手动扩展
             # logger.warning("时间戳似乎只有一维，假设为开始时间。结果可能不准确。")
             # 如果需要 (N, 14)，你需要知道如何从开始时间生成后续时间
             # 暂时保持原样，让 save_results 处理
             pass
        elif combined_times.ndim == 2 and combined_times.shape[1] != basemodel_cfg.output_len:
             logger.warning(f"时间戳第二维度 ({combined_times.shape[1]}) 与 output_len ({basemodel_cfg.output_len}) 不匹配")
             
        logger.info(f"时间戳加载成功。形状: {combined_times.shape}")
        return combined_times
    except Exception as e:
        logger.error(f"拼接时间戳时出错: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------------------------------------------------------------
# (!!) 模块4：扩散模型预测函数 (从 Canvas 复制并调整)
# ---------------------------------------------------------------------------------------------------------------------------------
def predict_wind_speed(
    x_past: torch.Tensor,
    base_model: nn.Module,
    diffusion_net: nn.Module,
    device: torch.device,
    # (!!) 从 config 读取这些参数
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    logger, # 添加 logger,
    cfg
) -> torch.Tensor: # (!!) 返回 Tensor 而不是 Numpy
    """
    使用基础模型和残差扩散模型预测未来风速 (返回标准化结果)。
    """
    batch_size = x_past.shape[0]
    logger.debug(f"predict_wind_speed: x_past shape={x_past.shape}")

    # --- 1. 基础模型预测 ---
    base_model.eval()
    diffusion_net.eval()
    with torch.no_grad():
        try:
            #decoder_input = torch.zeros_like((batch_size, cfg.basemodel_cfg.output_len, cfg.basemodel_cfg.n_labels))
            y_base_pred = base_model.predict_autoregressive(x_past.permute(0,2,1)).permute(0,2,1) # 形状 [B, C_out, L_out]
            logger.debug(f"predict_wind_speed: y_base_pred shape={y_base_pred.shape}")
        except Exception as e:
            logger.error(f"基础模型预测失败: {e}", exc_info=True)
            raise # 重新抛出异常

    # --- 2. 准备扩散采样 ---
    # a. Karras 时间表
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (sigma_max**(1/rho) + step_indices / (num_steps - 1) * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # 添加 t=0

    # b. 初始化: 从纯噪音开始
    y_t = torch.randn_like(y_base_pred) * t_steps[0] # [B, C_out, L_out]
    logger.debug(f"predict_wind_speed: Initial y_t shape={y_t.shape}, sigma0={t_steps[0]}")

    # --- 3. 迭代去噪循环 ---
    for i in tqdm(range(num_steps), desc="扩散采样中", leave=False):
        sigma_t = t_steps[i]
        sigma_next = t_steps[i+1]
        logger.debug(f"Step {i}: sigma_t={sigma_t}, sigma_next={sigma_next}")

        with torch.no_grad():
            sigma_t_batch = sigma_t.repeat(batch_size).to(device) # [B]
            logger.debug(f"  Input to diffusion_net: y_t={y_t.shape}, sigma={sigma_t_batch.shape}, x_past={x_past.shape}")
            try:
                y_pred_residual = diffusion_net(y_t, sigma_t_batch, x_past) # [B, C_out, L_out]
                logger.debug(f"  Output from diffusion_net: y_pred_residual={y_pred_residual.shape}")
            except Exception as e:
                logger.error(f"扩散模型预测失败 (Step {i}): {e}", exc_info=True)
                raise # 重新抛出异常

            # Karras (EDM) 采样器步骤
            d = (y_t - y_pred_residual) / sigma_t # 估计的 score
            y_next = y_t + d * (sigma_next - sigma_t)
            logger.debug(f"  d shape={d.shape}, y_next shape={y_next.shape}")

        y_t = y_next

    # --- 4. 结合基础预测和生成的残差 ---
    y_final_residual = y_t # 经过 num_steps 后，y_t 应该是预测的干净残差
    final_prediction_normalized = y_base_pred + y_final_residual
    logger.debug(f"predict_wind_speed: final_prediction_normalized shape={final_prediction_normalized.shape}")

    return final_prediction_normalized # 返回 PyTorch Tensor

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块5：测试循环 (修改为使用 predict_wind_speed)
# ---------------------------------------------------------------------------------------------------------------------------------
def test_epoch_ddpm(
    base_model,
    diffusion_net,
    dataloader,
    device,
    # (!!) 从 config 读取采样参数
    num_steps, sigma_min, sigma_max, rho,
    logger,
    cfg
):
    """执行一个完整的测试周期 (使用 DDPM 预测)"""
    base_model.eval()
    diffusion_net.eval()
    all_predictions_scaled = []
    all_ground_truth_scaled = []

    desc = "Testing Model (DDPM)"
    pbar = tqdm(dataloader, desc=desc, leave=False, total=len(dataloader))

    with torch.no_grad():
        for x_batch_scaled, y_batch_scaled in pbar:

            x_batch_scaled = x_batch_scaled.to(device).float()
            y_batch_scaled = y_batch_scaled.to(device).float()

            # --- (!!) 调整维度 ---
            # 假设 DataLoader 返回 [B, L, C], 模型需要 [B, C, L]
            if x_batch_scaled.dim() == 3 and x_batch_scaled.shape[1] == cfg.diffusionmodel_cfg.L_IN and x_batch_scaled.shape[2] == cfg.diffusionmodel_cfg.C_IN:
                 x_batch_scaled = x_batch_scaled.permute(0, 2, 1)
            if y_batch_scaled.dim() == 3 and y_batch_scaled.shape[1] == cfg.diffusionmodel_cfg.L_OUT and y_batch_scaled.shape[2] == cfg.diffusionmodel_cfg.C_OUT:
                 y_batch_scaled = y_batch_scaled.permute(0, 2, 1)
            # ---

            # --- (!!) 调用 DDPM 预测函数 ---
            try:
                predictions_scaled = predict_wind_speed(
                    x_past=x_batch_scaled,
                    base_model=base_model,
                    diffusion_net=diffusion_net,
                    device=device,
                    num_steps=num_steps,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho=rho,
                    logger=logger,
                    cfg=cfg # 传递 logger
                )
            except Exception as e:
                 logger.error(f"预测批次时出错: {e}", exc_info=True)
                 # 可以选择跳过这个批次或中止
                 continue # 跳过

            all_predictions_scaled.append(predictions_scaled.cpu().numpy())
            all_ground_truth_scaled.append(y_batch_scaled.cpu().numpy())
            pbar.set_postfix(batch_shape=f"{x_batch_scaled.shape[0]}") # 显示批次大小

    try:
        all_predictions_scaled = np.concatenate(all_predictions_scaled, axis=0)
        all_ground_truth_scaled = np.concatenate(all_ground_truth_scaled, axis=0)
    except ValueError as e:
        logger.error(f"!! 错误：无法拼接批次结果: {e}", exc_info=True)
        return None, None

    return all_predictions_scaled, all_ground_truth_scaled

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块6: 评估函数 (与 base model predict 脚本相同)
# ---------------------------------------------------------------------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred, logger):
    """评估预测结果 (PyTorch 版本)"""
    logger.info("正在评估预测结果...")
    try:
        #if y_true.ndim == 3: y_true = y_true.squeeze(-1)
        if y_pred.ndim == 3: y_pred = y_pred.squeeze(-1)
        mse = np.mean((y_pred - y_true) ** 2, axis=-1)
        rmse = np.sqrt(mse)
        rmse_score = np.mean(rmse)
        mae = np.mean(np.abs(y_pred - y_true), axis=-1)
        mae_score = np.mean(mae)
        logger.info(f"均方根误差 (RMSE): {rmse_score:.4f}")
        logger.info(f"平均绝对误差 (MAE): {mae_score:.4f}")
        return rmse_score, mae_score
    except Exception as e:
        logger.error(f"评估指标时出错: {e}", exc_info=True)
        return -1.0, -1.0

# ---------------------------------------------------------------------------------------------------------------------------------
# 模块7: 保存结果 (与 base model predict 脚本基本相同)
# ---------------------------------------------------------------------------------------------------------------------------------
def save_results(y_true_unscaled, y_pred_unscaled, all_times, output_path, output_len, model_version, logger):
    """按照你的 TF 格式保存所有结果"""
    logger.info("正在保存预测结果...")
    

    # 1. 保存 DataFrames
    logger.info("准备 DataFrames...")
    try:
        # 假设 y_true/y_pred 是 [N, L_out, C_out], 需要转置并squeeze
        if y_true_unscaled.ndim == 3 and y_true_unscaled.shape[2] == 1:
            y_true_flat = y_true_unscaled.squeeze(-1) # -> [N, L_out]
        else:
             y_true_flat = y_true_unscaled # 假设已经是 [N, L_out]
        if y_pred_unscaled.ndim == 3 and y_pred_unscaled.shape[2] == 1:
            y_pred_flat = y_pred_unscaled.squeeze(-1) # -> [N, L_out]
        elif y_pred_unscaled.ndim == 3 and y_pred_unscaled.shape[1] == 1:
            y_pred_flat = y_pred_unscaled.transpose(0,2,1).squeeze(-1)
        else:
             y_pred_flat = y_pred_unscaled

        # 时间戳处理: 假设 all_times 是 [N, L_out] 或 [N,]
        if all_times.ndim == 2 and all_times.shape[1] >= 1:
             index_list = all_times[:, 0] # 使用第一个时间戳作为索引
             time_cols_available = all_times.shape[1]
        elif all_times.ndim == 1:
             index_list = all_times # 假设只有开始时间
             time_cols_available = 1
             logger.warning("时间戳只有一维，假设为开始时间，summary 中的时间可能不准确。")
        else:
             logger.error("时间戳形状无法处理，将使用默认索引。")
             index_list = None
             time_cols_available = 0


        col_names = [f't{j+1}' for j in range(output_len)]
        true_df = pd.DataFrame(y_true_flat, index=index_list, columns=[f'true_{c}' for c in col_names])
        pred_df = pd.DataFrame(y_pred_flat, index=index_list, columns=[f'pred_{c}' for c in col_names])

        true_df.to_pickle(os.path.join(output_path, f"y_true_df_{model_version}.pkl"))
        pred_df.to_pickle(os.path.join(output_path, f"y_pred_df_{model_version}.pkl"))

        sample_size = min(100, len(true_df))
        true_df.head(sample_size).round(2).to_csv(os.path.join(output_path, f"y_true_df_{model_version}_sample.csv"))
        pred_df.head(sample_size).round(2).to_csv(os.path.join(output_path, f"y_pred_df_{model_version}_sample.csv"))
        logger.info("DataFrame (PKL 和 CSV 样本) 保存成功。")

    except Exception as e:
        logger.error(f"保存 DataFrame 时出错: {e}", exc_info=True)

    # 2. 保存 Summary
    logger.info("准备 Summary DataFrame...")
    try:
        y_pred_unscaled = y_pred_unscaled.transpose(0,2,1).squeeze(-1)
        summary_data = []
        num_samples = y_true_unscaled.shape[0]
        
        for i in range(num_samples):
            base_time = all_times[i] # (N, 14) -> 第 i 个样本的第 0 个时间
            for j in range(output_len):
                true_val = y_true_unscaled[i, j]
                pred_val = y_pred_unscaled[i, j]
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

    # 3. 保存评估指标
    logger.info("保存评估指标...")
    try:
        rmse_score, mae_score = evaluate_predictions(y_true_unscaled, y_pred_unscaled, logger)
        metrics_df = pd.DataFrame({'metric': ['RMSE', 'MAE'], 'value': [rmse_score, mae_score]})
        metrics_df.to_csv(os.path.join(output_path, f"metrics_{model_version}.csv"), index=False)
        logger.info("Metrics (CSV) 保存成功。")
    except Exception as e:
        logger.error(f"保存 Metrics 时出错: {e}", exc_info=True)

    logger.info(f"--- 所有结果已保存到: {output_path} ---")


# ---------------------------------------------------------------------------------------------------------------------------------
# 模块8：主函数
# ---------------------------------------------------------------------------------------------------------------------------------
def main():
    # 1. 初始化参数
    cfg = __init__()
    logger = cfg.logger

    # 2. 数据准备
    try:
        cfg.x_normalizer = load_normalizer(cfg.normalizer['x'])
        cfg.y_normalizer = load_normalizer(cfg.normalizer['y'])
        logger.info("标准化器加载成功。")
    except Exception as e:
        logger.error(f"加载标准化器失败: {e}", exc_info=True)
        return

    try:
        logger.info("正在创建 Test Dataloader (shuffle=False)...")
        test_loader = get_test_dataloaders(
            data_config=cfg.data_config,
            batch_size=cfg.batch_size,
            x_normalizer= cfg.x_normalizer,
            y_normalizer= cfg.y_normalizer,
            num_workers=cfg.num_workers
        )
        logger.info(f"Test Dataloader 创建成功。批次数: {len(test_loader)}")
    except Exception as e:
        logger.error(f"创建 Test Dataloader 失败: {e}", exc_info=True)
        return

    # 3. 加载时间戳
    all_times = load_all_time_info(cfg.data_config, logger)
    if all_times is None:
        logger.error("无法加载时间戳，预测中止。")
        return

    # 4. 验证数据一致性
    if len(test_loader.dataset) != len(all_times):
        logger.error("!! 严重错误: Dataloader 样本数与时间戳样本数不匹配 !!")
        logger.error(f"   Dataloader 样本数: {len(test_loader.dataset)}")
        logger.error(f"   时间戳样本数 (来自 .pkl): {len(all_times)}")
        # (!!) 如果不匹配，最好中止
        return

    # 5. 设置环境
    if use_gpu:
        try:
            gpu_id = 0
            if cfg.local_rank != -1: gpu_id = cfg.local_rank
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
    try:
        # a. 构建基础模型
        logger.info("构建基础模型...")
        # (!!) 使用 basemodel_cfg
        base_model = build_model(cfg.basemodel_cfg) # 假设 build_model 能区分
        logger.info("基础模型构建成功。")

        # b. 构建扩散模型
        logger.info("构建扩散模型...")
        # (!!) 使用 diffusionmodel_cfg
        diffusion_net = DiffusionNet1D(
        cfg.diffusionmodel_cfg)

        #diffusion_net = build_model(cfg.diffusionmodel_cfg, 'diffusion') # 假设 build_model 能区分
        logger.info("扩散模型构建成功。")

    except Exception as e:
        logger.error(f"构建模型失败: {e}", exc_info=True)
        return

    # 7. 加载模型权重
    try:
        # a. 加载基础模型权重
        base_model = load_checkpoint(cfg.base_model_path, base_model, logger)

        # b. 加载扩散模型权重
        diffusion_net = load_checkpoint(cfg.diffusion_model_path, diffusion_net, logger)

        # (!!) 确保两个模型都已加载
        #     load_checkpoint 内部已包含 .to(device) 和 .eval()

    except Exception as e:
        logger.error(f"加载模型权重时出错: {e}", exc_info=True)
        return

    # 8. 进行预测
    logger.info("--- 开始执行 DDPM 模型预测 ---")
    try:
        scaled_preds, scaled_true = test_epoch_ddpm(
            base_model=base_model,
            diffusion_net=diffusion_net,
            dataloader=test_loader,
            device=device,
            # (!!) 从 config 传递采样参数
            num_steps=cfg.num_steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            rho=cfg.rho,
            logger=logger, # 传递 logger,
            cfg=cfg
        )
        if scaled_preds is None:
             logger.error("预测失败，未返回任何结果。")
             return
        logger.info(f"预测完成。Scaled Preds 形状: {scaled_preds.shape}, Scaled True 形状: {scaled_true.shape}")

        # 9. 反归一化
        logger.info("正在反归一化预测结果...")
        unscaled_preds = cfg.y_normalizer.inverse_transform(scaled_preds)
        unscaled_true = cfg.y_normalizer.inverse_transform(scaled_true)
        logger.info("反归一化完成。")

        # 10. 保存所有结果
        # (自动创建 'predict' 目录 - 基于 project_path)
        output_dir = os.path.join(cfg.project_path, 'predict')
        os.makedirs(output_dir, exist_ok=True)

        save_results(
            y_true_unscaled=unscaled_true,
            y_pred_unscaled=unscaled_preds,
            all_times=all_times,
            output_path=output_dir,
            output_len=cfg.basemodel_cfg.output_len, # 从 config 获取
            model_version=cfg.diffusionmodel_cfg.model_name[0], # 使用 DDPM 模型名
            logger=logger
        )

    except Exception as e:
        logger.error(f"预测或保存过程中发生严重错误: {e}", exc_info=True)
        return

if __name__ == '__main__':
    main()
