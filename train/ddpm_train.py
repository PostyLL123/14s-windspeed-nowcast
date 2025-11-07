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
from main.model import build_optimizer, build_model
# (新导入)
from utils.data_loader import get_train_valid_dataloaders
from utils.data_normalization import create_normalizer, save_normalizer, load_normalizer
from utils.checkpoint import save_checkpoint, load_checkpoint

# --- 3. 设备设置 ---
use_gpu = torch.cuda.is_available()
print(f"检测到 GPU: {use_gpu}")


def __init__():
    parser = argparse.ArgumentParser(description='ddpm for residual training')
    parser.add_argument('--configs', type=str, default='diffusion_enhanced_s2s', help='configs of model')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training (DDP)')
    parser.add_argument('--project', type=str, default='14s', help='project name, e.g., 14s, 60min')
    args = parser.parse_args()
    
    module_name = f'main.configs.train.{args.project}.{args.configs}'
    try:
        xconfig = importlib.import_module(module_name)
    except ImportError:
        print(f"!! 错误: 无法加载配置文件: {module_name} !!")
        print(f"   请确保 'main/configs/train/{args.project}/{args.configs}.py' 文件存在。")
        sys.exit(1)
        
    config_save_path = os.path.join(xconfig.work_dir, args.project, args.configs)
    try:
        save_config(module_name, config_save_path)
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
# ---
# 阶段 2：1D 残差扩散模型 (Residual Diffusion Model)
# 这是我们要训练的新模型 (`net`)
# ---

class PositionalEmbedding(nn.Module):
    """
    用于噪音水平 `sigma` 的位置嵌入 (sin/cos 编码)。
    与 SongUNet 中的 `PositionalEmbedding` 类似。
    """
    def __init__(self, num_channels: int, endpoint: bool = False, amp_mode: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.endpoint = endpoint
        self.amp_mode = amp_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.float32 # 始终使用 float32 计算嵌入
        if self.endpoint:
            freqs = torch.arange(self.num_channels // 2, device=x.device, dtype=dtype)
            freqs = freqs / (self.num_channels // 2 - 1)
        else:
            freqs = torch.arange(self.num_channels // 2, device=x.device, dtype=dtype)
            freqs = freqs / (self.num_channels // 2)
        
        freqs = (10000. ** -freqs).to(dtype=dtype)
        args = x.to(dtype=dtype).ger(freqs) # [B] @ [C/2] -> [B, C/2]
        
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        return emb.to(dtype=x.dtype) # 转换回原始数据类型

class ResidualBlock1D(nn.Module):
    """
    1D 卷积残差块 (TCN / WaveNet 风格)。
    它接收 (y_noisy) 和 (emb) 作为输入。
    """
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, dilation: int):
        super().__init__()
        
        # 用于 `emb` 向量的仿射变换 (Affine transform)
        self.emb_proj = nn.Linear(emb_channels, out_channels * 2) # [B, E] -> [B, 2*C_out]
        
        # 1D 卷积
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=dilation, 
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1) # Dilation 1
        
        # 残差连接
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x 形状: [B, C, L]
        # emb 形状: [B, Emb]
        
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # 注入条件 (emb)
        # emb_proj 输出 [B, 2*C_out], 拆分为 scale 和 shift
        emb_out = self.emb_proj(emb) # [B, 2*C_out]
        scale, shift = torch.chunk(emb_out, 2, dim=1) # 2 x [B, C_out]
        
        # FiLM 层 (Feature-wise Linear Modulation)
        # [B, C_out, L] = [B, C_out, 1] * [B, C_out, L] + [B, C_out, 1]
        h = h * scale.unsqueeze(-1) + shift.unsqueeze(-1)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        # 残差连接
        return h + self.residual_conv(x)


class DiffusionNet1D(nn.Module):
    """
    1D 条件扩散模型 (`net`)。
    它接收 `y_noisy`, `sigma`, 和 `x_past`。
    """
    def __init__(
        self,
        c_out: int,
        l_out: int,
        c_in: int,
        l_in: int,
        model_channels: int,
        emb_channels: int
    ):
        super().__init__()
        
        # --- 1. 条件编码器 (Condition Encoder for x_past) ---
        # 目标：将 x_past [B, C_in, L_in] 压缩为上下文向量 c [B, Emb]
        self.condition_encoder = nn.Sequential(
            nn.Conv1d(c_in, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1), # 全局平均池化, [B, C_model, L_in] -> [B, C_model, 1]
            nn.Flatten(), # [B, C_model]
            nn.Linear(model_channels, emb_channels), # [B, Emb]
            nn.SiLU()
        )

        # --- 2. 噪音/时间步嵌入 (Noise/Time Embedding) ---
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.time_embedding = nn.Sequential(
            nn.Linear(model_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
            nn.SiLU()
        )
        
        # --- 3. 去噪主干网络 (Denoising Backbone for y_noisy) ---
        # 这是一个 1D TCN (WaveNet 风格)，处理 y_noisy [B, C_out, L_out]
        self.in_conv = nn.Conv1d(c_out, model_channels, kernel_size=3, padding=1)
        
        # 堆叠残差块
        self.blocks = nn.ModuleList([
            ResidualBlock1D(model_channels, model_channels, emb_channels, dilation=1),
            ResidualBlock1D(model_channels, model_channels, emb_channels, dilation=2),
            ResidualBlock1D(model_channels, model_channels, emb_channels, dilation=4),
        ])
        
        self.out_conv = nn.Conv1d(model_channels, c_out, kernel_size=3, padding=1)
        

    def forward(self, 
                y_noisy: torch.Tensor, 
                sigma: torch.Tensor, 
                x_past: torch.Tensor
               ) -> torch.Tensor:
        # y_noisy 形状: [B, C_out, L_out]
        # sigma 形状: [B]
        # x_past 形状: [B, C_in, L_in]

        # 1. 计算条件嵌入 `c` (来自 x_past)
        c = self.condition_encoder(x_past) # [B, Emb]

        # 2. 计算噪音嵌入 `emb` (来自 sigma)
        emb = self.map_noise(sigma) # [B, C_model]
        emb = self.time_embedding(emb) # [B, Emb]
        
        # 3. 合并嵌入
        total_emb = c + emb # [B, Emb]

        # 4. 运行去噪主干网络
        h = self.in_conv(y_noisy) # [B, C_model, L_out]
        
        for block in self.blocks:
            h = block(h, total_emb) # [B, C_model, L_out]
            
        y_pred_residual = self.out_conv(h) # [B, C_out, L_out]
        
        return y_pred_residual

# ---
# 阶段 2：1D 残差损失函数
# ---

class ResidualLoss1D(nn.Module):
    """
    1D 版本的 ResidualLoss。
    在内部持有冻结的 base_model。
    """
    def __init__(
        self,
        base_model: nn.Module,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5
    ):
        super().__init__()
        self.base_model = base_model # 冻结的基础模型
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def get_noise_params(self, y_residual: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        从 ResidualLoss (L575-L595) 中复制而来，用于采样 EDM 噪音。
        y_residual 是目标残差，形状 [B, C_out, L_out]
        """
        # (B, 1, 1) 以便广播到 (B, C_out, L_out)
        shape = (y_residual.shape[0], 1, 1) 
        
        # Sample noise level
        rnd_normal = torch.randn(shape, device=y_residual.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample noise
        n = torch.randn_like(y_residual) * sigma
        return n, sigma, weight

    def forward(
        self,
        net: nn.Module,
        y_true: torch.Tensor,
        x_past: torch.Tensor,
        # (可以添加 lead_time_label 等其他条件)
    ) -> torch.Tensor:
        # y_true 形状: [B, C_out, L_out]
        # x_past 形状: [B, C_in, L_in]
        # net 是 DiffusionNet1D 实例

        # 1. 获取基础预测 (在 no_grad 模式下，因为 base_model 已冻结)
        with torch.no_grad():
            y_base_pred = self.base_model.predict_autoregressive(x_past.permute(0,2,1)).permute(0,2,1) # [B, C_out, L_out]

        # 2. 计算残差 (这是扩散模型的目标)
        y_residual = y_true - y_base_pred # [B, C_out, L_out]

        # 3. 前向加噪
        n, sigma, weight = self.get_noise_params(y_residual)
        y_noisy_residual = y_residual + n

        # 4. 调用扩散模型 (net)
        # 我们需要 [B] 形状的 sigma 传递给 PositionalEmbedding
        sigma_time = sigma.squeeze().to(y_true.device)
        
        y_pred_residual = net(y_noisy_residual, sigma_time, x_past)

        # 5. 计算损失
        # 比较 预测的残差 和 真实的干净残差
        loss = weight * ((y_pred_residual - y_residual) ** 2)

        return loss

# ---
# 阶段 2：训练循环
# ---



if __name__ == "__main__":
    # 1. 初始化参数
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

    # 6. 构建模型


    # --- 2. 设置阶段 2 的训练 ---
    print("\n--- 阶段 2: 设置残差扩散模型训练 ---")
    
    # a. 加载并冻结基础模型
    print("加载并冻结 `base_model.pth`...")
    # TODO: 确保这里的 BaseModel 与您保存的 .pth 文件 *完全* 匹配
    base_model = build_model(cfg.basemodel_cfg)    
    # 7. 加载模型权重
    checkpoint_path = cfg.base_model_dir[0]
    base_model = load_checkpoint(checkpoint_path, base_model)
    logger.info("模型构建成功并已加载权重。")
    base_model.eval() # 设为评估模式
    for param in base_model.parameters(): # 冻结所有参数
        param.requires_grad = False
    
    # b. 实例化要训练的扩散模型 (net)
    print("实例化 1D 扩散模型 (net)...")
    net = build_model(cfg.model_cfg).to(device)

    if cfg.model_resume is not None:
        net = load_checkpoint(cfg.model_resume, net)

    logger.info(f'成功加载模型{cfg.model_resume}')



    # c. 实例化残差损失函数
    print("实例化 1D 残差损失函数 (loss_fn)...")
    loss_fn = ResidualLoss1D(base_model=base_model)

    # d. 设置优化器（只优化 net 的参数）
    optimizer, scheduler = build_optimizer(cfg, net)
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    logger.info("\n--- 阶段 2: 开始训练扩散模型 (net) ---")
    num_epochs = cfg.num_epochs # 训练轮数
    val_freq = cfg.val_freq # 验证频率

    best_val_loss = float('inf') # 用于保存最佳模型

    for epoch in range(num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")

        # --- 训练 ---
        net.train() # 设置为训练模式
        train_loss_accum = 0.0
        processed_batches = 0 # 用于跟踪进度

        pbar = tqdm(train_loader, desc=f"Training Epoch ", leave=False)

        for batch_idx, (x_past, y_true) in enumerate(pbar):
            # a. 将数据移动到设备
            x_past, y_true = x_past.to(device), y_true.to(device)

            if y_true.dim() != 3:
                y_true = y_true.unsqueeze(-1)

            if x_past.dim() == 3 and x_past.shape[1] == cfg.model_cfg.l_in and x_past.shape[2] == cfg.model_cfg.c_in:
                    x_past = x_past.permute(0, 2, 1)
            if y_true.dim() == 3 and y_true.shape[1] == cfg.model_cfg.l_out and y_true.shape[2] == cfg.model_cfg.c_out:
                    y_true = y_true.permute(0, 2, 1)

            # b. 清零梯度
            optimizer.zero_grad()

            # c. 计算损失
            loss = loss_fn(
                net=net,
                y_true=y_true,
                x_past=x_past
            )

            # d. 归约损失
            final_loss = loss.mean()

            # e. 反向传播
            final_loss.backward()

            # f. 更新权重
            optimizer.step()



            train_loss_accum += final_loss.item()
            processed_batches += 1
            pbar.set_postfix(loss=f"{final_loss.item():.6f}",lr=f"{optimizer.param_groups[0]['lr']:.7f}")


        avg_train_loss = train_loss_accum / len(train_loader)
        logger.info(f"Epoch {epoch+1} 完成, 平均训练损失: {avg_train_loss:.6f}")

        # --- 验证 ---
        if (epoch + 1) % val_freq == 0:
            net.eval() # 设置为评估模式
            val_loss_accum = 0.0
            logger.info("开始验证...")

            pbar = tqdm(val_loader, desc=f"Validing Epoch ", leave=False)

            with torch.no_grad(): # 在验证时不计算梯度
                for batch_idx, (x_past, y_true) in enumerate(pbar):
                    # a. 将数据移动到设备
                    x_past, y_true = x_past.to(device), y_true.to(device)

                    if y_true.dim() != 3:
                        y_true = y_true.unsqueeze(-1)

                    if x_past.dim() == 3 and x_past.shape[1] == cfg.model_cfg.l_in and x_past.shape[2] == cfg.model_cfg.c_in:
                            x_past = x_past.permute(0, 2, 1)
                    if y_true.dim() == 3 and y_true.shape[1] == cfg.model_cfg.l_out and y_true.shape[2] == cfg.model_cfg.c_out:
                            y_true = y_true.permute(0, 2, 1)


                    # b. 计算损失 (与训练时相同)
                    # 注意: loss_fn 内部的 base_model 已经是 eval 模式且 no_grad
                    val_loss = loss_fn(
                        net=net,
                        y_true=y_true,
                        x_past=x_past
                    )

                    # c. 归约损失
                    final_val_loss = val_loss.mean()
                    val_loss_accum += final_val_loss.item()

            avg_val_loss = val_loss_accum / len(val_loader)
            logger.info(f"Epoch {epoch+1} 验证完成, 平均验证损失: {avg_val_loss:.6f}")


            # --- (可选) 保存最佳模型 ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = "diffusion_net_1d_best.pth"
                torch.save(net.state_dict(), save_path)
                logger.info(f"*** 找到新的最佳验证损失，模型已保存到 {save_path} ***")
            cfg.logger.info(
                '[ Epoch %d ] ||| [lr: %.6f] [Loss: %.4f] [Correct: %.4f]|||  MaxMemory %dMB' %
                (epoch,
                optimizer.param_groups[0]['lr'],
                final_loss,
                avg_val_loss,
                torch.cuda.max_memory_allocated(device) / 1024 ** 2))

        if scheduler is not None:
            scheduler.step()


        
        cfg.logger.info(
            '[ Epoch %d ] ||| [lr: %.6f] [Loss: %.4f] |||  MaxMemory %dMB' %
            (epoch,
            optimizer.param_groups[0]['lr'],
            final_loss,
            torch.cuda.max_memory_allocated(device) / 1024 ** 2))

        
        if (epoch+1) % 5 == 0 and epoch>0: 
            save_file = os.path.join(cfg.model_save_dir, 'epoch_{}.pth'.format(epoch))
            save_checkpoint(save_file, net, epoch, optimizer=optimizer, scheduler=scheduler)
            cfg.logger.info('epoch_{}.pth'.format(epoch) + ' Saved')

    # --- 6. (可选) 保存最终模型 ---
    logger.info("\n--- 训练完成 ---")
    final_save_path = "diffusion_net_1d_final.pth"
    torch.save(net.state_dict(), final_save_path)
    logger.info(f"已保存最终训练好的 1D 扩散模型到 {final_save_path}。")


