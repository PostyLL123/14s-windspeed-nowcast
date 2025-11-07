import numpy as np
import pandas as pd
import os
import copy
import sys

# --- 路径设置 ---
# 添加项目根目录，以便 utils 等模块能被导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(project_root)

from utils.logger import create_logger

# 基础工作目录 (所有输出将保存在这里)
# TODO: (请确认) 这是你希望的输出路径
work_dir = '/home/luoew/model_output/'
# (确保 work_dir 存在)
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)
project = '14s'


local_rank = 0
# --- 数据路径 ---
# (已修改) 现在使用 data_config 指定文件夹
"""
,
        "DATA-16-20240601-0630-add-feature" 

,
        "DATA-16-20240523-0527-add-feature"
"""
data_config = {
    # (已修改) 使用你提供的 base_dir
    'base_dir': '/home/luoew/stat_data/haomibo/16-sampled-data',
    
    # TODO: (重要!) 训练和验证文件夹列表 (请使用你图片中的文件夹名)
    'train_folders': [
        "DATA-16-20240516-0522-add-feature",
        "DATA-16-20240601-0630-add-feature" 
        
        
    ],
    'val_folders': [
        "DATA-16-20240514-0516-add-feature",
        "DATA-16-20240523-0527-add-feature"
        
        
    ],
    
    # (已修改) 指定文件名后缀
    'x_suffix': '_x.npy',
    'y_suffix': '_y.npy'
}




# --- 训练设置 ---
start_epoch = 0
num_epochs = 100
batch_size = 512*10
num_workers = 32
save_interval = 10 # 每10个epoch保存一次
display_interval = 20 # (在 train.py 中未使用, 但保留)
rand_seed = 2024
val_freq = 5


# --- 优化器设置 ---
opt_type = 'Adam'
learning_rate = 1e-5
weight_decay = 1e-3
scheduler = 'CosineLR' # 'StepLR' 或 'CosineLR'
step_size = [50, 80] # 如果使用 StepLR

# --- 模型配置 (ModelCFG) ---
# (注意: 我已重写此类，使其参数与 EnhancedSeq2Seq 匹配)
class base_ModelCFG:
    def __init__(self):
        # 1. 关键维度 (必须与 .npy 文件匹配)
        # TODO: (重要!) 请根据你的数据修改这些值
        self.n_features = 14         # 编码器输入特征数 (例如: 风速, 温度等)
        self.n_labels = 1           # 解码器输出/输入特征数 (例如: 仅风速)
        self.input_len = 120        # 编码器输入序列长度
        self.output_len = 14         # 解码器输出序列长度

        # 2. 模型架构参数
        self.n_hiddensize = 128     # 隐藏层维度
        self.num_encoder_layers = 3 # 编码器LSTM层数
        self.num_decoder_layers = 2 # 解码器LSTM层数
        self.num_attention_heads = 4 # 多头注意力头数 (必须能整除 n_hiddensize)
        
        # 3. Dropout 和正则化
        self.attention_dropout = 0.1
        self.lstm_dropout = 0.2
        self.dropout = 0.0 # (config中的原始dropout, 也许在build_model中使用?)

        # 4. 架构开关
        self.use_layer_norm = True
        self.use_residual = True
        self.use_positional_encoding = True
        
        # 5. 模型名称 (用于 build_model)
        self.model_name = 'enhanced_s2s'
        
        # (保留的旧参数，以防 build_model 需要它们)
        self.batch_size = batch_size
        self.in_channels = self.input_len * self.n_features # (不推荐使用, 但保留)
        self.out_channels = self.n_labels # (重命名)

basemodel_cfg = base_ModelCFG()

class ModelCFG:
    def __init__(self):
        self.model_name = 'diffusion_enhanced_s2s'
        # --- 常量定义 (Constants) ---
        # TODO: 请根据您的数据修改这些值
        self.l_in = 120           # 过去 120 步
        self.l_out = 14           # 未来 14 步
        self.c_in = 14             # 假设输入有 3 个特征 (例如 风速u, 风速v, 温度)
        self.c_out = 1            # 假设我们只预测 1 个特征 (例如 未来风速u)
        self.batch_size = 16

        # 扩散模型超参数
        self.model_channels = 64  # U-Net 内部的基础通道数
        self.emb_channels = self.model_channels * 4 # 嵌入向量的维度

model_cfg = ModelCFG()
base_model_dir = ["/home/luoew/model_output/14s/enhanced_s2s/model/best_model.pth"]
# --- 日志记录器 Logger ---
eval_model = None # (保持你原来的逻辑)

if eval_model is not None:
    # (这似乎是一个错字, 'result_path' 未定义, 改为 'work_dir')
    logger = create_logger(work_dir+project+f'/{model_cfg.model_name}', 'test_stage_1') 
else:
    logger = create_logger(work_dir+project+f'/{model_cfg.model_name}', 'Train_stage_1')


#创建模型保存路径
model_output_dir = os.path.join(work_dir, project, model_cfg.model_name)
model_save_dir = os.path.join(work_dir+project+f'/{model_cfg.model_name}', 'model')
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)


normalize_config = {
    'method':'standard',
    'feature_range_min': -1,
    'feature_range_max': 1,
    'normalizer_save_dir' : os.path.join(work_dir+project+f'/{model_cfg.model_name}', 'normalizer')

}


normalizer = {
    'x': "/home/luoew/model_output/14s/enhanced_s2s/normalizer/normalizer_x_standard.pkl",
    'y': "/home/luoew/model_output/14s/enhanced_s2s/normalizer/normalizer_y_standard.pkl"
}


model_resume = '/home/luoew/model_output/14s/diffusion_enhanced_s2s/model/best_model.pth'