# -*- coding: utf-8 -*-
"""
DDPM (Diffusion Model) 预测阶段的配置文件
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import pickle

# --- 路径设置 ---
# 添加项目根目录，以便 utils 等模块能被导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.logger import create_logger

# --- 基础设置 ---
# 基础工作目录 (所有输出将保存在这里)
# TODO: (请确认) 这是你希望的输出路径
work_dir = '/home/luoew/model_output/'
project = '14s' # (新) 项目名称，区分于基础模型
model_name_str = 'diffusion_enhanced_s2s' # (新) 用于日志和文件名

# (确保 work_dir 和项目目录存在)
project_path = os.path.join(work_dir, project, model_name_str)
if not os.path.exists(project_path):
    os.makedirs(project_path, exist_ok=True)


local_rank = -1 # 假设单 GPU 或 CPU

# --- 数据路径 ---
data_config = {
    'base_dir': '/home/luoew/stat_data/haomibo/16-sampled-data',
    # (重要!) 确保 test_folders 指向你想要预测的数据
    'test_folders':[
        'DATA-16-20240511-0514-add-feature'
    ],
    'x_suffix': '_x.npy',
    'y_suffix': '_y.npy',
    'time_info_suffix': '_time_info.pkl',
    # (!!) 确认这个键名是否正确: 'output_start_times'
    'time_info_key': 'output_start_times'
}

# --- 标准化器路径 ---
# (重要!) 确保这些路径指向你训练 base model 时使用的标准化器
normalizer = {
    'x': "/home/luoew/model_output/14s/enhanced_s2s/normalizer/normalizer_x_standard.pkl",
    'y': "/home/luoew/model_output/14s/enhanced_s2s/normalizer/normalizer_y_standard.pkl"
}

# --- 模型路径 ---
# (重要!) 基础模型路径 (来自 base model config 的 eval_model)
base_model_path = "/home/luoew/model_output/14s/enhanced_s2s/model/best_model.pth"
# (重要!) 扩散模型路径 (指向你训练好的 DDPM 模型)
diffusion_model_path = "/home/luoew/diffusion_net_1d_best.pth" # <--- 修改为你的实际路径

# --- 基础模型配置 (BaseModelCFG) ---
# (直接从 base model config 复制，确保与加载的模型匹配)
class BaseModelCFG:
    def __init__(self):
        self.n_features = 14
        self.n_labels = 1
        self.input_len = 120
        self.output_len = 14
        self.n_hiddensize = 128
        self.num_encoder_layers = 3
        self.num_decoder_layers = 2
        self.num_attention_heads = 4
        self.attention_dropout = 0.1
        self.lstm_dropout = 0.2
        self.dropout = 0.0
        self.use_layer_norm = True
        self.use_residual = True
        self.use_positional_encoding = True
        self.model_name = 'enhanced_s2s' # 用于 build_model
        # --- (!!) 添加必要的 C_IN, C_OUT ---
        self.C_IN = self.n_features # Conv1d 需要
        self.C_OUT = self.n_labels   # Conv1d 需要
        self.L_IN = self.input_len   # 用于构建 Base Model 占位符
        self.L_OUT = self.output_len # 用于构建 Base Model 占位符


basemodel_cfg = BaseModelCFG()

# --- (新) 扩散模型配置 (DiffusionModelCFG) ---
class DiffusionModelCFG:
    def __init__(self):
        # 1. 维度 (必须与 BaseModelCFG 匹配!)
        self.C_IN = basemodel_cfg.C_IN
        self.L_IN = basemodel_cfg.L_IN
        self.C_OUT = basemodel_cfg.C_OUT
        self.L_OUT = basemodel_cfg.L_OUT

        # 2. DiffusionNet1D 内部参数 (基于训练脚本的常量)
        self.MODEL_CHANNELS = 64
        self.EMB_CHANNELS = self.MODEL_CHANNELS * 4
        self.model_name = [model_name_str] # 用于 build_model (如果需要)

diffusionmodel_cfg = DiffusionModelCFG()

# --- 预测/采样参数 ---
# 使用与训练脚本中 predict_wind_speed 匹配的参数
batch_size = 512 # 预测时可以使用更大的 batch size
num_workers = 64 # 根据你的系统调整
num_steps = 50 # 扩散采样步数
sigma_min = 0.002
sigma_max = 80.0
rho = 7.0

# --- 其他设置 ---
rand_seed = 2024
logger = create_logger(project_path, 'predict_ddpm')

logger.info(f'[Work Dir]: {work_dir}')
logger.info(f'[Project Dir]: {project_path}')
logger.info(f'[Base Model Path]: {base_model_path}')
logger.info(f'[Diffusion Model Path]: {diffusion_model_path}')
