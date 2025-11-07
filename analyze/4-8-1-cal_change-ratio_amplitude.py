import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import importlib

# 添加项目根目录到系统路径
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from typing import List, Tuple, Dict
import random
from datetime import datetime
import re

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def __init__():
    parser = argparse.ArgumentParser(description='calculate the wind speed change ratio and amplitude')

    parser.add_argument('--model_version', type=str, default='enhanced_s2s', help='configs of model')
    parser.add_argument('--project', type=str, default='14s', help='project name, 14s, 60min or ....')
    args = parser.parse_args()

    home_dir = os.path.expanduser("~")
    work_dir = os.path.join(home_dir,'model_output', '14s', args.model_version, 'analyze')

    args.input_file = os.path.join(work_dir, 'reorganized_predictions.csv')
    args.output_file = os.path.join(work_dir, 'change-ratio&amplitude.csv' )
    #os.makedirs(args.output_dir, exist_ok=True)
        

    return args

def calculate_change_ratio_and_amplitude(df: pd.DataFrame, ratio_interval: List[int]) -> pd.DataFrame:
    
# --- 1. 数据准备 ---
    df['time'] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index('time')
    df = df.drop(df.columns[0], axis=1)

    # --- 2. 自动检测所有可用的时间步 ---
    true_t_cols = [c for c in df.columns if c.startswith('true_t')]
    
    try:
        t_steps = sorted(list(set(
            int(re.findall(r'\d+', c)[0]) for c in true_t_cols
        )))
    except IndexError:
        print("错误: 无法从 'true_t' 列名中解析出时间步。")
        return df

    if not t_steps:
        print("错误: 未找到 'true_t' 列。")
        return df
        
    print(f"检测到时间步: {t_steps}")

    # --- 3. 向量化计算 ---
    print("开始向量化计算所有时间步的变率和幅度 (无法计算时填充 NaN)...")
    
    # 循环 1: 遍历你希望的间隔
    for interval in tqdm(ratio_interval, desc="Processing Intervals"):
        
        # 循环 2: 遍历所有可用的基础时间步 (例如 [1, 2, ..., 14])
        for base_t in t_steps:
            
            future_t = base_t + interval
            
            # --- 定义所有相关列名 ---
            base_true_col = f'true_t{base_t}'
            base_pred_col = f'pred_t{base_t}'
            future_true_col = f'true_t{future_t}'
            future_pred_col = f'pred_t{future_t}'

            # 新的输出列名 (无论如何都会创建)
            ratio_true_col_name = f'change_ratio_true_t{base_t}_{interval}s'
            ratio_pred_col_name = f'change_ratio_pred_t{base_t}_{interval}s'
            amp_true_col_name = f'change_amplitude_true_t{base_t}_{interval}s'
            amp_pred_col_name = f'change_amplitude_pred_t{base_t}_{interval}s'
            
            # --- 核心逻辑：检查所有必需的列是否存在 ---
            
            required_cols = [base_true_col, base_pred_col, future_true_col, future_pred_col]
            
            if all(c in df.columns for c in required_cols):
                # 存在: 执行计算
                based_speed_true = df[base_true_col]
                based_speed_pred = df[base_pred_col]
                future_speed_true = df[future_true_col]
                future_speed_pred = df[future_pred_col]

                df[ratio_true_col_name] = (future_speed_true - based_speed_true) / interval
                df[ratio_pred_col_name] = (future_speed_pred - based_speed_pred) / interval
                df[amp_true_col_name] = future_speed_true - based_speed_true
                df[amp_pred_col_name] = future_speed_pred - based_speed_pred
            
            else:
                # 不存在: 创建新列并填充 np.nan
                # 这会处理 t14 (base) + 1 (interval) -> t15 (不存在) 的情况
                df[ratio_true_col_name] = np.nan
                df[ratio_pred_col_name] = np.nan
                df[amp_true_col_name] = np.nan
                df[amp_pred_col_name] = np.nan

    return df
        
def main():
    args = __init__()

    df = pd.read_csv(args.input_file)

    ratio_interval = [3, 5, 7]

    result_df = calculate_change_ratio_and_amplitude(df, ratio_interval)

    result_df.to_csv(args.output_file, index=True)
    print(f"Saved change ratio and amplitude data to {args.output_file}")

if __name__ == "__main__":
    main()



