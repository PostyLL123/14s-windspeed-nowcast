#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
风速数据处理脚本

本脚本用于处理风速观测数据，主要功能包括：
1. 读取原始CSV格式的风速数据文件
2. 对数据进行预处理和质控：
   - 将时间列转换为datetime格式
   - 计算逐秒平均风速
   - 对异常值进行处理（负值设为0，超过50的值设为NaN）
3. 将处理后的数据按目录合并，并保存为pickle格式
4. 自动记录处理日志，包括成功和失败的处理记录

输入：原始CSV格式的风速数据文件
输出：处理后的pickle格式数据文件
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime

# 设置日志
log_dir = "/home/luoew/stat_data/haomibo/scripts/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"wind_data_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%/home/luoew/project/nowcasting/data_preprocess(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 文件处理器
        logging.StreamHandler()  # 控制台处理器
    ]
)
logger = logging.getLogger(__name__)

def process_csv_file(file_path):
    """
    处理单个CSV文件，返回处理后的DataFrame
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, skiprows=1)
        
        # 选择需要的列并重命名
        df = df[['time', '.st_WindSpdOut.Glv.WindSpd.r3sAvg']]
        df.columns = ['time', 'WindSpeed']
        
        # 将time列转换为datetime并设置为索引
        df['time'] = pd.to_datetime(df['time'])  # 直接解析datetime字符串
        df.set_index('time', inplace=True)
        
        # 计算逐秒平均
        df = df.resample('1S').mean()
        
        # 数据质控
        # 风速质控：低于0的设为0，高于50的设为NaN
        df.loc[df['WindSpeed'] < 0, 'WindSpeed'] = 0
        df.loc[df['WindSpeed'] > 50, 'WindSpeed'] = np.nan
        
        logger.info(f"成功处理文件: {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def process_subdirectory(subdir_path, output_dir):
    """
    处理单个子目录下的所有CSV文件，并将结果合并为一个DataFrame
    """
    subdir_path = Path(subdir_path)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已经处理过
    output_file = output_dir / f"{subdir_path.name}.pkl"
    if output_file.exists():
        logger.info(f"目录 {subdir_path.name} 已经处理过，跳过")
        return True
    
    # 存储所有处理后的DataFrame
    dfs = []
    
    # 处理目录下的所有CSV文件
    for file_path in subdir_path.rglob("*.csv"):
        if not file_path.name.startswith("processed_"):  # 跳过已处理的文件
            df = process_csv_file(file_path)
            if df is not None:
                dfs.append(df)
    
    if not dfs:
        logger.warning(f"目录 {subdir_path.name} 中没有找到有效的CSV文件")
        return False
    
    # 合并所有DataFrame
    combined_df = pd.concat(dfs)
    # 按时间排序
    combined_df.sort_index(inplace=True)
    # 删除重复的索引
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    # 保存为pickle文件
    combined_df.to_pickle(output_file)
    logger.info(f"成功保存目录 {subdir_path.name} 的处理结果到 {output_file}")
    
    return True

def process_all_directories(base_dir, output_dir):
    """
    处理基础目录下的所有一级子目录
    """
    base_dir = Path(base_dir)
    processed_count = 0
    error_count = 0
    
    # 获取所有一级子目录
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        if process_subdirectory(subdir, output_dir):
            processed_count += 1
        else:
            error_count += 1
    
    return processed_count, error_count

if __name__ == "__main__":
    base_dir = "/home/luoew/stat_data/haomibo/16-unzip"
    output_dir = "/home/luoew/stat_data/haomibo/16-processed"
    
    logger.info("开始处理数据...")
    processed_count, error_count = process_all_directories(base_dir, output_dir)
    
    logger.info(f"处理完成！成功处理 {processed_count} 个目录，失败 {error_count} 个目录。") 