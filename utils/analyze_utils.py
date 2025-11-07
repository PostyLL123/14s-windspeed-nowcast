test_import = False #用于测试导入是否成功

import os
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

# 添加项目根目录到系统路径
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

if test_import:
    print('导入测试成功！')
    os._exit(0)

def extract_and_group_predictions(df: pd.DataFrame, 
                                pred_interval: int = 5, 
                                time_interval: str = '1S') -> List[List[pd.Series]]:
    """
    提取并分组预测序列
    
    参数:
        df: 包含预测数据的DataFrame
        pred_interval: 预测间隔(秒)
        time_interval: 时间序列间隔,默认为1秒
        
    返回:
        分组后的预测序列列表,每组包含多个预测序列
    """
    # 获取所有预测列
    pred_cols = [col for col in df.columns if col.startswith('pred_')]
    n_steps = len(pred_cols)
    print(f"检测到{n_steps}个预测时间步")
    
    # 计算分组数量
    n_groups = int(np.ceil((n_steps + 1.1) / pred_interval))
    print(f"预测间隔: {pred_interval}秒")
    print(f"分组数量: {n_groups}")
    
    # 创建分组列表,每组存储多个预测序列
    grouped_predictions = [[] for _ in range(n_groups)]
    
    # 遍历每一行,生成预测序列
    for row_idx, (idx, row) in enumerate(df.iterrows()):
        # 确定该行属于哪个组
        group_idx = row_idx % n_groups
        
        # 生成该行的时间序列
        times = pd.date_range(start=idx, periods=n_steps, freq=time_interval)
        
        # 创建该行的预测序列
        pred_values = [round(row[col], 3) for col in pred_cols]
        pred_series = pd.Series(pred_values, index=times)
        
        # 将序列添加到对应的组
        grouped_predictions[group_idx].append(pred_series)
    
    # 打印分组信息
    for i, group in enumerate(grouped_predictions):
        print(f"第{i+1}组: {len(group)}个预测序列")
        if len(group) > 0:
            print(f"样例预测序列时间范围: {group[0].index[0]} 到 {group[0].index[-1]}")
    
    return grouped_predictions

def merge_prediction_groups(grouped_predictions: List[List[pd.Series]]) -> List[pd.Series]:
    """
    合并每组内的预测序列
    
    参数:
        grouped_predictions: 分组后的预测序列列表
        
    返回:
        合并后的预测序列列表
    """
    merged_predictions = []
    
    for group_idx, group in enumerate(grouped_predictions):
        # 直接concat所有序列
        merged = pd.concat(group)
        
        # 创建完整的时间索引(1秒间隔)
        full_index = pd.date_range(
            start=merged.index.min(),
            end=merged.index.max(),
            freq='1S'
        )
        
        # 重新索引,保留空值
        merged_series = merged.reindex(full_index)
        
        merged_predictions.append(merged_series)
        print(f"第{group_idx+1}组合并后的序列长度: {len(merged_series)}")
        print(f"时间范围: {merged_series.index[0]} 到 {merged_series.index[-1]}")
    
    return merged_predictions
