import numpy as np
import pandas as pd
from typing import List



# 1. 计算历史统计特征（使用滚动窗口，避免数据泄露）
def add_rolling_features(df, window_sizes=[10, 30, 60]):
    """
    添加滚动窗口特征，只使用历史数据
    window_sizes: 窗口大小（秒）
    """
    features = df.copy()
    
    # 重命名基础列
    features = features.rename(columns={
        'WindSpeed': 'wspd',
        # 'WindDir': 'wdir'
    })
    
    # 添加基础的风向三角函数特征
    # features['wdir_sin'] = np.sin(np.deg2rad(features['wdir']))
    # features['wdir_cos'] = np.cos(np.deg2rad(features['wdir']))
    
    for window in window_sizes:
        # 风速特征
        features[f'wspd_mean_{window}s'] = features['wspd'].rolling(
            window=window, min_periods=1, center=False).mean()
        features[f'wspd_std_{window}s'] = features['wspd'].rolling(
            window=window, min_periods=1, center=False).std()
        # features[f'wspd_max_{window}s'] = features['wspd'].rolling(
        #     window=window, min_periods=1, center=False).max()
        # features[f'wspd_min_{window}s'] = features['wspd'].rolling(
        #     window=window, min_periods=1, center=False).min()
        
        # # 风向特征（考虑循环性）
        # features[f'wdir_sin_mean_{window}s'] = features['wdir_sin'].rolling(
        #     window=window, min_periods=1, center=False).mean()
        # features[f'wdir_cos_mean_{window}s'] = features['wdir_cos'].rolling(
        #     window=window, min_periods=1, center=False).mean()
    
    return features

# 2. 添加时间特征
def create_time_features(timestamps: pd.DatetimeIndex, periods: List[int]) -> pd.DataFrame:
    """
    创建指定周期的时间特征
    
    参数:
        timestamps: 时间戳索引
        periods: 周期列表（秒）
        
    返回:
        包含时间特征的DataFrame
    """
    # 获取秒级时间信息
    second_in_minute = timestamps.second
    
    # 创建特征字典
    features = {}
    
    # 为每个周期创建正弦和余弦特征
    for period in periods:
        # 计算角频率（2π/周期）
        angular_freq = 2 * np.pi / period
        
        # 创建特征
        features.update({
            f'sec_sin_{period}s': np.sin(angular_freq * second_in_minute),
            f'sec_cos_{period}s': np.cos(angular_freq * second_in_minute)
        })
    
    return pd.DataFrame(features, index=timestamps)

def add_time_features(df: pd.DataFrame, periods: List[int], drop_existing: bool = False) -> pd.DataFrame:
    """
    为数据集添加或替换时间特征
    
    参数:
        df: 输入DataFrame（需要有DatetimeIndex）
        periods: 要添加的时间特征周期列表（秒）
        drop_existing: 是否删除现有的时间特征
        
    返回:
        添加了时间特征的DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入DataFrame必须有DatetimeIndex类型的索引")
    
    # 创建新的时间特征
    time_features = create_time_features(df.index, periods)
    
    # 如果需要删除现有的时间特征
    # if drop_existing:
    #     # 删除现有的sin/cos特征
    #     existing_cols = [col for col in df.columns 
    #                     if any(x in col for x in ['sin', 'cos'])]
    #     df = df.drop(columns=existing_cols)
    
    # 合并新特征
    df = pd.concat([df, time_features], axis=1)
    
    return df

def add_time_features_new(df):
    """添加时间特征"""
    features = df.copy()
    
    # 小时特征（保留）
    features['hour'] = features.index.hour
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    
    # 分钟特征（新增）
    features['minute'] = features.index.minute
    features['minute_sin'] = np.sin(2 * np.pi * features['minute'] / 60)
    features['minute_cos'] = np.cos(2 * np.pi * features['minute'] / 60)
    
    return features

# 3. 添加湍流强度特征
def add_turbulence_features(df, window_sizes=[60, 300]):
    """
    计算湍流强度特征
    """
    features = df.copy()
    
    for window in window_sizes:
        wind_std = features['wspd'].rolling(
            window=window, min_periods=1, center=False).std()
        wind_mean = features['wspd'].rolling(
            window=window, min_periods=1, center=False).mean()
        features[f'turb_{window}s'] = wind_std / wind_mean
    
    return features


def add_rate_features(df):
    """添加风速变化率和加速度特征"""
    features = df.copy()

    # 短期变化率（1秒、3秒、5秒）
    # features['wspd_change_1s'] = features['wspd'].diff(1)
    features['wspd_change_3s'] = features['wspd'].diff(3)
    features['wspd_change_5s'] = features['wspd'].diff(5)
    
    # 中期变化率（10秒、30秒）
    # features['wspd_change_10s'] = features['wspd'].diff(10)
    features['wspd_change_30s'] = features['wspd'].diff(30)
    
    # 加速度（变化率的变化率）
    # features['wspd_accel_1s'] = features['wspd_change_1s'].diff(1)
    features['wspd_accel_3s'] = features['wspd_change_3s'].diff(3)
    
    # 变化率与当前风速的比值（归一化变化率）
    # features['wspd_change_ratio_3s'] = features['wspd_change_1s'] / (features['wspd'] + 1e-6)
    features['wspd_change_ratio_5s'] = features['wspd_change_5s'] / (features['wspd'] + 1e-6)
    
    return features

# 1. 计算历史统计特征（使用滚动窗口，避免数据泄露）
def add_rolling_features(df, window_sizes=[10, 30, 60]):
    """
    添加滚动窗口特征，只使用历史数据
    window_sizes: 窗口大小（秒）
    """
    features = df.copy()
    
    # 重命名基础列
    features = features.rename(columns={
        'WindSpeed': 'wspd',
        #   'WindDir': 'wdir'
    })
    
    # 添加基础的风向三角函数特征
    # features['wdir_sin'] = np.sin(np.deg2rad(features['wdir']))
    # features['wdir_cos'] = np.cos(np.deg2rad(features['wdir']))
    
    for window in window_sizes:
        # 风速特征
        features[f'wspd_mean_{window}s'] = features['wspd'].rolling(
            window=window, min_periods=1, center=False).mean()
        features[f'wspd_std_{window}s'] = features['wspd'].rolling(
            window=window, min_periods=1, center=False).std()
        # features[f'wspd_max_{window}s'] = features['wspd'].rolling(
        #     window=window, min_periods=1, center=False).max()
        # features[f'wspd_min_{window}s'] = features['wspd'].rolling(
        #     window=window, min_periods=1, center=False).min()
        
        # # 风向特征（考虑循环性）
        # features[f'wdir_sin_mean_{window}s'] = features['wdir_sin'].rolling(
        #     window=window, min_periods=1, center=False).mean()
        # features[f'wdir_cos_mean_{window}s'] = features['wdir_cos'].rolling(
        #     window=window, min_periods=1, center=False).mean()
    
    return features

# 2. 添加时间特征
def create_time_features(timestamps: pd.DatetimeIndex, periods: List[int]) -> pd.DataFrame:
    """
    创建指定周期的时间特征
    
    参数:
        timestamps: 时间戳索引
        periods: 周期列表（秒）
        
    返回:
        包含时间特征的DataFrame
    """
    # 获取秒级时间信息
    second_in_minute = timestamps.second
    
    # 创建特征字典
    features = {}
    
    # 为每个周期创建正弦和余弦特征
    for period in periods:
        # 计算角频率（2π/周期）
        angular_freq = 2 * np.pi / period
        
        # 创建特征
        features.update({
            f'sec_sin_{period}s': np.sin(angular_freq * second_in_minute),
            f'sec_cos_{period}s': np.cos(angular_freq * second_in_minute)
        })
    
    return pd.DataFrame(features, index=timestamps)

def add_time_features(df: pd.DataFrame, periods: List[int], drop_existing: bool = True) -> pd.DataFrame:
    """
    为数据集添加或替换时间特征
    
    参数:
        df: 输入DataFrame（需要有DatetimeIndex）
        periods: 要添加的时间特征周期列表（秒）
        drop_existing: 是否删除现有的时间特征
        
    返回:
        添加了时间特征的DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入DataFrame必须有DatetimeIndex类型的索引")
    
    # 创建新的时间特征
    time_features = create_time_features(df.index, periods)
    
    # # 如果需要删除现有的时间特征
    # if drop_existing:
    #     # 删除现有的sin/cos特征
    #     existing_cols = [col for col in df.columns 
    #                     if any(x in col for x in ['sin', 'cos'])]
    #     df = df.drop(columns=existing_cols)
    
    # 合并新特征
    df = pd.concat([df, time_features], axis=1)
    
    return df

# 3. 添加湍流强度特征
def add_turbulence_features(df, window_sizes=[60, 300]):
    """
    计算湍流强度特征
    """
    features = df.copy()
    
    for window in window_sizes:
        wind_std = features['wspd'].rolling(
            window=window, min_periods=1, center=False).std()
        wind_mean = features['wspd'].rolling(
            window=window, min_periods=1, center=False).mean()
        features[f'turb_{window}s'] = wind_std / wind_mean
    
    return features
