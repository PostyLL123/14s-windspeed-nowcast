'''
特征工程脚本

功能：对预处理后的数据进行特征工程
    1. 计算历史统计特征（使用滚动窗口，避免数据泄露）：
       - 10s、30s、60s、120s的平均风速
       - 10s、30s、60s、120s的风速标准差
    2. 添加时间相关特征：
       - 小时、分钟、星期几、月份
       - 7s和14s的周期性特征（sin和cos）
    3. 添加湍流强度特征：
       - 60s和300s的湍流强度
    4. 添加变化率特征：
       - 3s、5s、30s的风速变化率
    5. 生成特征可视化图表：
       - 风速时间序列图
       - 特征相关性热图

输入：1-2检查后的数据文件（支持.pkl和.csv格式）
输出：
    1. 特征工程后的数据文件（保存在16-add-feature/data目录）
    2. 特征可视化图表（保存在16-add-feature/plots目录）
    3. 20行示例数据（保存在16-add-feature/data目录）



'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import glob
import sys

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号

# 添加项目根目录到系统路径
project_root = '/home/luoew/stat_data/haomibo/scripts'
print(f"项目根目录: {project_root}")
sys.path.append(project_root)

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
from utils.add_features import add_time_features_new, add_rate_features, add_rolling_features, add_time_features, add_turbulence_features


# 设置输出目录
OUTPUT_BASE_DIR = '/home/luoew/stat_data/haomibo/16-add-feature'
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_BASE_DIR, 'plots')
OUTPUT_DATA_DIR = os.path.join(OUTPUT_BASE_DIR, 'data')

# 全局变量
input_file = None  # 当前处理的文件路径
df = None  # 当前处理的数据框

# 仅处理选择的列
col_select = ['WindSpeed']  # 只保留风速列
col_standard = ['WindSpeed']
col_dict = dict(zip(col_select, col_standard))

#=================================================================================================================
# 数据读取与预处理
#=================================================================================================================

def read_data(file_path):
    """
    读取并预处理数据文件
    
    参数:
        file_path: 数据文件路径
    返回:
        df: 预处理后的数据框
    """
    global input_file, df
    input_file = file_path
    
    print(f"\n{'='*50}")
    print(f"处理文件: {file_path}")
    print(f"{'='*50}")
    
    # 读取数据
    try:
        if file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"不支持的文件格式: {file_path}")
            return None
        
        # 确保数据包含所需的列
        if 'WindSpeed' not in df.columns:
            print(f"文件 {file_path} 中不包含 WindSpeed 列")
            return None
        
        # 选择需要的列并重命名
        df = df[col_select].dropna()
        df.index = df.index.round('1s')
        df.rename(columns=col_dict, inplace=True)
        
        # 检查时间间隔
        df.index = df.index.round('1s')
        # 获取开始和结束时间
        start_time = pd.to_datetime(df.index[0])
        end_time = pd.to_datetime(df.index[-1])
        # 创建完整的时间索引
        full_index = pd.date_range(start=start_time, end=end_time, freq='1S')
        # 重新索引数据
        df = df.reindex(full_index)

        df['TIMESTAMP'] = df.index
        
        # 输出数据基本信息
        print(f"\n数据基本信息:")
        print(f"数据点数量: {len(df)}")
        print(f"时间范围: {df['TIMESTAMP'].min()} 到 {df['TIMESTAMP'].max()}")
        print(f"缺失值统计:\n{df.isnull().sum()}")
        
        return df
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def plot_wind_data(df, output_path=None):
    """
    绘制风速时间序列图
    
    参数:
        df: 数据框
        output_path: 图片保存路径
    """
    plt.figure(figsize=(15, 6))

    # 风速时间序列图
    plt.plot(df['TIMESTAMP'], df['WindSpeed'])
    plt.title('风速时间序列')
    plt.xlabel('时间')
    plt.ylabel('风速 (m/s)')

    plt.tight_layout()
    
    if output_path:
        print(f'图片保存路径: {output_path}')
        plt.savefig(output_path)
    else:
        # 从输入文件名中提取信息来命名输出图片
        file_basename = os.path.basename(input_file).split('.')[0]  # 获取不含扩展名的文件名
        output_path = os.path.join(OUTPUT_PLOTS_DIR, f'wind_{file_basename}.png')
        print(f'图片保存路径: {output_path}')
        plt.savefig(output_path)
    
    plt.close()

#=================================================================================================================
# 特征工程
#=================================================================================================================

def add_features(df):
    """
    对数据进行特征工程
    
    参数:
        df: 原始数据框
    返回:
        df_basic: 添加特征后的数据框
    """
    
    # 应用特征工程
    df_features = df.copy()
    
    # 添加滚动窗口特征
    df_features = add_rolling_features(df_features, window_sizes=[10,30,60,120])
    
    # 添加时间特征
    df_features = add_time_features(df_features, periods=[7,14])
    
    # 添加湍流强度特征
    df_features = add_turbulence_features(df_features, window_sizes=[60])
    
    # 添加新的时间特征
    df_features = add_time_features_new(df_features)
    
    # 添加变化率特征
    df_features = add_rate_features(df_features)

    # 检查新生成的特征
    print("\n=== 特征工程后的数据概览 ===")
    print(f"特征数量: {len(df_features.columns)}")
    print("\n特征列表:")
    print(df_features.columns.tolist())
    print("\n数据预览:")
    print(df_features.round(4))

    # 选择特征子集（目前不选择，保留所有特征）
    df_basic = df_features.dropna()

    print("\n=== 基础特征集概览 ===")
    print(f"特征数量: {len(df_basic.columns)}")
    print("\n特征列表:")
    print(df_basic.columns.tolist())
    print("\n数据预览:")
    print(df_basic.round(4))

    return df_basic

#=================================================================================================================
# 数据保存
#=================================================================================================================

def save_features(df_basic):
    """
    保存特征工程后的数据
    
    参数:
        df_basic: 特征工程后的数据框
    """
    # 从input_file路径中提取子目录与文件名
    file_basename = os.path.basename(input_file).split('.')[0]  # 获取不含扩展名的文件名

    # 确保输出目录存在

    # 保存文件，保持相同的命名规则
    pkl_path = os.path.join(OUTPUT_DATA_DIR, f'{file_basename}-add-feature.pkl')
    df_basic.to_pickle(pkl_path)
    print(f"\n数据已保存至: {pkl_path}")
    
    # 保存示例数据
    example_path = os.path.join(OUTPUT_DATA_DIR, f'{file_basename}-add-feature.20line-example.csv')
    df_basic.head(20).round(4).to_csv(example_path)
    print(f"示例数据已保存至: {example_path}")

#=================================================================================================================
# 主函数
#=================================================================================================================

def process_file(file_path):
    """
    处理单个数据文件
    
    参数:
        file_path: 数据文件路径
    """
    # 1. 读取数据
    df = read_data(file_path)
    if df is None:
        return
    
    # 2. 绘制风速时间序列图
    file_basename = os.path.basename(file_path).split('.')[0]
    plot_output_path = os.path.join(OUTPUT_PLOTS_DIR, f'wind_{file_basename}.png')
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    plot_wind_data(df, output_path=plot_output_path)
    
    # 3. 特征工程
    df_basic = add_features(df)
    
    # 4. 保存数据
    save_features(df_basic)

def process_directory(directory_path):
    """
    处理指定目录下的所有数据文件
    
    参数:
        directory_path: 数据文件目录路径
    """
    print(f"开始处理目录: {directory_path}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    # 查找所有pkl和csv文件
    pkl_files = glob.glob(os.path.join(directory_path, "**", "*.pkl"), recursive=True)
    csv_files = glob.glob(os.path.join(directory_path, "**", "*.csv"), recursive=True)
    
    all_files = pkl_files + csv_files
    
    if not all_files:
        print(f"在目录 {directory_path} 中未找到任何pkl或csv文件")
        return
    
    print(f"找到 {len(all_files)} 个文件:")
    for file_path in all_files:
        print(f"  - {file_path}")
    
    # 处理每个文件
    for file_path in all_files:
        process_file(file_path)

if __name__ == "__main__":
    # 指定要处理的目录
    input_dir = "/home/luoew/stat_data/haomibo/16-processed"
    
    # 处理目录中的所有文件
    process_directory(input_dir)
