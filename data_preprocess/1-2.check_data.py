'''
数据检查脚本

功能：对预处理后的数据进行质量检查和可视化分析
    1. 检查数据时间间隔，确保数据为1秒间隔
    2. 计算基本统计信息（均值、标准差、分位数等）
    3. 检查异常值：
       - 风速小于0的值设为0
       - 风速大于100的值设为NaN
    4. 统计缺失值情况
    5. 生成风速时间序列图
    6. 输出数据质量检查摘要，包括：
       - 总记录数
       - 时间范围
       - 缺失值统计

输入：1-1处理后的数据文件（支持.pkl和.csv格式）
输出：
    1. 数据质量检查报告（控制台输出）
    2. 风速时间序列图（保存在16-add-feature/plots目录）
    3. 处理后的数据文件（保存在16-add-feature目录）
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import glob
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号


# 添加项目根目录到系统路径
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(project_root)
sys.path.append(project_root)


# 仅处理选择的列
col_select = ['WindSpeed']  # 只保留风速列
col_standard = ['WindSpeed']
col_dict = dict(zip(col_select, col_standard))



def plot_wind_data(df, wspd_col='WindSpeed', output_path=None):
    plt.figure(figsize=(15, 6))

    # 风速时间序列图
    plt.plot(df['TIMESTAMP'], df['WindSpeed'])
    plt.title('风速时间序列')
    plt.xlabel('时间')
    plt.ylabel('风速 (m/s)')

    plt.tight_layout()
    os.makedirs('data/plots', exist_ok=True)
    
    if output_path:
        print(f'图片保存路径: {output_path}')
        plt.savefig(output_path)
    else:
        # 从输入文件名中提取信息来命名输出图片
        file_basename = os.path.basename(input_file).split('.')[0]  # 获取不含扩展名的文件名
        output_path = f'/home/luoew/stat_data/plots/wind_{file_basename}.png'
        print(f'图片保存路径: {output_path}')
        plt.savefig(output_path)
    
    plt.close()

def process_file(file_path):
    """
    处理单个数据文件
    
    参数:
        file_path: 数据文件路径
    """
    global input_file
    input_file = file_path
    
    print(f"\n{'='*50}")
    print(f"处理文件: {file_path}")
    
    # 读取数据
    try:
        if file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"不支持的文件格式: {file_path}")
            return
        
        # 确保数据包含所需的列
        if 'WindSpeed' not in df.columns:
            print(f"文件 {file_path} 中不包含 WindSpeed 列")
            return
        
        # 选择需要的列并重命名
        df = df[col_select].dropna()
        df.index = df.index.round('1s')
        df.rename(columns=col_dict, inplace=True)
        
        # 1. 检查时间间隔
        df.index = df.index.round('1s')
        # 获取开始和结束时间
        start_time = pd.to_datetime(df.index[0])
        end_time = pd.to_datetime(df.index[-1])
        # 创建完整的时间索引
        full_index = pd.date_range(start=start_time, end=end_time, freq='1S')
        # 重新索引数据
        df = df.reindex(full_index)

        df['TIMESTAMP'] = df.index

        # 2. 基本统计信息
        print("\n=== 数据基本统计 ===")
        print(df.describe())

        # 3. 检查异常值
        print("\n=== 异常值检查 ===")
        # 风速通常不会超过100m/s
        print("\n超出范围的风速值 (>100 m/s):")
        print(df[(df['WindSpeed'] > 100) | (df['WindSpeed'] < 0)][['TIMESTAMP', 'WindSpeed']])

        # 异常值处理
        # 统计处理前的异常值数量
        wind_speed_abnormal_count = len(df[(df['WindSpeed'] > 100) | (df['WindSpeed'] < 0)])
        print(f"\n处理前异常值数量 - 风速: {wind_speed_abnormal_count}")

        # 处理异常值
        # 将低于0的风速设为0
        df.loc[df['WindSpeed'] < 0, 'WindSpeed'] = 0
        # 将大于100的风速设为NaN
        df.loc[df['WindSpeed'] > 100, 'WindSpeed'] = np.nan

        # 统计处理后的异常值和NaN数量
        print("\n处理后的数据统计:")
        print(f"NaN值数量 - 风速: {df['WindSpeed'].isna().sum()}")
        print(f"风速<0的数量: {len(df[df['WindSpeed'] < 0])}")
        print(f"风速>100的数量: {len(df[df['WindSpeed'] > 100])}")

        print("\n缺失值统计:")
        print(df.isna().sum())

        # 创建输出目录
        output_dir = os.path.join('/home/luoew/stat_data/haomibo/', '16-processed', 'plots')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        file_basename = os.path.basename(file_path).split('.')[0]
        plot_output_path = os.path.join(output_dir, f'wind_{file_basename}.png')
        
        plot_wind_data(df, wspd_col='WindSpeed', output_path=plot_output_path)

        # 输出检查结果摘要
        print("\n=== 数据质量检查摘要 ===")
        print(f"总记录数: {len(df)}")
        print(f"时间范围: {df['TIMESTAMP'].min()} 到 {df['TIMESTAMP'].max()}")
        print(f"缺失值统计:\n{df.isnull().sum()}")
        
        # 保存处理后的数据
        output_data_dir = os.path.join('/home/luoew/stat_data/haomibo/','16-processed','data')
        os.makedirs(output_data_dir, exist_ok=True)
        
        output_data_path = os.path.join(output_data_dir, f'processed_{file_basename}.csv')
        df.to_csv(output_data_path)
        print(f"\n处理后的数据已保存至: {output_data_path}")
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

def process_directory(directory_path):
    """
    处理指定目录下的所有数据文件
    
    参数:
        directory_path: 数据文件目录路径
    """
    print(f"开始处理目录: {directory_path}")
    
    # 查找所有pkl和csv文件
    pkl_files = glob.glob(os.path.join(directory_path, "**", "*.pkl"), recursive=True)
    csv_files = glob.glob(os.path.join(directory_path, "**", "*.csv"), recursive=True)
    
    all_files = pkl_files + csv_files
    
    if not all_files:
        print(f"在目录 {directory_path} 中未找到任何pkl或csv文件")
        return
    
    print(f"找到 {len(all_files)} 个文件")
    for file_path in all_files:
        print(file_path)
        
    
    # 处理每个文件
    for file_path in all_files:
        process_file(file_path)

if __name__ == "__main__":
    # 指定要处理的目录
    input_dir = "/home/luoew/stat_data/haomibo/16-processed"
    
    # 处理目录中的所有文件
    process_directory(input_dir)
