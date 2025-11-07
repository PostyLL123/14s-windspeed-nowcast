'''
数据准备脚本

功能：准备训练数据，包括特征选择、数据标准化等
    1. 读取特征工程后的数据文件
    2. 选择关键特征：
       - 残差和高频分量
       - 不同时间窗口的平均风速（10s、30s、120s）
       - 不同时间尺度的风速变化率（3s、5s、30s）
       - 风速标准差（120s）
       - 湍流强度（60s）
       - 时间周期特征（7天和14天）
    3. 生成训练序列：
       - 按照指定的输入长度（默认120s）和输出长度（默认14s）采样
       - 在输入长度上加入缓冲区（默认240s），确保后续滤波时特征计算的准确性
    4. 生成可视化图表：
       - 特征时间序列图
       - 特征相关性热图
    5. 保存处理后的数据集：
       - 输入序列（X）
       - 输出序列（y）
       - 采样信息（时间戳、序列长度等）

输入：1-3特征工程后的数据集（.pkl格式）
输出：
    1. 训练数据集（保存在16-sampled-data目录）：
       - X.pkl：输入特征序列
       - y.pkl：输出目标序列
       - sampling_info.pkl：采样信息
    2. 特征可视化图表（保存在16-sampled-data/plots目录）
'''

import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from tqdm import tqdm
import glob
import seaborn as sns

# 设置matplotlib中文显示
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号

# 添加项目根目录到系统路径
project_root = '/home/luoew/stat_data/haomibo/scripts'
print(f"项目根目录: {project_root}")
sys.path.append(project_root)

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
from utils.utils_data_preprocessing import process_features_x_data_with_features_engineering, process_features_y_data_with_features_engineering

# 设置输出目录
OUTPUT_BASE_DIR = '/home/luoew/stat_data/haomibo/16-sampled-data'

# 定义挑选的特征列表
SELECTED_FEATURES = [
    'residual','high_freq',
    'wspd_mean_10s',       # 10秒平均风速
    'wspd_mean_30s',       # 30秒平均风速
    'wspd_mean_120s',      # 120秒平均风速
    'wspd_change_ratio_5s', # 5秒风速变化率
    'wspd_change_3s',      # 3秒风速变化
    'wspd_change_30s',     # 30秒风速变化
    'wspd_std_120s',       # 120秒风速标准差
    'turb_60s',             # 60秒湍流强度
    'sec_sin_7s', 'sec_cos_7s', 'sec_sin_14s', 'sec_cos_14s',
]

# 全局变量
input_file = None  # 当前处理的文件路径
df = None  # 当前处理的数据框

#=================================================================================================================
# 数据读取与预处理
#=================================================================================================================

def read_data(file_path):
    """
    读取数据文件
    
    参数:
        file_path: 数据文件路径
    返回:
        df: 数据框
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
        
        # 输出数据基本信息
        print(f"\n数据基本信息:")
        print(f"数据点数量: {len(df)}")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        
        return df
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def plot_features(df, output_dir, file_basename):
    """
    绘制选择的特征
    
    参数:
        df: 数据框
        output_dir: 输出目录
        file_basename: 文件名（不含扩展名）
    """
    # 创建plots目录
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 检查哪些特征在数据中可用
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    
    if not available_features:
        print("没有找到可用的特征进行绘图")
        return
    
    # 计算每个特征的时间序列长度
    n_features = len(available_features)
    
    # 设置每张图最多显示的行数
    max_rows_per_plot = 5
    
    # 计算需要多少张图
    n_plots = (n_features + max_rows_per_plot - 1) // max_rows_per_plot
    
    for plot_idx in range(n_plots):
        # 计算当前图要显示的特征
        start_idx = plot_idx * max_rows_per_plot
        end_idx = min((plot_idx + 1) * max_rows_per_plot, n_features)
        current_features = available_features[start_idx:end_idx]
        
        # 创建当前图的布局
        n_rows = len(current_features)
        fig, axes = plt.subplots(n_rows, 1, figsize=(15, 3 * n_rows))
        if n_rows == 1:
            axes = [axes]
        
        # 绘制每个特征
        for i, feature in enumerate(current_features):
            ax = axes[i]
            ax.plot(df.index, df[feature])
            ax.set_title(feature)
            ax.set_xlabel('时间')
            ax.set_ylabel('值')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(plots_dir, f'{file_basename}_features_part{plot_idx+1}.png')
        plt.savefig(plot_path)
        print(f"特征图已保存至: {plot_path}")
        plt.close()
    
    # 绘制特征相关性热图
    if len(available_features) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[available_features].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('特征相关性热图')
        plt.tight_layout()
        
        # 保存热图
        heatmap_path = os.path.join(plots_dir, f'{file_basename}_correlation.png')
        plt.savefig(heatmap_path)
        print(f"相关性热图已保存至: {heatmap_path}")
        plt.close()

#=================================================================================================================
# 数据采样
#=================================================================================================================

def prepare_batch_sequences(data_df, start_times, input_len, output_len, buffer_len=240):
    """
    从数据中批量准备输入和输出序列
    
    参数:
        data_df: 数据DataFrame
        start_times: 输入序列的结束时间列表
        input_len: 输入序列长度
        output_len: 输出序列长度
        buffer_len: 用于特征处理的缓冲区长度
        
    返回:
        input_sequences: 输入序列DataFrame列表
        output_sequences: 输出序列DataFrame列表
        valid_times: 有效的结束时间列表
        sampling_info: 采样信息列表
    """
    input_sequences = []
    output_sequences = []
    valid_times = []
    sampling_info = []
    invalid_count = 0
    
    for end_time in tqdm(start_times, desc="准备序列"):
        # 计算输入序列的起始时间（使用buffer_len）
        input_start_time = end_time - pd.Timedelta(seconds=buffer_len-1)
        
        # 计算输出序列的起始和结束时间
        output_start_time = end_time + pd.Timedelta(seconds=1)
        output_end_time = end_time + pd.Timedelta(seconds=output_len)
        
        # 计算输出序列的缓冲区起始时间（以output_end_time为最后一个时间点，向前取buffer_len长度）
        output_buffer_start_time = output_end_time - pd.Timedelta(seconds=buffer_len-1)
        
        # 从数据中提取输入序列（包含缓冲区）
        input_sequence = data_df.loc[input_start_time:end_time].copy()
        
        # 从数据中提取输出序列（包含缓冲区）
        output_sequence = data_df.loc[output_buffer_start_time:output_end_time].copy()
        
        # 检查是否获取到足够的数据
        if len(input_sequence) >= buffer_len and len(output_sequence) >= buffer_len:
            input_sequences.append(input_sequence)
            output_sequences.append(output_sequence)
            valid_times.append(end_time)
            
            # 记录采样信息（使用最终输出序列的起止时间）
            sampling_info.append({
                'sequence': len(valid_times),
                'input_start_time': end_time - pd.Timedelta(seconds=input_len-1),  # 最终输入序列的起始时间
                'input_end_time': end_time,  # 最终输入序列的结束时间
                'output_start_time': output_start_time,  # 最终输出序列的起始时间
                'output_end_time': output_end_time,  # 最终输出序列的结束时间
                'input_len': input_len,
                'output_len': output_len,
                'buffer_len': buffer_len
            })
        else:
            invalid_count+=1
            # pass
            # print(f"跳过样本: 输入序列长度={len(input_sequence)}/{buffer_len}, 输出序列长度={len(output_sequence)}/{buffer_len}")
    
    
    print(f' 跳过了{invalid_count}个不合格样本')
    return input_sequences, output_sequences, valid_times, sampling_info

def process_sequences(input_sequences, output_sequences, process_features_func=None, output_len=None, input_len=None):
    """
    对输入和输出序列进行预处理
    
    参数:
        input_sequences: 输入序列DataFrame列表
        output_sequences: 输出序列DataFrame列表
        process_features_func: 特征处理函数
        output_len: 输出序列长度，用于截取处理后的序列
        input_len: 输入序列长度，用于截取处理后的输入序列
    
    返回:
        processed_input_sequences: 处理后的输入序列列表
        processed_output_sequences: 处理后的输出序列列表
    """
    processed_input_sequences = []
    processed_output_sequences = []
    
    for i, (input_seq, output_seq) in enumerate(tqdm(zip(input_sequences, output_sequences), desc="预处理序列")):
        # 处理输入序列
        if process_features_func is not None:
            processed_input = process_features_func(input_seq)
        else:
            processed_input = input_seq
        
        # 只保留挑选的特征
        available_features = [f for f in SELECTED_FEATURES if f in processed_input.columns]
        processed_input = processed_input[available_features]
        
        # 如果指定了input_len，则截取最后input_len个时间步
        if input_len is not None and len(processed_input) > input_len:
            processed_input = processed_input.iloc[-input_len:]
        
        # 处理输出序列
        if process_features_func is not None:
            processed_output = process_features_func(output_seq)
        else:
            processed_output = output_seq
        
        # 只保留residual列
        processed_output = processed_output['residual']
        
        # 如果指定了output_len，则截取最后output_len个时间步（因为我们需要的是最后output_len个时间步）
        if output_len is not None and len(processed_output) > output_len:
            processed_output = processed_output.iloc[-output_len:]
        
        processed_input_sequences.append(processed_input)
        processed_output_sequences.append(processed_output)
    
    return processed_input_sequences, processed_output_sequences

def save_dataset(data_x, data_y, sampling_info, output_dir, dataset_name, input_len, output_len):
    """
    保存数据集及其采样信息
    
    参数:
        data_x: 输入数据numpy数组
        data_y: 输出数据numpy数组
        sampling_info: 采样信息列表
        output_dir: 输出目录
        dataset_name: 数据集名称
        input_len: 输入序列长度
        output_len: 输出序列长度
    """
    # 创建采样信息DataFrame
    sampling_info_df = pd.DataFrame(sampling_info)
    
    # 分别保存x和y的采样信息到Excel
    x_sampling_info = sampling_info_df.copy()
    y_sampling_info = sampling_info_df.copy()
    
    # 保存x的采样信息到Excel
    x_sampling_info_path = os.path.join(output_dir, f'{dataset_name}_x_sampling_info.xlsx')
    x_sampling_info.to_excel(x_sampling_info_path, index=False)
    print(f"x采样信息已保存至: {x_sampling_info_path}")
    
    # 保存y的采样信息到Excel
    y_sampling_info_path = os.path.join(output_dir, f'{dataset_name}_y_sampling_info.xlsx')
    y_sampling_info.to_excel(y_sampling_info_path, index=False)
    print(f"y采样信息已保存至: {y_sampling_info_path}")
    
    # 保存数据集信息
    dataset_info = {
        'x_shape': data_x.shape,
        'y_shape': data_y.shape,
        'input_len': input_len,
        'output_len': output_len,
        'selected_features': SELECTED_FEATURES  # 添加挑选的特征列表
    }
    
    # 保存数据集信息
    dataset_info_path = os.path.join(output_dir, f'{dataset_name}_dataset_info.pkl')
    with open(dataset_info_path, 'wb') as f:
        pickle.dump(dataset_info, f)
    print(f"数据信息已保存至: {dataset_info_path}")
    
    # 分别保存x和y数据集为npy格式
    data_x_path = os.path.join(output_dir, f'{dataset_name}_x.npy')
    data_y_path = os.path.join(output_dir, f'{dataset_name}_y.npy')
    np.save(data_x_path, data_x)
    np.save(data_y_path, data_y)
    print(f"x已保存至: {data_x_path}")
    print(f"y已保存至: {data_y_path}")
    
    # 分别保存x和y的示例数据
    x_sample_path = os.path.join(output_dir, f'{dataset_name}_x_sample.csv')
    y_sample_path = os.path.join(output_dir, f'{dataset_name}_y_sample.csv')
    
    # 保存x示例数据
    pd.DataFrame(data_x[0], columns=SELECTED_FEATURES).to_csv(x_sample_path, index=True)
    print(f"x示例数据已保存至: {x_sample_path}")
    
    # 保存y示例数据
    pd.DataFrame(data_y[0], columns=['residual']).to_csv(y_sample_path, index=True)
    print(f"y示例数据已保存至: {y_sample_path}")
    
    # 保存采样时间信息
    time_info = {
        'input_start_times': [info['input_start_time'] for info in sampling_info],
        'input_end_times': [info['input_end_time'] for info in sampling_info],
        'output_start_times': [info['output_start_time'] for info in sampling_info],
        'output_end_times': [info['output_end_time'] for info in sampling_info]
    }
    
    # 保存时间信息
    time_info_path = os.path.join(output_dir, f'{dataset_name}_time_info.pkl')
    with open(time_info_path, 'wb') as f:
        pickle.dump(time_info, f)
    print(f"时间信息已保存至: {time_info_path}")

def prepare_data(file_path, output_dir, input_len=120, output_len=14, interval=5, buffer_len=240):
    """
    准备数据序列
    
    参数:
        file_path: 输入数据路径
        output_dir: 输出目录
        input_len: 输入序列长度
        output_len: 输出序列长度
        interval: 采样间隔（秒）
        buffer_len: 用于特征处理的缓冲区长度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名）
    file_basename = os.path.basename(file_path).split('.')[0]
    
    # 加载数据
    print(f"加载数据: {file_path}")
    data = pd.read_pickle(file_path)
    
    # 计算可用的预测起始时间点
    start_times = []
    current_time = data.index[0] + pd.Timedelta(seconds=buffer_len-1)
    end_time = data.index[-1] - pd.Timedelta(seconds=output_len)
    
    while current_time <= end_time:
        start_times.append(current_time)
        current_time += pd.Timedelta(seconds=interval)
    
    # 同时准备输入和输出序列
    input_sequences, output_sequences, valid_times, sampling_info = prepare_batch_sequences(data, start_times, input_len, output_len, buffer_len)
    
    # 处理输入和输出序列
    processed_input_sequences, processed_output_sequences = process_sequences(
        input_sequences, output_sequences, 
        process_features_func=process_features_x_data_with_features_engineering,
        output_len=output_len,
        input_len=input_len
    )
    
    # 将序列转换为numpy数组
    print("将序列转换为numpy数组...")
    x_array = np.stack([seq.iloc[-input_len:].values for seq in processed_input_sequences])
    y_array = np.stack([seq.values for seq in processed_output_sequences])
    
    # 保存数据集
    save_dataset(x_array, y_array, sampling_info, output_dir, file_basename, input_len, output_len)
    
    print(f"数据处理完成，共处理 {len(processed_input_sequences)} 个序列")
    print(f"处理后的数据已保存到: {output_dir}")

#=================================================================================================================
# 主函数
#=================================================================================================================

def process_file(file_path, input_len=120, output_len=14, interval=5, buffer_len=240):
    """
    处理单个数据文件
    
    参数:
        file_path: 数据文件路径
        input_len: 输入序列长度
        output_len: 输出序列长度
        interval: 采样间隔（秒）
        buffer_len: 用于特征处理的缓冲区长度
    """
    # 1. 读取数据
    df = read_data(file_path)
    if df is None:
        return
    
    # 2. 准备输出目录
    file_basename = os.path.basename(file_path).split('.')[0]
    output_dir = os.path.join(OUTPUT_BASE_DIR, file_basename)
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否已处理过该文件
    y_sample_path = os.path.join(output_dir, f'{file_basename}_y_sample.csv')
    if os.path.exists(y_sample_path):
        print(f"文件 {file_basename} 已处理过，跳过处理")
        return
    
    # 3. 绘制特征图
    plot_features(df, output_dir, file_basename)
    
    # 4. 准备数据序列
    prepare_data(file_path, output_dir, input_len, output_len, interval, buffer_len)

def process_directory(directory_path, input_len=120, output_len=14, interval=5, buffer_len=240):
    """
    处理指定目录下的所有数据文件
    
    参数:
        directory_path: 数据文件目录路径
        input_len: 输入序列长度
        output_len: 输出序列长度
        interval: 采样间隔（秒）
        buffer_len: 用于特征处理的缓冲区长度
    """
    print(f"开始处理目录: {directory_path}")
    
    # 查找所有pkl和csv文件
    pkl_files = glob.glob(os.path.join(directory_path, "**", "DATA-16-20240601-0630-add-feature.pkl"), recursive=True)
    # csv_files = glob.glob(os.path.join(directory_path, "**", "*.csv"), recursive=True)
    
    all_files = pkl_files  # + csv_files
    
    if not all_files:
        print(f"在目录 {directory_path} 中未找到任何pkl或csv文件")
        return
    
    print(f"找到 {len(all_files)} 个文件:")
    for file_path in all_files:
        print(f"  - {file_path}")
    
    # 处理每个文件
    for file_path in all_files:
        process_file(file_path, input_len, output_len, interval, buffer_len)

if __name__ == "__main__":
    # 指定要处理的目录
    input_dir = "/home/luoew/stat_data/haomibo/16-add-feature/data"
    
    # 设置参数
    input_len = 120
    output_len = 14
    interval = 5
    buffer_len = 240  # 用于特征处理的缓冲区长度
    
    # 处理目录中的所有文件
    process_directory(input_dir, input_len, output_len, interval, buffer_len)
    
    print("数据准备完成!")
