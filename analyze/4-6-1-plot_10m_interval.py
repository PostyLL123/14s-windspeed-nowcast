
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.dates as mdates

# --- 1. 数据加载与预处理 ---
try:
    # 加载数据集
    model_version = 'enhanced_s2s'
    df = pd.read_csv(f'/home/luoew/model_output/14s/{model_version}/analyze/reorganized_predictions.csv')
     
    
    # 将时间列转换为datetime对象并设为索引
    df['time'] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index('time')
    df = df.drop(df.columns[0], axis=1)

    # --- 2. 按10分钟间隔分组并“拉平”数据绘图 ---
    
    # 为新的曲线图创建一个专用目录
    output_dir = f'/home/luoew/model_output/14s/{model_version}/analyze/visualization/wind_speed_plots_10min_flattened'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按10分钟频率分组
    grouper = pd.Grouper(freq='10T')
    
    generated_files = []

    # 遍历每个10分钟的时间组
    for group_name, group_df in df.groupby(grouper):
        
        if group_df.empty:
            continue

        # 初始化列表，用于存放“拉平”后的数据点和时间戳
        all_true_timestamps, all_true_values = [], []
        all_pred_timestamps, all_pred_values = [], []

        # 遍历组内的每一行（每行代表一个5秒的预测起点）
        for index, row in group_df.iterrows():
            # 提取未来1到5秒的真实值，并计算其对应的时间戳
            true_timestamps_for_row = [index + pd.Timedelta(seconds=i) for i in range(1, 6)]
            true_values_for_row = row['true_t1':'true_t5'].values
            
            # 提取未来1到5秒的预测值，并计算其对应的时间戳
            pred_timestamps_for_row = [index + pd.Timedelta(seconds=i) for i in range(1, 6)]
            pred_values_for_row = row['pred_t1':'pred_t5'].values

            # 将这5个数据点追加到总列表中
            all_true_timestamps.extend(true_timestamps_for_row)
            all_true_values.extend(true_values_for_row)
            all_pred_timestamps.extend(pred_timestamps_for_row)
            all_pred_values.extend(pred_values_for_row)

        # 如果没有收集到任何数据点，则跳过
        if not all_true_timestamps:
            continue

        # 将列表转换为Pandas Series，并按时间排序，以便绘制连续曲线
        true_series = pd.Series(all_true_values, index=all_true_timestamps).sort_index()
        pred_series = pd.Series(all_pred_values, index=all_pred_timestamps).sort_index()

        # --- 开始绘图 ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        # 绘制“拉平”后的真实风速曲线
        ax.plot(true_series.index, true_series.values, label='True Wind Speed', color='black', linewidth=2)
        # 绘制“拉平”后的预测风速曲线
        ax.plot(pred_series.index, pred_series.values, label='Predicted Wind Speed', color='royalblue', linewidth=1.8, alpha=0.9)
        
        # --- 美化图表 ---
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
        
        start_time = group_name
        end_time = start_time + pd.Timedelta(minutes=10)
        start_time_str_file = start_time.strftime('%Y-%m-%d_%H-%M')
        end_time_str_file = end_time.strftime('%H-%M')
        title = f'Continuous Wind Speed Prediction ({start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%H:%M")})'
        ax.set_title(title, fontsize=16)
        
        ax.set_xlim(start_time, end_time)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        ax.legend(loc='best')
        plt.tight_layout()
        
        # 保存图表
        filename = f"{output_dir}/wind_speed_{start_time_str_file}_to_{end_time_str_file}.png"
        plt.savefig(filename, dpi=200)
        plt.close(fig)
        generated_files.append(filename)

    print(f"成功生成 {len(generated_files)} 张图表，已保存至 '{output_dir}' 文件夹。")

except FileNotFoundError:
    print("错误：无法找到 'reorganized_predictions.csv' 文件。")
except Exception as e:
    print(f"处理过程中发生错误: {e}")  