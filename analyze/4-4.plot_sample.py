#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
案例挑选与绘图展示脚本
用于从整合输出文件中截取指定时间范围的数据并绘制折线图
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='案例挑选与绘图展示')
    
    # 输入文件路径
    parser.add_argument('--input_file', type=str, required=True,
                        help='整合输出文件路径，通常为combined_predictions.csv')
    
    # 时间范围参数
    parser.add_argument('--start_time', type=str, required=True,
                        help='开始时间，格式为YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--end_time', type=str, required=True,
                        help='结束时间，格式为YYYY-MM-DD HH:MM:SS')
    
    # 可选参数
    parser.add_argument('--dpi', type=int, default=300,
                        help='图像DPI，默认为300')
    parser.add_argument('--figsize', type=str, default='16,8',
                        help='图像尺寸，格式为width,height，默认为16,8')
    
    return parser.parse_args()

def load_data(file_path):
    """
    加载整合输出文件
    
    参数:
        file_path: 文件路径
        
    返回:
        加载的DataFrame，带有时间索引
    """
    print(f"正在加载数据: {file_path}")
    
    # 读取CSV文件，第一列作为索引
    df = pd.read_csv(file_path, index_col=0)
    
    # 将索引转换为datetime类型
    df.index = pd.to_datetime(df.index)
    
    print(f"数据加载完成，形状: {df.shape}")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"列名: {df.columns.tolist()}")
    
    return df

def extract_time_range(df, start_time, end_time):
    """
    提取指定时间范围的数据
    
    参数:
        df: 带有时间索引的DataFrame
        start_time: 开始时间字符串
        end_time: 结束时间字符串
        
    返回:
        截取后的DataFrame
    """
    # 转换时间字符串为datetime对象
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    
    # 截取指定时间范围的数据
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    extracted_df = df.loc[mask].copy()
    
    # 计算时间范围的秒数
    time_diff_seconds = (end_dt - start_dt).total_seconds()
    
    print(f"提取的时间范围: {start_dt} 到 {end_dt}")
    print(f"时间范围长度: {time_diff_seconds}秒")
    print(f"提取的数据点数: {len(extracted_df)}")
    
    return extracted_df, time_diff_seconds

def plot_data(df, time_diff_seconds, output_path, dpi=300, figsize=(12, 6)):
    """
    绘制折线图并保存
    
    参数:
        df: 要绘制的DataFrame
        time_diff_seconds: 时间范围的秒数
        output_path: 输出文件路径
        dpi: 图像DPI
        figsize: 图像尺寸元组
    """
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 48})  # 增加默认字体大小
    
    # 绘制原始风速（粗黑色实线）
    if 'wspd' in df.columns:
        plt.plot(df.index, df['wspd'], 'k-', linewidth=3.0, label='原始风速')
    else:
        plt.plot(df.index, df['true'], 'k-', linewidth=3.0, label='原始风速')
    
    # 获取预测组列名
    pred_cols = [col for col in df.columns if col.startswith('pred_group_')]
    
    # 颜色列表
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 绘制各预测组（彩色带点实线，稍细于原始风速）
    for i, col in enumerate(pred_cols):
        color = colors[i % len(colors)]
        plt.plot(df.index, df[col], color=color, linewidth=1.0, 
                 label=f'预测序列组{i+1}', alpha=0.9)#, '-o'markersize=3, 
    
    # 设置x轴格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # 根据数据点数量和时间范围调整x轴刻度密度
    total_points = len(df)
    
    # 计算合适的间隔，确保至少有30个标注点
    if time_diff_seconds <= 120:  # 2分钟以内
        # 每4秒一个刻度
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=4))
    elif time_diff_seconds <= 300:  # 5分钟以内
        # 每10秒一个刻度
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=10))
    elif time_diff_seconds <= 600:  # 10分钟以内
        # 每20秒一个刻度
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=20))
    elif time_diff_seconds <= 1800:  # 30分钟以内
        # 每分钟一个刻度
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    elif time_diff_seconds >= 3600:  # 30分钟以内
        # 每分钟一个刻度
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    else:  # 30分钟以上
        # 每2分钟一个刻度
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    
    # 旋转x轴标签以避免重叠
    plt.xticks(rotation=45)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 获取时间范围字符串
    time_range_str = f"{df.index[0].strftime('%Y-%m-%d %H:%M:%S')} 至 {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}"
    
    # 设置标题和标签（增加字体大小）
    plt.title(f"风速预测结果对比 ({time_range_str}, 共{int(time_diff_seconds)}秒)", fontsize=18)
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('风速 (m/s)', fontsize=16)
    
    # 增加图例字体大小和点大小
    plt.legend(loc='best', fontsize=14, markerscale=1.5)
    
    # 增加刻度标签大小
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"图像已保存至: {output_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 解析figsize参数
    figsize = tuple(map(int, args.figsize.split(',')))
    
    # 加载数据
    df = load_data(args.input_file)
    
    # 提取指定时间范围的数据
    extracted_df, time_diff_seconds = extract_time_range(df, args.start_time, args.end_time)
    
    if len(extracted_df) == 0:
        print("错误: 指定的时间范围内没有数据!")
        return
    
    # 创建输出文件路径
    input_dir = os.path.dirname(args.input_file)
    example_dir = os.path.join(input_dir, 'example')
    os.makedirs(example_dir, exist_ok=True)
    
    # 生成文件名
    start_str = args.start_time.replace(' ', '_').replace(':', '')
    end_str = args.end_time.replace(' ', '_').replace(':', '')
    output_filename = f"example_{start_str}_to_{end_str}.png"
    output_path = os.path.join(example_dir, output_filename)
    
    # 绘制并保存图像
    plot_data(extracted_df, time_diff_seconds, output_path, args.dpi, figsize)
    
    print("处理完成!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
