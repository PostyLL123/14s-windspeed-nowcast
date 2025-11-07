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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def __init__():
    parser = argparse.ArgumentParser(description='torch_version enhanced s2s model for 14s wind forecast')

    parser.add_argument('--configs', type=str, default='enhanced_s2s', help='configs of model')
    parser.add_argument('--local rank', type=int, default=0, help='local rank for distributed training') # (暂不启用)
    parser.add_argument('--project', type=str, default='14s', help='project name, 14s, 60min or ....')
    args = parser.parse_args()
    
    # 动态加载 config
    module_name = 'main.configs.train.' + args.project + '.' + args.configs
    try:
        xconfig = importlib.import_module(module_name)
    except ImportError:
        print(f"错误: 无法加载配置文件: {module_name}")
        print(f"请确保 config文件{module_name}存在")
        sys.exit(1)
        

    return xconfig

def load_prediction_data(predict_out_dir: str, model_version: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载预测结果数据"""
    #predict_out_dir = f'/home/luoew/project/nowcasting/model_output/{model_version}/predict_out'
    y_true = pd.read_pickle(os.path.join(predict_out_dir, f"y_true_df_{model_version}.pkl"))
    y_pred = pd.read_pickle(os.path.join(predict_out_dir, f"y_pred_df_{model_version}.pkl"))
    return y_true, y_pred

def reorganize_data_new(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    重组数据为统一的DataFrame
    每行包含一组预测的真值和预测值
    
    参数:
        y_true: 真值DataFrame
        y_pred: 预测值DataFrame
        test_start_time: 测试集开始时间（参考时间）
        args: 配置参数
        sampling_info: 采样信息DataFrame
    """
    # 重命名列
    true_cols = [f'true_t{i+1}' for i in range(y_true.shape[1])]
    pred_cols = [f'pred_t{i+1}' for i in range(y_pred.shape[1])]
    
    y_true.columns = true_cols
    y_pred.columns = pred_cols
    
    test_start_time = y_true.index[0]
    # 合并数据
    combined_df = pd.concat([y_true, y_pred], axis=1)
    
    # 如果样本数量不匹配，则使用默认的时间索引
    #print(f"警告: 样本数量不匹配，使用默认时间索引 (combined_df: {len(combined_df)}, sampling_info: {len(sampling_info)})")
    combined_df.index = pd.date_range(
        start=test_start_time,
        periods=len(combined_df),
        freq='5S'  # 预测间隔为5秒
    )
    
    return combined_df


def calculate_error_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算每个预测时间步的误差统计
    返回误差统计和分位数统计
    """
    n_steps = len([col for col in data.columns if col.startswith('true')])
    error_stats = []
    quantile_stats = []
    
    for i in range(n_steps):
        true_col = f'true_t{i+1}'
        pred_col = f'pred_t{i+1}'
        
        # 计算误差
        errors = data[pred_col] - data[true_col]
        abs_errors = np.abs(errors)
        
        # 基本统计量
        stats = {
            'timestep': i+1,
            'mae': abs_errors.mean(),
            'rmse': np.sqrt((errors**2).mean()),
            'bias': errors.mean(),
            'std': errors.std()
        }
        error_stats.append(stats)
        
        # 分位数统计
        quantiles = {
            'timestep': i+1,
            'q25': errors.quantile(0.25),
            'median': errors.median(),
            'q75': errors.quantile(0.75)
        }
        quantile_stats.append(quantiles)
    
    error_df = pd.DataFrame(error_stats)
    quantile_df = pd.DataFrame(quantile_stats)
    
    return error_df, quantile_df

def plot_random_samples(data: pd.DataFrame, n_samples_per_plot: int = 20, n_plots: int = 5, save_path: str = None):
    """
    绘制随机样本的预测对比图，生成多张图表
    
    参数:
        data: 包含真实值和预测值的DataFrame
        n_samples_per_plot: 每张图中显示的样本数量
        n_plots: 要生成的图表数量
        save_path: 保存路径（不包含扩展名）
    """
    # 确定总样本数
    total_samples = n_samples_per_plot * n_plots
    # 随机选择不重复的样本
    all_sample_indices = random.sample(range(len(data)), min(total_samples, len(data)))
    n_steps = len([col for col in data.columns if col.startswith('true')])
    
    # 为每张图创建单独的样本集
    for plot_idx in range(n_plots):
        # 获取当前图表的样本索引
        start_idx = plot_idx * n_samples_per_plot
        end_idx = start_idx + n_samples_per_plot
        current_samples = all_sample_indices[start_idx:end_idx]
        
        # 创建图表
        n_cols = 4
        n_rows = (n_samples_per_plot + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        for idx, sample_idx in enumerate(current_samples):
            ax = axes[idx]
            sample = data.iloc[sample_idx]
            
            # 提取真值和预测值
            true_values = [sample[f'true_t{i+1}'] for i in range(n_steps)]
            pred_values = [sample[f'pred_t{i+1}'] for i in range(n_steps)]
            
            # 绘制对比图
            time_points = range(1, n_steps + 1)
            ax.plot(time_points, true_values, 'b-', label='真实值', marker='o')
            ax.plot(time_points, pred_values, 'r--', label='预测值', marker='x')
            
            ax.set_title(f'样本 {sample_idx} ({data.index[sample_idx].strftime("%Y-%m-%d %H:%M:%S")})')
            ax.set_xlabel('预测时间步')
            ax.set_ylabel('风速 (m/s)')
            ax.grid(True)
            ax.legend()
        
        # 隐藏多余的子图
        for idx in range(len(current_samples), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            # 添加编号后缀
            plot_save_path = f"{save_path[:-4]}_{plot_idx+1:02d}.png"
            plt.savefig(plot_save_path)
            plt.close()
    else:
            plt.show()

def plot_error_statistics(error_stats: pd.DataFrame, quantile_stats: pd.DataFrame, data: pd.DataFrame, save_dir: str):
    """绘制误差统计图"""
    # 1. MAE和RMSE对比图
    plt.figure(figsize=(10, 6))
    plt.plot(error_stats['timestep'], error_stats['mae'], 'b-', label='MAE', marker='o')
    plt.plot(error_stats['timestep'], error_stats['rmse'], 'r-', label='RMSE', marker='s')
    plt.fill_between(quantile_stats['timestep'], 
                    quantile_stats['q25'], 
                    quantile_stats['q75'], 
                    alpha=0.2, color='gray')
    plt.xlabel('预测时间步')
    plt.ylabel('误差 (m/s)')
    plt.title('预测误差随时间步的变化')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'error_metrics.png'))
    plt.close()
    
    # 2. 误差箱型图
    plt.figure(figsize=(12, 6))
    error_data = []
    for i in range(len(error_stats)):
        true_col = f'true_t{i+1}'
        pred_col = f'pred_t{i+1}'
        errors = data[pred_col] - data[true_col]
        error_data.append(errors)
    
    plt.boxplot(error_data, labels=[f't{i+1}' for i in range(len(error_stats))])
    plt.xlabel('预测时间步')
    plt.ylabel('误差 (m/s)')
    plt.title('预测误差分布箱型图')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'error_boxplot.png'))
    plt.close()

def main():
    
    cfg = __init__()

    analyze_dir = os.path.join(cfg.work_dir, cfg.project, cfg.model_cfg.model_name, 'analyze')
    os.makedirs(analyze_dir, exist_ok=True)

    predict_dir = os.path.join(cfg.work_dir, cfg.project, cfg.model_cfg.model_name, 'predict')
    y_true, y_pred = load_prediction_data(predict_dir, cfg.model_cfg.model_name)

    print("重组数据...")
    data = reorganize_data_new(y_true, y_pred)
    data.to_excel(os.path.join(analyze_dir, 'reorganized_predictions.xlsx'))

    print("计算误差统计...")
    error_stats, quantile_stats = calculate_error_statistics(data)
    
    # 8. 保存统计结果
    error_stats.to_csv(os.path.join(analyze_dir, 'error_statistics.csv'), index=False)
    quantile_stats.to_csv(os.path.join(analyze_dir, 'error_quantiles.csv'), index=False)
    
    # 9. 绘制统计图表
    print("绘制统计图表...")
    plot_error_statistics(error_stats, quantile_stats, data, analyze_dir)
    
    # 10. 绘制随机样本对比图
    print("绘制随机样本对比图...")
    plot_random_samples(data, n_samples_per_plot=20, n_plots=5, 
                       save_path=os.path.join(analyze_dir, 'random_samples_comparison'))
    
    # 11. 保存重组后的数据
    data.to_csv(os.path.join(analyze_dir, 'reorganized_predictions.csv'))
    
    print(f"分析完成！结果已保存至: {analyze_dir}")
    
    # 12. 打印关键统计信息
    print("\n预测误差统计概要:")
    print("\nMAE统计:")
    print(error_stats[['timestep', 'mae']].to_string(index=False))
    print("\nRMSE统计:")
    print(error_stats[['timestep', 'rmse']].to_string(index=False))

if __name__ =='__main__':
    main()

