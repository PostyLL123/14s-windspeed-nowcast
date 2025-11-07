import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# ==============================================================================
# 1. 设置命令行参数
# ==============================================================================
parser = argparse.ArgumentParser(
    description='Compare and visualize metrics for two models across different wind speed intervals and prediction steps.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter # 使帮助信息更友好
)
parser.add_argument('--model_version1', '-mv1', type=str, required=True, help='Version name of the first model.')
parser.add_argument('--model_version2', '-mv2', type=str, required=True, help='Version name of the second model.')
parser.add_argument('--intervals', '-i', type=str, nargs='+', default=['1-3', '3-5', '5-10', '10-15'],
                    help='List of wind speed intervals to plot (e.g., "0-1" "1-3" "3-5"). Plots all if not specified.')
parser.add_argument('--start_step', '-ss', type=int, default=1, help='The starting prediction step to plot (e.g., 1 for t1).')
parser.add_argument('--end_step', '-es', type=int, default=14, help='The ending prediction step to plot (e.g., 14 for t14).')
parser.add_argument('--base_dir', '-bd', type=str, default='/home/rika/project/2025/wind-nowcasting25/leye16/model_output',
                    help='Base directory where model outputs are stored.')
parser.add_argument('--output_file', '-o', type=str, default='model_comparison.png', help='Name of the output plot file.')

args = parser.parse_args()


# ==============================================================================
# 2. 辅助函数和数据加载
# ==============================================================================

def load_metric_data(model_version, base_dir):
    """加载一个模型的所有四个指标CSV文件"""
    metrics = ['rmse', 'mae', 'mse', 'r2']
    metric_data = {}
    print(f"Loading data for model: {model_version}...")
    for metric in metrics:
        file_path = os.path.join(base_dir, model_version, 'metrics', f'{metric}_by_interval.csv')
        try:
            metric_data[metric] = pd.read_csv(file_path, index_col=0)
        except FileNotFoundError:
            print(f"ERROR: Cannot find file {file_path}")
            exit()
    return metric_data

# 为绘图定义更美观的标题
METRIC_TITLES = {
    'rmse': 'Root Mean Squared Error (RMSE)',
    'mae': 'Mean Absolute Error (MAE)',
    'mse': 'Mean Squared Error (MSE)',
    'r2': 'R-squared (R²)'
}

# ==============================================================================
# 3. 主逻辑
# ==============================================================================

# 加载两个模型的数据
data1 = load_metric_data(args.model_version1, args.base_dir)
data2 = load_metric_data(args.model_version2, args.base_dir)

# 确定要绘制的步长和区间
steps_to_plot = [f't{i}' for i in range(args.start_step, args.end_step + 1)]
all_intervals = data1['rmse'].index.tolist()

# 如果用户指定了区间，则进行过滤，否则使用所有可用区间
if args.intervals:
    # 检查用户指定的区间是否存在
    intervals_to_plot = [i for i in args.intervals if i in all_intervals]
    missing = set(args.intervals) - set(intervals_to_plot)
    if missing:
        print(f"Warning: The following specified intervals were not found and will be ignored: {missing}")
else:
    intervals_to_plot = all_intervals
    print(f"No intervals specified, plotting all available intervals: {intervals_to_plot}")


# 创建一个 2x2 的子图布局
fig, axes = plt.subplots(2, 2, figsize=(20, 15), constrained_layout=True)
axes = axes.flatten()  # 将 2x2 的数组转换为 1D 数组，方便循环

# 获取颜色循环，为每个风速区间分配一个颜色
colors = plt.cm.viridis(np.linspace(0, 1, len(intervals_to_plot)))

# 循环绘制四个指标
for i, metric in enumerate(['rmse', 'mae', 'mse', 'r2']):
    ax = axes[i]
    df1 = data1[metric].loc[intervals_to_plot, steps_to_plot]
    df2 = data2[metric].loc[intervals_to_plot, steps_to_plot]
    
    n_intervals = len(intervals_to_plot)
    n_steps = len(steps_to_plot)
    
    # 设置柱状图的位置
    x = np.arange(n_steps)  # X轴位置，代表每个步长
    total_width = 0.8  # 每个步长下所有柱子的总宽度
    interval_width = total_width / n_intervals  # 每个风速区间的柱子宽度
    
    for j, interval in enumerate(intervals_to_plot):
        # 计算每个区间柱子组的中心偏移量
        offset = (j - n_intervals / 2) * interval_width + interval_width / 2
        
        # 提取两个模型在该区间的数据
        values1 = df1.loc[interval].values
        values2 = df2.loc[interval].values
        
        # 绘制模型1的柱子 (实心)
        ax.bar(x + offset - interval_width / 4, values1, width=interval_width / 2, 
               color=colors[j], label=f'{interval} ({args.model_version1})' if i == 0 else "") # 仅在第一个子图添加标签以防重复
        
        # 绘制模型2的柱子 (带斜线图案)
        ax.bar(x + offset + interval_width / 4, values2, width=interval_width / 2, 
               color=colors[j], hatch='//', label=f'{interval} ({args.model_version2})' if i == 0 else "")

    # 设置子图的格式
    ax.set_title(METRIC_TITLES[metric], fontsize=16, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_xlabel('Prediction Step', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(steps_to_plot, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 创建一个统一的图例
# 我们需要为颜色（区间）和图案（模型）创建单独的图例
from matplotlib.patches import Patch
legend_elements_intervals = [Patch(facecolor=colors[j], label=interval) for j, interval in enumerate(intervals_to_plot)]
legend_elements_models = [
    Patch(facecolor='gray', label=f'Model 1: {args.model_version1}'),
    Patch(facecolor='gray', hatch='//', label=f'Model 2: {args.model_version2}')
]

fig.legend(handles=legend_elements_intervals + legend_elements_models, 
           loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(intervals_to_plot) + 2, fontsize='medium')

fig.suptitle('Model Performance Comparison', fontsize=24, y=1.03)

# 保存图像
plt.savefig(os.path.join(args.output_dir, args.output_file), dpi=300, bbox_inches='tight')
print(f"\nPlot saved successfully to {args.output_file}")

# 显示图像
plt.show()