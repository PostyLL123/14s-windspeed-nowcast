import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
# ==============================================================================
# 1. 设置参数和路径
# ==============================================================================
# --- 请将 model_version 设置为您要处理的模型版本 ---

parser = argparse.ArgumentParser(description='calculate error to different steps and wpsd interval')
parser.add_argument(
                    '--model_version',
                    '--mv',#short input
                    type=str,
                    required=True
                    )
args = parser.parse_args()

model_version = args.model_version
print(f'正在计算模型{model_version}的误差')
#model_version = 'enhanced-s2s-batchsize-256-hidden-128-traindata-2-sampling-revised-residual' # 示例，请替换为您自己的

# 定义指标和风速区间
metrics = ['rmse', 'mae', 'mse', 'r2']
bin_list_str = ['0', '1', '3', '5', '10', '15', '20', '30']

# 构建文件路径
base_dir = '/home/rika/project/2025/wind-nowcasting25/leye16/model_output'
csv_file_path = os.path.join(base_dir, model_version, 'analyze', 'reorganized_predictions.csv')
output_dir = os.path.join(base_dir, model_version, 'metrics')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# 2. 准备 Bins 和 Labels
# ==============================================================================
# 为 pd.cut 准备数值型的 bins，并在末尾添加无穷大以包含所有 >30 的值
cut_bins = [int(b) for b in bin_list_str] + [np.inf]

# 自动生成标签 (修正了原始代码的错误)
bin_labels = [f"{bin_list_str[i]}-{bin_list_str[i+1]}" for i in range(len(bin_list_str)-1)] + [f">= {bin_list_str[-1]}"]

# ==============================================================================
# 3. 读取和预处理数据
# ==============================================================================
try:
    # 使用第一列作为索引并直接解析日期，这是更高效的方式
    print(f"正在读取数据文件: {csv_file_path}")
    df_predictions = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
    print("数据读取成功。")
except FileNotFoundError:
    print(f"错误: 文件未找到 {csv_file_path}")
    exit()

# ==============================================================================
# 4. 初始化用于存储结果的 DataFrame
# ==============================================================================
# 创建一个字典，每个指标对应一个空的 DataFrame
# DataFrame 的索引是风速区间，列是预测步长
metric_results = {
    metric: pd.DataFrame(
        index=bin_labels, 
        columns=[f't{i}' for i in range(1, 15)]
    ) for metric in metrics
}

# ==============================================================================
# 5. 主循环：计算每个步长和区间的指标
# ==============================================================================
# 循环14个预测步长 (t1 到 t14)
for step in range(1, 15):
    print(f"正在处理预测步长: t{step}...")
    
    # 动态获取当前步长的真实值和预测值列名
    true_col = f'true_t{step}'
    pred_col = f'pred_t{step}'
    
    # 检查列是否存在于DataFrame中
    if true_col not in df_predictions.columns or pred_col not in df_predictions.columns:
        print(f"警告: 在文件中找不到 {true_col} 或 {pred_col}，跳过此步长。")
        continue

    # 创建一个临时DataFrame，只包含当前步长所需的数据
    temp_df = df_predictions[[true_col, pred_col]].copy()
    
    # 根据真实风速进行分箱
    temp_df['interval'] = pd.cut(
        x=temp_df[true_col],
        bins=cut_bins,
        labels=bin_labels,
        right=False  # 区间为左闭右开 [a, b)，例如 [1, 3) 包含1但不包含3
    )
    
    # 按风速区间分组
    grouped_by_interval = temp_df.groupby('interval')
    
    # 为每个区间计算指标
    for interval_name, group_df in grouped_by_interval:
        # 如果某个分组为空，则跳过
        if group_df.empty:
            continue
            
        y_true = group_df[true_col]
        y_pred = group_df[pred_col]
        
        # 计算所有指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        # R²可能在样本很少时没有意义，但我们仍然计算它
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

        # 将计算结果存入对应的DataFrame
        col_name = f't{step}'
        metric_results['mse'].loc[interval_name, col_name] = mse
        metric_results['mae'].loc[interval_name, col_name] = mae
        metric_results['rmse'].loc[interval_name, col_name] = rmse
        metric_results['r2'].loc[interval_name, col_name] = r2

# ==============================================================================
# 6. 保存结果到 CSV 文件
# ==============================================================================
print("\n计算完成，正在保存结果...")

for metric_name, result_df in metric_results.items():
    output_filename = os.path.join(output_dir, f'{metric_name}_by_interval.csv')
    result_df.to_csv(output_filename)
    print(f" - 已保存: {output_filename}")

print("\n所有文件已成功保存。")