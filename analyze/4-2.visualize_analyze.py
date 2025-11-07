test_import = False #用于测试导入是否成功

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Line
from typing import List
import argparse
import importlib
# 添加项目根目录到系统路径
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.analyze_utils import extract_and_group_predictions, merge_prediction_groups

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if test_import:
    print('导入测试成功！')
    os._exit(0)

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

def load_analyze_data(model_output_dir: str) -> pd.DataFrame:
    """
    加载analyze目录下的重建数据
    
    参数:
        model_output_dir: 模型输出目录
        
    返回:
        重建数据DataFrame
    """
    # 构建analyze目录路径
    analyze_dir = os.path.join(model_output_dir, 'analyze')
    if not os.path.exists(analyze_dir):
        raise FileNotFoundError(f"找不到analyze目录: {analyze_dir}")
    
    # 读取重组后的预测数据
    data_path = os.path.join(analyze_dir, 'reorganized_predictions.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到重建数据文件: {data_path}")
    
    # 读取数据,将时间列设置为索引
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    print(f"数据加载成功!")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"列名: {df.columns.tolist()}")
    
    return df

def extract_true_sequence(df: pd.DataFrame, interval: str = '1S') -> pd.Series:
    """
    从预测数据中提取并拼接真值序列
    
    参数:
        df: 包含预测数据的DataFrame
        interval: 时间间隔,默认为1秒
        
    返回:
        完整的真值序列
    """
    # 获取所有真值列
    true_cols = [col for col in df.columns if col.startswith('true_')]
    n_steps = len(true_cols)
    print(f"检测到{n_steps}个预测时间步")
    
    # 创建空的序列字典,用于存储所有时间点的值
    sequence_dict = {}
    
    # 遍历每一行,提取真值序列
    for idx, row in df.iterrows():
        # 对于每一行,根据起始时间和间隔生成时间序列
        times = pd.date_range(start=idx, periods=n_steps, freq=interval)
        
        # 将该行的真值添加到对应的时间点(保留3位小数)
        for t, col in zip(times, true_cols):
            sequence_dict[t] = round(row[col], 3)
    
    # 转换为Series并排序
    true_sequence = pd.Series(sequence_dict)
    true_sequence.sort_index(inplace=True)
    
    # 去除重复的时间点(如果有的话),保留第一次出现的值
    true_sequence = true_sequence[~true_sequence.index.duplicated(keep='first')]
    
    print(f"真值序列提取完成:")
    print(f"序列长度: {len(true_sequence)}")
    print(f"时间范围: {true_sequence.index[0]} 到 {true_sequence.index[-1]}")
    print(f"时间间隔: {interval}")
    
    return true_sequence

def create_combined_dataframe(true_sequence: pd.Series, 
                            merged_predictions: List[pd.Series]) -> pd.DataFrame:
    """
    将真值序列和预测序列合并为DataFrame
    
    参数:
        true_sequence: 真值序列
        merged_predictions: 合并后的预测序列列表
        
    返回:
        合并后的DataFrame
    """
    # 创建以真值序列index为基准的DataFrame
    combined_df = pd.DataFrame(index=true_sequence.index)
    
    # 添加真值序列
    combined_df['true'] = true_sequence
    
    # 添加各组预测序列
    for i, pred_series in enumerate(merged_predictions):
        combined_df[f'pred_group_{i+1}'] = pred_series
    
    return combined_df

def split_dataframe(df: pd.DataFrame, segment_size: int = 30000) -> List[pd.DataFrame]:
    """
    将DataFrame按照指定大小拆分成多段
    
    参数:
        df: 要拆分的DataFrame
        segment_size: 每段数据的长度
        
    返回:
        拆分后的DataFrame列表
    """
    total_len = len(df)
    n_segments = int(np.ceil(total_len / segment_size))
    print(f"数据总长度: {total_len}, 拆分为{n_segments}段")
    
    segments = []
    for seg_idx in range(n_segments):
        start_idx = seg_idx * segment_size
        end_idx = min((seg_idx + 1) * segment_size, total_len)
        
        segment = df.iloc[start_idx:end_idx].copy()
        segments.append(segment)
        
        print(f"第{seg_idx+1}段:")
        print(f"数据范围: {start_idx} - {end_idx}")
        print(f"时间范围: {segment.index[0]} - {segment.index[-1]}")
        print("---")
    
    return segments

def plot_combined_data(combined_df: pd.DataFrame, save_dir: str, title: str = "风速预测结果对比"):
    """
    使用pyecharts可视化合并后的数据
    
    参数:
        combined_df: 合并后的DataFrame
        save_dir: 保存目录
        title: 图表标题
    """
    # 创建图表
    line = Line(init_opts=opts.InitOpts(width="1600px", height="800px"))
    
    # 设置x轴数据
    timestamps = [t.strftime("%Y-%m-%d %H:%M:%S") for t in combined_df.index]
    line.add_xaxis(timestamps)
    
    # 计算所有数据的最大值和最小值
    all_values = []
    all_values.extend(combined_df['true'].dropna().values)
    pred_cols = [col for col in combined_df.columns if col.startswith('pred_group_')]
    for col in pred_cols:
        all_values.extend(combined_df[col].dropna().values)
    
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    
    # 计算Y轴范围,留出10%的空白
    y_range = y_max - y_min
    y_min = y_min - y_range * 0.1
    y_max = y_max + y_range * 0.1
    
    print(f"Y轴范围: [{y_min:.3f}, {y_max:.3f}]")
    
    # 添加真值数据
    line.add_yaxis(
        series_name="真实值",
        y_axis=[round(v, 3) if pd.notna(v) else "-" for v in combined_df['true']],
        symbol_size=2,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1),
        itemstyle_opts=opts.ItemStyleOpts(color='#000000'),  # 设置真值为黑色
        is_symbol_show=False
    )
    
    # 添加各组预测结果
    colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc']
    for i, col in enumerate(pred_cols):
        # 确保颜色索引不会超出范围
        color_idx = i % len(colors)
        line.add_yaxis(
            series_name=f"预测组{i+1}",
            y_axis=[round(v, 3) if pd.notna(v) else "-" for v in combined_df[col]],
            symbol_size=4,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=1),
            itemstyle_opts=opts.ItemStyleOpts(color=colors[color_idx]),
            is_symbol_show=True,
            is_connect_nones=False  # 不连接空值点
        )
    
    # 设置全局选项
    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            subtitle=f"时间范围: {combined_df.index[0]} 至 {combined_df.index[-1]}"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_top="5%"),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100)
        ],
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=False,
            axislabel_opts=opts.LabelOpts(rotate=45)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="风速 (m/s)",
            min_=round(y_min, 3),
            max_=round(y_max, 3),
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    )
    
    # 保存图表
    line.render(save_dir)

def main():
    try:
        # 1. 加载配置和数据
        cfg = __init__()

        print("\n模型配置:")
        print(f"模型版本: {cfg.model_cfg.model_name}")
        print(f"模型输出目录: {cfg.model_output_dir}")
        
        # 2. 加载数据
        print("\n加载分析数据...")
        df = load_analyze_data(cfg.model_output_dir)

        # 3. 提取真值序列
        print("\n提取真值序列...")
        true_sequence = extract_true_sequence(df, interval='1S')
        print(true_sequence)
        
                
        # 4. 提取和分组预测序列
        print("\n提取和分组预测序列...")
        grouped_predictions = extract_and_group_predictions(df, pred_interval=5, time_interval='1S')
        
        # 5. 合并每组内的预测序列
        print("\n合并预测序列...")
        merged_predictions = merge_prediction_groups(grouped_predictions)
        
        # 6. 创建合并后的DataFrame
        print("\n创建合并DataFrame...")
        combined_df = create_combined_dataframe(true_sequence, merged_predictions)
        
        # 7. 创建visualization目录
        visualization_dir = os.path.join(cfg.model_output_dir, 'analyze', 'visualization')
        os.makedirs(visualization_dir, exist_ok=True)
        
        # 8. 保存数据
        print("\n保存数据...")
        combined_df.to_csv(os.path.join(visualization_dir, 'combined_predictions.csv'))
        
        # 9. 拆分数据并分段可视化
        print("\n拆分数据并生成可视化图表...")
        segments = split_dataframe(combined_df, segment_size=30000)
        for i, segment in enumerate(segments, 1):
            save_path = os.path.join(visualization_dir, f'prediction_visualization_seg{i:02d}.html')
            plot_combined_data(
                segment, 
                save_path,
                title=f"风速预测结果对比 (第{i}/{len(segments)}段)"
            )
        
        print("\n处理完成!")
        print(f"结果已保存至: {visualization_dir}")
        print(f"- 数据文件: combined_predictions.csv")
        print(f"- 可视化文件: prediction_visualization_seg[01-{len(segments):02d}].html")
        
    except Exception as e:
        print(f"\n处理过程中出现错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        raise

if __name__ == "__main__":
    #model_out_dir = "enhanced-s2s-batchsize-256-hidden-128-traindata-2-sampling-revised-residual"#'enhanced-s2s-batchsize-512-hidden-512-sampling-revised-residual'#'S2S-decom-251011_1104-sampling-revised-residual'
    main()