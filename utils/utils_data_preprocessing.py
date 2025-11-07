import pandas as pd
# 添加项目根目录到系统路径
import sys, os
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from add_features import add_rolling_features, add_time_features, add_turbulence_features
from utils_decompose import decompose_signal
# from utils.fukong_logging_utils import log_or_print
# from WriteReadLog import ky_WriteLog,info,error,warning,critical,debug,ModelLog


USE_LOGGER = False #仅使用print

# 从已计算的特征中选择基础特征集
basic_features = [
    'wspd',                    # 当前风速
    'wdir_sin', 'wdir_cos',   # 当前风向的三角分量2个
    'wspd_mean_10s',          # 10秒平均风速
    'wspd_std_10s',           # 10秒风速标准差
    # 'hour_sin', 'hour_cos',   # 时间周期特征2个
    'turb_60s'                # 1分钟湍流强度
]


#=================================================================================================================
# STEP1: 数据特征工程
#=================================================================================================================

def data_feature_engineering(   
    df: pd.DataFrame,
    logger=None
) -> pd.DataFrame:
    # 应用特征工程
    df_features = df.copy()
    df_features = add_rolling_features(df_features, window_sizes=[10])
    df_features = add_time_features(df_features, periods=[3,7,14,60])
    df_features = add_turbulence_features(df_features, window_sizes=[60])

    # 选择特征子集并处理空值
    df_basic = df_features[basic_features]
    
    # 使用线性插值填充空值
    df_basic = df_basic.interpolate(method='linear')
    # 对于序列开始和结束的空值，使用前向/后向填充
    df_basic = df_basic.fillna(method='ffill').fillna(method='bfill')
    
    return df_basic

#=================================================================================================================
# STEP2: 数据分解
#=================================================================================================================

def decompose_wind_data(
    df: pd.DataFrame,
    logger=None
) -> pd.DataFrame:
    """
    使用滤波器方法对风速数据进行分解
    
    参数:
        df: 输入的DataFrame，必须包含'wspd'列
        logger: 可选，日志记录器
        
    返回:
        包含原始数据和分解结果的DataFrame
    """
    # 读取数据

    # log_or_print(f"开始处理数据，形状: {df.shape}", logger)
    if 'TIMESTAMP' not in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df.index).round('1s')
    
    original = df['wspd'].values
    # log_or_print(f"使用数据长度: {len(original)}", logger)
    
    # 设置分解参数
    decompose_params = {
        'cutoff_freq': 1/3,  # 3秒周期的截止频率
        'sampling_period': 1.0  # 1秒采样周期
    }
    
    # # 执行分解
    # log_or_print("开始基于滤波器的信号分解...", logger)
    # log_or_print(f"数据长度: {len(original)}", logger)
    # log_or_print(f"截止频率: {decompose_params['cutoff_freq']} Hz (周期 {1/decompose_params['cutoff_freq']} 秒)", logger)
    
    residual, components = decompose_signal(
        data=original,
        method='filter',
        params=decompose_params,
        print_stats=False
    )
    
    # 创建结果DataFrame
    result_df = df.copy()
    result_df['residual'] = residual  # 低频分量作为残差
    result_df['high_freq'] = components[0]  # 高频分量

    return result_df

def save_component_datasets(
    result_df: pd.DataFrame,
    component_name: str = 'residual',
    logger=None
) -> pd.DataFrame:
    """
    为指定分量创建数据集，并添加合适的时间特征
    
    参数:
        result_df: 包含分解结果的DataFrame
        component_name: 要处理的分量名称，默认为'residual'
        logger: 可选，日志记录器
        
    返回:
        处理后的分量数据集DataFrame
    """
    # 删除包含NaN的行
    clean_df = result_df#.dropna()
    # log_or_print(f"删除NaN后的数据长度: {len(clean_df)}", logger)
    
    # 验证请求的分量是否存在
    if component_name not in clean_df.columns:
        raise ValueError(f"分量 '{component_name}' 不在数据集中。可用分量: {[col for col in clean_df.columns if col in ['residual', 'high_freq'] or col.startswith('component_')]}")
    
    # 获取特征列（排除分量、残差和重构列）
    feature_cols = [col for col in clean_df.columns 
                   if not any(col.startswith(prefix) 
                             for prefix in ['component_', 'imf_', 'residual', 'reconstructed'])]
    
    # 准备数据集：将分量列放在第一列，然后是wspd，最后是其他特征
    cols = [component_name, 'wspd'] + [col for col in feature_cols if col not in ['wspd', 'TIMESTAMP']]
    comp_df = clean_df[cols].copy()
    
    # 设置时间戳为索引
    comp_df = comp_df.set_index(pd.to_datetime(clean_df['TIMESTAMP']))
    comp_df.index.name = 'TIMESTAMP'
    
    # # 根据分量类型添加合适的时间特征
    # if component_name == 'high_freq':
    #     # 高频分量使用3秒和7秒周期特征
    #     comp_df = add_time_features(comp_df, periods=[3,7])
    # elif component_name == 'residual':
    #     # 低频分量使用7秒、14秒和60秒周期特征
    #     comp_df = add_time_features(comp_df, periods=[7,14,60])
    
    
    return comp_df

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理数据集，包括特征工程和分解
    
    参数:
        df: 输入的DataFrame 
        periods: 时间特征周期列表
        
    返回:
        处理后的数据集DataFrame
    """ 
    # step 1: 特征工程
    df_basic = data_feature_engineering(df)
    
    # step 2: 数据分解
    result_df = decompose_wind_data(df_basic)

    # step 3: 保存分解结果
    comp_df = save_component_datasets(result_df, 'residual')
    
    return comp_df

def process_features_x_data_with_features_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理数据集，包括特征工程和分解
    
    参数:
        df: 输入的DataFrame 
        periods: 时间特征周期列表
        
    返回:
        处理后的数据集DataFrame
    """ 
    # step 1: 特征工程
    # df_basic = data_feature_engineering(df)
    
    # step 2: 数据分解
    result_df = decompose_wind_data(df)

    # step 3: 保存分解结果
    comp_df = save_component_datasets(result_df, 'residual')
    
    return comp_df

def process_features_y_data_with_features_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理数据集，包括特征工程和分解
    
    参数:
        df: 输入的DataFrame 
        periods: 时间特征周期列表
        
    返回:
        处理后的数据集DataFrame
    """ 
    # # step 1: 特征工程
    # df_basic = data_feature_engineering(df)
    
    # step 2: 数据分解
    result_df = decompose_wind_data(df)['residual']

    # # step 3: 保存分解结果
    # comp_df = save_component_datasets(result_df, 'residual')
    
    return result_df

if __name__ == "__main__":
    data_path = "data/1.data_raw/02-liuyuan-fukong/01/liuyuan-fukong_0207-0221.pkl" 

    df = pd.read_pickle(data_path)
    print(df)
    
    print(process_features(df))