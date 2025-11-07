import numpy as np
from typing import List, Tuple, Dict
import time
from scipy.signal import filtfilt # 使用filtfilt进行零相位滤波


__all__ = ['filter_decompose', 'decompose_signal']

def filter_decompose(data: np.ndarray, cutoff_freq: float = 1/1.5, sampling_period: float = 1.0, print_stats: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    使用低通滤波器将信号分解为高频和低频两个部分
    
    参数:
        data: 输入的时间序列数据，形状为 (n_samples,)
        cutoff_freq: 截止频率，默认为1/1.5 Hz（周期1.5秒）
        sampling_period: 采样周期，默认为1.0秒
        
    返回:
        low_freq: 低频分量（周期>1.5秒）
        [high_freq]: 高频分量（周期<1.5秒）的单元素列表
    """
    if print_stats==True:
        print(f"开始基于滤波器的信号分解...")
        print(f"数据长度: {len(data)}")
        print(f"截止频率: {cutoff_freq} Hz (周期 {1/cutoff_freq:.1f} 秒)")
        start_time = time.time()
    
    # 检查数据长度
    if len(data) < 240:  # 确保数据长度足够
        raise ValueError(f"输入数据长度({len(data)})过短，无法进行滤波分解。最小要求长度为240。")
    
    # 计算滤波器系数
    def lowpass_filter(T, freq):
        """设计二阶低通滤波器"""
        num0, num1, num2 = 1, 2, 1
        damp = 0.5  # 降低阻尼系数以减少过度平滑
        f_rwt = freq * T
        den0 = 1.0 + (4.0 * damp / f_rwt) + (4.0 / (f_rwt * f_rwt))
        den1 = 2.0 - (8.0 / (f_rwt * f_rwt))
        den2 = 1.0 - (4.0 * damp / f_rwt) + (4.0 / (f_rwt * f_rwt))
        b = np.array([num0, num1, num2])
        a = np.array([den0, den1, den2])
        return b, a
    
    # 获取滤波器系数
    b, a = lowpass_filter(sampling_period, cutoff_freq)
    
    # 为避免边缘效应，对数据进行镜像延拓
    n_extend = min(2000, len(data) // 4)  # 延拓长度
    data_extended = np.concatenate([data[n_extend-1::-1], data, data[:-n_extend-1:-1]])
    
    # 应用零相位滤波
    low_freq_extended = filtfilt(b, a, data_extended)
    
    # 提取原始长度的数据
    low_freq = low_freq_extended[n_extend:-n_extend]
    
    # 通过相减获取高频分量
    high_freq = data - low_freq
    
    # 计算耗时
    if print_stats==True:
        elapsed_time = time.time() - start_time
        print(f"分解完成！")
        print(f"耗时: {elapsed_time:.2f}秒")
    
    # 计算各分量的基本统计信息
    def print_stats(name: str, signal: np.ndarray):
        print(f"{name}统计信息:")
        print(f"- 均值: {np.mean(signal):.3f}")
        print(f"- 标准差: {np.std(signal):.3f}")
        print(f"- 最小值: {np.min(signal):.3f}")
        print(f"- 最大值: {np.max(signal):.3f}")
    
    if print_stats==True:
        print("\n分量统计:")
        print_stats("低频分量", low_freq)
        print_stats("高频分量", high_freq)
    
    # 验证重构
    reconstruction = low_freq + high_freq
    reconstruction_error = np.mean(np.abs(data - reconstruction))
    if print_stats==True:
        print(f"\n重构误差: {reconstruction_error:.6f}")
    
    return low_freq, [high_freq]

def decompose_signal(data: np.ndarray, 
                    method: str = 'eemd',
                    params: Dict = None,
                    print_stats: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    统一的信号分解接口
    
    参数:
        data: 输入的时间序列数据
        method: 分解方法，当前辅控版本代码仅可选 'filter'
        params: 分解方法的参数
        
    返回:
        residual: 残差序列
        components: 分解出的分量列表
    """
    if params is None:
        params = {}
    
    if method == 'filter':
        return filter_decompose(data, **params, print_stats=print_stats)
    else:
        raise ValueError(f"不支持的分解方法: {method}，目前只支持 'filter' 方法")

if __name__ == '__main__':
    # 示例用法
    data_path = '../data/2.data_processed/wind_basic_features-short2-0308-0320.csv'
    target_column = 'wind_speed'
    num_components = 3
    
    # 生成示例数据
    t = np.linspace(0, 10, 1000)
    data = np.sin(2 * np.pi * t) + 0.5 * np.sin(10 * np.pi * t)
    
    # 测试滤波器分解
    print("\n测试滤波器分解:")
    low_freq, [high_freq] = filter_decompose(data, cutoff_freq=1/3)
    
    # 重构信号
    reconstructed = low_freq + high_freq
    
    # 计算重构误差
    error = np.mean(np.abs(data - reconstructed))
    print(f'滤波器分解重构平均绝对误差：{error}')
