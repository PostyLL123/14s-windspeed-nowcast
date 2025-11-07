import numpy as np
import pandas as pd
import pickle
from typing import Union, List, Tuple, Dict
import os

class DataNormalizer:
    """数据标准化器，支持多种标准化方式和不同形状的数据"""
    def __init__(self, method: str = 'standard', feature_range: Tuple[float, float] = (-1, 1)):
        """
        初始化标准化器
        
        参数:
            method: 标准化方法，可选 'standard'（标准化）, 'minmax'（最小最大化）, 'robust'（稳健化）
            feature_range: 用于minmax方法的目标范围
        """
        self.method = method
        self.feature_range = feature_range
        self.params = {}
        self._is_fitted = False
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame], columns: List[str] = None) -> None:
        """
        计算标准化参数
        
        参数:
            data: 输入数据，可以是numpy数组或pandas DataFrame
            columns: 要标准化的列名列表（仅用于DataFrame）
        """
        # 将数据转换为numpy数组
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.columns.tolist()
            data = data[columns].values
        
        if self.method == 'standard':
            self.params['mean'] = np.nanmean(data, axis=0)
            self.params['std'] = np.nanstd(data, axis=0)
            self.params['std'][self.params['std'] == 0] = 1.0  # 避免除零
            
        elif self.method == 'minmax':
            self.params['min'] = np.nanmin(data, axis=0)
            self.params['max'] = np.nanmax(data, axis=0)
            self.params['range'] = self.params['max'] - self.params['min']
            self.params['range'][self.params['range'] == 0] = 1.0  # 避免除零
            
        elif self.method == 'robust':
            self.params['median'] = np.nanmedian(data, axis=0)
            self.params['q1'] = np.nanpercentile(data, 25, axis=0)
            self.params['q3'] = np.nanpercentile(data, 75, axis=0)
            self.params['iqr'] = self.params['q3'] - self.params['q1']
            self.params['iqr'][self.params['iqr'] == 0] = 1.0  # 避免除零
            
        self._is_fitted = True
        
    def transform(self, data: Union[np.ndarray, pd.DataFrame], columns: List[str] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        应用标准化转换
        
        参数:
            data: 输入数据
            columns: 要标准化的列名列表（仅用于DataFrame）
            
        返回:
            标准化后的数据
        """
        if not self._is_fitted:
            raise ValueError("请先调用fit方法")
        
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            if columns is None:
                columns = data.columns.tolist()
            values = data[columns].values
        else:
            values = data.copy()
        
        if self.method == 'standard':
            values = (values - self.params['mean']) / self.params['std']
            
        elif self.method == 'minmax':
            values = (values - self.params['min']) / self.params['range']
            values = values * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
            
        elif self.method == 'robust':
            values = (values - self.params['median']) / self.params['iqr']
        
        if is_dataframe:
            data = data.copy()
            data[columns] = values
            return data
        return values
    
    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame], columns: List[str] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        反向转换标准化的数据
        
        参数:
            data: 标准化后的数据
            columns: 要反向转换的列名列表（仅用于DataFrame）
            
        返回:
            原始尺度的数据
        """
        if not self._is_fitted:
            raise ValueError("请先调用fit方法")
        
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            if columns is None:
                columns = data.columns.tolist()
            values = data[columns].values
        else:
            values = data.copy()
        
        if self.method == 'standard':
            values = values * self.params['std'] + self.params['mean']
            
        elif self.method == 'minmax':
            values = (values - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            values = values * self.params['range'] + self.params['min']
            
        elif self.method == 'robust':
            values = values * self.params['iqr'] + self.params['median']
        
        if is_dataframe:
            data = data.copy()
            data[columns] = values
            return data
        return values
    
    def save(self, filepath: str) -> None:
        """保存标准化器参数"""
        save_dict = {
            'method': self.method,
            'feature_range': self.feature_range,
            'params': self.params,
            '_is_fitted': self._is_fitted
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataNormalizer':
        """加载标准化器参数"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        normalizer = cls(method=save_dict['method'], feature_range=save_dict['feature_range'])
        normalizer.params = save_dict['params']
        normalizer._is_fitted = save_dict['_is_fitted']
        return normalizer

def create_normalizer(data: Union[np.ndarray, pd.DataFrame], 
                     method: str = 'standard',
                     feature_range: Tuple[float, float] = (-1, 1),
                     columns: List[str] = None) -> Tuple[DataNormalizer, Union[np.ndarray, pd.DataFrame]]:
    """
    创建并应用标准化器
    
    参数:
        data: 输入数据
        method: 标准化方法
        feature_range: 用于minmax方法的目标范围
        columns: 要标准化的列名列表（仅用于DataFrame）
    
    返回:
        normalizer: 标准化器实例
        normalized_data: 标准化后的数据
    """
    normalizer = DataNormalizer(method=method, feature_range=feature_range)
    normalizer.fit(data, columns=columns)
    normalized_data = normalizer.transform(data, columns=columns)
    return normalizer, normalized_data

def save_normalizer(normalizer: DataNormalizer, filepath: str) -> None:
    """保存标准化器"""
    normalizer.save(filepath)

def load_normalizer(filepath: str) -> DataNormalizer:
    """加载标准化器"""
    return DataNormalizer.load(filepath)
