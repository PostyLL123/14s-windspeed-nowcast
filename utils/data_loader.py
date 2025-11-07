import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class WindSpeedDataset(Dataset):
    """
    用于风速预测的自定义PyTorch数据集。
    (已修改) 它加载一个或多个文件夹中的 X (输入) 和 Y (目标) Numpy数组。
    (已修改) 它会根据文件夹名 + 后缀来自动构建文件名。
    """
    def __init__(self, base_dir, folder_list, x_suffix, y_suffix, x_normalizer, y_normalizer, device=None):
        """
        初始化数据集。

        参数:
        base_dir (str): 存放所有数据文件夹的基础目录。
        folder_list (list): 要加载的数据文件夹名称列表 (例如 ['DATA-16-20240511...'])。
        x_suffix (str): X .npy 文件的后缀 (例如 '_x.npy')。
        y_suffix (str): Y .npy 文件的后缀 (例如 '_y.npy')。
        device (torch.device, 可选): 数据要加载到的设备 (例如 'cuda')。
        """
        print(f"正在从基础目录 '{base_dir}' 加载数据文件夹...")
        
        all_x_data = []
        all_y_data = []
        
        for folder in folder_list:
            # --- (新逻辑) ---
            # 根据文件夹名构建文件名
            x_filename = folder + x_suffix
            y_filename = folder + y_suffix
            # -----------------
            
            x_file_path = os.path.join(base_dir, folder, x_filename)
            y_file_path = os.path.join(base_dir, folder, y_filename)
            
            print(f"  -> 正在加载: {folder}/{x_filename} 和 {folder}/{y_filename}")
            try:
                x_data = np.load(x_file_path)
                y_data = np.load(y_file_path)
                
                # 验证样本数
                if x_data.shape[0] != y_data.shape[0]:
                    print(f"  -> 警告: 文件夹 {folder} 中的 X 和 Y 样本数不匹配! 跳过此文件夹。")
                    continue
                    
                all_x_data.append(x_data)
                all_y_data.append(y_data)
                
            except FileNotFoundError as e:
                print(f"  -> 错误: 找不到文件! {e}。跳过此文件夹。")
            except Exception as e:
                print(f"  -> 加载 {folder} 时出错: {e}。跳过此文件夹。")

        if not all_x_data or not all_y_data:
            raise RuntimeError("未能成功加载任何数据! 请检查 config 中的 'data_config' 路径。")

        # 将所有加载的数据连接成一个大数组
        combined_x = np.concatenate(all_x_data, axis=0)
        combined_y = np.concatenate(all_y_data, axis=0)
        scaled_x = x_normalizer.transform(combined_x)
        scaled_y = y_normalizer.transform(combined_y)
        
        # 将 Numpy 数组转换为 PyTorch 张量
        # 我们使用 .float() 来确保数据类型是 FloatTensor (float32)
        self.X = torch.tensor(scaled_x, dtype=torch.float32)
        self.Y = torch.tensor(scaled_y, dtype=torch.float32)
        '''
        # 将数据移动到指定设备 (如果提供)
        if device:
            self.X = self.X.to(device)
            self.Y = self.Y.to(device)
        '''
        # 验证样本数是否匹配
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"X 和 Y 的样本数不匹配! "
                             f"X 有 {self.X.shape[0]} 个样本, "
                             f"Y 有 {self.Y.shape[0]} 个样本。")

        print("数据加载完毕。")
        print(f"  输入 X 形状: {self.X.shape}")
        print(f"  目标 Y 形状: {self.Y.shape}")
        print(f"  总样本数: {self.X.shape[0]}")

    def __len__(self):
        """返回数据集中的样本总数"""
        return self.X.shape[0]

    def __getitem__(self, index):
        """
        根据索引获取一个 (x, y) 样本对
        """
        return self.X[index], self.Y[index]

def get_train_valid_dataloaders(
    data_config,
    batch_size, 
    x_normalizer,
    y_normalizer,
    num_workers=0,
    pin_memory=True
):
    """
    创建训练和验证 Dataloaders。

    参数:
    data_config (dict): 包含数据路径和文件夹信息的字典。
                        {
                            'base_dir': '/path/to/data/',
                            'train_folders': ['folder1', 'folder2'],
                            'val_folders': ['folder3'],
                            'x_suffix': '_x.npy',
                            'y_suffix': '_y.npy'
                        }
    batch_size (int): 批次大小。
    num_workers (int): 用于数据加载的子进程数。
    device (torch.device, 可选): 数据要加载到的设备。

    返回:
    (DataLoader, DataLoader): train_loader, val_loader
    """
    
    # 1. 创建训练数据集和加载器
    train_dataset = WindSpeedDataset(
        base_dir=data_config['base_dir'],
        folder_list=data_config['train_folders'],
        x_suffix=data_config['x_suffix'], # (修改)
        y_suffix=data_config['y_suffix'], # (修改)
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
    )
    
    # 如果设备是 'cuda'，pin_memory=True 会更快

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱顺序
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # 2. 创建验证数据集和加载器
    val_dataset = WindSpeedDataset(
        base_dir=data_config['base_dir'],
        folder_list=data_config['val_folders'],
        x_suffix=data_config['x_suffix'], # (修改)
        y_suffix=data_config['y_suffix'], # (修改)
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # 验证集不需要打乱顺序
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print("\nDataloaders 创建成功。")
    return train_loader, val_loader

def get_test_dataloaders(
    data_config,
    batch_size, 
    x_normalizer,
    y_normalizer,
    num_workers=0,
    pin_memory = True
):
    """
    创建训练和验证 Dataloaders。

    参数:
    data_config (dict): 包含数据路径和文件夹信息的字典。
                        {
                            'base_dir': '/path/to/data/',
                            'train_folders': ['folder1', 'folder2'],
                            'val_folders': ['folder3'],
                            'x_suffix': '_x.npy',
                            'y_suffix': '_y.npy'
                        }
    batch_size (int): 批次大小。
    num_workers (int): 用于数据加载的子进程数。
    device (torch.device, 可选): 数据要加载到的设备。

    返回:
    (DataLoader, DataLoader): train_loader, val_loader
    """
    
    # 1. 创建测试数据集和加载器
    test_dataset = WindSpeedDataset(
        base_dir=data_config['base_dir'],
        folder_list=data_config['test_folders'],
        x_suffix=data_config['x_suffix'], # (修改)
        y_suffix=data_config['y_suffix'], # (修改)
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 训练集需要打乱顺序
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    
    print("\nDataloaders 创建成功。")
    return test_loader

