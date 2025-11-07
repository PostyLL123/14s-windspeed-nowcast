import torch
import torch.nn as nn
#2025.11.05 新增smooth项，相比原有的一阶项，看看二阶项能不能让结果更平滑
class custom_loss(nn.Module):
    
    def __init__(self, r_rmse=0.8, r_mae=0.5, r_mask1=0.8, r_mask2=0.5, r_smooth=0.4,
                 range_min=7.5, range_max=10.5):
        """
        初始化损失函数的超参数。
        
        参数:
        r_rmse (float): RMSE 的权重
        r_mae (float): MAE 的权重
        r_mask1 (float): Masked MAE 的权重
        r_mask2 (float): Trend MAE 的权重
        range_min (float): Masked MAE 的范围下限
        range_max (float): Masked MAE 的范围上限
        """
        super(custom_loss, self).__init__()
        
        # 注册权重和范围参数
        self.r_rmse = r_rmse
        self.r_mae = r_mae
        self.r_mask1 = r_mask1
        self.r_mask2 = r_mask2
        self.r_smooth = r_smooth
        self.range_min = range_min
        self.range_max = range_max

    def forward(self, y_pred, y_true):
        """
        PyTorch 的前向传播方法，用于计算损失。
        
        参数:
        y_pred (torch.Tensor): 模型的预测输出
        y_true (torch.Tensor): 真实的标签
        
        返回:
        torch.Tensor: 一个标量的总损失值
        """
        
        # 1. 计算 MAE (Mean Absolute Error)
        mae = torch.mean(torch.abs(y_true - y_pred))
        
        # 2. 计算 RMSE (Root Mean Squared Error)
        # torch.square(x) 等同于 x ** 2
        rmse = torch.sqrt(torch.mean(torch.square(y_true - y_pred)))
        
        # 3. 计算 Masked MAE
        # 创建掩码 (mask)，注意 PyTorch 中使用 & 进行逻辑与
        mask1 = (y_true >= self.range_min) & (y_true <= self.range_max)
        
        abs_error = torch.abs(y_true - y_pred)
        masked_mae1 = torch.mean(abs_error * mask1.float())
        
        # 4. 计算 Trend MAE (趋势 MAE)
        # 假设 y 的 shape 为 [batch_size, sequence_length]
        diff_y_true = y_true[:, 1:] - y_true[:, :-1]
        diff_y_pred = y_pred[:, 1:] - y_pred[:, :-1]
        
        trend_mae = torch.mean(torch.abs(diff_y_true - diff_y_pred))

        smoothness_loss = 0.0
        # 需要至少3个时间步来计算二阶差分
        if y_pred.shape[1] > 2:
            # y_pred shape: [Batch, SeqLen, Features]
            # 我们只对第一个特征（风速）进行平滑
            pred_wind = y_pred[:, :, 0] # [Batch, SeqLen]
            true_wind = y_true[:, :, 0]
            
            # 一阶差分 (速度) [Batch, SeqLen-1]
            diff1_pred = pred_wind[:, 1:] - pred_wind[:, :-1]
            # 二阶差分 (加速度) [Batch, SeqLen-2]
            diff2_pred = diff1_pred[:, 1:] - diff1_pred[:, :-1]

            diff1_true = true_wind[:, 1:] - true_wind[:, :-1]
            diff2_true = diff1_true[:, 1:] - diff1_true[:, :-1]

            
            # 计算惩罚项 (L2 范数)
            smoothness_loss = torch.mean(torch.abs(diff2_true - diff2_pred))
        
        # 5. 计算加权总损失
        loss = (self.r_mae * mae + 
                self.r_rmse * rmse + 
                self.r_mask1 * masked_mae1 + 
                self.r_mask2 * trend_mae+
                self.r_smooth*smoothness_loss)
                
        return loss