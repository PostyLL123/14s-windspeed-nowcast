import torch 
from collections import OrderedDict
import os
import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def __mkdir__(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            pass

def load_checkpoint(checkpoint_path, model):
    """
    (修正版)
    正确加载一个 model.state_dict() 保存的 .pth 文件。
    """
    print(f"--- 正在加载模型检查点: {checkpoint_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"!! 加载 checkpoint 文件失败: {e} !!")
        return model

    new_state_dict = OrderedDict()
    is_ddp = False
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除 'module.'
            is_ddp = True
        else:
            new_state_dict[k] = v
    
    if is_ddp:
        print("检测到 'module.' 前缀 (DDP 模型), 已自动移除。")

    try:
        model.load_state_dict(new_state_dict)
        model.to(device) 
        model.eval()     
        print("模型参数加载成功并已切换到 .eval() 模式。")
    except Exception as e:
        print(f"!! 加载 state_dict 到模型时失败: {e} !!")
        
    return model


def save_checkpoint(file_name, model, epoch, optimizer=None, scheduler=None):

    #__mkdir__(file_name)
    save_dict = OrderedDict()
    if get_rank() == 0:
        if hasattr(model, 'module'):
            model = model.module
        save_dict['model'] = model.state_dict()
        save_dict['plan'] = dict(epoch=epoch)
        if optimizer is not None:
            save_dict['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            save_dict['scheduler'] = scheduler.state_dict()
        torch.save(save_dict, file_name)