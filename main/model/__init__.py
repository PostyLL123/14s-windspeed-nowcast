import torch
import torch.optim as optim
import importlib

def build_model(model_cfg):
    if hasattr(model_cfg, 'model_name'):
        Model = importlib.import_module(f'main.model.models.{model_cfg.model_name}')
        model = Model.Model(model_cfg)
    else:
        model = Model(model_cfg)
    return model

def build_optimizer(cfg, model):
    """
    根据 config 构建优化器和学习率调度器。
    (配套函数)
    """
    optimizer = None
    scheduler = None
    
    # 从 config 中获取参数
    opt_type = cfg.opt_type
    lr = cfg.learning_rate
    weight_decay = cfg.weight_decay

    # 1. 构建优化器
    if opt_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif opt_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif opt_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=0.9, # (SGD 通常需要 momentum)
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器类型: {opt_type}")

    # 2. 构建学习率调度器 (Scheduler)
    if cfg.scheduler == 'CosineLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.num_epochs, # (T_max 设为总 epoch 数)
            eta_min=lr * 0.01 # (例如, 最小衰减到 1%)
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=cfg.step_size, # (step_size 应该是 [epoch1, epoch2, ...])
            gamma=0.1 # (例如, 每次衰减 10 倍)
        )
    elif cfg.scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"不支持的调度器类型: {cfg.scheduler}")

    return optimizer, scheduler

