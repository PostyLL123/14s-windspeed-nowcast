import os
import torch
import torch.nn as nn
from collections import OrderedDict
import importlib
model_path = '/home/luoew/model_output/14s/diffusion_enhanced_s2s/model/epoch_4.pth'

module_name = 'main.configs.train.14s.diffusion_enhanced_s2s'
cfg = importlib.import_module(module_name)

if hasattr(cfg.model_cfg, 'model_name'):
    Model = importlib.import_module(f'main.model.models.{cfg.model_cfg.model_name}')
    model = Model.Model(cfg.model_cfg)

checkpoint = torch.load(model_path, map_location='cpu')

checkpoint_model_keys = [ikey for ikey in checkpoint['model'].keys()]
# check = checkpoint['model']
model_dict = model.state_dict()
check_ = OrderedDict()
for num, (k, v) in enumerate(model_dict.items()):
    if k != checkpoint_model_keys[num]:
        print('stop')
    check_[k] = checkpoint['model'][checkpoint_model_keys[num]]
model.load_state_dict(check_)
# 获取优化器的参数
optimizer = None
scheduler = None
if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
# 获取调度器的参数
if scheduler is not None:
    scheduler.load_state_dict(checkpoint['scheduler'])

if 'plan' in checkpoint:
    plan = checkpoint['plan']
else:
    plan = None