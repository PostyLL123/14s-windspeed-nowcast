import os
import random
import sys
sys.path.append('/')
sys.path.append('../')
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

from main.utils.commn import distributed, save_config, TimeCounter, __mkdir__
from main.utils.commn import get_local_rank
# from main.yolov3 import build_model, build_optimizer
from main.model import build_optimizer, build_model
from main.data_utils import build_dataloader
from main.api.train import train_onepart, valid
# from main.apis.test import test_onepart
# from main.apis.test import test_onepart
from main.utils.checkpoint import save_checkpoint, load_checkpoint
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"
print(device)

def draw_loss(loss_all,  epoch, cfg, name, val_loss_epochs=None):
    plt.figure()
    plt.plot(range(len(loss_all)), loss_all, label='train')
    if val_loss_epochs is not None:
        plt.plot(range(len(val_loss_epochs)), val_loss_epochs, label='valid')
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.title(f'Training Loss')
    s_name = os.path.join(cfg.work_dir, 'logs', 'imgs', f'loss_img_{name}.png')
    __mkdir__(s_name)
    plt.savefig(s_name)
    plt.close()
    cfg.logger.info(f'[Save Loss imgs Successed Epoch {epoch} ] ||| {s_name}]')


import torch.nn as nn

def __init__():
    parser = argparse.ArgumentParser()

    parser.add_argument('--configs', type=str, default='mlp')
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    args = parser.parse_args()
    module_name = 'main.configs.train.57.' + args.configs
    xconfig = importlib.import_module(module_name)

    save_config(module_name, xconfig.work_dir)
    __mkdir__(os.path.join(xconfig.work_dir, 'model', 'o.pth'))
    xconfig.logger.info('[Work Dir]: ' + xconfig.work_dir)
    xconfig.logger.info('[Configs]: ' + module_name)
    torch.manual_seed(xconfig.rand_seed)
    random.seed(xconfig.rand_seed)

    if use_gpu:
        torch.cuda.set_device(args.local_rank)
        torch.set_num_threads(2)
    # gpu = int(os.environ['LOCAL_RANK'])
    if distributed():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl',
                                world_size=world_size,
                                rank=rank,
                                init_method='env://')
    return xconfig



def main():

    cfg = __init__()
    model = build_model(cfg.model_cfg,0)
    # model = SimpleModel(72, 1, 1)
    model.cuda() if use_gpu else model.cpu()
    optimizer, scheduler = build_optimizer(cfg, model)



    start_epoch = 0
    start_part = 0
    if cfg.resume_model is not None:
        model, optimizer, scheduler, plan = load_checkpoint(cfg.resume_model, model, optimizer, scheduler)
        start_epoch = plan['epoch']
        start_part = plan['part']
        cfg.logger.info(f'[Resume Model] ||| {cfg.resume_model} ')
    elif cfg.pre_model is not None:
        if hasattr(cfg, 'start_epoch'):
            start_epoch = cfg.start_epoch
        model, _, _ , _ = load_checkpoint(cfg.pre_model, model)
        cfg.logger.info(f'[Pre Model] ||| {cfg.pre_model} ')

    if distributed():
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                broadcast_buffers=False,
                find_unused_parameters=False
        )

    train_loader = build_dataloader(cfg, cfg.train_data, cfg.batch_size, train=False)# test step 3 2189 test step 5 2189
    valid_loader = build_dataloader(cfg, cfg.test_data, cfg.test_batch_size, train=False)#test step 3 916 test step 5 589
    start_epoch = 0
    epoch_batch_num = len(train_loader)
    time_counter = TimeCounter(start_epoch, cfg.num_epochs, epoch_batch_num)
    time_counter.reset()

    batch_num = len(train_loader.batch_sampler)
    cfg.logger.info('{:#^75}'.format(' Data Information '))
    cfg.logger.info(f'[DATA INFO] ||| [All Batch Num] {batch_num} ||| [Batch Size] {cfg.batch_size} ')
    cfg.logger.info('{:#^75}'.format(' Data Information '))
    loss_epochs = []
    loss_batch = []
    val_loss_epochs = []
    correct_ls = []
    val_corr_ls = []
    for iepoch in range(start_epoch, cfg.num_epochs):
        cfg.logger.info('{:#^75}'.format(f' [Train Epoch] {iepoch} '))
        model, optimizer, scheduler, loss_item, correct = train_onepart(cfg, model, train_loader, optimizer, scheduler, iepoch, time_counter)
        train_loader.dataset.label_deal()
        save_file = os.path.join(cfg.work_dir, 'model', 'epoch_{}.pth'.format(iepoch))
        if iepoch % 5 == 0 and iepoch>0: 
            save_checkpoint(save_file, model, iepoch, optimizer=optimizer, scheduler=scheduler)
        cfg.logger.info('epoch_{}.pth'.format(iepoch) + ' Saved')
        valid_loss, val_cor = valid(cfg, model, valid_loader, iepoch)
        loss_batch.extend(loss_item)
        draw_loss(loss_batch, iepoch, cfg, 'Batches')
        loss_mean = np.asarray(loss_item).mean()
        loss_epochs.append(loss_mean)
        correct_ls.append(np.asarray(correct).mean())
        val_corr_ls.append(val_cor)
        val_loss_epochs.append(valid_loss)
        if len(loss_epochs) >= 3:
            draw_loss(loss_epochs, iepoch, cfg, 'Epoches', val_loss_epochs)
        if len(correct) >= 3:
            draw_loss(correct_ls, iepoch, cfg, 'Correct', val_corr_ls)
        cfg.logger.info(
            '[ Epoch %d ] ||| [lr: %.6f] [Loss: %.4f] [Correct: %.4f]|||  MaxMemory %dMB' %
            (iepoch,
             optimizer.param_groups[0]['lr'],
             loss_mean,
             val_cor,
             torch.cuda.max_memory_allocated(device) / 1024 ** 2))

        if cfg.scheduler == 'StepLR' or cfg.scheduler == 'CosineLR':
            if iepoch >= cfg.step_size[0]:
                scheduler.step()
    valid_loss = valid(cfg, model, valid_loader, iepoch, True)
    save_file = os.path.join(cfg.work_dir, 'model', 'epoch_{}.pth'.format(iepoch))
    # if iepoch % 5 == 0 and iepoch > 0:
    save_checkpoint(save_file, model, iepoch, optimizer=optimizer, scheduler=scheduler)


if __name__ == '__main__':

    main()