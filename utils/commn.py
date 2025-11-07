import os
import torch
import torch.distributed as dist


def dist_judge():
    if torch.distributed.is_initialized():
        # 分布式训练模式
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else None

        if world_size > 1:
            # 多节点多卡训练
            print(
                f"Multi-node multi-GPU training. Global Rank: {global_rank}, World Size: {world_size}, Local Rank: {local_rank}")
        else:
            # 单机多卡训练
            print(
                f"Single-node multi-GPU training. Global Rank: {global_rank}, World Size: {world_size}, Local Rank: {local_rank}")
        return True
    else:
        # 非分布式训练
        print("Single-GPU training.")
        return False


def distributed():
    num_gpus = int(os.environ['WORLD_SIZE']) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    return distributed


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if 'LOCAL_RANK' not in os.environ:
        return get_rank()
    else:
        return int(os.environ['LOCAL_RANK'])


def synchronize():
    '''
    Helper function to synchronize among all processes when using distributed training
    '''
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def __mkdir__(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            pass


import os


class FileUtils(object):
    def __init__(self):
        super().__init__()

    pass

    @staticmethod
    def makedir(dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    @staticmethod
    def makefile(dirs, filename):
        f = open(os.path.join(dirs, filename), "a")
        f.close()

    def make_updir(file_name):
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


import time
import datetime


class TimeCounter:
    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.epoch_iters = epoch_iters
        self.start_time = None

    def reset(self):
        self.start_time = time.time()

    def step(self, epoch, batch):
        used = time.time() - self.start_time
        finished_batch_nums = (epoch - self.start_epoch) * self.epoch_iters + batch
        batch_time_cost = used / finished_batch_nums
        total = (self.num_epochs - self.start_epoch) * self.epoch_iters * batch_time_cost
        left = total - used
        return str(datetime.timedelta(seconds=left))


def folder_mkdir(file_path):
    if os.path.exists(file_path) is False and get_rank() == 0:
        try:
            os.makedirs(file_path)
        except:
            print(f'[LOCAL RANK {get_rank()}] {file_path} is  exist')


import shutil


def save_config(module_name, work_dir):
    '''
    Helper function to save the config setting
    '''
    log_path = os.path.join(work_dir, 'config')
    folder_mkdir(work_dir)
    folder_mkdir(log_path)
    if get_rank() == 0:
        # datestr = datetime.datetime.now().strftime('%Y%m%d%H')
        datestr = (datetime.datetime.now()).strftime("%Y%m%d%H")
        save_path = os.path.join(log_path, f"{datestr}_config.py")
        ori_config_path = module_name.replace('.', '/') + '.py'
        shutil.copyfile('/home/luoew/project/nowcasting/'+ori_config_path, save_path)    