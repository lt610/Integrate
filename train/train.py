import os
import time
from subprocess import Popen, DEVNULL
from multiprocessing import Process
import pynvml
import torch.nn.functional as F
import numpy as np
import torch as th
import random


def train(model, graph, features, labels, train_mask, optimizer):
    model.train()
    logits = model(graph, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model


def print_tv():
    pass


def print_tvt(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, cva, cta):
    print("Epoch {:04d} | Train Loss {:.4f} | Val Loss {:.4f} | Test Loss {:.4f} | train Acc {:.2f} | "
          "Val Acc {:.2f} | Test Acc {:.2f} | Cur {:.2f}({:.2f})".format(epoch, train_loss, val_loss, test_loss,
                                                                          train_acc*100, val_acc*100, test_acc*100,
                                                                          cva*100, cta*100))


def log_tvt(ex, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    ex.log_scalar('train_loss', train_loss, epoch)
    ex.log_scalar('val_loss', val_loss, epoch)
    ex.log_scalar('test_loss', test_loss, epoch)
    ex.log_scalar('train_acc', train_acc*100, epoch)
    ex.log_scalar('val_acc', val_acc*100, epoch)
    ex.log_scalar('test_acc', test_acc*100, epoch)


def print_log_tvt(ex, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, cva, cta):
    log_tvt(ex, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
    print_tvt(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, cva, cta)


def print_split(content="-" * 10, n=45):
    print("\n{} {} {}\n".format("-" * n, content, "-" * n))


def generate_random_seeds(seed, nums):
    random.seed(seed)
    return [random.randint(1, 999999999) for _ in range(nums)]


def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True


def get_gpu_proc_num(gpu=0, max_proc_num=2):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    process = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return len(process)


def get_free_gpu(gpus=[0], max_proc_num=2, max_wait=100):
    waited = 0
    while True:
        for i in range(max_proc_num):
            for gpu in gpus:
                if get_gpu_proc_num(gpu) == i:
                    return gpu
        time.sleep(10)
        waited += 10
        if waited > max_wait:
            raise Exception("There is no free gpu.")


def exec_cmd(cmd):
    print("Running cmd: {}".format(cmd))
    proc = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    proc.wait()


# 等价于用&&拼接命令行，但是可以多开几个进程运行，从而实现并行化
def exec_cmds(cmds):
    for cmd in cmds:
        exec_cmd(cmd)


def parallel_exec_cmds(parallel_proc_num, wait_time, cmds):
    if parallel_proc_num > len(cmds):
        parallel_proc_num = len(cmds)

    procs = []
    gap = int(len(cmds) / parallel_proc_num + 0.5)
    for i in range(parallel_proc_num):
        start, end = i * gap, min(len(cmds), (i+1)*gap)
        if start >= len(cmds):
            break
        batch_cmds = cmds[start:end]
        procs.append(Process(target=exec_cmds, args=(batch_cmds, )))
    for proc in procs:
        proc.start()
        time.sleep(wait_time)
    for proc in procs:
        proc.join()



# # 留作纪念吧，果然还是不能随便造轮子
# def get_gpus_info():
#     gpu_processes = os.popen('nvidia-smi | grep python').read().strip()
#     if gpu_processes != "":
#         gpu_processes = gpu_processes.split("\n")
#     else:
#         gpu_processes = []
#     return gpu_processes
#
#
# # 后续还可以加上对memory, util等的估计
# # 还可以加上cpu
# # 这个方法主要用于debug场景，因此不严格考察进程正在加载的情形
# def get_one_free_gpu(gpus=[0, 1, 2, 3], max_proc_num=2):
#     gpu_processes = get_gpus_info()
#     used_gpus_pros = dict.fromkeys(gpus, 0)
#
#     for s in gpu_processes:
#         id = int(s.split()[1])
#         used_gpus_pros[id] += 1
#
#     for i in range(max_proc_num):
#         for gpu in gpus:
#             if used_gpus_pros[gpu] == i:
#                 return gpu
#     raise Exception("There is no free gpu.")
#
# # 这里用于长时间实验，因此会多花一些时间仔细检查进程正在加载的情形
# def get_free_gpus(gpus=[0, 1, 2, 3], max_proc_num=2):
#     print("正在寻找最有可能空闲的GPU...")
#     slice = 6
#     free_gpus_by_time = []
#     for _ in range (slice):
#         free_gpus = set()
#         gpu_processes = get_gpus_info()
#         used_gpus_pros = dict.fromkeys(gpus, 0)
#         for s in gpu_processes:
#             id = int(s.split()[1])
#             used_gpus_pros[id] += 1
#
#         for i in range(max_proc_num):
#             for gpu in gpus:
#                 if used_gpus_pros[gpu] == i:
#                     free_gpus.add(gpu)
#
#         free_gpus_by_time.append(free_gpus)
#         time.sleep(0.5)
#     final_free_gpus = free_gpus_by_time[0]
#     for i in range(1, slice):
#         final_free_gpus &= free_gpus_by_time[i]
#     return list(final_free_gpus)


