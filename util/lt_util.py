import os
import time
from subprocess import Popen, DEVNULL
from multiprocessing import Process
import pynvml
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch as th
import random
from sacred import Experiment


def cal_gain(fun, param=None):
    gain = 1
    if fun is F.sigmoid:
        gain = nn.init.calculate_gain('sigmoid')
    if fun is F.tanh:
        gain = nn.init.calculate_gain('tanh')
    if fun is F.relu:
        gain = nn.init.calculate_gain('relu')
    if fun is F.leaky_relu:
        gain = nn.init.calculate_gain('leaky_relu', param)
    return gain


def log_metric(epoch, degree, **metric):
    info = "Epoch {:04d}".format(epoch)
    for key, value in metric.items():
        # 自带四舍五入功能，而且对于0.5也有处理，round对0.5就没有处理
        info += eval('" | {{}} {{:.{}f}}".format("{}", {})'.format(degree, key, value))
    print(info)


def rec_metric(ex: Experiment, epoch, degree, **metric):
    for key, value in metric.items():
        # 这里逢5不会进1
        value = round(value, degree)
        ex.log_scalar(key, value, epoch)


def log_rec_metric(ex: Experiment, epoch, degree, metric):
    rec_metric(ex, epoch, degree, **metric)
    log_metric(epoch, degree, **metric)


def log_split(content="-" * 10, n=30):
    print("\n{} {} {}\n".format("-" * n, content, "-" * n))


def generate_random_seeds(seed, nums):
    random.seed(seed)
    # return [random.randint(1, 999999999) for _ in range(nums)]
    return [random.randint(0, 233333333) for _ in range(nums)]


def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def get_gpu_proc_num(gpu=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    process = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return len(process)


def get_free_gpu(gpus=[0], max_proc_num=2, max_wait=3600):
    if th.cuda.is_available():
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
    else:
        return "cpu"


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
    # python list数组不存在越界问题，将来这里可以优化一下代码
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
