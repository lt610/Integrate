import multiprocessing
import os
from subprocess import Popen, DEVNULL
import random

import dgl
import torch as th
from util.data_util import load_file_data
import numpy as np
import pandas as pd
import time
from dgl import function as fn
import tqdm

# graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes = load_data("cora")
# print(features.shape)
# graph.ndata['h'] = features
# graph.apply_edges(fn.u_dot_v("h", "h", "dot_"))
# w = graph.edata["dot_"]
# print(w.shape)


def log_metric(epoch, degree=4, **metric):
    info = "Epoch {:04d}".format(epoch)
    for key, value in metric.items():
        info += " | {} {}".format(key, value)
    print(info)

print([i for i in range(65, 101)])




