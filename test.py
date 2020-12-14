import multiprocessing
import os
from subprocess import Popen, DEVNULL
import random

from util.data_util import load_data

graph, features, labels, train_mask, \
    val_mask, test_mask, num_feats, num_classes = load_data('ogbn-arxiv')
print(features.shape)