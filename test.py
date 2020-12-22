import multiprocessing
import os
from subprocess import Popen, DEVNULL
import random

from util.data_util import load_data

seed = 42
random.seed(seed)
r1 = [random.randint(1, 999999999) for _ in range(10)]
print(r1)
