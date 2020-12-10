import multiprocessing
import os
from subprocess import Popen, DEVNULL
import random

a = ['a', 'b', 'c', 'd', 'e']
random.shuffle(a)
print(a)