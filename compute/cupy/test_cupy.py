"""
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""
import timeit
import cupy as cp
import numpy as np
from cupyx.profiler import benchmark


def my_func(a):
    return cp.sqrt(cp.sum(a ** 2, axis=-1))


def my_func_np(a):
    return np.sqrt(np.sum(a ** 2, axis=-1))


a = cp.random.random((10, 1024, 1024))
b = np.random.random((10, 1024, 1024))


number = 500

print(benchmark(my_func, (a,), n_repeat=number))

#
total_time = timeit.timeit(lambda: my_func_np(b), number=number)
print("Time to minimize loss = %.2f" % (total_time / number * 1000), " (ms)")
