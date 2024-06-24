"""
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""
import timeit
import scipy
import cupy as cp
import numpy as np
from cupyx.profiler import benchmark
import matplotlib.pyplot as plt

from cupyx.scipy.signal import convolve2d as convolve2d
from cupyx.scipy.signal import convolve as convolve


ima = scipy.datasets.ascent()
ima_noise = ima + np.random.normal(0,20, ima.shape)
ima_noise_D = cp.asarray(ima_noise)
ker = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=cp.float32)
#ker /= ker.sum()
if True:
    a_ima = np.array([ima_noise, ima_noise])
    a_ima.shape
    a_ker = np.array([ker, ker])
    a_ima.shape
    ker_D = cp.asarray(a_ker)
    ima_D = cp.asarray(a_ima)
    ima_dx_D = convolve(ima_noise_D, ker_D)
    ima_dx_D.shape

if False:
    ima_dx_D = convolve2d(ima_noise_D, ker_D, 'same','symm')
    ima_dx2_D = convolve2d(ima_dx_D, ker_D, 'same','symm')    
    plt.figure()
    plt.imshow(ima_noise_D.get())
    
    plt.figure()
    plt.imshow(ima_dx2_D.get())
    plt.show()
    
    



