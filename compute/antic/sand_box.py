"""
Created on 19 juin 2023

@author: jcolley
"""

import time
import numpy as np
import scipy.fft as sf
import matplotlib.pylab as plt


def laplacien_df(a_2d):
    return a_2d[:-2, 1:-1] + a_2d[1:-1, :-2] - 4 * a_2d[1:-1, 1:-1] + a_2d[1:-1, 2:] + a_2d[2:, 1:-1]

def fft_stencil_laplacian(n_x, n_y):
    c_x = n_x // 2
    c_y = n_y // 2
    cross_lap = np.zeros((n_x, n_y), dtype=np.float32)
    cross_lap[c_x, c_y] = -4
    cross_lap[c_x - 1, c_y] = 1
    cross_lap[c_x + 1, c_y] = 1
    cross_lap[c_x, c_y - 1] = 1
    cross_lap[c_x, c_y + 1] = 1
    return sf.rfft2(cross_lap)


def laplacian_df_fft(a_2d, fft_lap):
    return sf.fftshift(sf.irfft2(fft_lap*sf.rfft2(a_2d)))

if __name__ == "__main__":
    plt.show()
