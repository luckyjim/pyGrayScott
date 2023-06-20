"""
Created on 17 juin 2023

from https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
"""
import time

import numpy as np
import matplotlib.pylab as plt

import compute.common as gsc


def periodic_bc(u):
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]


def laplacian(u):
    """
    second order finite differences
    """
    return u[:-2, 1:-1] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1] + u[1:-1, 2:] + +u[2:, 1:-1]


def grayscott_numpy(U, V, Du, Dv, F, k, delta_l2, delta_t):
    u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]
    Lu = laplacian(U)
    Lv = laplacian(V)
    Du_n = Du/delta_l2
    Dv_n = Dv/delta_l2
    uvv = u * v * v
    u += delta_t * (Du_n * Lu - uvv + F * (1 - u))
    v += delta_t * (Dv_n * Lv + uvv - (F + k) * v)
    periodic_bc(U)
    periodic_bc(V)
    return U, V


def grayscott_loop(U, V, delta_l, delta_t, nb_frame):
    Du, Dv = 0.1, 0.05
    #F, k = 0.0565, 0.062
    F, k = 0.03, 0.062
    delta_l2 = delta_l ** 2
    step_frame = 40
    frames = np.empty((nb_frame // step_frame, U.shape[0], U.shape[1]), dtype=np.uint8)
    print(frames.shape)
    idx_fr = -1
    for idx in range(nb_frame):
        U, V = grayscott_numpy(U, V, Du, Dv, F, k, delta_l2, delta_t)
        if idx % step_frame == 0:
            V_min = V.min()
            V_scaled = np.uint8(255 * (V - V_min / (V.max() - V_min)))
            idx_fr += 1
            frames[idx_fr] = V_scaled
    return frames


if __name__ == "__main__":
    n_size = 300
    U, V, delta_x = gsc.init_gray_scott(1920, 1080)
    t0 = time.process_time()
    # redfinition de delta_x pour convergence
    delta_x = 1
    delta_t = 1
    nb_iteration = 40 * 10
    frames = grayscott_loop(U, V, delta_x,delta_t , nb_iteration)
    duration = time.process_time() - t0
    print(f"CPU time= {duration} s")
    fps = frames.shape[0]/(nb_iteration*delta_t)
    print(f"Total time {nb_iteration*delta_t}, nb frames {frames.shape[0]}")
    print(f"fps={fps}")
    gsc.frames_to_video(frames, "gs_numpy", 25)
    # last image
    plt.figure()
    plt.imshow(frames[0])
    # last image
    plt.figure()
    plt.imshow(frames[-1])
    plt.show()
