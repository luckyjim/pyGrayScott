"""
Created on 17 juin 2023

from https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
"""

import numpy as np

import compute.common as gsc


def periodic_bc(u):
    u[0,:] = u[-2,:]
    u[-1,:] = u[1,:]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]


def laplacian(u):
    return u[:-2, 1:-1] + u[1:-1,:-2] - 4 * u[1:-1, 1:-1] + u[1:-1, 2:] + +u[2:, 1:-1]


def grayscott_core(U, V, Du, Dv, F, k, delta_t):
    u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]
    Lu = laplacian(U)
    Lv = laplacian(V)
    uvv = u * v * v
    u += delta_t * (Du * Lu - uvv + F * (1 - u))
    v += delta_t * (Dv * Lv + uvv - (F + k) * v)
    periodic_bc(U)
    periodic_bc(V)
    return U, V


def grayscott_numpy(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=V.dtype)
    for idx_fr  in range(nb_frame):
        for _ in range(step_frame):
            U, V = grayscott_core(U, V, Du, Dv, F, k, delta_t)
            frames_V[idx_fr ,:,:] = V
    return frames_V


if __name__ == "__main__":
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 100
    frames_ui = gsc.grayscott_main(grayscott_numpy, gs_pars, U, V, nb_frame)
    
