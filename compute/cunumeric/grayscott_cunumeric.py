#import numpy as np
import cunumeric as np

import compute.common as gsc


def periodic_bc(u):
    u[0,:] = u[-2,:]
    u[-1,:] = u[1,:]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]


def laplacian(u):
    """
    second order finite differences
    """
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


def grayscott_cunumeric(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=V.dtype)
    idx_fr = -1
    for idx_fr  in range(nb_frame):
        for _ in range(step_frame):
            U, V = grayscott_core(U, V, Du, Dv, F, k, delta_t)
            frames_V[idx_fr ,:,:] = V
    return frames_V


if __name__ == "__main__":
    n_size = 500
    U, V, _ = gsc.grayscott_init(n_size, n_size)
    # U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 100
    frames_ui = gsc.grayscott_main(grayscott_cunumeric, gs_pars, U, V, nb_frame)
    
