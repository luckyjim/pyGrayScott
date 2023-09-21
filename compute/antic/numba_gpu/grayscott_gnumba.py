"""
Created on 17 juin 2023

from https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
"""

import numpy as np
import numba as nb


import compute.common as gsc

kwd = {"cache": True, "fastmath": {"reassoc", "contract", "arcp"}}

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
    idx_fr = -1
    for idx_fr  in range(nb_frame):
        for _ in range(step_frame):
            U, V = grayscott_core(U, V, Du, Dv, F, k, delta_t)
            frames_V[idx_fr ,:,:] = V
    return frames_V

#
# with @stencil
#

@nb.stencil
def stencil_lap4(u):
    return u[0, 1] + u[1, 0] + u[-1, 0] + u[0, -1] - 4 * u[0, 0]

@nb.njit(fastmath=True, parallel=True)
def grayscott_stencil_numba_par(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
    # Init constante
    n_x, n_y = m_U.shape[0], m_U.shape[1]
    # alloc
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=m_V.dtype)
    range_step = range(step_frame)
    for idx_f in range(nb_frame):
        for _ in range_step:
            uvv = m_U * m_V * m_V
            m_U += delta_t * (Du * stencil_lap4(m_U) - uvv + Feed * (1 - m_U))
            m_V += delta_t * (Dv * stencil_lap4(m_V) + uvv - (Feed + Kill) * m_V)
        frames_V[idx_f,:,:] = m_V
    return frames_V

@nb.njit(fastmath={"reassoc", "contract", "arcp"})
def grayscott_stencil_numba(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
    # Init constante
    n_x, n_y = m_U.shape[0], m_U.shape[1]
    # alloc
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=m_V.dtype)
    range_step = range(step_frame)
    for idx_f in range(nb_frame):
        for _ in range_step:
            uvv = m_U * m_V * m_V
            m_U += delta_t * (Du * stencil_lap4(m_U) - uvv + Feed * (1 - m_U))
            m_V += delta_t * (Dv * stencil_lap4(m_V) + uvv - (Feed + Kill) * m_V)
        frames_V[idx_f,:,:] = m_V
    return frames_V
  


if __name__ == "__main__":
    n_size = 500
    #U, V, _ = gsc.grayscott_init(n_size, n_size)
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 100
    #frames_ui = gsc.grayscott_main(grayscott_numpy, gs_pars, U, V, nb_frame)
    frames_ui = gsc.grayscott_main(grayscott_stencil_numba, gs_pars, U, V, nb_frame)
