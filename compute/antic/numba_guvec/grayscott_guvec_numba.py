"""
C
"""

import numpy as np
import matplotlib.pylab as plt
import numba as nb

import compute.common as gsc

kwd = {"cache": True, "fastmath": {"reassoc", "contract", "arcp"}}


@nb.guvectorize([(nb.float32[:,:], nb.float32[:,:], nb.float32[4], nb.float32[:,:])],
             '(n, m),(n, m),(4) -> (n, m)',
             target='cuda')
def grayscott_gu(m_U, m_V, pars, out_array):
    # Init constante
    Du, Dv, Feed, Kill, delta_t = pars[0], pars[1], pars[2], pars[3]
    n_x, n_y = m_U.shape[0], m_U.shape[1]
    alpha_u = -Feed - 4 * Du
    alpha_v = -Feed - Kill - 4 * Dv
    dtFeed = delta_t * Feed
    # alloc
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=m_V.dtype)
    range_i = range(1, n_x - 1)
    range_j = range(1, n_y - 1)
    step_frame = 34
    range_step = range(step_frame)
    c_U = np.empty_like(m_U)
    c_V = np.empty_like(m_U)
    for idx_f in range(nb_frame):
        for _ in range_step:
            c_U[:,:] = m_U
            c_V[:,:] = m_V
            for li in range_i:
                for lj in range_j:
                    uvv = c_U[li, lj] * (c_V[li, lj] ** 2)
                    # update m_U
                    dudt = ((c_U[li, lj] * alpha_u) - uvv) + Du * (
                        c_U[li - 1, lj] + c_U[li + 1, lj] + c_U[li, lj - 1] + c_U[li, lj + 1]
                    )
                    m_U[li, lj] += (delta_t * dudt) + dtFeed
                    # update m_V
                    dvdt = ((c_V[li, lj] * alpha_v) + uvv) + Dv * (
                        c_V[li - 1, lj] + c_V[li + 1, lj] + c_V[li, lj - 1] + c_V[li, lj + 1]
                    )
                    m_V[li, lj] += delta_t * dvdt
        frames_V[idx_f,:,:] = m_V
    return frames_V


@nb.njit(**kwd)
def grayscott_numba_ufunc(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
    pars = np.array([Du, Dv, Feed, Kill, delta_t], dtype=np.float32)
    out_array = np.zeros_like(m_U)
    grayscott_gu(m_U, m_V, pars, out_array)


if __name__ == "__main__":
    n_size = 500
    U, V, _ = gsc.grayscott_init(n_size, n_size)
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 200
    frames_ui = gsc.grayscott_main(grayscott_numba_ufunc, gs_pars, U, V, nb_frame)
    # last image
    plt.figure()
    plt.imshow(frames_ui[-1])
    plt.show()
