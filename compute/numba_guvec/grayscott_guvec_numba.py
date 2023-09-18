"""
Created on 17 juin 2023


In [2]: numba.__version__
Out[2]: '0.56.4'


bench 1:
1920 1080
step_x=0.0005211047420531526
step_frame=34
nb_frame=500
CPU time= 213.75834463799998 s
(500, 1920, 1080)


bench 2:
step_frame=34
nb_frame=1000
CPU time= 439.44128702300003 s
(1000, 1920, 1080)

bench 3: with short vector math library (SVML) 
step_frame=34
nb_frame=1000
CPU time= 352.219058588 s
(1000, 1920, 1080)
$ numba -s | grep SVML
__SVML Information__
SVML State, config.USING_SVML                 : True
SVML Library Loaded                           : True
llvmlite Using SVML Patched LLVM              : True
SVML Operational                              : True
"""

import numpy as np
import matplotlib.pylab as plt
import numba as nb

import compute.common as gsc

kwd = {"cache": True, "fastmath": {"reassoc", "contract", "arcp"}}


@nb.guvectorize(['(float32[:,:], float32[:,:], float32[4], float32[:,:]) '],
             '(n, m)(n, m)(4) -> (n, m)',
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
def grayscott_numba_guvec(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
    pass


if __name__ == "__main__":
    n_size = 500
    U, V, _ = gsc.grayscott_init(n_size, n_size)
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 200
    frames_ui = gsc.grayscott_main(grayscott_numba_guvec, gs_pars, U, V, nb_frame)
    # last image
    plt.figure()
    plt.imshow(frames_ui[-1])
    plt.show()
