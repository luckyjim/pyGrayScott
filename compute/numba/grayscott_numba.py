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


@nb.njit(**kwd)
def grayscott_numba(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
    # Init constante
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


@nb.njit(fastmath={"reassoc", "contract", "arcp"}, cache=True, parallel=True)
def grayscott_numba_par(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
    # Init constante
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
            for li in nb.prange(1, n_x - 1):
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




# @nb.njit(**kwd)
def grayscott_vec_fma_add(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
    # Init constante
    n_x, n_y = m_U.shape[0], m_U.shape[1]
    qy_4 = (n_y - 2) // 4
    ry_4 = (n_y - 2) % 4
    alpha_u = -Feed - 4 * Du
    alpha_v = -Feed - Kill - 4 * Dv
    dtFeed = delta_t * Feed
    # alloc
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=m_V.dtype)
    range_i = range(1, n_x - 1)
    range_j4 = range(1, n_y - 5, 4)
    range_4 = range(4)
    range_step = range(step_frame)
    c_U = np.empty_like(m_U)
    c_V = np.empty_like(m_U)
    # U array for vectorial calculation
    vec_u1 = np.zeros((8, 4), dtype=m_V.dtype)
    vec_u2 = np.zeros((3, 4), dtype=m_V.dtype)
    vec_u1[4] = Dv
    vec_u1[6] = alpha_u
    vec_u2[0] = dtFeed
    vec_u2[1] = delta_t
    # V array for vectorial calculation
    vec_v1 = np.zeros((8, 4), dtype=m_V.dtype)
    vec_v1[4] = Dv
    vec_v1[6] = alpha_v
    for idx_f in range(nb_frame):
        for _ in range_step:
            c_U[:,:] = m_U
            c_V[:,:] = m_V
            for li in range_i:
                for lj4 in range_j4:
                    for lk in range_4:
                        lj = lj4 + lk
                        uvv = c_U[li, lj] * (c_V[li, lj] ** 2)
                        # fill vec_u
                        vec_u1[7, lk] = -uvv
                        vec_u1[0, lk] = c_U[li - 1, lj]
                        vec_u1[1, lk] = c_U[li + 1, lj]
                        vec_u1[5, lk] = c_U[li, lj]
                        vec_u1[2, lk] = c_U[li, lj - 1]
                        vec_u1[3, lk] = c_U[li, lj + 1]
                        # fill vec_v
                        vec_v1[7, lk] = uvv
                        vec_v1[0, lk] = c_V[li - 1, lj]
                        vec_v1[1, lk] = c_V[li + 1, lj]
                        vec_v1[5, lk] = c_V[li, lj]
                        vec_v1[2, lk] = c_V[li, lj - 1]
                        vec_v1[3, lk] = c_V[li, lj + 1]
                    # vectorial calculuation for m_U
                    vec_u1[1] += vec_u1[0]
                    vec_u1[2] += vec_u1[1]
                    vec_u1[3] += vec_u1[2]
                    vec_u1[5] = vec_u1[5] * vec_u1[6] + vec_u1[7]
                    vec_u1[5] += vec_u1[3] * vec_u1[4]
                    # do m_U[li, lj] += (delta_t * dudt) + dtFeed
                    m_U[li, lj4: lj4 + 4] = (
                        c_U[li, lj4: lj4 + 4] + vec_u2[0] + vec_u1[5] * vec_u2[1]
                    )
                    # vectorial calculuation for m_V
                    vec_v1[1] += vec_v1[0]
                    vec_v1[2] += vec_v1[1]
                    vec_v1[3] += vec_v1[2]
                    vec_v1[5] = vec_v1[5] * vec_v1[6] + vec_v1[7]
                    vec_v1[5] += vec_v1[3] * vec_v1[4]
                    # do m_V[li, lj] += (delta_t * dvdt)
                    # we used dt vec of U vec_u2[ 1]
                    m_V[li, lj4: lj4 + 4] = c_V[li, lj4: lj4 + 4] + vec_v1[5] * vec_u2[1]
        frames_V[idx_f,:,:] = m_V
    return frames_V


if __name__ == "__main__":
    n_size = 500
    #U, V, _ = gsc.grayscott_init(n_size, n_size)
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 100
    frames_ui = gsc.grayscott_main(grayscott_numba, gs_pars, U, V, nb_frame)
    # last image
    plt.figure()
    plt.imshow(frames_ui[-1])
    # plt.show()
