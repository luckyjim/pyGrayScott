"""
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""

from functools import partial


import numpy as np
import jax.numpy as jnp
from jax import jit

import compute.common as gsc




#@partial(jit, static_argnums=(0,1,8,9))
@partial(jit,static_argnums=(2,3,4,5,6,7,8))
def grayscott_jax(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
    # Init constante
    n_x, n_y = m_U.shape[0], m_U.shape[1]
    alpha_u = -Feed - 4 * Du
    alpha_v = -Feed - Kill - 4 * Dv
    dtFeed = delta_t * Feed
    # alloc
    frames_V = jnp.zeros((nb_frame, n_x, n_y), dtype=m_V.dtype)
    range_i = range(1, n_x - 1)
    range_j = range(1, n_y - 1)
    range_step =  jnp.arange(0,int(step_frame))
    c_U = jnp.empty_like(m_U)
    c_V = jnp.empty_like(m_U)
    for idx_f in jnp.arange(0,nb_frame):
        for _ in range_step:
            #c_U.at[:,:] = m_U
            c_U = c_U.at[:,:].set(m_U)
            c_V = c_V.at[:,:].set(m_V)
            for li in range_i:
                for lj in range_j:
                    uvv = c_U[li, lj] * (c_V[li, lj] ** 2)
                    # update m_U
                    dudt = ((c_U[li, lj] * alpha_u) - uvv) + Du * (
                        c_U[li - 1, lj] + c_U[li + 1, lj] + c_U[li, lj - 1] + c_U[li, lj + 1]
                    )
                    m_U.at[li, lj].add((delta_t * dudt) + dtFeed)
                    # update m_V
                    dvdt = ((c_V[li, lj] * alpha_v) + uvv) + Dv * (
                        c_V[li - 1, lj] + c_V[li + 1, lj] + c_V[li, lj - 1] + c_V[li, lj + 1]
                    )
                    m_V.at[li, lj].add(delta_t * dvdt)
        frames_V[idx_f,:,:] = m_V
    return frames_V


#
# multi-core version 
#


if __name__ == "__main__":
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 2
    frames_ui = gsc.grayscott_main(grayscott_jax, gs_pars, U, V, nb_frame)
