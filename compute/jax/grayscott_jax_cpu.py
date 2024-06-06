"""
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""

"""
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import correlate2d

import compute.common as gsc

#@jit
def grayscott_jax_kernel(Lu, Lv, u, v, Du, Dv, F, k, delta_t):
    uvv = u * v * v
    out_u = u + delta_t * ((Du * Lu - uvv) + F * (1 - u))
    out_v = v + delta_t * ((Dv * Lv + uvv) - v * (F + k))
    return out_u, out_v


#@jit
def grayscott_inner(stencil, j_u, j_v, Du, Dv, F, k, delta_t):
    for _ in jnp.arange(34):
        # compute laplacians with convolution provided by cupy
        Lu = correlate2d(j_u, stencil, "same", "fill", 0)
        Lv = correlate2d(j_v, stencil, "same", "fill", 0)
        j_u, j_v = grayscott_jax_kernel(Lu, Lv, j_u, j_v, Du, Dv, F, k, delta_t)
    return  j_u, j_v

def grayscott_jax_2(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    """
       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    """
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=np.float32)
    # Laplacian stencil
    stencil = jnp.array([[0, 1, 0], 
                         [1, -4, 1], 
                         [0, 1, 0]], dtype=np.float32)

    j_u = jnp.array(U)
    j_v = jnp.array(V)
    for idx_fr in range(nb_frame):
        j_u, j_v = grayscott_inner(stencil, j_u, j_v, Du, Dv, F, k, delta_t)
        frames_V[idx_fr, :, :] = j_v
    return frames_V

def grayscott_jax_ok(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    """
       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    """
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=np.float32)
    # Laplacian stencil
    stencil = jnp.array([[0, 1, 0], 
                         [1, -4, 1], 
                         [0, 1, 0]], dtype=np.float32)

    j_u = jnp.array(U)
    j_v = jnp.array(V)
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            Lu = correlate2d(j_u, stencil, "same", "fill", 0)
            Lv = correlate2d(j_v, stencil, "same", "fill", 0)
            j_u, j_v = grayscott_jax_kernel(Lu, Lv, j_u, j_v, Du, Dv, F, k, delta_t)
        frames_V[idx_fr, :, :] = j_v
    return frames_V


if __name__ == "__main__":
    #U, V, _ = gsc.grayscott_init(1920, 1080)
    U, V, _ = gsc.grayscott_init(500, 500)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 300
    frames_ui = gsc.grayscott_main(grayscott_jax_2, gs_pars, U, V, nb_frame)
