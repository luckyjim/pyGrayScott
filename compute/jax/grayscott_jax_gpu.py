"""
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d as conv2d_gpu
import numpy as np

import compute.common as gsc

flag_gpu = False

if flag_gpu:
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
        '--xla_gpu_enable_async_collectives=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
    )


def grayscott_kernel(Lu, Lv, u, v, Du, Dv, F, k, delta_t):
    uvv = u * v * v
    u += delta_t * ((Du * Lu - uvv) + F * (1 - u))
    v += delta_t * ((Dv * Lv + uvv) - v * (F + k))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def grayscott_jax(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    '''
    Solve Gray-Scott equation with finite difference,
    hardward target GPU wiht cupy library :
    
       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    '''
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = jnp.empty((nb_frame, n_x, n_y), dtype=jnp.float32)
    # Laplacian stencil
    stencil = jnp.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]],
                        dtype=jnp.float32)
    # Load in GPU memory
    stl_gpu = jax.device_put(stencil)
    # u_gpu = jax.device_put(U)
    # v_gpu = jax.device_put(V)
    u_gpu = U
    v_gpu = V
    for idx_fr in range(nb_frame):
        print(f"process frame {idx_fr}")
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            Lu = conv2d_gpu(u_gpu, stl_gpu, 'same', 'fill', 0)
            Lv = conv2d_gpu(v_gpu, stl_gpu, 'same', 'fill', 0)
            #grayscott_kernel(Lu, Lv, u_gpu, v_gpu, Du, Dv, F, k, delta_t)
            uvv = u_gpu * v_gpu * v_gpu
            u_gpu += delta_t * ((Du * Lu - uvv) + F * (1 - u_gpu))
            v_gpu += delta_t * ((Dv * Lv + uvv) - v_gpu * (F + k))

        # x = x.at[idx].set(y)
        # frames_V[idx_fr ,:,:] = jax.device_get(v_gpu)
        #frames_V = frames_V.at[idx_fr].set(jax.device_get(v_gpu))
        frames_V = frames_V.at[idx_fr].set(v_gpu)
    print(f"grayscott_jax END, shape out {frames_V.shape}")
    return frames_V


if __name__ == "__main__":
    U, V, _ = gsc.grayscott_init(1920, 1080)
    U_jax = jnp.array(U)
    V_jax = jnp.array(V)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 3
    frames_ui = gsc.grayscott_main(grayscott_jax, gs_pars, U_jax, V_jax, nb_frame)
