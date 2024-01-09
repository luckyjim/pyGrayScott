"""
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""
import jax
import numpy as np 
import compute.common as gsc

from jax.scipy.signal import convolve2d as conv2d_gpu

import os
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


def grayscott_jax_gpu(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    '''
    Solve Gray-Scott equation with finite difference,
    hardward target GPU wiht cupy library :
    
       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    '''
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=np.float32)
    # Laplacian stencil
    stencil = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]],
                        dtype=np.float32)
    # Load in GPU memory
    stl_gpu = jax.device_put(stencil)
    u_gpu = jax.device_put(U)
    v_gpu = jax.device_put(V)
    for idx_fr  in range(nb_frame):
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            Lu = conv2d_gpu(u_gpu, stl_gpu, 'same', 'fill', 0)
            Lv = conv2d_gpu(v_gpu, stl_gpu, 'same', 'fill', 0)
            grayscott_kernel(Lu, Lv, u_gpu, v_gpu, Du, Dv, F, k, delta_t)
        frames_V[idx_fr ,:,:] = jax.device_get(v_gpu)
    return frames_V


if __name__ == "__main__":
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 100
    frames_ui = gsc.grayscott_main(grayscott_jax_gpu, gs_pars, U, V, nb_frame)
