"""
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""
import cupy as cp
import numpy as np 
import compute.common as gsc

from cupyx.scipy.signal import convolve2d as conv2d_gpu


@cp.fuse()
def grayscott_kernel(Lu, Lv, u, v, Du, Dv, F, k, delta_t):
    uvv = u * v * v
    u += delta_t * ((Du * Lu - uvv) + F * (1 - u))
    v += delta_t * ((Dv * Lv + uvv) - v * (F + k))


def grayscott_cupy(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    '''
    Solve Gray-Scott equation with finite difference,
    hardward target GPU wiht cupy library :
    
       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    '''
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=np.float32)
    # Laplacian stencile
    stencil = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]],
                        dtype=np.float32)
    # Load in GPU memory
    stl_gpu = cp.array(stencil)
    u_gpu = cp.asarray(U)
    v_gpu = cp.asarray(V)
    for idx_fr  in range(nb_frame):
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            Lu = conv2d_gpu(u_gpu, stl_gpu, 'same', 'fill', 0)
            Lv = conv2d_gpu(v_gpu, stl_gpu, 'same', 'fill', 0)
            grayscott_kernel(Lu, Lv, u_gpu, v_gpu, Du, Dv, F, k, delta_t)
        frames_V[idx_fr ,:,:] = v_gpu.get()
    return frames_V


if __name__ == "__main__":
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 100
    frames_ui = gsc.grayscott_main(grayscott_cupy, gs_pars, U, V, nb_frame)
