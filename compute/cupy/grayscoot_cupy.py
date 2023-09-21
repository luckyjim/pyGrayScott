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
    hardward target GPU wiht cupy libraray :
    
       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    
    :param U: array 2D 
    :param V: array 2D 
    :param Du: float
    :param Dv: float 
    :param F: float
    :param k: float
    :param delta_t: float
    :param nb_frame: number of frame in video
    :param step_frame: copy frame each 'step_frame' finit difference iteration
    '''
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=V.dtype)
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
            Lu = conv2d_gpu(u_gpu, stl_gpu, 'same', 'fill', 0)
            Lv = conv2d_gpu(v_gpu, stl_gpu, 'same', 'fill', 0)
            grayscott_kernel(Lu, Lv, u_gpu, v_gpu, Du, Dv, F, k, delta_t)
        frames_V[idx_fr ,:,:] = v_gpu.get()
    return frames_V


if __name__ == "__main__":
    n_size = 500
    # U, V, _ = gsc.grayscott_init(n_size, n_size)
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 100
    # frames_ui = gsc.grayscott_main(grayscott_numpy, gs_pars, U, V, nb_frame)
    frames_ui = gsc.grayscott_main(grayscott_cupy, gs_pars, U, V, nb_frame)