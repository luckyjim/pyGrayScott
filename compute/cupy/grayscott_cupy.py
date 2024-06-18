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


def grayscott_cupy_1c(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    """
    Solve Gray-Scott equation with finite difference,
    hardward target GPU wiht cupy library :

       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    """
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = cp.empty((nb_frame, n_x, n_y), dtype=np.float32)
    # Laplacian stencil
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Load in GPU memory
    stl_gpu = cp.array(stencil)
    # u_gpu = cp.asarray(U)
    # v_gpu = cp.asarray(V)
    uv_gpu = cp.zeros((n_x, 2 * n_y + 4), dtype=np.float32)
    u_gpu = uv_gpu[:, :n_y]
    u_gpu[:, :] = cp.asarray(U)
    v_gpu = uv_gpu[:, n_y + 4 :]
    v_gpu[:, :] = cp.asarray(V)

    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            Lap = conv2d_gpu(uv_gpu, stl_gpu, "same", "fill", 0)
            # Lv = conv2d_gpu(v_gpu, stl_gpu, 'same', 'fill', 0)
            grayscott_kernel(
                Lap[:, :n_y], Lap[:, n_y + 4 :], u_gpu, v_gpu, Du, Dv, F, k, delta_t
            )
        # frames_V[idx_fr ,:,:] = v_gpu.get()
        frames_V[idx_fr, :, :] = v_gpu

    return frames_V.get()


if __name__ == "__main__":
    U, V, _ = gsc.grayscott_init(1920, 1080)
    # U, V, _ = gsc.grayscott_init(500, 500)
    gs_pars = gsc.grayscott_pars()
    nb_frame = 300
    frames_ui = gsc.grayscott_main(grayscott_cupy_1c, gs_pars, U, V, nb_frame)
