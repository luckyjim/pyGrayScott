#!/usr/bin/env python3
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
    u += delta_t * ((Du * Lu - uvv) + F * (1.0 - u))
    v += delta_t * ((Dv * Lv + uvv) - v * (F + k))


def grayscott_cupy_fm(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    """
    Solve Gray-Scott equation with finite difference,
    hardward target GPU wiht cupy library :

       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    """
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = cp.empty((nb_frame, n_x, n_y), dtype=cp.float32)
    # Laplacian stencil
    #stl_gpu = cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32)
    stl_gpu = cp.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=cp.float32)
    u_gpu = cp.asarray(U)
    v_gpu = cp.asarray(V)
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            Lu = conv2d_gpu(u_gpu, stl_gpu, "same", "fill", 0)
            Lv = conv2d_gpu(v_gpu, stl_gpu, "same", "fill", 0)
            grayscott_kernel(Lu, Lv, u_gpu, v_gpu, Du, Dv, F, k, delta_t)
        frames_V[idx_fr] = v_gpu
    return frames_V.get()


def grayscott_cupy_fm2(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    """
    Solve Gray-Scott equation with finite difference,
    hardward target GPU wiht cupy library :

       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    """
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = cp.empty((nb_frame, n_x, n_y), dtype=cp.float32)
    # Laplacian stencil
    stl_gpu = cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32)
    u_gpu = cp.asarray(U)
    v_gpu = cp.asarray(V)
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            # Lu = conv2d_gpu(u_gpu, stl_gpu, 'same', 'fill', 0)
            # Lv = conv2d_gpu(v_gpu, stl_gpu, 'same', 'fill', 0)
            grayscott_kernel(
                conv2d_gpu(u_gpu, stl_gpu, "same", "fill", 0),
                conv2d_gpu(v_gpu, stl_gpu, "same", "fill", 0),
                u_gpu,
                v_gpu,
                Du,
                Dv,
                F,
                k,
                delta_t,
            )
        frames_V[idx_fr] = v_gpu
    return frames_V.get()

def grayscott_cupy_fm3(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    """
    Solve Gray-Scott equation with finite difference,
    hardward target GPU wiht cupy library :

       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    """
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = cp.empty((nb_frame, n_x, n_y), dtype=cp.float32)
    # Laplacian stencil
    #stl_gpu = cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32)
    print("3x3 1, -8")
    stl_gpu = cp.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=cp.float32)
    u_gpu = cp.asarray(U)
    v_gpu = cp.asarray(V)
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            # Lu = conv2d_gpu(u_gpu, stl_gpu, 'same', 'fill', 0)
            # Lv = conv2d_gpu(v_gpu, stl_gpu, 'same', 'fill', 0)
            grayscott_kernel(
                conv2d_gpu(u_gpu, stl_gpu, "same", "fill", 0),
                conv2d_gpu(v_gpu, stl_gpu, "same", "fill", 0),
                u_gpu,
                v_gpu,
                Du,
                Dv,
                F,
                k,
                delta_t,
            )
        frames_V[idx_fr] = v_gpu
    #print(frames_V[nb_frame-1].max())
    frames_V *= 255/0.7
    #print(frames_V[nb_frame-1].max())
    frames_int = frames_V.astype(np.uint8)
    return frames_int.get()


def grayscott_cupy_fm4(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    """
    Solve Gray-Scott equation with finite difference,
    hardward target GPU wiht cupy library :

       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    """
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = cp.empty((nb_frame, n_x, n_y), dtype=cp.uint8)
    # Laplacian stencil
    stl_gpu = cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32)
    u_gpu = cp.asarray(U)
    v_gpu = cp.asarray(V)
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            # compute laplacians with convolution provided by cupy
            # Lu = conv2d_gpu(u_gpu, stl_gpu, 'same', 'fill', 0)
            # Lv = conv2d_gpu(v_gpu, stl_gpu, 'same', 'fill', 0)
            grayscott_kernel(
                conv2d_gpu(u_gpu, stl_gpu, "same", "fill", 0),
                conv2d_gpu(v_gpu, stl_gpu, "same", "fill", 0),
                u_gpu,
                v_gpu,
                Du,
                Dv,
                F,
                k,
                delta_t,
            )
        frames_scale = v_gpu*(255/0.7)
        frames_V[idx_fr] = frames_scale.astype(cp.uint8)
    return frames_V.get()

if __name__ == "__main__":
    U, V, _ = gsc.grayscott_init(1920, 1080)
    #U, V, _ = gsc.grayscott_init(500, 500)
    name_pars = "Pulsating spots"
    #name_pars = ""
    gs_pars = gsc.grayscott_pars(name_pars)
    nb_frame = 1000
    frames_ui = gsc.grayscott_main(grayscott_cupy_fm3, gs_pars, U, V, nb_frame)
