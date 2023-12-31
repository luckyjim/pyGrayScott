"""
Created on 17 juin 2023




"""
import time

import numpy as np
import matplotlib.pylab as plt

import compute.common as gsc

# def grayscott_naive(m_U, m_V, Du, Dv, Feed, Kill, delta_t, nb_frame, step_frame):
#     n_x, n_y = m_U.shape[0], m_U.shape[1]
#     alpha_u = -Feed - 4 * Du
#     alpha_v = -Feed - Kill - 4 * Dv
#     dtFeed = delta_t * Feed 
#     frames_V = np.empty((nb_frame, n_x, n_y), dtype=m_V.dtype)
#     range_i = range(1, n_x - 1)
#     range_j = range(1, n_y - 1)
#     range_step = range(step_frame)
#     c_U = np.empty_like(m_U)
#     c_V = np.empty_like(m_U)
#     for idx_f in range(nb_frame):
#         for _ in range_step:
#             c_U[:, :] = m_U
#             c_V[:, :] = m_V
#             for li in range_i:
#                 for lj in range_j:
#                     uvv = c_U[li, lj] * (c_V[li, lj] ** 2)
#                     dudt = ((c_U[li, lj] * alpha_u) - uvv) + Du * (
#                         c_U[li - 1, lj] + c_U[li + 1, lj] + c_U[li, lj - 1] + c_U[li, lj + 1]
#                     )
#                     m_U[li, lj] += ((delta_t * dudt) + dtFeed)
#                     dvdt = ((c_V[li, lj] * alpha_v) + uvv) + Dv * (
#                         c_V[li - 1, lj] + c_V[li + 1, lj] + c_V[li, lj - 1] + c_V[li, lj + 1]
#                     )
#                     m_V[li, lj] += delta_t * dvdt
#         frames_V[idx_f,:,:] = m_V
#     return frames_V



def grayscott_loop(U, V, nb_frame, step_frame=34):
    Du, Dv = 0.1, 0.05
    F, k = 0.0565, 0.062
    #F, k = 0.03, 0.062
    step_frame = 34
    delta_t = 1
    print(f"step_frame={step_frame}")
    print(f"nb_frame={nb_frame}")
    t0 = time.process_time()
    frames_V = grayscott_naive(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame)
    duration = time.process_time() - t0
    print(f"CPU time= {duration} s")
    frames_ui = np.empty((nb_frame, U.shape[0], U.shape[1]), dtype=np.uint8)
    for idx in range(nb_frame):
        V = frames_V[idx]
        V_min = V.min()
        V_scaled = np.uint8(255 * (V - V_min / (V.max() - V_min)))
        frames_ui[idx] = V_scaled
    return frames_ui


if __name__ == "__main__":
    n_size = 300
    U, V, _ = gsc.grayscott_init(n_size, n_size)
    #U, V, _ = gsc.grayscott_init(1920, 1080)
    frames = grayscott_loop(U, V, 5)
    gsc.frames_to_video(frames, "gray_scott_python")
    # last image
    plt.figure()
    plt.imshow(frames[-1])
    plt.show()
