"""
Created on 18 juin 2023

@author: jcolley
"""
import time

import cv2
import numpy as np
import matplotlib.pylab as plt


def grayscott_init(n_x, n_y, t_float=np.float32):
    """
    Return 2 2D-array (n_x, n_y)

    :param n_x:
    :type n_x:
    :param n_y:
    :type n_y:
    :param n_pix:
    :type n_pix:
    """
    print(n_x, n_y)
    u_ar = np.ones((n_x, n_y), dtype=t_float)
    v_ar = np.zeros((n_x, n_y), dtype=t_float)
    mesh_x, step_x = np.linspace(0, 1, n_x, True, True)
    max_y = n_y * step_x
    mesh_y = np.linspace(0, max_y, n_y, True)
    x, y = np.meshgrid(mesh_x, mesh_y)
    mask = (0.4 < x) & (x < 0.6) & ((0.4 * max_y) < y) & (y < (0.6 * max_y))
    u_ar[mask.T] = 0.70
    v_ar[mask.T] = 0.25
    print(f"step_x={step_x}")
    return u_ar, v_ar, step_x


def grayscott_pars(name=""):
    d_pars = {}
    d_pars["Du"] = 0.1
    d_pars["Dv"] = 0.05
    d_pars["feed"] = 0.0565
    d_pars["kill"] = 0.062
    if name == "point":
        d_pars["feed"] = 0.03
        d_pars["kill"] = 0.062
    return d_pars


def frames_to_video(frames, file_video, fps=24):
    '''
    
    :param frames:
    :param file_video:
    :param fps:
    '''
   
    print(frames.shape)
    n1, n2 = frames.shape[1], frames.shape[2]
    video = cv2.VideoWriter(f"{file_video}.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, (n1, n2))
    frame_rgb = np.zeros((n2, n1, 3), dtype=np.uint8)
    for idx in range(frames.shape[0]):
        frame_rgb[:,:, 1] = frames[idx].T
        video.write(frame_rgb)
    video.release()


def grayscott_main(func_grayscott, gs_pars, u_ar, v_ar, nb_frame, step_frame=34):
    '''
    :param func_grayscott:
    :param gs_pars:
    :param u_ar:
    :param v_ar:
    :param nb_frame:
    :param step_frame:
    '''
    Du = gs_pars["Du"]
    Dv = gs_pars["Dv"]
    F = gs_pars["feed"]
    k = gs_pars["kill"]
    delta_t = 1
    print(f"step_frame={step_frame}")
    print(f"nb_frame={nb_frame}")
    print(func_grayscott.__name__)
    t0 = time.process_time()
    frames_v_ar = func_grayscott(u_ar, v_ar, Du, Dv, F, k, delta_t, nb_frame, step_frame)
    duration = time.process_time() - t0
    print(f"CPu_ar time= {duration} s")
    frames_ui = np.empty((nb_frame, u_ar.shape[0], u_ar.shape[1]), dtype=np.uint8)
    for idx in range(nb_frame):
        v_ar = frames_v_ar[idx]
        v_ar_min = v_ar.min()
        v_ar_scaled = np.uint8(255 * (v_ar - v_ar_min / (v_ar.max() - v_ar_min)))
        frames_ui[idx] = v_ar_scaled
    file_video = func_grayscott.__name__ + f"_{u_ar.shape[0]}x{u_ar.shape[1]}_{nb_frame}_{int(duration+0.5)}"
    frames_to_video(frames_ui, file_video)
    plt.figure()
    plt.imshow(frames_ui[-1])
    #plt.show()    
    return frames_ui
