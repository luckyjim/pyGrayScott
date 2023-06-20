"""
Created on 18 juin 2023

@author: jcolley
"""
import cv2
import numpy as np


def init_gray_scott(n_x, n_y, t_float=np.float32):
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
    u = np.ones((n_x, n_y), dtype=t_float)
    v = np.zeros((n_x, n_y), dtype=t_float)
    mesh_x, step_x = np.linspace(0, 1, n_x, True, True)
    max_y = n_y * step_x
    mesh_y = np.linspace(0, max_y, n_y, True)
    x, y = np.meshgrid(mesh_x, mesh_y)
    mask = (0.4 < x) & (x < 0.6) & ((0.4 * max_y) < y) & (y < (0.6 * max_y))
    u[mask.T] = 0.70
    v[mask.T] = 0.25
    print(f"step_x={step_x}")
    return u, v, step_x


def frames_to_video(frames, file_video, fps=24):
    """
    n1 is abscisse
frames_to_video
    :param frames:
    :type frames: float (nb_frame, n1, n2)
    :param file_video:
    :type file_video: string
    """
    print(frames.shape)
    n1, n2 = frames.shape[1], frames.shape[2]
    video = cv2.VideoWriter(f"{file_video}.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, (n1, n2))
    frame_rgb = np.zeros((n2, n1, 3), dtype=np.uint8)
    for idx in range(frames.shape[0]):
        frame_rgb[:, :, 1] = frames[idx].T
        video.write(frame_rgb)
    video.release()
