import time
from datetime import datetime

import cv2
import numpy as np
#import matplotlib.pylab as plt


def grayscott_init(n_x, n_y, t_float=np.float32):
    """
    init rectangle
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
    
    if name  == "Labyrinthine":
        l_pars = [0.1,0.05,0.037,0.06]
    elif name  == "Spots":
        l_pars = [0.1,0.05,0.014,0.054]
    elif name  == "Pulsating spots":
        l_pars = [0.1,0.05,0.025,0.06]        
    elif name  == "Worms":
        l_pars = [0.1,0.05,0.078,0.061]        
    elif name  == "Holes":
        l_pars = [1,1,0.039,0.058]
    elif name  == "Spatiotemporal chaos":
        l_pars = [1,1,0.026,0.051]
    elif name  == "Intermittent chaos/holes":
        l_pars = [1,1,0.034,0.056]
    else:
        l_pars = [0.1,0.05,0.0565,0.062]
    d_pars = {}
    d_pars["Du"] = l_pars[0]
    d_pars["Dv"] = l_pars[1]
    d_pars["feed"] = l_pars[2]
    d_pars["kill"] = l_pars[3]
    return d_pars


def frames_to_video(frames, file_video, fps=24):
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
    * measure time to compute Gray-Scott
    * convert array frame to video with name of method, size image, time compute
    '''
    Du = gs_pars["Du"]
    Dv = gs_pars["Dv"]
    F = gs_pars["feed"]
    k = gs_pars["kill"]
    delta_t = 1
    print(f"step_frame={step_frame}")
    print(f"nb_frame={nb_frame}")
    print(func_grayscott.__name__)
    t_cpu = time.process_time()
    t_wall = datetime.now()
    frames_v_ar = func_grayscott(u_ar, v_ar, Du, Dv, F, k, delta_t, nb_frame, step_frame)
    duration_cpu = time.process_time() - t_cpu
    duration_wall = datetime.now()-t_wall
    print(f"CPU time= {duration_cpu} s")
    print(f"Wall time= {duration_wall} s") 
    frames_ui = np.empty((nb_frame, u_ar.shape[0], u_ar.shape[1]), dtype=np.uint8)
    #return frames_ui
    if frames_v_ar.dtype == np.float32:
        for idx in range(nb_frame):
            v_ar = frames_v_ar[idx]
            v_ar_min = v_ar.min()
            v_ar_max = v_ar.max()
            #print(v_ar_min, v_ar_max)
            v_ar_scaled = np.uint8(255 * (v_ar - v_ar_min / (v_ar_max - v_ar_min)))
            frames_ui[idx] = v_ar_scaled
    else:
        frames_ui = frames_v_ar
    file_video = func_grayscott.__name__ + f"_{u_ar.shape[0]}x{u_ar.shape[1]}_{nb_frame}_{int(duration_wall.total_seconds()+0.5)}"
    frames_to_video(frames_ui, file_video)
    #plt.figure()
    #plt.imshow(frames_ui[-1])
    # plt.show()    
    return frames_ui
