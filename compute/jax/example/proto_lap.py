'''
Created on 5 juin 2024

@author: jcolley
'''
from matplotlib.pylab import plt
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import correlate2d

import compute.common as gsc

def fill_inner(u):
    j1 = jnp.zeros((4,4), dtype=np.float32)
    j1[1:-1,1:-1] = u 
    return j1

def test_fill():
    uu = jnp.arange(9).reshape(3,3)
    print(uu)
    print(fill_inner(uu))
    

def grayscott_kernel_nip(Lu, Lv, u, v, Du, Dv, F, k, delta_t):
    uvv = u * v * v
    out_u = u + delta_t * ((Du * Lu - uvv) + F * (1 - u))
    out_v = v + delta_t * ((Dv * Lv + uvv) - v * (F + k))
    return out_u, out_v

def laplacian(u):
    return u[:-2, 1:-1] + u[1:-1,:-2] - 4 * u[1:-1, 1:-1] + u[1:-1, 2:] + +u[2:, 1:-1]

def plot_array(m_a, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(m_a)
    
def proto_lap():
    U, V, _ = gsc.grayscott_init(1920, 1080)
    j_u = jnp.array(U)
    j_v = jnp.array(V)
    plot_array(j_u, 'j_u')
    # compute laplacians with convolution provided by cupy
    Lu = laplacian(j_u)
    plot_array(Lu, 'Laplacian by jax')
    print(Lu.max(), Lu.min())

def grayscott_jax_lap(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=V.dtype)
    for idx_fr  in range(nb_frame):
        for _ in range(step_frame):
            pass
        frames_V[idx_fr ,:,:] = V
    return frames_V

def grayscott_inner(j_u, j_v, Du, Dv, F, k, delta_t):
    for _ in jnp.arange(34):
        # compute laplacians with convolution provided by cupy
        Lu = laplacian(j_u)
        Lv = laplacian(j_v)
        
    return  j_u, j_v


def grayscott_jax_2(U, V, Du, Dv, F, k, delta_t, nb_frame, step_frame):
    """
       * laplacian is computed with 2D convolution provided by cupy
       * update U, V with elementwise GPU kernel
    """
    # output video frames
    n_x, n_y = U.shape[0], U.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=np.float32)
    # Laplacian stencil
    stencil = jnp.array([[0, 1, 0], 
                         [1, -4, 1], 
                         [0, 1, 0]], dtype=np.float32)

    j_u = jnp.array(U)
    j_v = jnp.array(V)
    for idx_fr in range(nb_frame):
        j_u, j_v = grayscott_inner(j_u, j_v, Du, Dv, F, k, delta_t)
        frames_V[idx_fr, :, :] = j_v
    return frames_V

if __name__ == '__main__':
    test_fill()
    plt.show()
