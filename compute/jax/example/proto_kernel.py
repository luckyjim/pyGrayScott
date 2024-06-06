'''
Created on 4 juin 2024

@author: jcolley
'''

from matplotlib.pylab import plt
import numpy as np
import jax.numpy as jnp
from jax import jit

import compute.common as gsc



def inplace(u, v):
    u += v
    v += 2


def compute_copy(u, v):
    ou = u+v
    ov = v+2
    return ou, ov


def test_jax_ite():
    print("=========== test_jax_ite")
    j_au= jnp.arange(16).reshape(4,4).astype(np.float32)
    j_av= jnp.ones(16).reshape(4,4).astype(np.float32)
    jit_compute_copy = jit(compute_copy)
    print(j_au)
    j_au,j_av = jit_compute_copy(j_au,j_av)
    print(j_au)
    j_au,j_av = jit_compute_copy(j_au,j_av)
    print(j_au)

def test_jax_inplace():
    print("=========== test_jax_inplace")
    j_au= jnp.arange(16).reshape(4,4).astype(np.float32)
    j_av= jnp.ones(16).reshape(4,4).astype(np.float32)
    print(j_au)
    inplace(j_au,j_av )
    print(j_au)
    
def test_np_inplace():
    print("=========== test_np_inplace")
    au= np.arange(16).reshape(4,4).astype(np.float32)
    av= np.ones(16).reshape(4,4).astype(np.float32)
    print(au)
    inplace(au,av )
    print(au)

def laplacian(u):
    return u[:-2, 1:-1] + u[1:-1,:-2] - 4 * u[1:-1, 1:-1] + u[1:-1, 2:] + +u[2:, 1:-1]


def grayscott_kernel(Lu, Lv, u, v, Du, Dv, F, k, delta_t):
    uvv = u * v * v
    u += delta_t * ((Du * Lu - uvv) + F * (1 - u))
    v += delta_t * ((Dv * Lv + uvv) - v * (F + k))

def grayscott_kernel_nip(Lu, Lv, u, v, Du, Dv, F, k, delta_t):
    uvv = u * v * v
    out_u = u + delta_t * ((Du * Lu - uvv) + F * (1 - u))
    out_v = v + delta_t * ((Dv * Lv + uvv) - v * (F + k))
    return out_u, out_v

def jit_grayscott_kernel_nip(Lu, Lv, u, v, Du, Dv, F, k, delta_t):
    return jit(grayscott_kernel_nip(Lu, Lv, u, v, Du, Dv, F, k, delta_t))

def test_kernel():
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    j_u = jnp.array(U)
    j_v = jnp.array(V)
    # j_u = U
    # j_v = V
    j_Lu = laplacian(j_u)
    j_Lv = laplacian(j_v)
    Lu = laplacian(U)
    plot_array(j_u,'array U')
    plot_array(j_Lu,'Laplacian array j_U')
    plot_array(Lu,'Laplacian array U')
    Du = gs_pars["Du"]
    Dv = gs_pars["Dv"]
    F = gs_pars["feed"]
    k = gs_pars["kill"]
    delta_t = 1
    copy_j_u = j_u.copy()
    grayscott_kernel(j_Lu, j_Lv, j_u[1:-1, 1:-1], j_v[1:-1, 1:-1], Du, Dv, F, k, delta_t)
    plot_array(j_u,'array U')
    diff = j_u-copy_j_u
    plot_array(diff,'array diff U')
    print(diff.max(), diff.min())
    

def test_kernel_nip():
    U, V, _ = gsc.grayscott_init(1920, 1080)
    gs_pars = gsc.grayscott_pars()
    j_u = jnp.array(U)
    j_v = jnp.array(V)
    # j_u = U
    # j_v = V
    j_Lu = laplacian(j_u)
    j_Lv = laplacian(j_v)
    Lu = laplacian(U)
    plot_array(j_u,'array U')
    plot_array(j_Lu,'Laplacian array j_U')
    plot_array(Lu,'Laplacian array U')
    Du = gs_pars["Du"]
    Dv = gs_pars["Dv"]
    F = gs_pars["feed"]
    k = gs_pars["kill"]
    delta_t = 1
    copy_j_u = j_u.copy()
    jit_grayscott_kernel_nip = jit(grayscott_kernel_nip)
    ou, ov = jit_grayscott_kernel_nip(j_Lu, j_Lv, j_u[1:-1, 1:-1], j_v[1:-1, 1:-1], Du, Dv, F, k, delta_t)
    plot_array(j_u,'array U')
    diff =ou-copy_j_u[1:-1, 1:-1]
    plot_array(diff,'array diff U')
    print(diff.max(), diff.min())
    
def plot_array(m_a, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(m_a)


def test_only_laplacian():
    U, V, _ = gsc.grayscott_init(1920, 1080)
    u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]
    Lu = laplacian(U)
    plot_array(Lu, "Laplacian")
    print(Lu.max(), Lu.min())
    
if __name__ == '__main__':
    #test_kernel_nip()
    #test_jax_ite()
    #test_jax_inplace()
    #test_np_inplace()
    test_only_laplacian()
    plt.show()
