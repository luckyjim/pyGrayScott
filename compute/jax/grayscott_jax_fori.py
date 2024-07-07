"""
Alice Faure
Colley Jean-Marc CNRS/IN2P3/LPNHE
"""

import timeit
import os
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax

from jax.lax import fori_loop

import compute.common as gsc


# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=True "
#     "--xla_gpu_enable_async_collectives=true "
#     "--xla_gpu_enable_latency_hiding_scheduler=true "
#     "--xla_gpu_enable_highest_priority_async_stream=true "
# )


@jax.jit
def laplacian_9_jax(u):
    """
    stencil = [[0.25, 0.5, 0.25],
               [0.5 , -3 , 0.5],
               [0.25, 0.5, 0.25]])
    :param u: tableau 2D
    """
    u = u / 2.0
    lap = u.at[:-2, 1:-1].get()
    lap = lap + u.at[1:-1, :-2].get()
    lap = lap + u.at[1:-1, 2:].get()
    lap = lap + u.at[2:, 1:-1].get()
    lap = lap - 6.0 * u.at[1:-1, 1:-1].get()
    u = u / 2.0
    lap = lap + u.at[:-2, :-2].get()
    lap = lap + u.at[2:, 2:].get()
    lap = lap + u.at[2:, :-2].get()
    lap = lap + u.at[:-2, 2:].get()
    lap_full = jnp.zeros_like(u)
    return lap_full.at[1:-1, 1:-1].set(lap)


@jax.jit
def lap_9_with_1_jax(u):
    """
    stencil 9 point with 1 around
       1, 1,1
       1,-8,1
       1, 1,1

    fill 0 border

    :param u: tableau 2D
    """
    lap = u.at[:-2, 1:-1].get()
    lap = lap + u.at[1:-1, :-2].get()
    lap = lap + u.at[1:-1, 2:].get()
    lap = lap + u.at[2:, 1:-1].get()
    lap = lap - 8.0 * u.at[1:-1, 1:-1].get()
    lap = lap + u.at[:-2, :-2].get()
    lap = lap + u.at[2:, 2:].get()
    lap = lap + u.at[2:, :-2].get()
    lap = lap + u.at[:-2, 2:].get()
    lap_full = jnp.zeros_like(u)
    return lap_full.at[1:-1, 1:-1].set(lap)



@jax.jit
def gray_scott_jax_uv(UV, Du, Dv, f, k, dt,UVV):
    
    UVV = UV[0] * UV[1] * UV[1]
    UV = UV.at[0].add((Du * lap_9_with_1_jax(UV[0]) - UVV + f * (1 - UV[0])) * dt)
    UV = UV.at[1].add((Dv * lap_9_with_1_jax(UV[1]) + UVV - UV[1] * (f + k)) * dt)
    
    return UV


@jax.jit
def compute_iter_uv(UV, UVV, Du, Dv, f, k, dt, dsteps):
    def core_gs(n, UV):
        return gray_scott_jax_uv(UV, Du, Dv, f, k, dt,UVV)
    return fori_loop(0, dsteps, core_gs, UV)


def gray_scott_fori(U, V, Du, Dv, f, k, dt, nimages, dsteps):
    """Compute the Gray-Scott diffusion reaction for each time step and save images

    Args:
        U (array of floats): The initialized U array
        V (array of floats): The initialized V array
        Du (float): Diffusion rate of the u species
        Dv (float): Diffusion rate of the v species
        f (float): Feed rate 
        k (float): Kill rate
        dt (float): Time interval between two steps
        nimages (int): Number of images to be created
        dsteps (int): Number of extra steps to be computed between images
        nrow (int): Number of rows of the arrays to be created
        ncol (int): Number of columns of the arrays to be created

    Returns:
        image_array (array of floats): 3D array to store all the images
    """
    image_array = np.zeros((nimages, U.shape[0],U.shape[1]),dtype=np.float32)
    UV = jnp.empty((2,U.shape[0],U.shape[1]),dtype=jnp.float32)
    UV = UV.at[0].set(U)
    UV = UV.at[1].set(V)
    UVV = jnp.empty_like(U,dtype=jnp.float32)
    for idx in range(nimages):
        UV = compute_iter_uv(UV,UVV, Du, Dv, f, k, dt,dsteps)
        image_array[idx] = UV[1]
            
    return image_array  




if __name__ == "__main__":
    if True:
        U, V, _ = gsc.grayscott_init(1920, 1080)
        #U, V, _ = gsc.grayscott_init(500, 500)
        gs_pars = gsc.grayscott_pars()
        nb_frame = 100
        gsc.grayscott_main(gray_scott_fori, gs_pars, U, V, nb_frame)
        
