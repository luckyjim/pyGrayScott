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


import compute.common as gsc


# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=True "
#     "--xla_gpu_enable_async_collectives=true "
#     "--xla_gpu_enable_latency_hiding_scheduler=true "
#     "--xla_gpu_enable_highest_priority_async_stream=true "
# )


@jax.jit
def laplacian_9_with_1_jax_pad(u, upad):
    """
    stencil 9 point with 1 around
       1, 1,1
       1,-8,1
       1, 1,1

    fill 0 border

    :param u: tableau 2D
    """
    upad = upad.at[1:-1, 1:-1].set(u)
    lap = upad.at[:-2, 1:-1].get()
    lap += upad.at[1:-1, :-2].get()
    lap += upad.at[1:-1, 2:].get()
    lap += upad.at[2:, 1:-1].get()
    lap -= 8.0 * u
    lap += upad.at[:-2, :-2].get()
    lap += upad.at[2:, 2:].get()
    lap += upad.at[2:, :-2].get()
    return lap + upad.at[:-2, 2:].get()


@jax.jit
def gray_scott_jax_core(U, V, Du, Dv, f, k, dt, UVV, tpad):
    """ """

    UVV = U * V * V
    U += (Du * laplacian_9_with_1_jax_pad(U, tpad) - UVV + f * (1 - U)) * dt
    V += (Dv * laplacian_9_with_1_jax_pad(V, tpad) + UVV - V * (f + k)) * dt
    return U, V


def gray_scott_jax_pad(U_np, V_np, Du, Dv, f, k, dt, nb_frame, step_frame):
    """Propagate the u and v species in U and V arrays

    Args:
        U (array of floats): Input array of u species
        V (array of floats): Input array of v species
        Du (float): Diffusion rate of the u species
        Dv (float): Diffusion rate of the v species
        f (float): Feed rate
        k (float): Kill rate
        dt (float): Time interval between two steps
        stencil (array of floats): the stencil describing the propagation


    Returns:
        U (array of floats): The updated U array
        V (array of floats): The updated V array
    """
    n_x, n_y = U_np.shape[0], U_np.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=np.float32)
    UVV = jnp.zeros((n_x, n_y), dtype=jnp.float32)
    U = jnp.array(U_np)
    V = jnp.array(V_np)
    tpad = jnp.zeros(tuple(np.array(U.shape) + 2), dtype=np.float32)
    print(tpad.shape)
    print("device array V ", V.devices())
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            U, V = gray_scott_jax_core(U, V, Du, Dv, f, k, dt, UVV, tpad)
        frames_V[idx_fr] = V
    return frames_V


if __name__ == "__main__":
    if True:
        U, V, _ = gsc.grayscott_init(1920, 1080)
        #U, V, _ = gsc.grayscott_init(624, 500)
        gs_pars = gsc.grayscott_pars("Pulsating spots")
        nb_frame = 100
        gsc.grayscott_main(gray_scott_jax_pad, gs_pars, U, V, nb_frame)
