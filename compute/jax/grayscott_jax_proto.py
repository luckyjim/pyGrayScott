"""
From Alice Faure work


1920 1080
step_x=0.0005211047420531526
step_frame=34
nb_frame=1000
gray_scott_jaxlap
CPU time= 760.005607155 s
Wall time= 0:06:29.966826 s
<class 'numpy.ndarray'>
(1000, 1920, 1080)
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.signal import correlate2d
from functools import partial
import compute.common as gsc
import timeit

import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

# https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
# 9-point stencil
# stencil = jnp.array([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]], dtype=jnp.float32)
# 5 point stencil
# stencil = jnp.array([[0, 1.0, 0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)


@jax.jit
def laplacian_5_jax(u):
    """
    the best implementation

    :param u: tableau 2D
    """
    lap = u.at[:-2, 1:-1].get()
    lap = lap + u.at[1:-1, :-2].get()
    lap = lap + u.at[1:-1, 2:].get()
    lap = lap + u.at[2:, 1:-1].get()
    lap = lap - 4 * u.at[1:-1, 1:-1].get()
    lap_full = jnp.zeros_like(u)
    return lap_full.at[1:-1, 1:-1].set(lap)


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
def laplacian_9_with_1_jax(u):
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


def laplacian_5_jax_slow(u):
    """
    very slow

    :param u: tableau 2D
    """
    lap = -4 * u.at[1:-1, 1:-1].get()
    lap = (
        lap
        + u.at[1:-1, :-2].get()
        + u.at[1:-1, 2:].get()
        + u.at[2:, 1:-1].get()
        + u.at[:-2, 1:-1].get()
    )
    lap_full = jnp.zeros_like(u)
    return lap_full.at[1:-1, 1:-1].set(lap)


@jax.jit
def laplacian_5_jax_at(u):
    """
    not the best
    """
    lap = u.at[:-2, 1:-1].get()
    lap = lap.at[:, :].add(u.at[1:-1, :-2].get())
    lap = lap.at[:, :].add(u.at[1:-1, 2:].get())
    lap = lap.at[:, :].add(u.at[2:, 1:-1].get())
    lap = lap.at[:, :].add(-4 * u.at[1:-1, 1:-1].get())
    lap_full = jnp.zeros_like(u)
    return lap_full.at[1:-1, 1:-1].set(lap)


@jax.jit
def laplacian_5_jax_at_first(u):
    """
    not the best
    """
    lap = jnp.zeros_like(u)
    lap = lap.at[1:-1, 1:-1].set(u.at[:-2, 1:-1].get())
    lap = lap.at[1:-1, 1:-1].add(u.at[1:-1, :-2].get())
    lap = lap.at[1:-1, 1:-1].add(u.at[1:-1, 2:].get())
    lap = lap.at[1:-1, 1:-1].add(u.at[2:, 1:-1].get())
    lap = lap.at[1:-1, 1:-1].add(-4 * u.at[1:-1, 1:-1].get())
    return lap


@jax.jit
def laplacian_numpy_jax(U, stencil):
    """Computes the laplacian in vectorized loops

    Args:
        U (array of floats): The array for which we will compute the laplacian
        stencil (3x3 array of floats): The stencil to be applied to U

    Returns:
        lap: The laplacian of U
    """
    lap = jnp.zeros((len(U), len(U[0])))

    for stencil_row in [-1, 0, 1]:
        for stencil_col in [-1, 0, 1]:
            lap = lap.at[1:-1, 1:-1].add(
                stencil[stencil_row + 1, stencil_col + 1]
                * U[
                    (1 + stencil_row) : (len(U) - 1 + stencil_row),
                    (1 + stencil_col) : (len(U[0]) - 1 + stencil_col),
                ]
            )

    return lap


@jax.jit
def gray_scott_jax_core(U, V, Du, Dv, f, k, dt, UVV):
    """ """

    UVV = U * V * V
    U = U + ((Du * laplacian_9_jax(U) - UVV + f * (1 - U)) * dt)
    V = V + ((Dv * laplacian_9_jax(V) + UVV - V * (f + k)) * dt)
    return U, V


@partial(jax.jit, static_argnames=["step_frame"])
def gray_scott_jax_loop(U, V, Du, Dv, f, k, dt, step_frame):
    """ """
    UVV = jnp.empty_like(U)
    for _ in jnp.arange(step_frame):
        # UVV = UVV.at[:, :].set(U * V * V)
        # U = U.at[:, :].add((Du * laplacian_5_jax(U) - UVV + f * (1 - U)) * dt)
        # V = V.at[:, :].add((Dv * laplacian_5_jax(V) + UVV - V * (f + k)) * dt)
        UVV = U * V * V
        U = U + ((Du * laplacian_5_jax(U) - UVV + f * (1 - U)) * dt)
        V = V + ((Dv * laplacian_5_jax(V) + UVV - V * (f + k)) * dt)
    return U, V


def gray_scott_jax_front(U_np, V_np, Du, Dv, f, k, dt, nb_frame, step_frame):
    """
    nok for   nb_frame >100
    """
    n_x, n_y = U_np.shape[0], U_np.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=np.float32)
    print(U_np.min(), U_np.max())
    print(V_np.min(), V_np.max())
    U = jnp.array(U_np)
    V = jnp.array(V_np)
    for idx_fr in jnp.arange(nb_frame):
        U, V = gray_scott_jax_loop(U, V, Du, Dv, f, k, dt, step_frame)
        frames_V[idx_fr] = V
    return frames_V


def gray_scott_jaxlap(U_np, V_np, Du, Dv, f, k, dt, nb_frame, step_frame):
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
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            UVV = U * V * V
            U = U + ((Du * laplacian_5_jax(U) - UVV + f * (1 - U)) * dt)
            V = V + ((Dv * laplacian_5_jax(V) + UVV - V * (f + k)) * dt)
        frames_V[idx_fr] = V
    return frames_V


def gray_scott_jax_fast(U_np, V_np, Du, Dv, f, k, dt, nb_frame, step_frame):
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
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            U, V = gray_scott_jax_core(U, V, Du, Dv, f, k, dt, UVV)
        frames_V[idx_fr] = V
    return frames_V


def gray_scott_jaxlap_gen(U_np, V_np, Du, Dv, f, k, dt, nb_frame, step_frame):
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
    # stencil = jnp.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    # https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    # 9-point stencil
    stencil = jnp.array([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]], dtype=jnp.float32)
    # 5 point stencil
    stencil = jnp.array([[0, 1.0, 0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    UVV = jnp.zeros((n_x, n_y), dtype=jnp.float32)
    U = jnp.array(U_np)
    V = jnp.array(V_np)
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            UVV = U * V * V
            U = U + ((Du * laplacian_numpy_jax(U, stencil) - UVV + f * (1 - U)) * dt)
            V = V + ((Dv * laplacian_numpy_jax(V, stencil) + UVV - V * (f + k)) * dt)
        frames_V[idx_fr] = V
    return frames_V


def test_lap():
    stencil = jnp.array([[0, 1.0, 0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    stencil = jnp.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    ma = 2 * jnp.array(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 2, 1.0, 1.0], [1.0, 1.0, -2, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=jnp.float32,
    )
    print(ma)
    print(laplacian_5_jax(ma))
    nb_ite = 10000
    print(timeit.timeit(lambda: laplacian_5_jax(ma), number=nb_ite))
    print(laplacian_5_jax_at(ma))
    print(timeit.timeit(lambda: laplacian_5_jax_at(ma), number=nb_ite))
    print(laplacian_numpy_jax(ma, stencil))
    print(timeit.timeit(lambda: laplacian_numpy_jax(ma, stencil), number=nb_ite))
    print(laplacian_9_jax(ma))
    print(timeit.timeit(lambda: laplacian_9_jax(ma), number=nb_ite))


def test_lap2():
    stencil = jnp.array([[0, 1.0, 0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    # stencil = jnp.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    U, V, _ = gsc.grayscott_init(2000, 1000)
    ma = jnp.array(U)
    print(ma.device)
    r1 = laplacian_5_jax(ma)
    print(r1)
    nb_ite = 1000
    print(timeit.timeit(lambda: laplacian_5_jax(ma), number=nb_ite))
    r2 = laplacian_5_jax_at(ma)
    print(r2)
    print(timeit.timeit(lambda: laplacian_5_jax_at(ma), number=nb_ite))
    r3 = laplacian_numpy_jax(ma, stencil)
    print(r3)
    print(timeit.timeit(lambda: laplacian_numpy_jax(ma, stencil), number=nb_ite))
    r4 = laplacian_5_jax_slow(ma)
    print(r4)
    print(timeit.timeit(lambda: laplacian_5_jax_slow(ma), number=nb_ite))
    r5 = laplacian_9_jax(ma)
    print(r5)
    print(timeit.timeit(lambda: laplacian_9_jax(ma), number=nb_ite))

    print(np.allclose(r1, r2))
    print(np.allclose(r1, r3))
    print(np.allclose(r1, r4))
    print(np.allclose(r1, r5))


if __name__ == "__main__":
    # test_lap()
    if True:
        U, V, _ = gsc.grayscott_init(1920, 1080)
        #U, V, _ = gsc.grayscott_init(500, 500)
        gs_pars = gsc.grayscott_pars()
        nb_frame = 1000
        # gsc.grayscott_main(gray_scott_jaxlap, gs_pars, U, V, nb_frame)
        # gsc.grayscott_main(gray_scott_jax_front, gs_pars, U, V, nb_frame)
        gsc.grayscott_main(gray_scott_jax_fast, gs_pars, U, V, nb_frame)
