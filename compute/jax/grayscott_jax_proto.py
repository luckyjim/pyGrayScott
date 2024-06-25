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


@jax.jit
def laplacian_4_jax(u):
    lap = u.at[:-2, 1:-1].get()
    lap = lap + u.at[1:-1, :-2].get()
    lap = lap + u.at[1:-1, 2:].get()
    lap = lap + u.at[2:, 1:-1].get()
    lap = lap - 4 * u.at[1:-1, 1:-1].get()
    lap_full = jnp.zeros_like(u)
    return lap_full.at[1:-1, 1:-1].set(lap)


@jax.jit
def laplacian_4_jax_at(u):
    lap = u.at[:-2, 1:-1].get()
    lap = lap.at[:, :].add(u.at[1:-1, :-2].get())
    lap = lap.at[:, :].add(u.at[1:-1, 2:].get())
    lap = lap.at[:, :].add(u.at[2:, 1:-1].get())
    lap = lap.at[:, :].add(-4 * u.at[1:-1, 1:-1].get())
    lap_full = jnp.zeros_like(u)
    return lap_full.at[1:-1, 1:-1].set(lap)


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


@partial(jax.jit, static_argnames=["step_frame"])
def gray_scott_jax_loop(U, V, Du, Dv, f, k, dt, step_frame):
    """ """
    UVV = jnp.empty_like(U)
    for _ in jnp.arange(step_frame):
        # UVV = UVV.at[:, :].set(U * V * V)
        # U = U.at[:, :].add((Du * laplacian_4_jax(U) - UVV + f * (1 - U)) * dt)
        # V = V.at[:, :].add((Dv * laplacian_4_jax(V) + UVV - V * (f + k)) * dt)
        UVV = U * V * V
        U = U + ((Du * laplacian_4_jax(U) - UVV + f * (1 - U)) * dt)
        V = V + ((Dv * laplacian_4_jax(V) + UVV - V * (f + k)) * dt)
    return U,V


def gray_scott_jax_front(U_np, V_np, Du, Dv, f, k, dt, nb_frame, step_frame):
    """ """
    n_x, n_y = U_np.shape[0], U_np.shape[1]
    frames_V = np.empty((nb_frame, n_x, n_y), dtype=np.float32)
    print(U_np.min(), U_np.max())
    print(V_np.min(), V_np.max())
    U = jnp.array(U_np)
    V = jnp.array(V_np)
    for idx_fr in jnp.arange(nb_frame):
        U,V = gray_scott_jax_loop(U, V, Du, Dv, f, k, dt, step_frame)
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
    stencil = jnp.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    stencil = jnp.array([[0, 1.0, 0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    UVV = jnp.zeros((n_x, n_y), dtype=jnp.float32)
    U = jnp.array(U_np)
    V = jnp.array(V_np)
    for idx_fr in range(nb_frame):
        for _ in range(step_frame):
            UVV = U * V * V
            U = U + ((Du * laplacian_4_jax(U) - UVV + f * (1 - U)) * dt)
            V = V + ((Dv * laplacian_4_jax(V) + UVV - V * (f + k)) * dt)

            # UVV = UVV.at[:, :].set(U * V * V)
            # # U = U.at[:, :].add((Du * laplacian_numpy_jax(U, stencil) - UVV + f * (1 - U)) * dt)
            # # V = V.at[:, :].add((Dv * laplacian_numpy_jax(V, stencil) + UVV - V * (f + k)) * dt)
            # U = U.at[:, :].add((Du * laplacian_4_jax(U) - UVV + f * (1 - U)) * dt)
            # V = V.at[:, :].add((Dv * laplacian_4_jax(V) + UVV - V * (f + k)) * dt)

        frames_V[idx_fr] = V
    return frames_V


def test_lap():
    stencil = jnp.array([[0, 1.0, 0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    stencil = jnp.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    ma = jnp.array(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.5, 1.0, 1.0], [1.0, 1.0, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=jnp.float32,
    )
    print(laplacian_4_jax(ma))
    nb_ite = 10000
    print(timeit.timeit(lambda: laplacian_4_jax(ma), number=nb_ite))
    print(laplacian_4_jax_at(ma))
    print(timeit.timeit(lambda: laplacian_4_jax_at(ma), number=nb_ite))
    print(laplacian_numpy_jax(ma, stencil))
    print(timeit.timeit(lambda: laplacian_numpy_jax(ma, stencil), number=nb_ite))


def test_lap2():
    stencil = jnp.array([[0, 1.0, 0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    # stencil = jnp.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    U, V, _ = gsc.grayscott_init(500, 500)
    ma = jnp.array(U)
    r1 = laplacian_4_jax(ma)
    print(r1)
    nb_ite = 1000
    print(timeit.timeit(lambda: laplacian_4_jax(ma), number=nb_ite))
    r2 = laplacian_4_jax_at(ma)
    print(r2)
    print(timeit.timeit(lambda: laplacian_4_jax_at(ma), number=nb_ite))
    r3 = laplacian_numpy_jax(ma, stencil)
    print(r3)
    print(timeit.timeit(lambda: laplacian_numpy_jax(ma, stencil), number=nb_ite))
    print(np.allclose(r1, r2))
    print(np.allclose(r1, r3))


if __name__ == "__main__":
    # test_lap2()
    if True:
        U, V, _ = gsc.grayscott_init(1920, 1080)
        #U, V, _ = gsc.grayscott_init(500, 500)
        gs_pars = gsc.grayscott_pars()
        nb_frame = 1000
        gsc.grayscott_main(gray_scott_jaxlap, gs_pars, U, V, nb_frame)
        #gsc.grayscott_main(gray_scott_jax_front, gs_pars, U, V, nb_frame)
