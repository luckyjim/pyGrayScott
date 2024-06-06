'''
Created on 4 juin 2024

@author: jcolley
'''

from matplotlib.pylab import plt
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import correlate2d

import compute.common as gsc

def plot_array(m_a, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(m_a)
    
def proto_convol_1():
    stencil = jnp.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]],
                        dtype=np.float32)
    U, V, _ = gsc.grayscott_init(1920, 1080)
    print(stencil)
    j_u = jnp.array(U)
    j_v = jnp.array(V)
    plot_array(j_u, 'j_u')
    # compute laplacians with convolution provided by cupy
    Lu = correlate2d(j_u, stencil, 'same', 'fill', 0)
    plot_array(Lu, 'Laplacian by jax')
    print(Lu.max(), Lu.min())

if __name__ == '__main__':
    proto_convol_1()
    plt.show()
