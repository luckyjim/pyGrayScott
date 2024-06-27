'''

'''
import numpy as np 

# import jax.numpy as jnp
# import jax
#from jax import jit

#seed_key = jax.random.PRNGKey(2)


def gb_to_size(size_gb, size_dtype=4):
    return int(size_gb * (1024 ** 3) / size_dtype)


# def proto_jax_norm_mc():
#     size_gb = 3
#     size_ar = gb_to_size(size_gb)
#     biga = jax.random.normal(seed_key, (size_ar,), dtype=jnp.float32)
#     print(size_ar)
#     def norm(a_in):
#         return jnp.sum(a_in * a_in)
#
#     ret = norm(biga)
#     print(ret)


def proto_np_norm_mc():
    size_gb = 1
    size_ar = gb_to_size(size_gb)
    biga = np.random.normal(0,1,size_ar).astype(np.float32)
    print(size_ar)
    def norm(a_in):
        return np.sum(np.sin(a_in)**2 + np.cos(a_in)**2)
    
    ret = norm(biga)
    print(ret)

       
if __name__ == "__main__":
    #proto_jax_norm_mc()
    proto_np_norm_mc()
