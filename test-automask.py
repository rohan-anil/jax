import jax.numpy as np
from jax import vmap
from jax.interpreters.masking import apply_masked, pad_and_stack


def f(x):
  return np.max(np.cos(x))

vals = [np.arange(n) + 1 for n in range(5)]

def f_padded(x, mask):
  return apply_masked(f, (x,), (mask,))

padded_val, mask = pad_and_stack(vals)
print vmap(f_padded)(padded_val, mask)


# # bug in vmap. This fails:
# print vmap(np.any)(np.array([[True, False], [False, False]]))
