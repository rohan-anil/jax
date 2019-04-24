# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Helpers for indexed updates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from ..abstract_arrays import ShapedArray, ConcreteArray
from .. import core
from .. import lax
from ..numpy import lax_numpy as np


def _scatter_update(x, idx, y, scatter_op):
  """Helper for indexed updates.

  Computes the value of x that would result from computing::
    x[idx] op= y
  except in a pure functional way, with no in-place updating.

  Support NumPy-style basic indexing only, i.e., `idx` must be
  `None`, an integer, a `slice` object, or ellipses, or a tuple of the above.

  TODO(phawkins): support advanced indexing.
  """

  x = np.asarray(x)
  y = np.asarray(y)
  x_shape = np.shape(x)
  y_shape = np.shape(y)
  y = lax.convert_element_type(y, lax.dtype(x))

  if not isinstance(idx, tuple):
    idx = (idx,)

  # Test for unsupported advanced indexing and report an error.
  if any(onp.ndim(elt) != 0 for elt in idx):
    raise NotImplementedError("Unimplemented case for indexed update. Advanced "
                              "indexing is not yet implemented.")

  # Remove ellipses and add trailing slice(None)s.
  idx = np._canonicalize_tuple_index(x, idx)

  _int = lambda aval: not aval.shape and onp.issubdtype(aval.dtype, onp.integer)

  x_axis = 0
  y_axis = 0  # Current axis in y, before collapsing. See below.
  collapsed_y_axis = 0  # Current axis in y, after collapsing.

  # Scatter dimension numbers.
  update_window_dims = []
  inserted_window_dims = []
  scatter_dims_to_operand_dims = []

  scatter_indices = np.zeros((0,), dtype=np.int32)

  # We perform three transformations to y before the scatter op, in order:
  # First, y is broadcast to slice_shape. In general `y` only need broadcast to
  # the right shape.
  slice_shape = []
  # Next, y is reshaped to collapsed_slice_shape. This is to handle `None`
  # indices, which the scatter cannot remove itself.
  collapsed_slice_shape = []
  # Finally, we reverse reversed_y_dims to handle slices with negative strides.
  reversed_y_dims = []

  for i in idx:
    try:
      abstract_i = core.get_aval(i)
    except TypeError:
      abstract_i = None
    if (isinstance(abstract_i, ConcreteArray) or
        isinstance(abstract_i, ShapedArray)) and _int(abstract_i):
      i = np.mod(i, np._constant_like(i, x.shape[x_axis]))
      i = lax.convert_element_type(i, np.int32)
      i = np.broadcast_to(i, tuple(scatter_indices.shape[:-1]) + (1,))
      scatter_indices = np.concatenate((scatter_indices, i), -1)
      inserted_window_dims.append(x_axis)
      scatter_dims_to_operand_dims.append(x_axis)
      x_axis += 1
    elif i is None:
      slice_shape.append(1)
      y_axis += 1
    elif np._is_slice_none(i):
      slice_shape.append(x_shape[x_axis])
      collapsed_slice_shape.append(x_shape[x_axis])
      update_window_dims.append(collapsed_y_axis)
      collapsed_y_axis += 1
      y_axis += 1
      x_axis += 1
    elif isinstance(i, slice):
      start, limit, stride, needs_rev = np._static_idx(i, x.shape[x_axis])
      if needs_rev:
        reversed_y_dims.append(collapsed_y_axis)
      if stride == 1:
        i = lax.convert_element_type(start, np.int32)
        i = np.broadcast_to(i, tuple(scatter_indices.shape[:-1]) + (1,))
        scatter_indices = np.concatenate((scatter_indices, i), -1)
        slice_shape.append(limit - start)
        collapsed_slice_shape.append(limit - start)
        update_window_dims.append(collapsed_y_axis)
        scatter_dims_to_operand_dims.append(x_axis)
      else:
        i = np.arange(start, limit, stride, dtype=np.int32)
        size = i.shape[0]
        slice_shape.append(size)
        collapsed_slice_shape.append(size)
        scatter_indices_shape = tuple(scatter_indices.shape[:-1]) + (size,)
        i = lax.broadcast_in_dim(
            i, shape=scatter_indices_shape + (1,),
            broadcast_dimensions=(len(scatter_indices_shape) - 1,))
        scatter_indices = lax.broadcast_in_dim(
            scatter_indices,
            shape=scatter_indices_shape + (len(scatter_dims_to_operand_dims),),
            broadcast_dimensions=(
              tuple(range(len(scatter_indices_shape) - 1)) +
              (len(scatter_indices_shape),)))
        scatter_indices = np.concatenate(
          (scatter_indices, i), len(scatter_indices_shape))
        scatter_dims_to_operand_dims.append(x_axis)
        inserted_window_dims.append(x_axis)

      collapsed_y_axis += 1
      y_axis += 1
      x_axis += 1
    else:
      raise IndexError("Unknown index type ", i)

  y = np.broadcast_to(y, tuple(slice_shape))
  y = lax.reshape(y, collapsed_slice_shape)
  if reversed_y_dims:
    y = lax.rev(y, reversed_y_dims)

  dnums = lax.ScatterDimensionNumbers(
    update_window_dims = tuple(update_window_dims),
    inserted_window_dims = tuple(inserted_window_dims),
    scatter_dims_to_operand_dims = tuple(scatter_dims_to_operand_dims)
  )
  return scatter_op(x, scatter_indices, y, dnums)


class _Indexable(object):
  """Helper object for building indexes for indexed update functions.

  This is a singleton object that overrides the :code:`__getitem__` method
  to return the index it is passed.

  >>> jax.ops.index[1:2, 3, None, ..., ::2]
  (slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
  """
  __slots__ = ()

  def __getitem__(self, index):
    return index

#: Index object singleton
index = _Indexable()


def index_add(x, idx, y):
  """Pure equivalent of :code:`x[idx] += y`.

  Returns the value of `x` that would result from the
  NumPy-style :mod:`indexed assignment <numpy.doc.indexing>`::
    x[idx] += y

  Note the `index_add` operator is pure; `x` itself is
  not modified, instead the new value that `x` would have taken is returned.

  Unlike the NumPy code :code:`x[idx] += y`, if multiple indices refer to the
  same location the updates will be summed. (NumPy would only apply the last
  update, rather than summing the updates.) The order in which conflicting
  updates are applied is implementation-defined and may be nondeterministic
  (e.g., due to concurrency on some hardware platforms).

  Args:
    x: an array.
    idx: a Numpy-style basic index, consisting of `None`, integers, `slice`
      objects, ellipses, or a tuple of the above. A convenient syntactic sugar
      for forming indices is via the :data:`jax.ops.index` object.
    y: the array of updates. `y` must be broadcastable to the shape of the
      array that would be returned by `x[idx]`.

  Returns:
    An array.

  >>> x = jax.numpy.ones((5, 6))
  >>> jax.ops.index_add(x, jax.ops.index[2:4, 3:], 6.)
  array([[1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 7., 7., 7.],
         [1., 1., 1., 7., 7., 7.],
         [1., 1., 1., 1., 1., 1.]], dtype=float32)
  """
  return _scatter_update(x, idx, y, lax.scatter_add)

def index_update(x, idx, y):
  """Pure equivalent of :code:`x[idx] = y`.

  Returns the value of `x` that would result from the
  NumPy-style :mod:`indexed assignment <numpy.doc.indexing>`::
    x[idx] = y

  Note the `index_update` operator is pure; `x` itself is
  not modified, instead the new value that `x` would have taken is returned.

  Unlike NumPy's :code:`x[idx] = y`, if multiple indices refer to the same
  location it is undefined which update is chosen; JAX may choose the order of
  updates arbitrarily and nondeterministically (e.g., due to concurrent
  updates on some hardware platforms).

  Args:
    x: an array.
    idx: a Numpy-style basic index, consisting of `None`, integers, `slice`
      objects, ellipses, or a tuple of the above. A convenient syntactic sugar
      for forming indices is via the :data:`jax.ops.index` object.
    y: the array of updates. `y` must be broadcastable to the shape of the
      array that would be returned by `x[idx]`.

  Returns:
    An array.

  >>> x = jax.numpy.ones((5, 6))
  >>> jax.ops.index_update(x, jax.ops.index[::2, 3:], 6.)
  array([[1., 1., 1., 6., 6., 6.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 6., 6., 6.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 6., 6., 6.]], dtype=float32)
  """
  return _scatter_update(x, idx, y, lax.scatter)
