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
"""
Parallelization primitives.
"""

from jax import ad_util
from jax.lax import lax
from jax.abstract_arrays import ShapedArray
from jax.core import Primitive
from jax.interpreters import ad
from jax.interpreters import parallel
from jax.interpreters import pxla
from jax.util import partial


### parallel traceables

def psum(x, axis_name):
  return psum_p.bind(x, axis_name=axis_name)

def pmax(x, axis_name):
  return pmax_p.bind(x, axis_name=axis_name)

def pswapaxes(x, axis_name, axis):
  """Analogue to `np.swapaxes` involving a hidden axis.

  Specifically, transposes the operand along the axis that's currently hidden
  and the given concrete axis. The implicit position of the hidden axis remains
  unchanged.
  """
  return pswapaxes_p.bind(x, axis_name=axis_name, axis=axis)

def psplit(x, axis_name, axis):
  """Merge operand along the hidden axis and split it along `axis`.

  The newly split axis becomes the hidden axis for the output, and in particular
  the implicit position of the hidden axis changes.
  """
  # lowering should be:
  # return xla_all_to_all(x, hidden axis, axis)
  return psplit_p.bind(x, axis_name=axis_name, axis=axis)

def psplit_like(x, y, axis_name):
  """Split `x` along any axis on which `y` is split, if it is."""
  return psplit_like_p.bind(x, y, axis_name=axis_name)

def pcollect(x, axis_name):
  # lowering should be:
  # x = xla_broadcast(x, (xb.get_replica_count(),))
  # return xla_all_to_all(x, 0, dim(axis_name), **params)
  return pcollect_p.bind(x, axis_name=axis_name)


### parallel primitives

def _unbound_name_error(primitive_name, *args, **kwargs):
  axis_name = kwargs['axis_name']
  msg = "axis name '{}' is unbound for primitive {}."
  raise NameError(msg.format(axis_name, primitive_name))

def PmapPrimitive(name):
  prim = Primitive(name)
  prim.def_impl(partial(_unbound_name_error, name))
  prim.def_abstract_eval(lambda x, *args, **kwargs: x)
  return prim


def _psum_serial_pmap_rule(vals, axes):
  val, = vals
  axis, = axes
  return lax._reduce_sum(val, [axis]), None

def _psum_transpose_rule(t, axis_name):
  return [t]

def _psum_parallel_translation_rule(c, val, device_groups):
  if len(device_groups) > 1:
    return c.CrossReplicaSum(val, device_groups)
  else:
    return c.CrossReplicaSum(val)

psum_p = PmapPrimitive('psum')
psum_p.def_impl(partial(_unbound_name_error, 'psum'))
psum_p.def_abstract_eval(lambda x, *args, **kwargs: x)
parallel.serial_pmap_primitive_rules[psum_p] = _psum_serial_pmap_rule
pxla.parallel_translation_rules[psum_p] = _psum_parallel_translation_rule
ad.deflinear(psum_p, _psum_transpose_rule)
parallel.defreducer(lax.reduce_sum_p, psum_p)


def _pmax_serial_pmap_rule(vals, axes):
  val, = vals
  axis, = axes
  return lax._reduce_max(val, [axis]), None

pmax_p = PmapPrimitive('pmax')
pmax_p.def_impl(partial(_unbound_name_error, 'pmax'))
pmax_p.def_abstract_eval(lambda x, *args, **kwargs: x)
parallel.serial_pmap_primitive_rules[pmax_p] = _pmax_serial_pmap_rule
parallel.defreducer(lax.reduce_max_p, pmax_p)


def _pswapaxes_serial_pmap_rule(vals, axes, axis):
  x, = vals
  axis_in, = axes
  if x.shape[axis_in] != x.shape[axis]:
    raise ValueError("pswapaxes between non-square dimensions")
  perm = list(range(x.ndim))
  perm[axis_in] = axis
  perm[axis] = axis_in
  return lax.transpose(x, perm), axis_in

pswapaxes_p = PmapPrimitive('pswapaxes')
parallel.serial_pmap_primitive_rules[pswapaxes_p] = _pswapaxes_serial_pmap_rule


def _psplit_serial_pmap_rule(vals, axes, axis):
  x, = vals
  axis_in, = axes
  if x.shape[axis_in] != x.shape[axis]:
    raise ValueError(
        "psplit between non-square dimensions {} and {} of {}".format(
            axis_in, axis, x.shape))
  return x, axis

psplit_p = PmapPrimitive('psplit')
parallel.serial_pmap_primitive_rules[psplit_p] = _psplit_serial_pmap_rule


def _psplit_like_serial_pmap_rule(vals, axes):
  x, y = vals
  xaxis, yaxis = axes
  if xaxis is not None and x.shape[xaxis] != x.shape[yaxis]:
    raise ValueError(
        "psplit_like is a non-square re-split along {} and {} of {}".format(
            xaxis, yaxis, x.shape))
  return x, yaxis

psplit_like_p = PmapPrimitive('psplit_like')
psplit_like_p.def_abstract_eval(
    lambda x, y, *args, **kwargs: ShapedArray(y.shape, x.dtype))
parallel.serial_pmap_primitive_rules[psplit_like_p] = _psplit_like_serial_pmap_rule


def _pcollect_serial_pmap_rule(vals, axes):
  x, = vals
  return x, None

pcollect_p = PmapPrimitive('pcollect')
parallel.serial_pmap_primitive_rules[pcollect_p] = _pcollect_serial_pmap_rule


### papply rules
# TODO(skye): it would be nice if we could put these with their corresponding
# primitives, but that currently causes circular dependencies. More refactoring
# might fix this.

def _dot_papply_rule(name, size, vals, dims):
  x, y = vals
  xdim, ydim = dims
  if xdim is None:
    return lax.dot(x, y), ydim
  elif ydim is None:
    return lax.dot(x, y), xdim
  elif ydim == 0:
    if xdim != x.ndim:
      x = psplit(x, name, x.ndim)
    x = x[..., None]
    y = y[..., None, :]
    return psum(x * y, name), None
  else:
    y = pcollect(y, name)
    return lax.dot(x, y), xdim


def _dot_general_papply_rule(name, size, vals, dims, dimension_numbers):
  x, y = vals
  xdim, ydim = dims

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  if len(lhs_batch) > 0 or len(rhs_batch) > 0:
    raise NotImplementedError

  def adjust_dims(dims, thresh):
    return tuple(i - 1 if i >= thresh else i for i in dims if i != thresh)

  sub_lhs_contract, sub_rhs_contract = lhs_contract, rhs_contract
  if xdim is not None:
    sub_lhs_contract = adjust_dims(lhs_contract, xdim)
  if ydim is not None:
    sub_rhs_contract = adjust_dims(rhs_contract, ydim)

  sub_dimension_numbers = (
      (sub_lhs_contract, sub_rhs_contract), (lhs_batch, rhs_batch))

  if xdim in lhs_contract and ydim in rhs_contract:
    z = lax.dot_general(x, y, sub_dimension_numbers)
    return psum(z, name), None
  elif xdim in lhs_contract:
    if ydim is not None:        # Cannot hide two dimensions, so collect one
      y = pcollect(y, name)
    return lax.dot_general(x, y, sub_dimension_numbers), xdim
  elif ydim in rhs_contract:
    if xdim is not None:        # Cannot hide two dimensions, so collect one
      x = pcollect(x, name)
    return lax.dot_general(x, y, sub_dimension_numbers), ydim
  elif xdim is not None:
    if ydim is not None:        # Cannot hide two dimensions, so collect one
      y = pcollect(y, name)
    return lax.dot_general(x, y, sub_dimension_numbers), xdim
  elif ydim is not None:
    return lax.dot_general(x, y, sub_dimension_numbers), ydim
  else:
    return lax.dot_general(x, y, sub_dimension_numbers), None


def _reshape_papply_rule(name, size, vals, axes, new_sizes, dimensions,
                         old_sizes):
  operand, = vals
  axis, = axes

  def filter_ones(xs):
    return filter(lambda x: x != 1, xs)

  def find_new_axis(old_axis, old_sizes, new_sizes):
    if len(filter_ones(new_sizes)) != len(filter_ones(old_sizes)):
      return None
    num_before = len(filter_ones(old_sizes[:old_axis]))
    sz = old_sizes[old_axis]
    for i, new_sz in enumerate(new_sizes):
      if num_before == 0:
        if new_sz == sz:
          return i
        elif new_sz != 1:
          if sz == 1:
            return i - 1
          else:
            return None
      elif new_sz != 1:
        num_before -= 1
    return None

  err = NotImplementedError(
      'papply of reshape that would change hidden dimension size')

  if dimensions is None:
    new_axis = find_new_axis(axis, old_sizes, new_sizes)
    if new_axis is not None:
      if (lax.prod(old_sizes[:axis]) != lax.prod(new_sizes[:new_axis]) or
          lax.prod(old_sizes[axis + 1:]) != lax.prod(new_sizes[new_axis + 1:])):
        raise err
      new_sizes_ = new_sizes[:new_axis] + new_sizes[new_axis + 1:]
      if new_axis == -1:  # reshape squeezes only and all major singleton axes
        new_axis = None
      return lax.reshape(operand, new_sizes_, dimensions=dimensions), new_axis
    else:
      raise err
  else:
    raise NotImplementedError('papply of reshape with `dimensions`')


def _transpose_papply_rule(name, size, vals, dims, permutation):
  x, = vals
  xdim, = dims
  perm = list(permutation)
  if perm[xdim] == xdim:
    x = lax.transpose(x, perm)
    out_dim = xdim
  else:
    in_dim, = [i for i in range(len(perm)) if perm[i] == xdim]
    out_dim = perm[xdim]
    perm[in_dim] = out_dim
    perm[out_dim] = in_dim
    perm = perm[:xdim] + perm[xdim + 1:]
    perm = [i - 1 if i > xdim else i for i in perm]
    x = lax.transpose(x, perm)
    x = pswapaxes(x, name, in_dim)
  return x, xdim


def _select_papply_rule(name, size, vals, dims):
  dimset = set([d for d in dims if d is not None])
  if len(dimset) != 1:
    raise NotImplementedError(
        'papply of select with operands split along different dimensions')
  like_val, like_dim = [(v, d) for v, d in zip(vals, dims) if d is not None][0]

  def normalize_split(val, dim):
    return psplit_like(val, like_val, name) if dim is None else val

  vals = [normalize_split(v, d) for v, d in zip(vals, dims)]
  return lax.select_p.bind(*vals), like_dim


def _add_jaxvals_papply_rule(name, size, vals, dims):
  x, y = vals
  xdim, ydim = dims
  if xdim == ydim:
    out_dim = xdim
  elif ydim is None:
    y = lax.psplit_like(y, x, name)
    out_dim = xdim
  else:
    x = lax.psplit_like(x, y, name)
    out_dim = ydim
  return ad_util.add_jaxvals_p.bind(x, y), out_dim


parallel.papply_primitive_rules[lax.dot_p] = _dot_papply_rule
parallel.papply_primitive_rules[lax.dot_general_p] = _dot_general_papply_rule
parallel.papply_primitive_rules[lax.reshape_p] = _reshape_papply_rule
parallel.papply_primitive_rules[lax.transpose_p] = _transpose_papply_rule
parallel.papply_primitive_rules[lax.select_p] = _select_papply_rule
parallel.papply_primitive_rules[ad_util.add_jaxvals_p] = (
    _add_jaxvals_papply_rule)
