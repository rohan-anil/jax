# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from ..util import unzip2, unzip3
from .. import core
from ..core import Trace, Tracer, new_master
from ..abstract_arrays import UnshapedArray, ShapedArray
from ..linear_util import transformation, wrap_init
import jax.lax as lax
import jax.numpy as np
import numpy as onp

class MaskTracer(Tracer):
  def __init__(self, trace, val, mask):
    self.trace = trace
    self.val = val
    self.mask = mask
    self.ndim = self.val.ndim

  @property
  def aval(self):
    shape = (None,)*self.ndim  # TODO(dougalm): provide some shape information
    return ShapedArray(shape, self.val.dtype)

  def unpack(self):
    assert False

  def full_lower(self):
    if self.mask is None:
      return core.full_lower(self.val)
    else:
      return self


class MaskTrace(Trace):
  def pure(self, val):
    return MaskTracer(self, val, None)

  def lift(self, val):
    return MaskTracer(self, val, None)

  def sublift(self, val):
    return MaskTracer(self, val.val, val.mask)

  def process_primitive(self, primitive, tracers, params):
    vals_in, masks_in = unzip2((t.val, t.mask) for t in tracers)
    if all(m is None for m in masks_in):
      assert False
      return primitive.bind(*vals_in, **params)
    else:
      masked_primitive = get_primitive_masker(primitive)
      val_out, mask_out = masked_primitive(vals_in, masks_in, **params)
      return MaskTracer(self, val_out, mask_out)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO(dougalm)

  def post_process_call(self, _, out_tracer):
    raise NotImplementedError  # TODO(dougalm)

  def pack(self, tracers):
    raise NotImplementedError  # TODO(dougalm)


primitive_maskers = {}

def get_primitive_masker(p):
  try:
    return primitive_maskers[p]
  except KeyError:
    raise NotImplementedError("Masking rule for {} not yet implemented".format(p))

def cos_mask_rule((x,), (mask,)):
  return np.cos(x), mask

def reduce_max_mask_rule((operand,), (mask,), axes):
  shape = operand.shape
  mzero = lax.broadcast(lax._get_max_identity(operand.dtype), shape)
  masked_operand = lax.select(mask, operand, mzero)
  val_out = lax._reduce_max(masked_operand, axes=axes)
  mask_out = lax.reduce(mask, False, np.logical_or, axes)
  return val_out, mask_out

def convert_element_type_max_mask_rule((operand,), (mask,), new_dtype, old_dtype):
  val_out = lax.convert_element_type(operand, new_dtype)
  return val_out, mask


primitive_maskers[lax.cos_p] = cos_mask_rule
primitive_maskers[lax.reduce_max_p] = reduce_max_mask_rule

primitive_maskers[lax.convert_element_type_p] = convert_element_type_max_mask_rule


@transformation
def mask_transform(vals, masks):
  with new_master(MaskTrace) as master:
    trace = MaskTrace(master, core.cur_sublevel())
    in_tracers = map(partial(MaskTracer, trace), vals, masks)
    out_tracer = yield in_tracers
    out_tracer = trace.full_raise(out_tracer)
  yield out_tracer.val, out_tracer.mask

def apply_masked(f, xs, masks):
  fun = wrap_init(f, {})
  out_val, _ = mask_transform(fun).call_wrapped(xs, masks)
  return out_val

def pad_to_shape(shape, x):
  blank = np.zeros(shape, dtype=x.dtype)
  return lax.dynamic_update_slice(blank, x, (0,)*len(shape))

def make_mask(shape, valid_shape):
  if shape == ():
    return True
  else:
    valid, valid_rest = valid_shape[0], valid_shape[1:]
    full, full_rest = shape[0], shape[1:]
    suffix_mask = make_mask(full_rest, valid_rest)
    repeats = (valid,) + (1,) * len(full_rest)
    return onp.concatenate([onp.tile(suffix_mask, repeats),
                            onp.zeros((full - valid,) + full_rest, dtype=bool)])

def pad_and_stack(xs):
  max_shape = tuple(map(max, zip(*(x.shape for x in xs))))
  masks = onp.stack([make_mask(max_shape, x.shape) for x in xs])
  xs_padded = np.stack([pad_to_shape(max_shape, x) for x in xs])
  return xs_padded, masks
