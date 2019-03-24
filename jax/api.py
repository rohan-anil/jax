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

"""
User-facing transformations.

These mostly wrap internal transformations, providing convenience flags to
control behavior and handling Python containers (tuples/lists/dicts) of
arguments and outputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import operator as op
import os

import numpy as onp
from contextlib import contextmanager
from distutils.util import strtobool
from six.moves import reduce

from . import core
from . import linear_util as lu
from .core import pack, eval_jaxpr
from .api_util import (pytree_fun_to_jaxtupletree_fun, pytree_to_jaxtupletree,
                       apply_jaxtree_fun, wraps)
from .tree_util import (process_pytree, node_types, build_tree, PyTreeDef,
                        tree_map, tree_flatten, tree_unflatten, tree_structure,
                        tree_transpose, leaf)
from .util import (unzip2, unzip3, curry, partial, safe_map, safe_zip,
                   WrapHashably, prod)
from .lib.xla_bridge import canonicalize_dtype, device_count
from .abstract_arrays import ShapedArray
from .interpreters import partial_eval as pe
from .interpreters import xla
from .interpreters import pxla
from .interpreters import ad
from .interpreters import batching
from .interpreters import parallel
from .interpreters import cse
from .config import flags, config

map = safe_map
zip = safe_zip

FLAGS = flags.FLAGS
flags.DEFINE_bool("jax_disable_jit",
                  strtobool(os.getenv("JAX_DISABLE_JIT", "False")),
                  "Disable JIT compilation and just call original Python.")


def jit(fun, static_argnums=()):
  """Sets up `fun` for just-in-time compilation with XLA.

  Args:
    fun: Function to be jitted. Should be a pure function, as side-effects may
      only be executed once. Its positional arguments and return value should be
      arrays, scalars, or standard Python containers (tuple/list/dict) thereof.
      Keyword arguments and positional arguments specified by `static_argnums`
      can be anything at all. These are treated as static (see below).
    static_argnums: A tuple of ints. Specifies which arguments to treat as
      static (compile-time constant). Operations that only depend on static
      arguments will be constant-folded. Calling the jitted function with
      different values for these constants will trigger recompilation.

  Returns:
    A wrapped version of `fun`, set up for just-in-time compilation.

  In the following example, `selu` can be compiled into a single fused kernel by
  XLA:

  >>> @jax.jit
  >>> def selu(x, alpha=1.67, lmbda=1.05):
  >>>   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
  >>>
  >>> key = jax.random.PRNGKey(0)
  >>> x = jax.random.normal(key, (10,))
  >>> selu(x)
  array([-0.54485154,  0.27744263, -0.29255125, -0.91421586, -0.62452525,
         -0.2474813 , -0.8574326 , -0.7823267 ,  0.7682731 ,  0.59566754],
        dtype=float32)

  """
  @wraps(fun)
  def f_jitted(*args, **kwargs):
    if _jit_is_disabled or config.read('jax_disable_jit'):
      return fun(*args, **kwargs)
    f = lu.wrap_init(fun, kwargs)
    dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
    f, dyn_args = _argnums_partial(f, dyn_argnums, args)
    jaxtupletree_args, in_trees = unzip2(map(pytree_to_jaxtupletree, dyn_args))
    _check_args(jaxtupletree_args)
    jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(f, in_trees)
    jaxtupletree_out = xla.xla_call(jaxtree_fun, *jaxtupletree_args)
    return build_tree(out_tree(), jaxtupletree_out)

  f_jitted.__name__ = "jit({})".format(f_jitted.__name__)
  return f_jitted


@contextmanager
def disable_jit():
  """Context manager that disables `jit`.

  For debugging purposes, it is useful to have a mechanism that disables `jit`
  everywhere in a block of code, namely the `disable_jit` decorator.

  Inside a `jit`-ted function the values flowing through
  traced code can be abstract (i.e., shaped arrays with an unknown values),
  instead of concrete (i.e., specific arrays with known values).

  For example:

  >>> @jax.jit
  >>> def f(x):
  >>>   y = x *2
  >>>   print("Value of y is", y)
  >>>   return y + 3
  >>>
  >>> print(f(jax.numpy.array([1, 2, 3])))
  Value of y is Traced<ShapedArray(int32[3]):JaxprTrace(level=-1/1)>
  [5 7 9]

  Here `y` has been abstracted by `jit` to a `ShapedArray`, which represents an
  array with a fixed shape and type but an arbitrary value. If we want to see a
  concrete values while debugging, we can use the `disable_jit` decorator, at
  the cost of slower code:

  >>> with jax.disable_jit():
  >>>   print(f(np.array([1, 2, 3])))
  >>>
  Value of y is [2 4 6]
  [5 7 9]
  """
  global _jit_is_disabled
  _jit_is_disabled, prev_val = True, _jit_is_disabled
  yield
  _jit_is_disabled = prev_val
_jit_is_disabled = False


def xla_computation(fun, static_argnums=()):
  def pv_like(x):
    aval = xla.abstractify(x)
    return pe.PartialVal((aval, core.unit))

  @wraps(fun)
  def computation_maker(*args, **kwargs):
    jax_args, in_trees = unzip2(map(pytree_to_jaxtupletree, args))
    wrapped = lu.wrap_init(fun)
    jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(wrapped, in_trees)
    pvals = map(pv_like, jax_args)
    jaxpr, _, consts = pe.trace_to_jaxpr(jaxtree_fun, pvals, **kwargs)
    return xla.build_jaxpr(jaxpr, consts, *map(xla.abstractify, args))

  return computation_maker


def grad(fun, argnums=0, has_aux=False):
  """Creates a function which evaluates the gradient of `fun`.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      `argnums` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape `()` but not
      arrays with shape `(1,)` etc.)
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether `fun` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.

  Returns:
    A function with the same arguments as `fun`, that evaluates the gradient of
    `fun`. If `argnums` is an integer then the gradient has the same shape and
    type as the positional argument indicated by that integer. If argnums is a
    tuple of integers, the gradient is a tuple of values with the same shapes
    and types as the corresponding arguments. If `has_aux` is True then a pair
    of (gradient, auxiliary_data) is returned.

  For example:

  >>> grad_tanh = jax.grad(jax.numpy.tanh)
  >>> grad_tanh(0.2)
  array(0.961043, dtype=float32)

  """
  value_and_grad_f = value_and_grad(fun, argnums, has_aux=has_aux)

  docstr = ("Gradient of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "gradient, which has the same shape as the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def grad_f(*args, **kwargs):
    if not has_aux:
      _, g = value_and_grad_f(*args, **kwargs)
      return g
    else:
      (_, aux), g = value_and_grad_f(*args, **kwargs)
      return g, aux

  return grad_f

def value_and_grad(fun, argnums=0, has_aux=False):
  """Creates a function which evaluates both `fun` and the gradient of `fun`.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      `argnums` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape `()` but not
      arrays with shape `(1,)` etc.)
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether `fun` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.

  Returns:
    A function with the same arguments as `fun` that evaluates both `fun` and
    the gradient of `fun` and returns them as a pair (a two-element tuple). If
    `argnums` is an integer then the gradient has the same shape and type as the
    positional argument indicated by that integer. If argnums is a tuple of
    integers, the gradient is a tuple of values with the same shapes and types
    as the corresponding arguments.
  """

  docstr = ("Value and gradient of {fun} with respect to positional "
            "argument(s) {argnums}. Takes the same arguments as {fun} but "
            "returns a two-element tuple where the first element is the value "
            "of {fun} and the second element is the gradient, which has the "
            "same shape as the arguments at positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def value_and_grad_f(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = _argnums_partial(f, argnums, args)
    if not has_aux:
      ans, vjp_py = vjp(f_partial, *dyn_args)
    else:
      ans, vjp_py, aux = vjp(f_partial, *dyn_args, has_aux=True)
    _check_scalar(ans)
    g = vjp_py(onp.ones((), onp.result_type(ans)))
    g = g[0] if isinstance(argnums, int) else g
    if not has_aux:
      return ans, g
    else:
      return (ans, aux), g

  return value_and_grad_f


def jacfwd(fun, argnums=0):
  """Jacobian of `fun` evaluated column-by-column using forward-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default `0`).

  Returns:
    A function with the same arguments as `fun`, that evaluates the Jacobian of
    `fun` using forward-mode automatic differentiation.

  >>> def f(x):
  >>>   return jax.numpy.asarray(
  >>>     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jax.numpy.sin(x[0])])
  >>> jax.jacfwd(f)(np.array([1., 2., 3.]))
  array([[ 1.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  5.        ],
         [ 0.        , 16.        , -2.        ],
         [ 1.6209068 ,  0.        ,  0.84147096]], dtype=float32)
  """

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = _argnums_partial(f, argnums, args)
    pushfwd = partial(jvp, f_partial, dyn_args)
    y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    return tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)

  return jacfun

def jacrev(fun, argnums=0):
  """Jacobian of `fun` evaluated row-by-row using reverse-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default `0`).

  Returns:
    A function with the same arguments as `fun`, that evaluates the Jacobian of
    `fun` using reverse-mode automatic differentiation.

  >>> def f(x):
  >>>   return jax.numpy.asarray(
  >>>     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jax.numpy.sin(x[0])])
  >>> jax.jacrev(f)(np.array([1., 2., 3.]))
  array([[ 1.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  5.        ],
         [ 0.        , 16.        , -2.        ],
         [ 1.6209068 ,  0.        ,  0.84147096]], dtype=float32)
  """
  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = _argnums_partial(f, argnums, args)
    y, pullback = vjp(f_partial, *dyn_args)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
    return tree_transpose(tree_structure(example_args), tree_structure(y), jac)

  return jacfun
jacobian = jacrev

def hessian(fun, argnums=0):
  """Hessian of `fun`.

  Args:
    fun: Function whose Hessian is to be computed.
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default `0`).

  Returns:
    A function with the same arguments as `fun`, that evaluates the Hessian of
    `fun`.

  >>> g = lambda(x): x[0]**3 - 2*x[0]*x[1] - x[1]**6
  >>> jax.hessian(g)(jax.numpy.array([1., 2.]))
  array([[   6.,   -2.],
         [  -2., -480.]], dtype=float32)
  """

  return jacfwd(jacrev(fun, argnums=argnums), argnums=argnums)

def _std_basis(pytree):
  leaves, _ = tree_flatten(pytree)
  ndim = sum(map(onp.size, leaves))
  return _unravel_array_into_pytree(pytree, 1, onp.eye(ndim))

def _unravel_array_into_pytree(pytree, axis, arr):
  leaves, treedef = tree_flatten(pytree)
  axis = axis % arr.ndim
  dtypes = map(_dtype, leaves)
  shapes = [arr.shape[:axis] + onp.shape(l) + arr.shape[axis+1:] for l in leaves]
  parts = _split(arr, onp.cumsum(map(onp.size, leaves[:-1])), axis)
  reshaped_parts = [onp.reshape(part.astype(dtype), shape)
                    for part, dtype, shape in zip(parts, dtypes, shapes)]
  return tree_unflatten(treedef, reshaped_parts)

def _split(x, indices, axis):
  if isinstance(x, onp.ndarray):
    return onp.split(x, indices, axis)
  else:
    return x.split(indices, axis)

def _dtype(x):
  return canonicalize_dtype(onp.result_type(x))


def vmap(fun, in_axes=0, out_axes=0):
  """Vectorizing map. Creates a function which maps `fun` over additional axes.

  Args:
    fun: Function to be mapped over additional axes.
    in_axes: Specifies which input axes to map over. These may be integers,
      `None`, or (possibly nested) tuples of integers or `None`.
    out_axes: Specifies which output axes to map over. These may be integers,
      `None`, or (possibly nested) tuples of integers or `None`.

  Returns:
    Batched/vectorized version of `fun` with arguments that correspond to those
    of `fun`, but with extra array axes at positions indicated by `in_axes`, and
    a return value that corresponds to that of `fun`, but with extra array axes
    at positions indicated by `out_axes`.

  For example, we can implement a matrix-matrix product using a vector dot
  product:

  >>> vv = lambda x, y: np.vdot(x, y)  #  ([a], [a]) -> []
  >>> mv = vmap(vv, (0, None), 0)      #  ([a,b], [b]) -> [a]
  >>> mm = vmap(mv, (None, 1), 1)      #  ([a,b], [b,c]) -> [a,c]

  (`[a,b]` indicates an array with shape (a,b))
  """

  docstr = ("Vectorized version of {fun}. Takes similar arguments as {fun} "
            "but with additional array axes over which {fun} is mapped.")

  if (not isinstance(in_axes, (list, tuple, type(None), int))
      or not isinstance(out_axes, (list, tuple, type(None), int))):
    msg = ("vmap arguments in_axes and out_axes must each be an integer, None, "
           "or a (nested) tuple of those types, got {} and {} respectively.")
    raise TypeError(msg.format(type(in_axes), type(out_axes)))

  @wraps(fun, docstr=docstr)
  def batched_fun(*args, **kwargs):
    if not isinstance(fun, lu.WrappedFun):
      f = lu.wrap_init(fun, kwargs)
    in_axes_ = in_axes if isinstance(in_axes, (list, tuple)) else (in_axes,) * len(args)
    in_flat, in_trees = unzip2(map(pytree_to_jaxtupletree, args))
    jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(f, in_trees)
    out_flat = batching.batch(jaxtree_fun, in_flat, in_axes_, out_axes)
    return build_tree(out_tree(), out_flat)

  return batched_fun


def pmap(fun, axis_name=None):
  """Set up SPMD function for JIT compilation and parallel execution with XLA."""
  axis_name = _TempAxisName() if axis_name is None else axis_name

  @wraps(fun)
  def f_jitted(*args, **kwargs):
    leaves, _ = tree_flatten(args)
    axis_sizes = set(onp.shape(leaf)[0] for leaf in leaves)
    if len(axis_sizes) != 1:
      msg = "pmap requires all leading axes to have equal length, got {}."
      raise TypeError(msg.format(axis_sizes))
    axis_size = axis_sizes.pop()

    jaxtupletree_args, in_trees = unzip2(map(pytree_to_jaxtupletree, args))
    _check_args(jaxtupletree_args)
    f = lu.wrap_init(fun, kwargs)
    f, out_tree = pytree_fun_to_jaxtupletree_fun(f, in_trees)
    jaxtupletree_out = pxla.xla_pcall(f, *jaxtupletree_args,
                                      axis_name=axis_name, axis_size=axis_size)
    return build_tree(out_tree(), jaxtupletree_out)

  namestr = "pmap({}, axis_name={})".format
  f_jitted.__name__ = namestr(f_jitted.__name__, axis_name)
  return f_jitted

def serial_pmap(fun, axis_name=None, in_axes=0, out_axes=0):
  """Vectorizing pseudo-map for single-program multiple-data (SPMD) functions."""
  axis_name = _TempAxisName() if axis_name is None else axis_name

  def map_fun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    in_axes_ = in_axes if isinstance(in_axes, (list, tuple)) else (in_axes,) * len(args)
    in_flat, in_trees = unzip2(map(pytree_to_jaxtupletree, args))
    jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(f, in_trees)
    out_flat = parallel.serial_pmap(jaxtree_fun, axis_name, in_flat, in_axes_, out_axes)
    return build_tree(out_tree(), out_flat)

  return map_fun

class _TempAxisName(object):
  def __repr__(self):
    return '<temp axis {}>'.format(hex(id(self)))


def papply(fun, axis_size, in_axes=0, out_axes=0):
  """Apply a function using parallel computation by sharding inputs."""
  axis_name = parallel.newvar()

  def papply_fun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    in_axes_ = in_axes if isinstance(in_axes, (list, tuple)) else (in_axes,) * len(args)
    args_flat, in_trees = unzip2(map(pytree_to_jaxtupletree, args))
    jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(f, in_trees)
    out_flat = parallel.papply(jaxtree_fun, axis_name, args_flat, axis_size,
                               in_axes_, out_axes)
    return build_tree(out_tree(), out_flat)

  return papply_fun, axis_name


def jvp(fun, primals, tangents):
  """Computes a (forward-mode) Jacobian-vector product of `fun`.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: The primal values at which the Jacobian of `fun` should be
      evaluated. Should be a tuple of arrays, scalar, or standard Python
      container thereof. The length of the tuple is equal to the number of
      positional parameters of `fun`.
    tangents: The tangent vector for which the Jacobian-vector product should be
      evaluated. Should be a tuple of arrays, scalar, or standard Python
      container thereof, with the same tree structure and array shapes as
      `primals`.

  Returns:
    A `(primals_out, tangents_out)` pair, where `primals_out` is
    `fun(*primals)`, and `tangents_out` is the Jacobian-vector product of
    `function` evaluated at `primals` with `tangents`. The `tangents_out` value
    has the same Python tree structure and shapes as `primals_out`.

  For example:

  >>> jax.jvp(jax.numpy.sin, (0.1,), (0.2,))
  (array(0.09983342, dtype=float32), array(0.19900084, dtype=float32))
  """
  def trim_arg(primal, tangent):
    primal_jtuple, tree_def = pytree_to_jaxtupletree(primal)
    tangent_jtuple, tree_def_2 = pytree_to_jaxtupletree(tangent)
    assert tree_def == tree_def_2, (tree_def, tree_def_2)
    return primal_jtuple, tangent_jtuple, tree_def

  if not isinstance(fun, lu.WrappedFun):
    fun = lu.wrap_init(fun)
  ps_flat, ts_flat, in_trees = unzip3(map(trim_arg, primals, tangents))
  jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(fun, in_trees)
  out_primal, out_tangent = cse.cse(ad.jvp(jaxtree_fun)).call_wrapped(ps_flat, ts_flat)
  return (build_tree(out_tree(), out_primal), build_tree(out_tree(), out_tangent))

def linearize(traceable, *primals):
  fun = lu.wrap_init(traceable)
  primals_flat, in_trees = unzip2(map(pytree_to_jaxtupletree, primals))
  jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(fun, in_trees)
  out_primal, out_pval, jaxpr, consts = ad.linearize(jaxtree_fun, *primals_flat)
  out_tree = out_tree()
  out_primal_py = build_tree(out_tree, out_primal)
  lifted_jvp = partial(lift_linearized, jaxpr, consts, (in_trees, out_tree), out_pval)
  return out_primal_py, lifted_jvp

def lift_linearized(jaxpr, consts, io_tree, out_pval, *py_args):
  def fun(*args):
    primals = pack(args) # doesn't matter what these are-they'll be ignored
    tangents = pack(args)
    _, ans = eval_jaxpr(jaxpr, consts, (), primals, tangents)
    return pe.merge_pvals(ans, out_pval)

  return apply_jaxtree_fun(fun, io_tree, *py_args)

def vjp(fun, *primals, **kwargs):
  """Compute a (reverse-mode) vector-Jacobian product of `fun`.

  `grad` is implemented as a special case of `vjp`.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: A sequence of primal values at which the Jacobian of `fun`
      should be evaluated. The length of `primals` should be equal to the number
      of positional parameters to `fun`. Each primal value should be a tuple of
      arrays, scalar, or standard Python containers thereof.
    has_aux: Optional, bool. Indicates whether `fun` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.

  Returns:
    A `(primals_out, vjpfun)` pair, where `primals_out` is `fun(*primals)`.
    `vjpfun` is a function from a cotangent vector with the same shape as
    `primals_out` to a tuple of cotangent vectors with the same shape as
    `primals`, representing the vector-Jacobian product of `fun` evaluated at
    `primals`.

  >>> def f(x, y):
  >>>   return jax.numpy.sin(x), jax.numpy.cos(y)
  >>> primals, g = jax.vjp(f, 0.5, 1.0)
  >>> g((-0.7, 0.3))
  (array(-0.61430776, dtype=float32), array(-0.2524413, dtype=float32))
  """
  has_aux = kwargs.pop('has_aux', False)
  assert not kwargs
  if not isinstance(fun, lu.WrappedFun):
    fun = lu.wrap_init(fun)
  primals_flat, in_trees = unzip2(map(pytree_to_jaxtupletree, primals))
  _check_args(primals_flat)
  jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(fun, in_trees)
  if not has_aux:
    out_primal, out_vjp = ad.vjp(jaxtree_fun, primals_flat)
  else:
    out_primal, out_vjp, aux = ad.vjp(jaxtree_fun, primals_flat, has_aux=True)
  out_tree = out_tree()
  if has_aux:
    out_tree, aux_tree = out_tree.children
  out_primal_py = build_tree(out_tree, out_primal)
  ct_in_trees = [out_tree]
  ct_out_tree = PyTreeDef(node_types[tuple], None, in_trees)
  def out_vjp_packed(cotangent_in):
    return out_vjp(cotangent_in)
  vjp_py = partial(apply_jaxtree_fun, out_vjp_packed, (ct_in_trees, ct_out_tree))
  if not has_aux:
    return out_primal_py, vjp_py
  else:
    return out_primal_py, vjp_py, build_tree(aux_tree, aux)


def trace_to_jaxpr(traceable, py_pvals, **kwargs):
  fun = lu.wrap_init(traceable)
  pvals, in_trees = unzip2(map(tree_to_pval_tuples, py_pvals))
  jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(fun, in_trees)
  jaxpr, out_pval, consts = pe.trace_to_jaxpr(jaxtree_fun, pvals, **kwargs)
  return jaxpr, consts, out_pval, (in_trees, out_tree())

def lift_jaxpr(jaxpr, consts, io_tree, pvals, py_args):
  def fun(*args):
    ans = eval_jaxpr(jaxpr, consts, (), *args)
    return pe.merge_pvals(ans, pvals)
  return apply_jaxtree_fun(fun, io_tree, *py_args)

def make_jaxpr(fun):
  """Adapts `fun` to return its `jaxpr` program representation.

  Args:
    fun: The function whose `jaxpr` is to be computed. Its positional arguments
      and return value should be arrays, scalars, or standard Python containers
      (tuple/list/dict) thereof.

  Returns:
    A wrapped version of `fun`, set up to return a `jaxpr`.

  A `jaxpr` is JAX's intermediate representation for program traces. The `jaxpr`
  language is based on the simply-typed first-order lambda calculus with
  let-bindings. `make_jaxpr` adapts a function to return its `jaxpr`, which we
  can inspect to understand what JAX is doing internally.

  The `jaxpr` returned is a trace of `fun` abstracted to `ShapedArray` level.
  Other levels of abstraction exist internally.

  We do not describe the semantics of the `jaxpr` language in detail here, but
  instead give a few examples.

  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> f(3.0)
  array(-0.83602184, dtype=float32)
  >>> jax.make_jaxpr(f)(3.0)
  { lambda  ;  ; a.
    let b = cos a
        c = sin b
    in c }
  >>> jax.make_jaxpr(jax.grad(f))(3.0)
  { lambda b ;  ; a.
    let c = pack a
        (d) = id c
        e = cos d
        f = cos e
        g = mul b f
        h = neg g
        i = sin d
        j = mul h i
        k = pack j
        (l) = id k
    in l }
  """
  def pv_like(x):
    aval = xla.abstractify(x)
    return pe.PartialVal((aval, core.unit))

  wrapped = lu.wrap_init(fun)

  @wraps(fun)
  def jaxpr_maker(*args, **kwargs):
    jax_args, in_trees = unzip2(map(pytree_to_jaxtupletree, args))
    jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(wrapped, in_trees)
    pvals = map(pv_like, jax_args)
    jaxpr, _, _ = pe.trace_to_jaxpr(jaxtree_fun, pvals, **kwargs)
    return jaxpr

  jaxpr_maker.__name__ = "make_jaxpr({})".format(jaxpr_maker.__name__)
  return jaxpr_maker

tree_to_pval_tuples = partial(process_pytree, pe.pack_pvals)


device_put = jit(lambda x: x)
device_get_array = lambda x: x.copy() if type(x) is xla.DeviceArray else x
device_get = partial(tree_map, device_get_array)
replicate = lambda x: pmap(lambda _: x)(onp.arange(device_count()))
unreplicate = lambda x: tree_map(op.itemgetter(0), x)


def _argnums_partial(f, dyn_argnums, args):
  if isinstance(dyn_argnums, int):
    dyn_argnums = (dyn_argnums,)
  else:
    dyn_argnums = tuple(dyn_argnums)
  fixed_args = tuple([None if i in dyn_argnums else WrapHashably(arg)
                      for i, arg in enumerate(args)])
  dyn_args = tuple(args[i] for i in dyn_argnums)
  return _argnums_partial_(f, dyn_argnums, fixed_args), dyn_args

@lu.transformation
def _argnums_partial_(dyn_argnums, fixed_args, *dyn_args):
  args = [None if arg is None else arg.val for arg in fixed_args]
  for i, arg in zip(dyn_argnums, dyn_args):
    args[i] = arg
  ans = yield args
  yield ans

def _check_args(args):
  for arg in args:
    if not (isinstance(arg, core.Tracer) or core.valid_jaxtype(arg)):
      raise TypeError("Argument '{}' of type {} is not a valid JAX type"
                      .format(arg, type(arg)))

def _check_scalar(x):
  msg = "Gradient only defined for scalar-output functions. Output was: {}".format
  try:
    aval = core.get_aval(x)
    if not (isinstance(aval, ShapedArray) and aval.shape == ()):
      raise TypeError(msg(x))
  except TypeError:
    raise TypeError(msg(x))


def _primitive(fun):
  name = getattr(fun, '__name__', '<unnamed user primitive>')
  fun_p = core.Primitive(name)
  fun_p.def_impl(fun)

  # generic transformation implementations that rely on traceability of `fun`
  fun_p.def_abstract_eval(partial(pe.abstract_eval_fun, fun))
  xla.translations[fun_p] = partial(xla.lower_fun, fun)
  ad.primitive_jvps[fun_p] = partial(jvp, fun)
  # TODO(mattjj): batching

  @wraps(fun)
  def traceable(*args, **kwargs):
    # TODO(mattjj): pytrees to jaxtupletrees
    return fun_p.bind(*args, **kwargs)
  traceable.primitive = fun_p

  return traceable
