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

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax import lax
from jax.util import partial, curry
from jax.interpreters.masking import apply_masked, pad_and_mask, pad_and_stack


from jax.config import config
config.parse_flags_with_absl()

class MaskingTest(jtu.JaxTestCase):

  def testMapReduce(self):
    x = np.arange(3) + 1
    x_padded, mask = pad_and_mask(x, (5,))
    ans = apply_masked(lambda x: np.max(np.cos(x)), (x_padded,), (mask,))
    assert ans == 0.5403023
