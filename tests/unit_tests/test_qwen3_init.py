# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch

from torchtitan.models.qwen3 import qwen3_configs


class TestQwen3DenseInit(unittest.TestCase):
    def test_dense_init_uses_fan_in_for_inputs_and_zero_for_outputs(self):
        model = qwen3_configs["debugmodel"]().build()
        layer0 = model.layers["0"]

        fan_in_weights = [
            layer0.attention.wq.weight,
            layer0.attention.wk.weight,
            layer0.attention.wv.weight,
            layer0.feed_forward.w1.weight,
            layer0.feed_forward.w3.weight,
        ]
        zero_weights = [
            layer0.attention.wo.weight,
            layer0.feed_forward.w2.weight,
            model.output.weight,
        ]

        with torch.no_grad():
            for weight in fan_in_weights + zero_weights:
                weight.fill_(1.0)
            model.tok_embeddings.weight.zero_()

        torch.manual_seed(0)
        model.init_states()

        for weight in zero_weights:
            self.assertTrue(torch.count_nonzero(weight) == 0)

        for weight in fan_in_weights:
            expected_std = 1 / math.sqrt(weight.shape[1])
            self.assertFalse(torch.all(weight == 0))
            self.assertAlmostEqual(weight.std().item(), expected_std, delta=0.002)

        self.assertFalse(torch.all(model.tok_embeddings.weight == 0))

