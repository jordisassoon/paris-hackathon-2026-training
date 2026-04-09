# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.config_registry import qwen3_debugmodel_param_groups
from torchtitan.models.qwen3.optimizer import (
    MultiGroupOptimizersContainer,
    OptimizerGroup,
)


def _param_names(param_group, param_to_name):
    return {param_to_name[param] for param in param_group}


class TestQwen3DenseParamGroups(unittest.TestCase):
    def test_block_attention_and_ffn_delegate_grouping(self):
        model = qwen3_configs["debugmodel"]().build()
        block = model.layers["0"]

        attention_groups = block.attention.get_param_groups()
        feed_forward_groups = block.feed_forward.get_param_groups()
        block_groups = block.get_param_groups()
        param_to_name = {param: name for name, param in model.named_parameters()}

        self.assertEqual(
            _param_names(
                attention_groups[OptimizerGroup.BACKBONE_1D].no_decay_params,
                param_to_name,
            ),
            {
                "layers.0.attention.q_norm.weight",
                "layers.0.attention.k_norm.weight",
            },
        )
        self.assertEqual(
            _param_names(
                feed_forward_groups[OptimizerGroup.BACKBONE_2D].decay_params,
                param_to_name,
            ),
            {
                "layers.0.feed_forward.w1.weight",
                "layers.0.feed_forward.w2.weight",
                "layers.0.feed_forward.w3.weight",
            },
        )
        self.assertTrue(
            {
                "layers.0.attention_norm.weight",
                "layers.0.ffn_norm.weight",
            }.issubset(
                _param_names(
                    block_groups[OptimizerGroup.BACKBONE_1D].no_decay_params,
                    param_to_name,
                )
            )
        )

    def test_dense_grouping(self):
        model = qwen3_configs["debugmodel"]().build()

        param_groups = model.get_param_groups()
        param_to_name = {param: name for name, param in model.named_parameters()}

        self.assertEqual(
            _param_names(
                param_groups[OptimizerGroup.EMBEDDING].no_decay_params, param_to_name
            ),
            {"tok_embeddings.weight"},
        )
        self.assertEqual(
            _param_names(param_groups[OptimizerGroup.HEADS].decay_params, param_to_name),
            {"output.weight"},
        )
        self.assertEqual(param_groups[OptimizerGroup.HEADS].no_decay_params, [])
        self.assertEqual(param_groups[OptimizerGroup.BACKBONE_1D].decay_params, [])
        self.assertEqual(param_groups[OptimizerGroup.BACKBONE_2D].no_decay_params, [])

        backbone_1d_names = _param_names(
            param_groups[OptimizerGroup.BACKBONE_1D].no_decay_params,
            param_to_name,
        )
        self.assertTrue(
            {
                "layers.0.attention.q_norm.weight",
                "layers.0.attention.k_norm.weight",
                "layers.0.attention_norm.weight",
                "layers.0.ffn_norm.weight",
                "norm.weight",
            }.issubset(backbone_1d_names)
        )

        backbone_2d_names = _param_names(
            param_groups[OptimizerGroup.BACKBONE_2D].decay_params,
            param_to_name,
        )
        self.assertTrue(
            {
                "layers.0.attention.wq.weight",
                "layers.0.attention.wk.weight",
                "layers.0.attention.wv.weight",
                "layers.0.attention.wo.weight",
                "layers.0.feed_forward.w1.weight",
                "layers.0.feed_forward.w2.weight",
                "layers.0.feed_forward.w3.weight",
            }.issubset(backbone_2d_names)
        )

        grouped_params = [
            param
            for group in param_groups.values()
            for bucket in (group.decay_params, group.no_decay_params)
            for param in bucket
        ]
        named_params = [
            param for _, param in model.named_parameters() if param.requires_grad
        ]
        self.assertEqual(len(grouped_params), len(named_params))
        self.assertEqual(
            {id(param) for param in grouped_params},
            {id(param) for param in named_params},
        )

    def test_param_group_debug_config_uses_multigroup_optimizer(self):
        config = qwen3_debugmodel_param_groups()
        self.assertIsInstance(config.optimizer, MultiGroupOptimizersContainer.Config)

    def test_model_grouping_works_after_activation_checkpoint_wrapping(self):
        model = qwen3_configs["debugmodel"]().build()
        apply_ac(
            model,
            ActivationCheckpointConfig(mode="selective"),
        )

        param_groups = model.get_param_groups()

        self.assertTrue(param_groups[OptimizerGroup.BACKBONE_2D])


if __name__ == "__main__":
    unittest.main()
