# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest

import torch

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.config_registry import qwen3_debugmodel
from torchtitan.models.qwen3.optimizer import (
    MultiGroupOptimizersContainer,
    OptimizerGroup,
    ParamGroup,
)
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter


def _param_names(param_group, param_to_name):
    return {param_to_name[param] for param in param_group}


class TestQwen3DenseParamGroups(unittest.TestCase):
    def test_biasful_projections_route_biases_to_backbone_1d(self):
        config = qwen3_configs["debugmodel"]()
        layer0 = config.layers[0]

        attention = dataclasses.replace(
            layer0.attention,
            wq=dataclasses.replace(layer0.attention.wq, bias=True),
            wkv=dataclasses.replace(layer0.attention.wkv, bias=True),
            wo=dataclasses.replace(layer0.attention.wo, bias=True),
        )
        feed_forward = dataclasses.replace(
            layer0.feed_forward,
            w1=dataclasses.replace(layer0.feed_forward.w1, bias=True),
            w2=dataclasses.replace(layer0.feed_forward.w2, bias=True),
            w3=dataclasses.replace(layer0.feed_forward.w3, bias=True),
        )
        layers = list(config.layers)
        layers[0] = dataclasses.replace(
            layer0,
            attention=attention,
            feed_forward=feed_forward,
        )
        model = dataclasses.replace(config, layers=layers).build()

        block_groups = model.layers["0"].get_param_groups()
        param_to_name = {param: name for name, param in model.named_parameters()}

        self.assertTrue(
            {
                "layers.0.attention.wq.bias",
                "layers.0.attention.wk.bias",
                "layers.0.attention.wv.bias",
                "layers.0.attention.wo.bias",
                "layers.0.feed_forward.w1.bias",
                "layers.0.feed_forward.w2.bias",
                "layers.0.feed_forward.w3.bias",
            }.issubset(
                _param_names(
                    block_groups[OptimizerGroup.BACKBONE_1D].no_decay_params,
                    param_to_name,
                )
            )
        )

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
        config = qwen3_debugmodel()
        self.assertIsInstance(config.optimizer, MultiGroupOptimizersContainer.Config)

    def test_model_grouping_works_after_activation_checkpoint_wrapping(self):
        model = qwen3_configs["debugmodel"]().build()
        apply_ac(
            model,
            ActivationCheckpointConfig(mode="selective"),
        )

        param_groups = model.get_param_groups()

        self.assertTrue(param_groups[OptimizerGroup.BACKBONE_2D])

    def test_validate_param_groups_rejects_duplicate_params(self):
        model = qwen3_configs["debugmodel"]().build()
        groups = {group: ParamGroup() for group in OptimizerGroup}
        groups[OptimizerGroup.EMBEDDING].no_decay_params.append(
            model.tok_embeddings.weight
        )
        groups[OptimizerGroup.HEADS].decay_params.append(model.tok_embeddings.weight)

        with self.assertRaisesRegex(
            ValueError,
            "Parameter appears in both 'embedding' and 'heads' groups",
        ):
            model._validate_param_groups(groups)

    def test_state_dict_adapter_roundtrip_keeps_embedding_and_head_separate(self):
        config = qwen3_configs["debugmodel"]()
        model = config.build()
        adapter = Qwen3StateDictAdapter(config, None)
        state_dict = {
            "tok_embeddings.weight": model.tok_embeddings.weight.detach().clone(),
            "output.weight": model.output.weight.detach().clone(),
            "norm.weight": model.norm.weight.detach().clone(),
            "layers.0.attention.wq.weight": (
                model.layers["0"].attention.wq.weight.detach().clone()
            ),
        }

        hf_state_dict = adapter.to_hf(state_dict)
        self.assertIn("model.embed_tokens.weight", hf_state_dict)
        self.assertIn("lm_head.weight", hf_state_dict)

        roundtrip_state_dict = adapter.from_hf(hf_state_dict)
        self.assertEqual(set(roundtrip_state_dict), set(state_dict))
        for key, value in state_dict.items():
            self.assertTrue(torch.equal(roundtrip_state_dict[key], value))


if __name__ == "__main__":
    unittest.main()
