# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict weight import for the published feedforward G1 ONNX actor."""

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def import_onnx_actor(actor, path, base_width=480):
    """Copy every actor layer; zero appended-input columns to preserve old outputs.

    Optimizer, critic and exploration variance are not present in an ONNX actor.
    Unsupported graphs fail explicitly instead of partially copying weights.
    """
    graph = onnx.load(str(path)).graph
    operations = [node.op_type for node in graph.node]
    if operations != ["Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]:
        raise ValueError(f"Unsupported ONNX graph: {operations}")
    if graph.node[-1].output[0] != graph.output[0].name:
        raise ValueError("Expected the final Gemm to produce the ONNX action output.")
    arrays = {value.name: numpy_helper.to_array(value).copy() for value in graph.initializer}
    linear_layers = [layer for layer in actor.mlp if isinstance(layer, torch.nn.Linear)]
    nodes = [node for node in graph.node if node.op_type == "Gemm"]
    if len(linear_layers) != len(nodes):
        raise ValueError("PPO actor architecture differs from the ONNX actor.")
    previous = graph.input[0].name
    for index, node in enumerate(graph.node):
        if node.input[0] != previous:
            raise ValueError("Expected a single feedforward chain.")
        previous = node.output[0]
        if node.op_type == "Elu":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get("alpha", 1.0) != 1.0:
                raise ValueError("Only ELU alpha=1 is supported.")
    with torch.no_grad():
        for index, (layer, node) in enumerate(zip(linear_layers, nodes, strict=True)):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get("transA", 0) or attrs.get("alpha", 1.0) != 1 or attrs.get("beta", 1.0) != 1:
                raise ValueError("Unsupported ONNX Gemm attributes.")
            weight = arrays[node.input[1]]
            weight = weight if attrs.get("transB", 0) else weight.T
            bias = arrays[node.input[2]]
            expected = (layer.out_features, base_width if index == 0 else layer.in_features)
            if weight.shape != expected or bias.shape != (layer.out_features,):
                raise ValueError(f"Layer {index} shape mismatch: {weight.shape} vs {expected}")
            layer.weight.zero_()
            layer.weight[:, : weight.shape[1]].copy_(torch.as_tensor(weight, device=layer.weight.device))
            layer.bias.copy_(torch.as_tensor(bias, device=layer.bias.device))
    return {
        "base_observation_width": base_width,
        "extra_observation_width": linear_layers[0].in_features - base_width,
        "copied_layers": len(nodes),
        "zero_initial_extra_input_weights": True,
    }


def check_onnx_parity(actor, observations, path, base_width=480):
    """Compare actual initialized PPO means against ONNX Runtime, without learning."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    values = observations["policy"]
    source = values[:, :base_width].detach().cpu().numpy().astype(np.float32)
    expected = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in source])
    with torch.inference_mode():
        actual = actor(observations).detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
    return float(np.abs(actual - expected).max())
