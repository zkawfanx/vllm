# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor import parameter


@pytest.mark.parametrize(
    ("param_cls", "kwargs"),
    [
        (parameter.BasevLLMParameter, {}),
        (parameter.ModelWeightParameter, {"input_dim": 1, "output_dim": 0}),
    ],
)
def test_parameter_ops_skip_python_dispatch(monkeypatch, param_cls, kwargs):
    """Weight loading and eager ops must retain plain Parameter dispatch semantics."""
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)

    def weight_loader(param, weight):
        param.load_column_parallel_weight(weight)

    param = param_cls(data=torch.zeros(4, 4), weight_loader=weight_loader, **kwargs)
    pointer = param.data_ptr()
    param.weight_loader(param, torch.ones(4, 4))

    assert type(param) is param_cls
    assert param.data_ptr() == pointer
    assert not param.requires_grad
    # Compare with Parameter because some platforms intercept all tensors.
    plain = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)
    assert torch.overrides.has_torch_function((param,)) == (
        torch.overrides.has_torch_function((plain,))
    )
    out = param + 1
    assert type(out) is torch.Tensor
    assert torch.equal(out, torch.full((4, 4), 2.0))
