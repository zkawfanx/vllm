# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small sequence-parallel collectives shared by model frontends."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    communicator = get_tp_group().device_communicator
    collective = getattr(communicator, name, None) if communicator else None
    return collective(x) if collective is not None else None


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    result = _custom_collective("custom_all_gather", x)
    return result if result is not None else tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    padding = (-x.shape[0]) % tp_size
    if padding:
        x = torch.nn.functional.pad(x, (0, 0, 0, padding))
    result = _custom_collective("custom_reduce_scatter", x)
    return result if result is not None else tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim in (1, 2)
    tp_size = get_tensor_model_parallel_world_size()
    padding = (-x.shape[0]) % tp_size
    if padding:
        pad = (0, padding) if x.ndim == 1 else (0, 0, 0, padding)
        x = torch.nn.functional.pad(x, pad)
    chunk = x.shape[0] // tp_size
    rank = get_tensor_model_parallel_rank()
    return x[rank * chunk : (rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens
    tp_size = get_tensor_model_parallel_world_size()
    padding = (-num_tokens) % tp_size
    if padding:
        is_padding = torch.nn.functional.pad(is_padding, (0, padding), value=True)
    chunk = is_padding.shape[0] // tp_size
    rank = get_tensor_model_parallel_rank()
    return is_padding[rank * chunk : (rank + 1) * chunk]
