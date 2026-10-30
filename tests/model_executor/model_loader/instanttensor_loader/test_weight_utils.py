# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.fixture
def mock_instanttensor(monkeypatch):
    from vllm.model_executor.model_loader import weight_utils

    native_open = MagicMock()
    reader = native_open.return_value.__enter__.return_value
    reader.total_tensor_size = 0
    reader.tensors.return_value = iter(())
    monkeypatch.setitem(
        sys.modules, "instanttensor", SimpleNamespace(safe_open=native_open)
    )
    group = SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setattr(weight_utils, "get_world_group", lambda: group)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "current_device", lambda: 0)
    return native_open, group


@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("separate_shard", [False, True])
def test_instanttensor_engram_routing(
    mock_instanttensor, tmp_path, cpu_offload, separate_shard
):
    native_open, group = mock_instanttensor
    cpu_weights = {
        "layers.1.engram.embed.weight": torch.arange(16)
        .reshape(4, 4)
        .to(torch.float8_e4m3fn),
        "layers.1.engram.embed.scale": torch.tensor(
            [0, 127, 128, 254], dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
    }
    gpu_weights = {
        "layers[1].engram.wkv.weight": torch.ones(2),
        "layers.1.other_engram.embed.weight": torch.ones(2),
        "layers.1.engram.embed.weight_scale": torch.ones(1),
    }
    path = tmp_path / "model.safetensors"
    save_file(gpu_weights if separate_shard else gpu_weights | cpu_weights, path)
    paths = [str(path)]
    if separate_shard:
        path = tmp_path / "engram.safetensors"
        save_file(cpu_weights, path)
        paths.append(str(path))
    expected_gpu = gpu_weights if cpu_offload else gpu_weights | cpu_weights
    reader = native_open.return_value.__enter__.return_value
    reader.tensors.return_value = iter(expected_gpu.items())

    loaded = list(
        instanttensor_weights_iterator(paths, False, load_engram_on_cpu=cpu_offload)
    )
    kwargs = native_open.call_args.kwargs
    assert kwargs["copy"] is True
    assert kwargs["process_group"] is (None if cpu_offload else group.device_group)
    if cpu_offload:
        assert set(kwargs["tensor_filter"]) == expected_gpu.keys()
        assert kwargs["tensor_name_match_mode"] == "exact"
    else:
        assert "tensor_filter" not in kwargs
    expected = gpu_weights | cpu_weights
    assert len(loaded) == len(expected)
    assert dict(loaded).keys() == expected.keys()
    for name, tensor in loaded:
        assert tensor.dtype == expected[name].dtype
        assert tensor.shape == expected[name].shape
        assert torch.equal(tensor.view(torch.uint8), expected[name].view(torch.uint8))
        if cpu_offload and name in cpu_weights:
            assert tensor.device.type == "cpu"


@pytest.mark.parametrize("name", ["engram.embed.weight", "layers.1.engram.embed.scale"])
def test_instanttensor_cpu_only_checkpoint(mock_instanttensor, tmp_path, name):
    native_open, _ = mock_instanttensor
    path = tmp_path / "engram.safetensors"
    weight = torch.arange(4).reshape(2, 2)
    save_file({name: weight}, path)
    loaded = dict(
        instanttensor_weights_iterator([str(path)], False, load_engram_on_cpu=True)
    )
    native_open.assert_not_called()
    torch.testing.assert_close(loaded[name], weight)


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_instanttensor_no_engram_keeps_collective(
    mock_instanttensor, tmp_path, monkeypatch, cpu_offload
):
    from vllm.model_executor.model_loader import weight_utils

    native_open, group = mock_instanttensor
    path = tmp_path / "model.safetensors"
    save_file({"attention.weight": torch.ones(2)}, path)
    if not cpu_offload:
        monkeypatch.setattr(
            weight_utils, "safe_open", MagicMock(side_effect=AssertionError)
        )
    list(
        instanttensor_weights_iterator(
            [str(path)], False, load_engram_on_cpu=cpu_offload
        )
    )
    assert native_open.call_args.kwargs["process_group"] is group.device_group
    assert "tensor_filter" not in native_open.call_args.kwargs


@pytest.mark.parametrize(
    ("cpu_offload", "architecture", "layers", "expected"),
    [
        (None, "DeepseekV41ForCausalLM", [1], True),
        (False, "DeepseekV41ForCausalLM", [1], False),
        (True, "DeepseekV41ForCausalLM", [1], True),
        (None, "DeepseekV41ForCausalLM", [], False),
        (None, "OtherModel", [1], False),
        (None, None, [], False),
    ],
)
def test_instanttensor_loader_config(
    monkeypatch, cpu_offload, architecture, layers, expected
):
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import default_loader

    config = SimpleNamespace(
        engram_config=(
            None if cpu_offload is None else SimpleNamespace(cpu_offload=cpu_offload)
        ),
        model_config=(
            SimpleNamespace(
                architecture=architecture,
                hf_text_config=SimpleNamespace(engram_layer_ids=layers),
            )
            if architecture
            else None
        ),
    )
    monkeypatch.setattr(
        default_loader, "get_current_vllm_config_or_none", lambda: config
    )
    iterator = MagicMock(return_value=iter(()))
    monkeypatch.setattr(default_loader, "instanttensor_weights_iterator", iterator)
    loader = default_loader.DefaultModelLoader(LoadConfig(load_format="instanttensor"))
    monkeypatch.setattr(
        loader, "_prepare_weights", lambda *args: ("", ["model.safetensors"], True)
    )
    list(loader._get_weights_iterator(loader.Source("model", None)))
    assert iterator.call_args.kwargs["load_engram_on_cpu"] is expected


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="InstantTensor requires NVIDIA GPUs"
)
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_instanttensor_engram_cuda(tmp_path, cpu_offload):
    weights = {
        "layers[1].engram.wkv.weight": torch.ones(2),
        "layers.1.engram.embed.weight": torch.arange(8)
        .reshape(4, 2)
        .to(torch.float8_e4m3fn),
        "layers.1.engram.embed.scale": torch.tensor(
            [0, 127, 128, 254], dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
    }
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    loaded = dict(
        instanttensor_weights_iterator(
            [str(path)], False, load_engram_on_cpu=cpu_offload
        )
    )
    assert loaded.keys() == weights.keys()
    for name, tensor in loaded.items():
        cpu_table = cpu_offload and ".embed." in name
        assert tensor.device.type == ("cpu" if cpu_table else "cuda")
        assert tensor.dtype == weights[name].dtype
        assert torch.equal(
            tensor.cpu().view(torch.uint8), weights[name].view(torch.uint8)
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    safetensors = glob.glob(f"{model_dir}/*.safetensors")
    assert len(safetensors) > 0

    instanttensor_tensors = {}
    hf_safetensors_tensors = {}

    for name, tensor in instanttensor_weights_iterator(safetensors, True):
        instanttensor_tensors[name] = tensor.to("cpu")

    for name, tensor in safetensors_weights_iterator(safetensors, True):
        hf_safetensors_tensors[name] = tensor

    assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

    for name, instanttensor_tensor in instanttensor_tensors.items():
        assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


if __name__ == "__main__":
    test_instanttensor_model_loader()
