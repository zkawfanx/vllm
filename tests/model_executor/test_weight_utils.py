# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile

import huggingface_hub.constants
import pytest
import torch
from huggingface_hub.utils import LocalEntryNotFoundError

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    initialize_single_dummy_weight,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform


def test_download_weights_from_hf():
    with tempfile.TemporaryDirectory() as tmpdir:
        # assert LocalEntryNotFoundError error is thrown
        # if offline is set and model is not cached
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        with pytest.raises(LocalEntryNotFoundError):
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )

        # download the model
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "facebook/opt-125m",
            allow_patterns=["*.safetensors", "*.bin"],
            cache_dir=tmpdir,
        )

        # now it should work offline
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        assert (
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )
            is not None
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    ("dtype", "value"),
    [(torch.float32, 0.0), (torch.float8_e4m3fn, 1.0), (torch.uint8, 127)],
)
def test_dummy_value_override_preserves_tensor_identity(device, dtype, value):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    parameter = torch.nn.Parameter(
        torch.empty(8, dtype=dtype, device=device), requires_grad=False
    )
    parameter.dummy_weight_value = value
    storage = parameter.data_ptr()
    initialize_single_dummy_weight(parameter, name="weight_scale")
    assert parameter.data_ptr() == storage
    assert parameter.dtype == dtype
    assert parameter.float().tolist() == [value] * 8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_default", [False, True])
def test_dummy_range_and_seed_are_reproducible(dtype, use_default):
    parameter = torch.empty(32, dtype=dtype)
    expected = torch.empty(
        32, dtype=torch.float16 if dtype == torch.float8_e4m3fn else dtype
    )
    bound = 1.0 if use_default and dtype == torch.float8_e4m3fn else 1e-3
    expected.uniform_(-bound, bound, generator=torch.Generator().manual_seed(1234))
    if use_default:
        initialize_single_dummy_weight(parameter)
    else:
        initialize_single_dummy_weight(parameter, low=-1e-3, high=1e-3)
    assert torch.equal(parameter.float(), expected.to(dtype).float())


def test_dummy_value_override_is_deferred_on_meta():
    parameter = torch.empty(8, device="meta")
    parameter.dummy_weight_value = 1.0
    initialize_single_dummy_weight(parameter)
    assert parameter.device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/UVA")
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_engram_dummy_values_only_apply_to_cpu_offload(monkeypatch, cpu_offload):
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.models.deepseek_v4_1.common import engram

    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: 0)
    with torch.device("cuda"):
        layer = engram.ParallelEngramEmbedding(8, 32, (4, 4), cpu_offload=cpu_offload)
    if not cpu_offload:
        assert not hasattr(layer.weight, "dummy_weight_value")
        assert not hasattr(layer.weight_scale_inv, "dummy_weight_value")
        return
    weight, scale = layer.weight, layer.weight_scale_inv
    storage = (weight.data_ptr(), scale.data_ptr())
    assert weight.is_pinned() and scale.is_pinned()
    DummyModelLoader(LoadConfig(load_format="dummy")).load_weights(layer, None)
    assert layer.weight is weight and layer.weight_scale_inv is scale
    assert (weight.data_ptr(), scale.data_ptr()) == storage
    assert torch.equal(weight.float(), torch.ones(8, 32))
    assert scale.tolist() == [[127]] * 8
    indices = torch.tensor([[0, 4], [3, 7]], device="cuda", dtype=torch.int64)
    output = torch.empty((2, 2, 32), dtype=torch.bfloat16, device="cuda")
    layer.lookup(indices, output)
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_dummy_fp8_weight_is_not_degenerate(dtype):
    """+-1e-3 underflows in FP8, so 8-bit floats default to +-1."""
    param = torch.empty(4096, dtype=dtype)
    initialize_single_dummy_weight(param)
    values = param.float()
    assert values.isfinite().all()
    assert 0.5 < values.abs().max() <= 1.0
    assert (values != 0).float().mean() > 0.99


@pytest.mark.parametrize(
    ("dtype", "name"),
    [(torch.float8_e8m0fnu, "weight_scale_inv"), (torch.uint8, "w13_weight_scale")],
)
def test_dummy_e8m0_scale_is_fixed_exponent(dtype, name):
    """E8M0 scales, including uint8-stored ones, get a finite 2**-7."""
    param = torch.empty(64, dtype=dtype)
    initialize_single_dummy_weight(param, name=name)
    assert (param.view(torch.uint8) == 120).all()


@pytest.mark.skipif(
    current_platform.is_tpu(), reason="TPU does not fill packed weights"
)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.float4_e2m1fn_x2])
def test_dummy_packed_weight_gets_random_bytes(dtype):
    """Packed weights are filled deterministically instead of left uninitialized."""
    # Some backends can't fill packed FP4 directly, so view zeroed bytes.
    params = [torch.zeros(4096, dtype=torch.uint8).view(dtype) for _ in range(2)]
    for param in params:
        initialize_single_dummy_weight(param, name="w13_weight")
    raw = params[0].view(torch.uint8)
    assert raw.unique().numel() > 250
    assert torch.equal(raw, params[1].view(torch.uint8))


@pytest.mark.skipif(current_platform.is_rocm(), reason="ROCm zeroes integer params")
def test_dummy_wide_integer_param_is_untouched():
    """Wider integer params may be indices, so random values are unsafe."""
    param = torch.arange(16, dtype=torch.int32)
    initialize_single_dummy_weight(param, name="g_idx")
    assert torch.equal(param, torch.arange(16, dtype=torch.int32))


class TestMaybeRemapKvScaleName:
    """Tests for maybe_remap_kv_scale_name covering all checkpoint formats."""

    PARAMS_DICT = {
        "model.layers.0.self_attn.attn.k_scale": None,
        "model.layers.0.self_attn.attn.v_scale": None,
        "model.layers.0.self_attn.attn.q_scale": None,
        "model.layers.0.self_attn.qkv_proj.weight": None,
    }

    def test_qkv_proj_k_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.k_scale -> attn.k_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qkv_proj_v_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.v_scale -> attn.v_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_modelopt_k_proj_k_scale(self):
        """ModelOpt format: k_proj.k_scale -> attn.k_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_modelopt_v_proj_v_scale(self):
        """ModelOpt format: v_proj.v_scale -> attn.v_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_deprecated_kv_scale(self):
        """Old format: kv_scale -> attn.k_scale (deprecated)"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.kv_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_default_bare_k_scale(self):
        """Default format: .k_scale -> .attn.k_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_non_scale_name_unchanged(self):
        """Non-scale names should be returned unchanged."""
        name = "model.layers.0.self_attn.qkv_proj.weight"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_modelopt_k_proj_k_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        k_proj.k_scale -> attn.k_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_nvfp4_modelopt_v_proj_v_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        v_proj.v_scale -> attn.v_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_qwen3_vl_moe_qkv_proj_k_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qwen3_vl_moe_qkv_proj_v_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_nvfp4_weight_scale_not_remapped(self):
        """NVFP4 weight_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.weight_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_input_scale_not_remapped(self):
        """NVFP4 input_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.input_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_missing_target_returns_none(self):
        """If remapped name not in params_dict, return None."""
        empty_params: dict[str, None] = {}
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", empty_params
        )
        assert result is None


class TestKvCacheScaleMapper:
    """The `WeightsMapper` returned by `get_cache_scale_mapper` replaces the
    per-model `maybe_remap_kv_scale_name` calls. It must remap the same set of
    checkpoint formats (the non-`params_dict`-dependent ones) and be idempotent
    so it composes safely with a model's own qkv/gate_up `hf_to_vllm_mapper`."""

    def _mapper(self):
        # `get_cache_scale_mapper` does not use `self`; call it on the base
        # class to get the default (non-config-specific) mapper.
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizationConfig,
        )

        return QuantizationConfig.get_cache_scale_mapper()

    def _map(self, name: str) -> str | None:
        return self._mapper()._map_name(name)

    @pytest.mark.parametrize(
        "name,expected",
        [
            # Qwen3-MoE / llm-compressor fused qkv_proj
            (
                "model.layers.0.self_attn.qkv_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.qkv_proj.v_scale",
                "model.layers.0.self_attn.attn.v_scale",
            ),
            # ModelOpt / NVFP4 k_proj/v_proj
            (
                "model.layers.0.self_attn.k_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.v_proj.v_scale",
                "model.layers.0.self_attn.attn.v_scale",
            ),
            # deprecated fused kv_scale and bare scales
            (
                "model.layers.0.self_attn.kv_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            # NemotronH mixer
            (
                "model.layers.0.mixer.k_proj.k_scale",
                "model.layers.0.mixer.attn.k_scale",
            ),
            # already in vLLM form -> unchanged (idempotent)
            (
                "model.layers.0.self_attn.attn.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            # non-kv scales must not be touched
            (
                "model.layers.0.self_attn.k_proj.weight_scale",
                "model.layers.0.self_attn.k_proj.weight_scale",
            ),
            (
                "model.layers.0.self_attn.k_proj.input_scale",
                "model.layers.0.self_attn.k_proj.input_scale",
            ),
            # regular weights untouched
            (
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ),
        ],
    )
    def test_remap(self, name, expected):
        assert self._map(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.k_proj.k_scale",
            "model.layers.0.self_attn.qkv_proj.v_scale",
            "model.layers.0.mixer.k_proj.k_scale",
        ],
    )
    def test_idempotent(self, name):
        once = self._map(name)
        assert once is not None
        assert self._map(once) == once

    def test_composes_with_qkv_mapper(self):
        """Applied together with a model's qkv/gate_up mapper, the regex scale
        rules run before the substr rename, so scales are normalized to `.attn.`
        and regular projections are still fused correctly."""
        from vllm.model_executor.models.utils import WeightsMapper

        model_mapper = WeightsMapper(
            orig_to_new_substr={
                ".q_proj": ".qkv_proj.q",
                ".k_proj": ".qkv_proj.k",
                ".v_proj": ".qkv_proj.v",
            }
        )
        # AutoWeightsLoader does `mapper |= cache_scale_mapper`
        combined = model_mapper | self._mapper()

        assert (
            combined._map_name("model.layers.0.self_attn.q_proj.weight")
            == "model.layers.0.self_attn.qkv_proj.q.weight"
        )
        assert (
            combined._map_name("model.layers.0.self_attn.k_proj.k_scale")
            == "model.layers.0.self_attn.attn.k_scale"
        )
        assert (
            combined._map_name("model.layers.0.self_attn.k_scale")
            == "model.layers.0.self_attn.attn.k_scale"
        )


if __name__ == "__main__":
    test_download_weights_from_hf()
