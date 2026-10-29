# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.kernels.mhc  # noqa: F401
from vllm.model_executor.kernels.mhc.tilelang import (
    _tilelang_hc_prenorm_gemm,
    _torch_hc_prenorm_gemm,
    mhc_pre_broadcast_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre reference kernel from tilelang repo: https://github.com/tile-ai/tilelang/blob/d135bd1cd2d2eee74fbb41dd0a0831a427194c86/examples/deepseek_mhc/example_mhc_pre.py#L303"""
    hc_mult = residual.shape[-2]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )

    hc_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ],
    )
    mixes = mixes * hc_scale + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

    res_mix = sinkhorn_normalize_ref(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )

    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post reference kernel from tilelang repo: https://github.com/tile-ai/tilelang/blob/d135bd1cd2d2eee74fbb41dd0a0831a427194c86/examples/deepseek_mhc/example_mhc_post.py#L68"""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def hc_head_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    residual_flat = residual.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * residual.float(), dim=-2).bfloat16()


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("fuse_norm", [False, True])
def test_mhc_pre_tilelang(num_tokens, hidden_size, hc_mult, fuse_norm):
    """``fuse_norm`` selects the RMSNorm-fused kernel, which is the variant
    DeepSeek V4 actually runs (the model always passes ``norm_weight``)."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = 2 * hc_mult + hc_mult2
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(hc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1

    hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0
    norm_eps = 1e-5
    norm_weight = (
        torch.randn((hidden_size,), dtype=torch.bfloat16) * 0.1 + 1.0
        if fuse_norm
        else None
    )

    post_mix_ref, res_mix_ref, layer_input_ref = mhc_pre_ref(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    if norm_weight is not None:
        li = layer_input_ref.float()
        layer_input_ref = (
            li
            * torch.rsqrt(li.square().mean(-1, keepdim=True) + norm_eps)
            * norm_weight.float()
        ).bfloat16()

    out = torch.ops.vllm.mhc_pre_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        1,
        norm_weight,
        norm_eps,
    )

    ref = (post_mix_ref, res_mix_ref, layer_input_ref)
    for actual, expected in zip(out, ref, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_broadcast_tilelang(num_tokens, hidden_size, hc_mult):
    """First-layer variant: the (T, H) residual is broadcast across the
    hc_mult streams, so the result must match ``mhc_pre_ref`` on the
    explicitly expanded residual. RMSNorm fusion is mandatory here (the
    wrapper asserts ``norm_weight``), matching how the model calls it."""
    _run_mhc_pre_broadcast_case(num_tokens, hidden_size, hc_mult)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_broadcast_tilelang_without_deep_gemm(
    num_tokens, hidden_size, hc_mult, monkeypatch
):
    """Same case with DeepGEMM forced unavailable.

    The broadcast variant is the one that reaches the prenorm GEMM through a
    separate branch from its two siblings, and on a pre-Hopper device the
    DeepGEMM branch aborts in ``hyperconnection.hpp`` rather than returning
    wrong numbers. Below SM90 this is what the test above already runs, so
    the point of forcing it is coverage on hardware where DeepGEMM *is*
    supported and the fallback would otherwise never execute.
    """
    monkeypatch.setattr(
        "vllm.utils.deep_gemm.is_deep_gemm_supported", lambda *a, **kw: False
    )
    _run_mhc_pre_broadcast_case(num_tokens, hidden_size, hc_mult)


def _run_mhc_pre_broadcast_case(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = 2 * hc_mult + hc_mult2
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(hc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    # The model precomputes fn_broadcast this way in
    # finalize_mhc_broadcast_weights: summing fn over the hc_mult axis is
    # exactly the GEMM against a residual that is identical in every stream.
    fn_broadcast = fn.view(hc_mult3, hc_mult, hidden_size).sum(dim=1)
    hc_scale = torch.randn((3,), dtype=torch.float) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1

    hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0
    norm_eps = 1e-5
    norm_weight = torch.randn((hidden_size,), dtype=torch.bfloat16) * 0.1 + 1.0

    residual_ref = x.unsqueeze(-2).expand(num_tokens, hc_mult, hidden_size)
    post_mix_ref, res_mix_ref, layer_input_ref = mhc_pre_ref(
        residual_ref,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    li = layer_input_ref.float()
    layer_input_ref = (
        li
        * torch.rsqrt(li.square().mean(-1, keepdim=True) + norm_eps)
        * norm_weight.float()
    ).bfloat16()

    residual_out, post_mix, res_mix, layer_input = mhc_pre_broadcast_tilelang(
        x,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        1,
        norm_weight,
        norm_eps,
        fn_broadcast=fn_broadcast,
    )

    # The materialized broadcast must be an exact copy of the input rows.
    torch.testing.assert_close(residual_out, residual_ref.contiguous(), rtol=0, atol=0)
    torch.testing.assert_close(post_mix, post_mix_ref, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(res_mix, res_mix_ref, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(layer_input, layer_input_ref, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize(
    ("num_tokens", "hidden_size"),
    [
        # T crosses the routing boundary (_PRENORM_SMALL_T = 32): below it the
        # one-CTA-per-token tilelang kernel runs (fp32 fn, strict tolerance);
        # at and above it the cuBLAS bf16 route runs, whose fn and output are
        # rounded to bf16, bounded at rel 5e-3 directly on `out` — the
        # downstream mhc_pre tolerance alone cannot distinguish bf16-fn
        # rounding from a broken kernel.
        (1, 1280),
        (31, 1280),
        (32, 1280),
        (512, 1280),
        (2048, 1280),
        (1, 4096),
        (31, 4096),
        (32, 4096),
        (64, 4096),
        (512, 4096),
        (2048, 4096),
        (1, 7168),
        (31, 7168),
        (32, 7168),
        (64, 7168),
        (512, 7168),
        (2048, 7168),
    ],
)
def test_hc_prenorm_gemm_tilelang(num_tokens, hidden_size):
    from vllm.model_executor.kernels.mhc.tilelang import _PRENORM_SMALL_T

    torch.set_default_device(DEVICE)
    set_random_seed(0)

    hc_mult = 4
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    x = torch.randn((num_tokens, hc_mult * hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult3, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    out_ref = torch.empty((1, num_tokens, hc_mult3), dtype=torch.float32)
    sqrsum_ref = torch.empty((1, num_tokens), dtype=torch.float32)
    out = torch.empty_like(out_ref)
    sqrsum = torch.empty_like(sqrsum_ref)

    _torch_hc_prenorm_gemm(x, fn, out_ref, sqrsum_ref)
    _tilelang_hc_prenorm_gemm(x, fn, out, sqrsum, hidden_size, hc_mult)

    if num_tokens < _PRENORM_SMALL_T:
        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-4)
    else:
        # bf16 fn plus bf16 GEMM output: rel 5e-3 against the tensor scale.
        scale = float(out_ref.abs().max())
        torch.testing.assert_close(out, out_ref, atol=5e-3 * scale, rtol=5e-3)
    torch.testing.assert_close(sqrsum, sqrsum_ref, atol=8.0, rtol=5e-4)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_post_tilelang(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    out = torch.ops.vllm.mhc_post_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )

    torch.testing.assert_close(out, ref, atol=5e-2, rtol=1e-2)


@pytest.mark.parametrize("sqrsum_ready", [False, True])
def test_prenorm_router_drops_sqrsum_when_already_produced(monkeypatch, sqrsum_ready):
    """``sqrsum_ready`` must reach the cuBLAS route as ``sqrsum=None``.

    This one line is what makes VLLM_MHC_POST_FUSE_SQRSUM and
    VLLM_MHC_PRENORM_SHARD safe together: the fold owns ``gemm_out_sqrsum`` and
    the shard owns ``gemm_out_mul``, and they stay disjoint only because the
    shard is handed None and returns before recomputing the fold's quantity. A
    refactor that let the shard recompute it would break the pair silently, so
    pin the routing rather than the arithmetic.
    """
    # Fetch the submodule through importlib: the package does
    # `from .triton import *`, which rebinds the name `triton` on the package
    # to the external library, so both the dotted-string form and
    # `from ...mhc import triton` hand back the wrong module.
    import importlib

    from vllm.model_executor.kernels.mhc import tilelang as tl_mod

    mhc_triton = importlib.import_module("vllm.model_executor.kernels.mhc.triton")

    seen = {}

    def _spy(x, fn, out, sqrsum):
        seen["sqrsum_is_none"] = sqrsum is None

    monkeypatch.setattr(mhc_triton, "hc_prenorm_gemm_cublas", _spy)

    hidden, hc_mult, num_tokens = 4096, 4, 64
    k = hc_mult * hidden
    x = torch.zeros(num_tokens, k, dtype=torch.bfloat16)
    fn = torch.zeros(24, k, dtype=torch.float32)
    out = torch.zeros(1, num_tokens, 24, dtype=torch.float32)
    sqrsum = torch.zeros(1, num_tokens, dtype=torch.float32)

    tl_mod._tilelang_hc_prenorm_gemm(
        x, fn, out, sqrsum, hidden, hc_mult, sqrsum_ready=sqrsum_ready
    )
    assert seen["sqrsum_is_none"] is sqrsum_ready


def test_prenorm_shard_requires_the_sqrsum_fold(monkeypatch):
    """The shard must stay off when it still owes a sqrsum.

    Sharding costs one all-gather per boundary. If the caller has not already
    produced the sqrsum, the shard would need a second gather and hand most of
    the win back, so the pairing with VLLM_MHC_POST_FUSE_SQRSUM is enforced in
    the kernel rather than left to whoever writes the serve flags. Reaching for
    the TP group at all is the observable: it only happens on the shard path.
    """
    import importlib

    mhc_triton = importlib.import_module("vllm.model_executor.kernels.mhc.triton")
    monkeypatch.setattr(mhc_triton.envs, "VLLM_MHC_PRENORM_SHARD", True)

    reached = []

    def model_parallel_is_initialized():
        reached.append(True)
        return False

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.model_parallel_is_initialized",
        model_parallel_is_initialized,
    )

    # Real device tensors: the not-None branch runs _row_sqrsum_kernel for
    # real. num_tokens is the smallest value that clears the shard threshold.
    torch.set_default_device(DEVICE)
    num_tokens, k = 512, 4096 * 4
    x = torch.zeros(num_tokens, k, dtype=torch.bfloat16)
    fn = torch.zeros(24, k, dtype=torch.float32)
    out = torch.zeros(1, num_tokens, 24, dtype=torch.float32)
    sqrsum = torch.zeros(1, num_tokens, dtype=torch.float32)

    mhc_triton.hc_prenorm_gemm_cublas(x, fn, out, sqrsum)
    assert not reached, "shard must not engage while it still owes a sqrsum"

    mhc_triton.hc_prenorm_gemm_cublas(x, fn, out, None)
    assert reached, "shard must engage once the fold has produced the sqrsum"


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [511, 512, 513, 1024, 8191, 8192])
def test_prenorm_shard_rows_partition_exactly(num_tokens, tp_size):
    """The prenorm shard's row split must tile the tokens exactly.

    ``all_gatherv`` reassembles the GEMM output from these counts, so a split
    that over- or under-covers would put a wrong row into the sinkhorn rather
    than fail loudly. Sizes are also kept within one row of each other because
    the slowest rank sets the collective's arrival time.
    """
    from vllm.model_executor.kernels.mhc.triton import (
        _PRENORM_SHARD_MIN_TOKENS,
        _prenorm_shard_rows,
    )

    rows = _prenorm_shard_rows(num_tokens, tp_size)
    if tp_size == 1 or num_tokens < _PRENORM_SHARD_MIN_TOKENS:
        assert rows is None, "must stay replicated below the threshold or at TP=1"
        return
    assert len(rows) == tp_size
    assert sum(rows) == num_tokens, "split must cover every token exactly once"
    assert max(rows) - min(rows) <= 1, "split must be balanced within one row"
    assert min(rows) > 0, "no rank may be given an empty slice"


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 128, 2048])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_post_sqrsum_matches_standalone_reduction(num_tokens, hidden_size, hc_mult):
    """The fused sqrsum must equal the separate pass it replaces.

    Above the cuBLAS crossover the prenorm GEMM's companion reduction is a
    second read of exactly what mhc_post just wrote, so folding it into
    mhc_post is only sound if it reduces the *stored* bf16 values -- squaring
    the fp32 pre-rounding value would drift from _row_sqrsum_kernel.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_post_sqrsum_tilelang,
    )
    from vllm.model_executor.kernels.mhc.triton import _row_sqrsum_kernel

    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    fused_out = torch.empty_like(residual)
    fused_sqrsum = torch.empty((1, num_tokens), dtype=torch.float32)
    mhc_post_sqrsum_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        fused_out,
        fused_sqrsum,
        hc_mult,
        hidden_size,
    )

    ref_out = torch.ops.vllm.mhc_post_tilelang(
        x, residual, post_layer_mix.unsqueeze(-1), comb_res_mix
    )
    ref_sqrsum = torch.empty((1, num_tokens), dtype=torch.float32)
    k = hc_mult * hidden_size
    ref_flat = ref_out.view(num_tokens, k)
    _row_sqrsum_kernel[(num_tokens,)](
        ref_flat, ref_sqrsum[0], ref_flat.stride(0), k, BLOCK_K=1024, num_warps=4
    )

    # The residual must be bit-identical: same arithmetic, same rounding.
    # The sqrsum only agrees to summation order -- the two kernels tree-reduce
    # hc_mult * hidden fp32 terms differently.
    torch.testing.assert_close(fused_out, ref_out, atol=0, rtol=0)
    torch.testing.assert_close(fused_sqrsum, ref_sqrsum, atol=0, rtol=1e-4)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_fused_post_pre(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(hc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1

    hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0

    def run_ref():
        residual_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
        post_mix_ref, res_mix_ref, layer_input_ref = mhc_pre_ref(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_alpha,
            sinkhorn_repeat,
        )
        return residual_ref, post_mix_ref, res_mix_ref, layer_input_ref

    residual_ref, post_mix_ref, res_mix_ref, layer_input_ref = run_ref()

    residual, post_mix, res_mix, x = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )

    torch.testing.assert_close(residual, residual_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(post_mix, post_mix_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(res_mix, res_mix_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(x, layer_input_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head_triton(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((1,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32) * 0.1
    rms_eps = hc_eps = 1e-6

    out = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16)
    out.fill_(float("nan"))

    result = torch.ops.vllm.hc_head_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )

    assert result is None
    assert not torch.isnan(out).any()

    out_ref = hc_head_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_eps)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head_tilelang(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((1,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32) * 0.1
    rms_eps = hc_eps = 1e-6

    out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
    )

    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()

    out_ref = hc_head_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_eps)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("tp_size", [4, 8])
def test_hc_collapse_before_gather_is_bitwise_equal(num_tokens, tp_size):
    """Simulate SP on one GPU, including padding and ranks without valid tokens."""
    set_random_seed(0)
    padded_tokens = num_tokens + (-num_tokens) % tp_size
    streams = torch.randn(padded_tokens, 4, 5120, dtype=torch.bfloat16, device=DEVICE)
    pre_mix = torch.rand(padded_tokens, 4, device=DEVICE)
    stream_shards = streams.chunk(tp_size)
    mix_shards = pre_mix.chunk(tp_size)
    before = hc_collapse_triton(
        torch.cat(stream_shards)[:num_tokens], torch.cat(mix_shards)[:num_tokens]
    )
    after = torch.cat(
        [hc_collapse_triton(x, pre) for x, pre in zip(stream_shards, mix_shards)]
    )[:num_tokens]
    assert after.shape == (num_tokens, 5120)
    torch.testing.assert_close(after, before, atol=0, rtol=0)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("tokens", [0, 1, 6, 8, 16, 17, 32])
@pytest.mark.parametrize("hidden", [2048, 4096, 5120, 7168, 5136])
def test_delayed_fusion_dispatch_covers_only_complete_tiles(tokens, hidden):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_delayed_fused_config

    config = mhc_delayed_fused_config(tokens, hidden, 4)
    if not 0 < tokens <= 16 or hidden == 5136:
        assert config is None
    else:
        assert config is not None
        tile, splits, threads = config
        assert 24 % tile == 0
        assert hidden % (splits * threads) == 0


@pytest.mark.skipif(not HAS_TILELANG_MHC, reason="TileLang MHC support required")
@pytest.mark.parametrize("tokens", [0, 1, 6, 8, 16, 17, 128])
@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_delayed_fusion_matches_projection_reference(
    tokens, hidden, carried, normalized
):
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_delayed_fused_config,
        mhc_fused_post_pre_delayed_tilelang,
        mhc_post_tilelang,
        mhc_pre_delayed_tilelang,
    )

    set_random_seed(0)
    residual = torch.randn(tokens, 4, hidden, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    previous_post = torch.rand(tokens, 4, 1, device=DEVICE)
    previous_comb = torch.rand(tokens, 4, 4, device=DEVICE)
    fn = torch.randn(24, 4 * hidden, device=DEVICE) * 0.02
    scale = torch.tensor([0.5, 0.25, 1.0], device=DEVICE)
    base = torch.randn(24, device=DEVICE)
    pre = torch.rand(tokens, 4, device=DEVICE) if carried else None
    norm = (
        torch.ones(hidden, device=DEVICE, dtype=torch.bfloat16) if normalized else None
    )
    args = (
        x,
        residual,
        previous_post,
        previous_comb,
        fn,
        scale,
        base,
        1e-20,
        1e-6,
        1e-6,
        2.0,
        20,
        pre,
        norm,
        1e-6,
    )
    actual = mhc_fused_post_pre_delayed_tilelang(*args)
    assert [v.shape for v in actual] == [
        (tokens, 4, hidden),
        (tokens, 4, 1),
        (tokens, 4, 4),
        (tokens, hidden),
        (tokens, 4),
    ]
    if not tokens:
        return
    post = mhc_post_tilelang(x, residual, previous_post, previous_comb)
    reference = mhc_pre_delayed_tilelang(
        post,
        fn,
        scale,
        base,
        1e-20,
        1e-6,
        1e-6,
        2.0,
        20,
        pre_mix=pre,
        norm_weight=norm,
    )
    torch.testing.assert_close(actual[0], post, atol=0, rtol=0)
    torch.testing.assert_close(actual[3], reference[2], atol=0, rtol=0)
    if mhc_delayed_fused_config(tokens, hidden, 4) is None:
        for value, expected in zip(actual[1:], reference, strict=True):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
    else:
        # Fused projection uses the unrounded post, not its stored BF16 value.
        post_float = torch.einsum("tij,tih->tjh", previous_comb, residual.float())
        post_float += previous_post * x.float().unsqueeze(1)
        flat = post_float.flatten(1)
        mixes = (flat @ fn.T) * (flat.square().mean(-1, keepdim=True) + 1e-20).rsqrt()
        coefficients = (
            mixes
            * torch.cat((scale[0].expand(4), scale[1].expand(4), scale[2].expand(16)))
            + base
        )
        expected = (
            coefficients[:, 4:8].sigmoid().unsqueeze(-1) * 2,
            sinkhorn_normalize_ref(coefficients[:, 8:].view(tokens, 4, 4), 20, 1e-6),
            coefficients[:, :4].sigmoid() + 1e-6,
        )
        for index, value in zip((1, 2, 4), expected, strict=True):
            torch.testing.assert_close(actual[index], value, atol=1e-5, rtol=1e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mhc_fused_post_pre_delayed_tilelang(*args)
    graph.replay()
    for value, expected in zip(captured, actual, strict=True):
        torch.testing.assert_close(value, expected, atol=0, rtol=0)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("tokens,hidden", [(0, 5120), (17, 5120), (1, 5136)])
def test_delayed_fusion_fallback_and_empty(monkeypatch, tokens, hidden):
    from vllm.model_executor.kernels.mhc import tilelang as module

    residual = torch.empty(tokens, 4, hidden, dtype=torch.bfloat16)
    x = torch.empty(tokens, hidden, dtype=torch.bfloat16)
    events = []

    def post(*args):
        events.append("post")
        return residual

    def pre(*args, **kwargs):
        events.append("pre")
        return ("post", "comb", "input", "next")

    monkeypatch.setattr(module, "mhc_post_tilelang", post)
    monkeypatch.setattr(module, "mhc_pre_delayed_tilelang", pre)
    out = module.mhc_fused_post_pre_delayed_tilelang(
        x, residual, None, None, None, None, None, 1e-6, 1e-6, 1e-6, 2.0, 20
    )
    assert len(out) == 5
    assert events == (["post", "pre"] if tokens else [])
    if not tokens:
        assert [t.shape for t in out] == [
            (0, 4, hidden),
            (0, 4, 1),
            (0, 4, 4),
            (0, hidden),
            (0, 4),
        ]


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_delayed_fusion_warmup_includes_fused_splits(monkeypatch):
    from vllm.model_executor.kernels.mhc import warmup

    monkeypatch.setattr(warmup, "is_deep_gemm_supported", lambda: False)
    keys = warmup.MHC_PRE_NORM_KERNEL.get_warmup_keys(
        max_tokens=128,
        extra_splits=(4, 16, 4),
        hidden_size=5120,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_eps=1e-6,
        hc_mult=4,
        use_pre_mix_in=True,
        rms_numel=20480,
    )
    assert sorted(key.n_splits for key in keys) == [1, 4, 16]
    assert all(key.rms_numel == 20480 for key in keys)
