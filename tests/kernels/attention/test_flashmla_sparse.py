# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


def test_deepseek_v4_c128a_dynamic_topk_packed_buffers():
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    device = torch.device("cuda")
    capacity_width = 256
    active_width = 128
    global_decode_buffer = torch.empty(
        (2, capacity_width), dtype=torch.int32, device=device
    )
    decode_lens_buffer = torch.empty(2, dtype=torch.int32, device=device)
    prefill_buffer = torch.empty((2, capacity_width), dtype=torch.int32, device=device)

    global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
        positions=torch.tensor([255, 511], dtype=torch.int64, device=device),
        compress_ratio=128,
        num_decode_tokens=1,
        token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32, device=device),
        block_table=torch.tensor([[3]], dtype=torch.int32, device=device),
        block_size=capacity_width,
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64, device=device),
        global_decode_buffer=global_decode_buffer,
        decode_lens_buffer=decode_lens_buffer,
        prefill_buffer=prefill_buffer,
        max_compressed_tokens=active_width,
    )

    assert global_decode.shape == (1, active_width)
    assert prefill_local.shape == (1, active_width)
    assert global_decode.stride() == (active_width, 1)
    assert prefill_local.stride() == (active_width, 1)
    assert global_decode[0, :2].cpu().tolist() == [768, 769]
    assert decode_lens.cpu().tolist() == [2]
    assert prefill_local[0, :4].cpu().tolist() == list(range(4))
    assert torch.all(global_decode[0, 2:] == -1)
    assert torch.all(prefill_local[0, 4:] == -1)


def test_sparse_flashmla_metadata_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 128
    num_heads_k = 1
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    topk = 128

    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )
    assert isinstance(tile_md, fm.FlashMLASchedMeta)
    assert tile_md.tile_scheduler_metadata is None
    assert tile_md.num_splits is None
    assert num_splits is None


def test_sparse_flashmla_decode_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 64
    head_dim_k = 576
    head_dim_v = 512
    num_heads_k = 1
    page_block_size = 64
    bytes_per_token = 656
    topk = 128

    # Metadata
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    # q_heads_per_hk = num_heads_q // num_heads_k
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )

    # Inputs
    q = torch.zeros(
        (batch_size, seqlen_q, num_heads_q, head_dim_k),
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = torch.zeros(
        (1, page_block_size, num_heads_k, bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    indices = torch.zeros(
        (batch_size, seqlen_q, topk), dtype=torch.int32, device=device
    )

    block_table = torch.zeros((batch_size, 128), dtype=torch.int32, device=device)
    out, lse = fm.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_md,
        num_splits,
        indices=indices,
        is_fp8_kvcache=True,
    )
    assert out.shape[0] == batch_size
    assert out.shape[-1] == head_dim_v
    assert lse.shape[0] == batch_size


@pytest.mark.parametrize("h_q", [64, 128])
def test_sparse_flashmla_prefill_smoke(h_q: int):
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    torch.manual_seed(0)
    s_q = 1
    s_kv = 8
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128
    q = torch.randn((s_q, h_q, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=torch.bfloat16, device=device)
    indices = torch.randint(s_kv, (s_q, h_kv, topk), dtype=torch.int32, device=device)
    reference_indices = indices.clone()
    reference_indices[..., 1:] = -1
    kwargs = {"topk_length": torch.ones(1, dtype=torch.int32, device=device)}
    reference = fm.flash_mla_sparse_fwd(q, kv, reference_indices, 1.0, d_v, **kwargs)
    actual = fm.flash_mla_sparse_fwd(q, kv, indices, 1.0, d_v, **kwargs)

    for actual_tensor, reference_tensor in zip(actual, reference):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)
    assert actual[0].shape == (s_q, h_q, d_v)


def test_deepseek_v4_prefill_chunk_planning_expands_for_short_sequences():
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        block_size=64,
        num_prefills=5,
        prefill_seq_lens_cpu=torch.tensor([80, 96, 112, 128, 144], dtype=torch.int32),
        prefill_query_lens_cpu=torch.tensor([4, 4, 4, 4, 4], dtype=torch.int32),
        prefill_window_size=64,
        prefill_max_model_len=1024,
        prefill_max_num_batched_tokens=128,
    )

    chunk_plan = metadata.get_prefill_chunk_plan(compress_ratio=4, prefill_chunk_size=4)

    # the adaptive plan keeps all 5 in one chunk
    assert chunk_plan == [(0, 5, 36, 103)]


def test_flashinfer_sparse_indices_cache(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as flashinfer_mod
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    builder_calls = 0

    def fake_build(*args, **kwargs):
        nonlocal builder_calls
        builder_calls += 1
        return (
            torch.tensor([[builder_calls]], dtype=torch.int32),
            torch.tensor([builder_calls], dtype=torch.int32),
        )

    monkeypatch.setattr(
        flashinfer_mod, "build_flashinfer_mixed_sparse_indices", fake_build
    )

    def make_attn(compress_ratio: int, topk_width: int):
        attn = object.__new__(flashinfer_mod.DeepseekV4FlashInferMLAAttention)
        attn.compress_ratio = compress_ratio
        attn.window_size = 4
        attn.topk_indices_buffer = torch.tensor(
            [[0, 1], [2, 3], [4, 5]], dtype=torch.int32
        )[:, :topk_width]
        return attn

    def make_swa_metadata():
        return DeepseekSparseSWAMetadata(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
            block_size=64,
            seq_lens=torch.tensor([8, 10], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1, 1], dtype=torch.int32),
            decode_swa_indices=torch.tensor([[5, 6, -1, -1]], dtype=torch.int32),
            decode_swa_lens=torch.tensor([2], dtype=torch.int32),
            is_valid_token=torch.tensor([True], dtype=torch.bool),
            num_decodes=1,
            num_prefills=1,
            num_decode_tokens=1,
            num_prefill_tokens=2,
        )

    def make_flashmla_metadata():
        return DeepseekV4FlashMLAMetadata(
            num_reqs=2,
            max_query_len=2,
            max_seq_len=10,
            num_actual_tokens=3,
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
            block_size=256,
            topk_tokens=2,
            c128a_global_decode_topk_indices=torch.tensor(
                [[[9, 10]]], dtype=torch.int32
            ),
            c128a_decode_topk_lens=torch.tensor([2], dtype=torch.int32),
            c128a_prefill_topk_indices=torch.tensor(
                [[0, 1], [1, 2]], dtype=torch.int32
            ),
        )

    swa_attn = make_attn(1, 0)
    swa_metadata = make_swa_metadata()
    _, _, sparse_indices_first, sparse_lens_first = (
        swa_attn._build_sparse_index_metadata(
            kv_cache=None,
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
        )
    )
    _, _, sparse_indices_second, sparse_lens_second = (
        swa_attn._build_sparse_index_metadata(
            kv_cache=None,
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
        )
    )
    assert builder_calls == 1
    assert sparse_indices_first is sparse_indices_second
    assert sparse_lens_first is sparse_lens_second

    c128a_attn = make_attn(128, 2)
    c128a_metadata = make_swa_metadata()
    c128a_flashmla_md = make_flashmla_metadata()
    _, _, sparse_indices_first, sparse_lens_first = (
        c128a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c128a_metadata,
            attn_metadata=c128a_flashmla_md,
            swa_only=False,
        )
    )
    _, _, sparse_indices_second, sparse_lens_second = (
        c128a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c128a_metadata,
            attn_metadata=c128a_flashmla_md,
            swa_only=False,
        )
    )

    assert builder_calls == 2
    assert sparse_indices_first is sparse_indices_second
    assert sparse_lens_first is sparse_lens_second

    c4a_attn = make_attn(4, 2)
    c4a_metadata = make_swa_metadata()
    c4a_flashmla_md = make_flashmla_metadata()
    c4a_flashmla_md.c128a_global_decode_topk_indices = None
    c4a_flashmla_md.c128a_decode_topk_lens = None
    c4a_flashmla_md.c128a_prefill_topk_indices = None
    _, _, sparse_indices_third, sparse_lens_third = (
        c4a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c4a_metadata,
            attn_metadata=c4a_flashmla_md,
            swa_only=False,
        )
    )
    _, _, sparse_indices_fourth, sparse_lens_fourth = (
        c4a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c4a_metadata,
            attn_metadata=c4a_flashmla_md,
            swa_only=False,
        )
    )

    assert builder_calls == 4
    assert sparse_indices_third is not sparse_indices_fourth
    assert sparse_lens_third is not sparse_lens_fourth


def _make_v41_swa_builder(device, num_tokens, vision_n_layers=2):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.sparse_mla import (
        DeepseekV41SparseSWAMetadataBuilder,
    )
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=4096,
            hf_config=DeepseekV41Config(
                text_config={
                    "sliding_window": 128,
                    "compress_ratios": [0, 1, 2],
                    "index_topk": 8,
                },
                vision_config={
                    "num_hidden_layers": vision_n_layers,
                    "max_image_tokens": 2048,
                },
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=num_tokens),
        speculative_config=None,
    )
    spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="auto",
        model_version="deepseek_v4",
    )
    return DeepseekV41SparseSWAMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer0"],
        vllm_config=config,
        device=device,
    )


@pytest.mark.cpu_test
def test_v41_image_swa_metadata_and_warmup_match_text():
    from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
        CombineTopkSwaIndicesKernel,
    )

    image_builder = _make_v41_swa_builder(torch.device("cpu"), 100)
    text_builder = _make_v41_swa_builder(torch.device("cpu"), 100, vision_n_layers=0)
    assert image_builder.max_image_tokens == text_builder.max_image_tokens == 0
    assert image_builder.prefill_index_width == text_builder.prefill_index_width == 128
    kernel = CombineTopkSwaIndicesKernel()
    image_keys = kernel.get_warmup_keys(image_builder.vllm_config)
    assert image_keys == kernel.get_warmup_keys(text_builder.vllm_config)
    assert {key.COMPRESS_RATIO for key in image_keys} == {0, 1, 2}


@pytest.mark.parametrize("compress_ratio", [0, 1, 2])
@pytest.mark.parametrize("query_len", [100, 4000])
@pytest.mark.parametrize("mixed", [False, True])
def test_v41_image_prefill_indices_are_causal(compress_ratio, query_len, mixed):
    """Full/chunked image prefills agree in gathered and paged sparse indices."""
    if not torch.accelerator.is_available():
        pytest.skip("requires CUDA or ROCm")
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
        build_flashinfer_mixed_sparse_indices,
        combine_topk_swa_indices,
    )

    device = torch.device("cuda")
    num_decode = int(mixed)
    seq_len, window = 4000, 128
    builder = _make_v41_swa_builder(device, query_len + num_decode)
    common = create_common_attn_metadata(
        BatchSpec(
            seq_lens=([10] if mixed else []) + [seq_len],
            query_lens=([1] if mixed else []) + [query_len],
        ),
        block_size=64,
        device=device,
        arange_block_indices=True,
    )
    common.mm_req_doc_ranges = {num_decode: [(1900, 3947)]}
    md = builder.build(0, common)
    assert builder.max_image_tokens == 0
    assert md.prefill_left_visible is md.prefill_right_visible is None
    assert md.prefill_swa_indices.shape[-1] == window
    [(start, end, n, m)] = md.get_prefill_chunk_plan(
        compress_ratio, prefill_chunk_size=4, has_compressed=compress_ratio > 0
    )
    gather_len = int(md.prefill_gather_lens[0])
    assert (start, end, m - n) == (0, 1, gather_len)
    assert gather_len == query_len + min(seq_len - query_len, window - 1)
    topk = torch.empty(query_len, 0, dtype=torch.int32, device=device)
    indices, lens = combine_topk_swa_indices(
        topk,
        md.query_start_loc[num_decode:],
        md.prefill_seq_lens,
        md.prefill_gather_lens,
        window,
        compress_ratio,
        0,
        m,
        n,
    )
    mixed_indices, mixed_lens = build_flashinfer_mixed_sparse_indices(
        md.decode_swa_indices[:, 0],
        None,
        None,
        topk,
        md.query_start_loc,
        md.seq_lens,
        md.token_to_req_indices,
        md.block_table,
        md.block_size,
        None,
        md.block_size,
        window,
        compress_ratio,
        0,
    )
    indices, lens = indices.cpu(), lens.cpu().tolist()
    paged_indices = md.prefill_swa_indices[:, 0].cpu()
    paged_lens = md.prefill_swa_lens.cpu().tolist()
    mixed_indices = mixed_indices.cpu()
    assert mixed_lens.cpu().tolist() == [window] * (query_len + num_decode)
    if mixed:
        assert torch.equal(mixed_indices[0], md.decode_swa_indices[0, 0].cpu())
    page_base = int(md.block_table[num_decode, 0]) * md.block_size
    for token, pos in enumerate(range(seq_len - query_len, seq_len)):
        lo, hi = max(0, pos - window + 1), pos + 1
        row = indices[token, : lens[token]]
        assert lens[token] == paged_lens[token] == hi - lo
        assert torch.all((row >= n) & (row < m))
        assert (row - n + seq_len - gather_len).tolist() == list(range(lo, hi))
        expected_slots = list(range(page_base + lo, page_base + hi))
        assert paged_indices[token, : hi - lo].tolist() == expected_slots
        assert mixed_indices[num_decode + token, : hi - lo].tolist() == expected_slots
        assert torch.all(mixed_indices[num_decode + token, hi - lo :] == -1)
