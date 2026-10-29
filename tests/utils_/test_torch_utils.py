# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.utils.torch_utils import (
    LayerName,
    common_broadcastable_dtype,
    current_stream,
    is_lossless_cast,
    weak_ref_tensors,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_weak_ref_tensors_does_not_keep_nested_tensors_alive() -> None:
    # Graph entries must not pin auxiliary outputs for every capture size.
    hidden, aux = (torch.empty(1 << 20, device="cuda") for _ in range(2))
    layer_name = LayerName("model.layers.0.self_attn.attn")
    out = weak_ref_tensors((hidden, [{"aux": (aux,)}], 3, layer_name))
    assert isinstance(out, tuple)
    weak_hidden, weak_list, size, name = out
    assert isinstance(weak_list, list) and size == 3 and name is layer_name
    assert weak_hidden.data_ptr() == hidden.data_ptr()
    assert weak_list[0]["aux"][0].data_ptr() == aux.data_ptr()
    nbytes = hidden.nbytes + aux.nbytes
    allocated = torch.accelerator.memory_allocated()
    del hidden, aux
    assert torch.accelerator.memory_allocated() == allocated - nbytes


def test_weak_ref_tensors_preserves_non_tensor_leaves() -> None:
    sentinel = object()
    values = [None, 1.5, "layer", 3, sentinel, {"metadata": (None, sentinel)}]
    out = weak_ref_tensors((values,))
    assert isinstance(out, tuple) and isinstance(out[0], list)
    assert out == (values,)
    assert out[0][4] is sentinel
    assert out[0][5]["metadata"][1] is sentinel


@pytest.mark.parametrize("container_type", [list, tuple])
def test_weak_ref_tensors_traverses_sequence_subclasses(monkeypatch, container_type):
    class Outputs(container_type):
        pass

    tensor, alias, metadata = torch.empty(1), object(), object()
    monkeypatch.setattr("vllm.utils.torch_utils.weak_ref_tensor", lambda _: alias)
    original = Outputs([tensor, (tensor, metadata)])
    out = weak_ref_tensors(original)
    assert type(out) is container_type
    assert out[0] is alias and out[1][0] is alias
    assert out[1][1] is metadata
    assert original[0] is tensor and original[1][0] is tensor


@pytest.mark.parametrize("value", [None, 1.5, "layer", 3, {}, object()])
def test_weak_ref_tensors_rejects_invalid_root_types(value) -> None:
    with pytest.raises(ValueError):
        weak_ref_tensors(value)


@pytest.mark.parametrize(
    ("src_dtype", "tgt_dtype", "expected_result"),
    [
        # Different precision_levels
        (torch.bool, torch.int8, True),
        (torch.bool, torch.float16, True),
        (torch.bool, torch.complex32, True),
        (torch.int64, torch.bool, False),
        (torch.int64, torch.float16, True),
        (torch.int64, torch.complex32, True),
        (torch.float64, torch.bool, False),
        (torch.float64, torch.int8, False),
        (torch.float64, torch.complex32, True),
        (torch.complex128, torch.bool, False),
        (torch.complex128, torch.int8, False),
        (torch.complex128, torch.float16, False),
        # precision_level=0
        (torch.bool, torch.bool, True),
        # precision_level=1
        (torch.int8, torch.int16, True),
        (torch.int16, torch.int8, False),
        (torch.uint8, torch.int8, False),
        (torch.int8, torch.uint8, False),
        # precision_level=2
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float16, False),
        (torch.bfloat16, torch.float32, True),
        (torch.float32, torch.bfloat16, False),
        # precision_level=3
        (torch.complex32, torch.complex64, True),
        (torch.complex64, torch.complex32, False),
    ],
)
def test_is_lossless_cast(src_dtype, tgt_dtype, expected_result):
    assert is_lossless_cast(src_dtype, tgt_dtype) == expected_result


@pytest.mark.parametrize(
    ("dtypes", "expected_result"),
    [
        ([torch.bool], torch.bool),
        ([torch.bool, torch.int8], torch.int8),
        ([torch.bool, torch.int8, torch.float16], torch.float16),
        ([torch.bool, torch.int8, torch.float16, torch.complex32], torch.complex32),  # noqa: E501
    ],
)
def test_common_broadcastable_dtype(dtypes, expected_result):
    assert common_broadcastable_dtype(dtypes) == expected_result


def _test_stream_thread(main_expected_stream: torch.cuda.Stream):
    import threading

    child_stream = torch.cuda.Stream()
    thread_stream_ready = threading.Event()
    thread_can_exit = threading.Event()

    def child_thread_func():
        with torch.cuda.stream(child_stream):
            thread_stream_ready.set()
            thread_can_exit.wait(timeout=10)

    child_thread = threading.Thread(target=child_thread_func)
    child_thread.start()

    try:
        assert thread_stream_ready.wait(timeout=5), (
            "Child thread failed to enter stream context in time"
        )

        main_current_stream = current_stream()

        assert main_current_stream != child_stream, (
            "Main thread's current_stream was contaminated by child thread"
        )
        assert main_current_stream == main_expected_stream, (
            f"Main thread's stream changed unexpectedly. "
            f"Expected {main_expected_stream}, got {main_current_stream}"
        )

        thread_can_exit.set()

    finally:
        child_thread.join(timeout=5)
        if child_thread.is_alive():
            pytest.fail("Child thread failed to exit properly")


def test_current_stream_multithread():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    main_dedicated_stream = current_stream()

    assert main_dedicated_stream.cuda_stream != 0, (
        "ROCm/CUDA should create a dedicated stream, not use default stream (0x0)"
    )

    main_stream_again = current_stream()
    assert main_stream_again == main_dedicated_stream, (
        "Multiple calls to current_stream should return the same dedicated stream"
    )

    _test_stream_thread(main_dedicated_stream)
