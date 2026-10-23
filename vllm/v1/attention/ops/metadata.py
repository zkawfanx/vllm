"""Device-side token-to-request mapping."""

from vllm.triton_utils import tl, triton


@triton.jit
def _token_request(query_start_loc, token, num_reqs):
    lower = tl.full(token.shape, 0, tl.int32)
    upper = tl.full(token.shape, num_reqs, tl.int32)
    while tl.sum((lower < upper).to(tl.int32), 0) > 0:
        middle = (lower + upper) // 2
        end = tl.load(query_start_loc + middle + 1, middle < num_reqs, other=0x7FFFFFFF)
        right = (lower < upper) & (end <= token)
        lower = tl.where(right, middle + 1, lower)
        upper = tl.where((lower < upper) & ~right, middle, upper)
    return tl.minimum(lower, num_reqs - 1)


@triton.jit(do_not_specialize=["num_reqs", "num_mapped", "num_tokens"])
def _token_request_mapping_kernel(
    query_start_loc, output, num_reqs, num_mapped, num_tokens
):
    token = tl.program_id(0) * 256 + tl.arange(0, 256)
    request = _token_request(query_start_loc, token, num_reqs)
    tl.store(
        output + token, tl.where(token < num_mapped, request, 0), token < num_tokens
    )
