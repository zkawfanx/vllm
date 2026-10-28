# Validation commands and protocol

Use a uv-managed `.venv/bin/python` with the source revision identified by
the relevant evidence row. Tests used the existing native build recorded in
`model-results.json`, not a newly compiled replacement. Install repository
and test dependencies through `uv`; XGrammar was pinned to 0.2.8 for both
baseline and candidate. Public Qwen3-8B and DFlash tokenizer/config fixtures
must be cached for the scheduler cases. Model weights are not included.

## Focused suites

Run from the corresponding repository checkout. The original provenance
wrapper verified source SHA, imports and native hashes before/after pytest;
the equivalent direct pytest invocations below omit that local wrapper.
Some subsets intentionally bypass unrelated root test fixtures. GPU detection
must work even for CPU controls importing CUDA-gated model code.

```bash
.venv/bin/python -m pytest --confcutdir=tests -o addopts= -o usefixtures= -q \
  tests/v1/worker/test_gpu_input_batch_v2.py

.venv/bin/python -m pytest --confcutdir=tests -o addopts= -o usefixtures= -q \
  tests/kernels/moe/test_topk_softplus_sqrt.py \
  -k 'v41_vision_routing or dsv4_routing_with_padding or fused_topk_softplus_sqrt_padding'

.venv/bin/python -m pytest --confcutdir=tests -o addopts= -o usefixtures= -q \
  tests/v1/spec_decode/test_rejection_sampler_utils.py \
  tests/v1/worker/test_gpu_rejection_sampler_chunking.py

.venv/bin/python -m pytest --confcutdir=tests -o addopts= -o usefixtures= -q \
  tests/models/test_deepseek_v41_pipeline.py
```

`focused-tests.json` records all ten focused run argument lists; the distributed
Engram case requires eight visible GPUs. `cpu-history.json` contains the
historical CPU argument lists. In the original runs CUDA was hidden for those
CPU cases, public fixture loading was offline, and source/XGrammar import
locations were checked. The additional suites retain every selected JUnit
case ID. The two-GPU E8M0 check separately exercised profile and rearrangement
transfers through NCCL and Gloo staging with all byte patterns.

For a source review, use changed-file hooks, then the manual typing check;
consult the known baseline failures in the report before interpreting a
nonzero mypy result:

```bash
.venv/bin/pre-commit run --files <changed-files>
.venv/bin/pre-commit run mypy-3.12 --hook-stage manual --files <changed-files>
git diff --check
```

## Serving configurations

Set `MODEL_PATH` to the unchanged DeepSeek-V4.1-Flash-1017 checkpoint.
The command below is the portable serving portion of the recorded TP8 launch.
The local worker extension used for identity and scratch-buffer probes is
omitted; it does not exist in this repository. `recorded_server_argv` retains
its name to make that instrumentation boundary explicit.

```bash
export VLLM_USE_RUST_FRONTEND=0
export VLLM_ENFORCE_STRICT_TOOL_CALLING=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export OMP_NUM_THREADS=4

.venv/bin/python -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --enable-expert-parallel --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 --host localhost --port 8000 \
  --served-model-name DeepSeek-V4.1-Flash-roadmap \
  --max-num-seqs 8 --max-model-len 262144 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.88 --mm-encoder-tp-mode data \
  --engram-config '{"cpu_offload":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

The frontend reports cache block size **8**, the minimum across heterogeneous
cache groups, despite the explicit CLI block size 256. This does not change
the physical SM80 KV584 layout or make every cache group use eight-token pages.
The model results contain the resolved cache/compilation configuration; CLI
arguments alone do not establish the actual graph execution path.

- Eager comparison: add `--enforce-eager` on the recorded eager revision.
- PP2: set `VLLM_PP_LAYER_PARTITION=14,26`, TP4, PP2, DP1; omit speculative
  configuration. Verify automatic V1 selection and stage ownership.
- Final SP/EPLB: TP4, PP1, DP2, EP enabled, DSpark5, and add
  `--enable-eplb --eplb-config
  '{"window_size":32,"step_interval":64,"log_balancedness":true,"use_async":false}'`.
  Verify SP on both models, 40 registered target EPLB layers per worker, and
  no draft EPLB state. The local validation used eight GPUs on one node.

## Client workload definitions

The existing validation clients and raw responses remain local; the protocols
below and per-case structured extracts are published. Reconstructing these
protocols is a new run, not proof of byte-identical reproduction of the local
instrumentation.

**Integration:** two solid-red images (560 by 560 and 42882 by 42), eight
concurrent mixed requests (two approximately 33K-token prefills, one also with
an image), one subsequent short arithmetic request, and two stochastic
one-sentence requests. Greedy cases require exact expected content and stop
finish reasons. All requests use max output 128, seed 42, thinking disabled
and explicit effort 50; stochastic cases use temperature 0.7, top-p 0.9 and
seeds 42/43. Selected token logprobs and token sequences are compared across
candidate modes. This is not maximum-context certification.

**API:** the 28 named cases in `model-results.json` cover strict auto/required/
named tools, streaming/nonstreaming, thinking on/off, Responses text types,
tool continuation and item lifecycle consistency. Two additional metadata
requests check hidden reasoning and output token/top-token logprobs. Compare
streamed done items with completed and stored output, including item/call IDs.

**GSM8K:** use `tests/evals/gsm8k/gsm8k_eval.py`'s existing
`_build_gsm8k_prompts(100, 5, "")` and `get_answer_value`, with the dataset
hashes recorded in `model-results.json`. Retain the original order. Send
`/v1/chat/completions` requests with temperature 0, seed 42, max output 2048,
concurrency 8, timeout 600 seconds and stop strings
`["Question", "Assistant:", "<|separator|>"]`. Set
`chat_template_kwargs={"thinking":true,"reasoning_effort":50}` for matched
effort, and `{"thinking":true}` for candidate default 75. Candidate requests
alternate variant order by even/odd question index on the same instance.
Score content only, retaining invalid answers and truncations. Do not rescore
selectively or pool quality wall time into throughput measurements.

**Throughput:** tokenize `(first frozen GSM8K prompt + newline) * 20` once.
Use the first 512 or 8192 token IDs; replace token index 1 with
`1000 + (5000 if length == 8192 else 0) + round*200 + concurrency*20 + index`.
Use `/v1/completions`, temperature 0, seed 42, max output 128 and
`ignore_eos=true`. For each length, warm up eight C8 requests at round 0;
then run rounds 1 and 2, each with four C1 requests and sixteen C8 requests.
Check actual prompt/output token counts and all HTTP outcomes. Aggregate
output tokens divided by summed batch durations, excluding warmups. Report
both rounds; this is descriptive and the baseline/candidate graph paths differ.

**EPLB load:** four forced 256-output-token requests plus one short arithmetic
control. Count completed synchronous rearrangements in the serving log and
verify target/draft registration before and after. Pair this with the byte-copy
and integration evidence; the short rebalance interval is a stress condition.

## Evidence integrity

From this evidence directory:

```bash
sha256sum --check SHA256SUMS
```

The digests identify published bytes and private source artifacts. They do
not attest to human review or replace the explicitly bounded validation scope.
