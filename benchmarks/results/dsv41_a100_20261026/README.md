# DeepSeek V4.1 A100 roadmap validation

This report supports human review of the fork's roadmap backports and the
correctness fixes found while validating them. Validation finished on
2026-10-26. All ten owned test servers were stopped afterward.

The focused tests and serving checks passed after the fixes below. **Model
quality parity has not been established:** the frozen 100-question GSM8K
comparison scored 95 for the baseline, 93 for the graph candidate and 94 for
the eager candidate at reasoning effort 50. These limitations remain review
items; this report is not a recommendation to merge or deploy.

## Source and review scope

- Target: `zkawfanx/vllm`, branch `codex/dsv41-flash-a100`.
- Baseline: `78da347b7dbcf3ba52d96cadd7fe3b16d65cde9f`.
- Final source in this historical validation: `d6c94fa2e8b9d5f5233b559c69feeab28c94b886`.
- [Ordered source commits and donor references](commits.json): 27 commits.
- [Roadmap issue](https://github.com/zkawfanx/vllm/issues/18) remains open;
  this bundle covers selected items, not the entire roadmap.

The original evidence commit adds only this report and data. Each result identifies
the source used for that run; earlier results are not relabelled as a complete
rerun at the final source revision. There are no native C++/CUDA, schema or
build-system changes in the source series. The existing SM80 KV584 ABI and
seven loaded native libraries were preserved and verified per successful run.

The subsequent review correction to image-profile geometry is not covered by
these historical model runs. It fixes custom aspect-ratio/min-pixel combinations
and has a separate focused validation handoff; the recorded outcomes below
remain attached to their original source revisions.

Review the source commits in these groups:

| Area | Resulting behavior / upstream references |
| --- | --- |
| Python frontend | 1017 reasoning defaults, tool-turn history and image/UUID order; #58316, #60792, #62199 |
| Tools and parser | Bounded schema/cardinality/filtering changes with XGrammar 0.2.8; omitted DSML `string` attribute; #56408, #50502, #59879, #58019, #56271 |
| Mapping and speculation | Int64 MRV2 offsets; never-proposed token rejection and independent sampling streams/seeds; #60736, #50939, #58784, #54282, #61161 |
| Routing and image input | Draft padding, V4.1 image-bias fallback, stable negative scores and maximum-feature profiling; #59166, #54566/#56599, #57898 |
| Causal image configuration | Remove obsolete bidirectional-prefix widening; #57152. The baseline already had causal index kernels; no baseline attention-leak reproduction is claimed. |
| Engram and pipeline stages | Stage-entry injection, asynchronous offload prefetch with per-layer completion, collapse before SP gather; #62208, #56512/#56926, #56903 |
| DSpark context and loader | Context combination/native KV graph capture; dtype-aware dummy initialization; #57632/#59511, #56688/#61134 |
| Validation corrections | Responses stream completion, 1D token sharding, MoE dispatch, E8M0 transfer and routing numerics, described below |

Rust frontend, full grammar/namespace/allowed-tool parity, encoder graph
migrations, FlashInfer upgrades, MegaMoE vision/native ABI changes,
DP-sharded/shared-host Engram tables, LiLiCorr and broader fusion work remain
outside this series. PP2 tests omit DSpark because this source explicitly
does not support DSpark with PP greater than one.

## Duplicate-work check

[The refreshed search record](related-work.json) contains commands, results
and dispositions. The only open fork PR was
[#19](https://github.com/zkawfanx/vllm/pull/19), the common-device mapping
optimization. That work is excluded here; MRV2 int64 overflow is a separate
correctness fix. Fork PRs #20 and #21 are already in the baseline.

This is a fork integration draft, not a competing upstream proposal. Related
upstream work is explicitly recorded, including merged
[#59307](https://github.com/vllm-project/vllm/pull/59307) for non-Harmony
streamed-item reuse, open Responses proposals #54064/#56285/#58018, routing
fallback #63687, and EPLB alias deduplication #63075. Merged
[#61786](https://github.com/vllm-project/vllm/pull/61786) is the direct upstream
analogue for byte-view EPLB transfers; this fork uses a narrower E8M0-only view
at `RoutedExperts.get_expert_weights`. It is distinct from alias deduplication.
The local corrections
must not be presented as novel upstream fixes. Broad searches capped at 100
results are discovery only; the record also contains narrower searches and
inspection of the related proposals.

## Tests and failure controls

[Focused results](focused-tests.json) contain case identifiers, outcomes,
timings, pytest arguments, source revisions and raw-artifact digests.

| Coverage | Result |
| --- | --- |
| Mapping / routing / sampling / rejection | 8 / 170 / 22 / 51 passed |
| Context and scheduler / Engram / collapse / dummy weights | 31 / 16 / 6 / 21 passed |
| Eight-GPU Engram collectives | 1 passed, exercising 12 offload/SP/graph-mode combinations |
| Distinct focused total | **326 passed**, including CPU/configuration controls |
| Responses CPU regressions | 65 passed, 1 existing Harmony expected failure |
| Token-ID sharding / existing MoE routing controls | 5 / 4 passed |
| E8M0/BF16 transfer storage cases | 4 passed |
| Two-GPU transfer harness | NCCL and Gloo staging, row/column-major scales, all 256 byte patterns passed |

The additional rows overlap earlier coverage and are not summed with 326.
[Historical CPU results](cpu-history.json) report 615 cumulative distinct
cases across the Oct25 source series. That is incremental coverage, not a
615-case run at final HEAD, and overlaps the Oct26 suites.

Sampling checks cover FP32/FP64 and 500,000-draw distributions. The routing
suite includes signed padding and CUDA graph replay. Distributed Engram
checks compare values bit for bit with offload and SP on/off in eager, full
and breakable graph modes.

[Failures and fixes](failures-and-fixes.json) retain the negative outcomes:

| Commit | Observed failure and correction |
| --- | --- |
| `2c9e9a3b51` | Correct a test's valid-proposal expectation for standard versus joint block verification; 51 rejection cases pass. No production change. |
| `0cea8a4474` | Four A100 routing cases fail near logits -20 through -17; vision-only `log1p` preserves small positive scores. All 170 routing cases pass. |
| `92b7822093` | Seven of 28 live API cases fail because completion regenerates streamed item IDs. Reuse completed stream items, retain logprobs and honor hidden reasoning; 28/28 plus two metadata checks pass. This implements the contract already addressed upstream by #59307 on the older fork. |
| `6dfc982c6e` | All eight TP4/DP2 workers assert when 1D input IDs reach `sp_shard`. Pad/shard IDs along the token dimension; five CPU cases and final serving pass. |
| `935c68509f` | All eight workers then assert on local dispatch token counts. Forward the model's SP setting to fused MoE; final serving passes. |
| `d6c94fa2e8` | EPLB profiling then fails because NCCL cannot gather E8M0. Expose storage-aliasing uint8 transfer views, preserving parameter dtype/storage. Two E8M0 unit controls fail before and four storage cases pass after. Distributed transfer and final serving pass. |

Resolved harness issues were missing public tokenizer/config fixtures, a
multiprocessing launcher missing its main guard, a graph-only component
probe mistakenly attempted on eager serving, and a 600-second startup timeout
during checkpoint I/O. The retry used 1800 seconds. The eager probe restored
scratch bindings in its cleanup; serving/API/quality/benchmark checks then
completed. These attempts are not counted as successful component evidence.

## Real-model evidence

[Model results and launch arguments](model-results.json) include all successful
run identities, per-case outcomes, graph/component checks and native hashes.
Hardware was eight A100-SXM4 80 GB GPUs with the unchanged
DeepSeek-V4.1-Flash-1017 checkpoint. The checkpoint was reused; the entire
checkpoint was not hashed. Runtime versions: Python 3.12.13, Torch
2.13.0+cu130, Triton 3.7.1, CUDA toolkit 13.1 and XGrammar 0.2.8.

| Run | Source | Topology | Integration |
| --- | --- | --- | --- |
| Baseline graph | `78da347b7d` | TP8/EP/DP1, DSpark5 | 13/13 |
| Candidate graph | `0cea8a4474` | TP8/EP/DP1, DSpark5 | 13/13 |
| Candidate eager | `92b7822093` | TP8/EP/DP1, DSpark5 | 13/13; API 28/28; metadata 2/2 |
| Candidate PP2 | `92b7822093` | TP4/PP2, partition 14/26, target only | 13/13 |
| Final SP/EPLB | `d6c94fa2e8` | TP4/DP2/EP8, SP, DSpark5, target-only EPLB | 13/13; load check passed |

Integration includes real images, the legal 42882-by-42 maximum-feature aspect
ratio, approximately 33K-token mixed/chunked prefill, shrinking batches and
greedy/stochastic decoding. All 11 greedy candidate graph/eager token
sequences agree (maximum selected-token logprob difference 3.35e-5). All 11
final SP/EPLB greedy sequences agree with TP8 graph (maximum 2.66e-5).

Checkpoint probes compare both real Engram layers to synchronous lookup and
DSpark context preparation to eager execution. Three replays with changed
inputs/padding/slots match bit for bit, including three uint8 native KV caches
of shape `[8, 32, 584]`. Eight graph workers passed. Final DP2 probes execute
on all eight workers; the RPC returns only the first DP engine's four results.
PP2 stage ownership was verified on all workers, with Engram layer 14 at the
entry of the second pipeline stage.

The final EPLB run registers 40 target layers and zero draft layers per worker.
Five load requests include four forced 256-token outputs. Eight synchronous
rearrangements completed in **37.97–40.58 seconds** each, with window size 32,
step interval 64 and zero redundant experts. This intentionally frequent
correctness/load stress does not establish deployment latency or throughput.
Async EPLB, redundant experts, multiple nodes and elastic EP were not tested.

## Quality and throughput

[Every GSM8K scorer outcome](quality-cases.json) is included without prompts,
answers or generated reasoning. The dataset, prompt bundle and evaluator
hashes are in `model-results.json`. Selection: first 100 test questions, first
five training examples, temperature 0, seed 42, concurrency 8, max output
2048; original last-number scorer and stop strings retained.

| Source/mode | Effort 50 | Default 75 | Transport/server errors |
| --- | --- | --- | --- |
| Baseline graph | 95/100 | Not run | 0/100 |
| Candidate graph | 93/100 | 90/100 | 0/200 |
| Candidate eager | 94/100 | 92/100 | 0/200 |

Each default-75 run has one token-cap truncation; effort 50 has none. The
`Question` stop string can end reasoning before the final answer. The legacy
last-number scorer can misread currency decimals and answers containing unit
conversions. Scores were not selectively corrected. Relative to the baseline,
the graph run loses scored questions 58/93/98 and gains 73; eager loses 93/98
and gains 73 (zero-based indices). This sample establishes neither quality
parity nor a statistically supported regression. Quality was measured on the
recorded TP8 revisions, not rerun in the final SP/EPLB topology.

| Input tokens | Concurrency | Baseline output tok/s | Graph output tok/s | Eager output tok/s |
| --- | --- | --- | --- | --- |
| 512 | 1 | 113.0 | 111.3 | 16.8 |
| 512 | 8 | 386.9 | 428.6 | 117.1 |
| 8192 | 1 | 81.2 | 94.1 | 23.9 |
| 8192 | 8 | 116.4 | 142.7 | 97.8 |

Each server completed 96 benchmark requests: 16 warmups and 80 measured,
two rounds per input/concurrency pair, fixed 128 output tokens, zero failures.
Raw token prompts bypass effort defaults and vary early tokens to avoid prefix
reuse. Baseline 512-token/C8 rounds varied from 330.9 to 465.8 tok/s.
The baseline actually captured 15 PIECEWISE target graphs; the graph candidate
captured seven FULL target and six FULL draft graphs. Final SP/EPLB used 15
PIECEWISE target captures. The results describe combined changes with different
effective graph paths, not an isolated optimization or a general speedup.

## Reproduction and static checks

[Commands and workload protocol](REPRODUCE.md) describe the test environment,
representative pytest invocations, serving configurations and client protocol.
The JSON records contain the full per-suite pytest arguments and recorded
server arguments. Machine-specific runtime scaffolding and instrumentation
are not published as portable harnesses.

Changed-file hooks and manual Python 3.12 typing checks were run during the
source series. **Typing is not wholly green:** two pre-existing MegaMoE
keyword errors (`shared_x_sf`, `shared_block_m`) were reproduced at the exact
baseline, as were three FlashInfer metadata/API typing errors on the causal
image subset. No suppression was added. The affected commits used documented
hook skips; passing subset checks do not imply a clean repository-wide mypy
run. The existing zero-delta Harmony lifecycle expected failure also remains.

## Sanitization and accountability

These are allowlisted structured extracts from the recorded validation, not
replacement raw logs or an independent attestation. They preserve test names,
source revisions, configuration, failures, counts, metrics and content hashes.
Machine/user paths, host addresses, process IDs, request IDs, account/credit
information, dataset answers and raw generated text/reasoning are omitted.
Placeholders in server arguments identify redacted runtime inputs. Digests of
private raw inputs support later matching without disclosing their contents.
Raw artifacts remain local. `SHA256SUMS` covers every published file here
except itself.

AI assistance (Codex) was used for implementation, tests, analysis and this
report. This draft is for the human submitter's line-by-line review and relevant
test reruns under `AGENTS.md`; those human actions have not been attested by
the agent. No merge or deployment is authorized by this report.
