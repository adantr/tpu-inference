# Fixed TPU contract

- Repository: `adantr/tpu-inference`, upstream base
  `1bd693780c5fda65d85cc35f44473bdfa4a98f59`.
- Editable file: `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`.
  The default production caller must still reach it. No dispatcher edits,
  wrapper indirection, experimental batched route, or evaluation-aware branches.
- Target: RPA v3 decode for `Qwen/Qwen3-4B`, BF16 weights/activations/KV cache,
  tensor parallelism 1. Prefer a compatible `v5e-1`; use `v6e-1` if unavailable.
  Freeze the actual device and runtime before measuring; CPU checks establish
  no TPU performance or correctness result.
- Math: preserve scaled attention, softmax, causal/window masks, soft capping,
  tensor shapes/dtypes, accumulation policy, padding treatment, and all supported
  control paths. No lowered precision or tolerance relaxation. A decode edit
  must preserve prefill/mixed behavior through the same public function.
- State: preserve exact KV-cache writes, unchanged cache regions, alias/donation
  behavior, sequence/page indexing, and `update_kv_cache=False` semantics.
  No cached answers, skipped requests, shortened decoding, hard-coded outputs,
  host/network lookups, benchmark-state reads, or altered reference paths.
- Correctness: finalize a focused check exercising actual DECODE routing and
  production shapes before freezing. The upstream test named `decode_only`
  currently supplies distribution `[0,0,N]`, which routes through MIXED; its
  name alone is not decode coverage. Use its independent reference and exact
  initialized-KV comparison as a starting point. Its BF16 output tolerance is
  `atol=rtol=0.2`; this is an upstream floor, not a new precision allowance.
  Preserve the current arithmetic/precision by review and require a stricter
  production-shape comparison if the proposed change can alter numerical error.
  Freeze any such additional check before research. No failed, empty, or skipped
  suite qualifies. Supply a fixed pytest command; the driver requests JUnit
  output and hashes the check's source. Until this is finalized, freeze is blocked.
  Correct serving output counts alone are insufficient.
- Initial workload draft: 16 deterministic requests each at input lengths 256,
  2048, and 4096 crossed with concurrency 1 and 8; output length 128,
  seed 42, greedy decoding with EOS
  ignored, prefix caching disabled. Confirm these settings fit the selected TPU
  and freeze them before edits. Fixed vLLM source, tokenizer, seed, and arguments
  define the saved random workload; per-request lengths must remain identical.
  The cheap `screen` stage uses contexts 256 and 2048 at concurrency 8 and one
  pair; its output is feedback only. The full gate uses all six fixed workloads.
- Primary feedback: detailed JSON from the installed `vllm bench serve` command.
  Baseline and candidate get fresh servers and full discarded warmup passes,
  in four pairs ordered AB, BA, AB, BA. All runs and failures remain available.
- Selected metrics: output-token throughput (higher is better); mean, median
  (p50), and p99 TTFT, TPOT, ITL, and end-to-end latency (lower is better), for
  every selected workload. Missing, zero, or nonfinite values invalidate a run.
- Integrity: no request failures/errors or changed per-request input/output token
  counts; generated output counts and aggregate token counts must match. Freeze
  server/client settings, model revision, tokenizer, dependencies, source,
  correctness checks, metric set, and thresholds before candidate evaluation.
- Acceptance: zero permitted regression. Every observed paired gain must be
  nonnegative on every selected metric/workload, and at least one metric must
  improve by at least 1% in every pair. A metric slower in every pair rejects
  the candidate; mixed signs or insufficient gains are inconclusive. This
  conservative observed-range rule is not a statistical confidence bound and
  cannot prove population-wide zero regression. Never silently add a noise
  allowance. Require the same outcome in an independent confirmation plus the
  fixed correctness gates and manual mechanism review before retaining a winner.
- A later candidate must also preserve the previous accepted candidate's selected
  metrics. Base-relative results alone do not authorize replacing a better winner.
- Out of scope: model/runtime/benchmark changes, quantization, changed attention
  semantics, moving work outside the timed serving path, or infrastructure tuning
  disguised as a kernel change. Claims remain scoped to the frozen TPU/workload.
