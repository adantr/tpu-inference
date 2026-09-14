# Kernel loop

Read `problem.md`, `experiment.py`, `candidate.py`, and the last twenty rows of
`results.tsv`. Read the selected TPU kernel and its production caller before
forming an optimization plan. Use the most specific applicable kernel skill
before implementation; this serving driver itself does not implement a kernel.

Before freezing, confirm the single-chip hardware, immutable model/tokenizer
revision, installed vLLM commit, local import path, default RPA v3 route, and
workload memory fit. Record exact device/runtime identity and verify the server
logs show the default RPA route. Disable the experimental batched route. Resolve
any upstream test failure before research. The selected native test suite must
exercise real TPU execution, reference outputs, exact KV updates, masks, tails,
decode, mixed batches, and prefill. Add a missing production-shape check before
freezing; never modify the correctness contract during a search.

For each attempt:

1. State one mechanism and the decode regime it affects. Prefer removal of work,
   redundant transfers, synchronization, or layout conversion over new machinery.
2. Write a short proof card: claim, assumptions, expected Pallas/Mosaic/XLA
   lowering, evidence to collect, correctness result, decision. TPU work does not
   use PTX/SASS or CUDA Compute Sanitizer.
3. Edit only the real whitelisted `kernel.py`. Keep at most 40 added/deleted lines
   against the fixed base. Do not rename existing ABI names or introduce new
   leading-underscore names. `candidate.py` remains frozen settings.
4. Run `screen` with a new attempt name for the modest serving-only signal.
   Inspect its metrics and retain failures as well as faster results. Run the
   full `run` command only for a promising candidate; `screen_only` is never
   acceptance. Both stages and their workload subsets are frozen in advance.
5. Investigate `invalid` or `inconclusive` outcomes. Do not pool only favorable
   repeats, change the workload, raise regression allowances, omit a metric,
   reduce output length, or declare a winner from one run. A new fixed protocol
   requires a new baseline session, not retroactive rescoring.
6. For an apparent winner, inspect the actual TPU compiled route/profile and
   perform an independent confirmation using the same frozen protocol. Review
   math, precision, masking, padding, and KV ownership line by line. A benchmark
   success flag cannot establish numerical correctness or that the intended
   kernel ran. Use kernelguard only when it has a relevant supported check;
   the native TPU tests and serving measurements are the regular loop.
7. Append one row to `results.tsv`, linking the attempt directory and proof card.
   Keep a candidate only when the fixed comparison, confirmation, and correctness
   contract all pass. Restore the prior accepted kernel after rejected variants;
   preserve its source snapshot. Follow the user's Git authorization separately.

The upstream base remains immutable across attempts. Also inspect the previous
accepted result before replacing it: beating the original base alone does not
establish that a later candidate improves on the current best. After two rejected
variants of one mechanism, collect new evidence or try a different mechanism.
