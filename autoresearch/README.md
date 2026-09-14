# TPU autoresearch

A six-file workspace for small edits to the real RPA v3 kernel in
`adantr/tpu-inference`. `vllm bench serve` supplies the hillclimb feedback.
`experiment.py` uses the standard library; it does not pretend the CUDA
kernelguard evaluator supports TPU.

On the selected single-chip TPU, install the pinned tpu-inference/vLLM stack,
read `problem.md`, and finalize the settings in `candidate.py`, including the
focused real-DECODE correctness command. That file holds
settings only. The editable candidate is
`tpu_inference/kernels/ragged_paged_attention/v3/kernel.py` in the real checkout.
Download the pinned model snapshot, including weights and tokenizer, to the local `TOKENIZER` path. The server reads this snapshot and its weight hashes are frozen.
Validate the saved commands and workload fit before freezing; hardware execution
and workload finalization have not been performed by this template.

From the mini-kernelguard root, with the same Python environment as vLLM:

```bash
export TPU_INFERENCE_ROOT=/opt/tpu-research/tpu-inference
python -m autoresearch.experiment freeze /opt/tpu-research/evidence/rpa-session
# Make one small edit to the real kernel.py, then:
python -m autoresearch.experiment screen /opt/tpu-research/evidence/rpa-session screen-001
# Only for a promising candidate:
python -m autoresearch.experiment run /opt/tpu-research/evidence/rpa-session attempt-001
python -m autoresearch.experiment compare /opt/tpu-research/evidence/rpa-session attempt-001
```

`freeze` requires the exact upstream baseline kernel. It saves that source,
commands, workload, settings, and source/runtime hashes. `screen` measures one
baseline/candidate pair on the preselected discovery workload, without the full
correctness suite. Its verdict is always `screen_only` or a failure; it cannot
establish correctness or a winner. `run` preserves the
candidate, swaps the actual kernel while servers are stopped, runs the fixed
TPU reference/KV-cache check, and collects four balanced baseline/candidate pairs.
Every run starts a fresh server and discards a full workload pass before timing.
All commands, server logs, detailed vLLM JSONs, test reports, and source hashes
remain in the session directory. Run directories cannot be overwritten.

Use a session directory outside the tpu-inference checkout. Run exclusively:
do not edit the checkout or run another evaluator/server during an attempt.
Normal exit and exceptions restore the candidate. After an external kill or host
loss, check the real kernel against the saved attempt's `candidate.py` before
resuming. The driver never invokes Git.

The full result reports each metric's median and observed range of paired gains.
These are descriptive repeatability checks, not confidence intervals or a proof
of zero regression. See `problem.md` for the strict acceptance rule. A winner
still requires manual review of unchanged semantics and actual compiled routing.
