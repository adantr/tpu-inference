"""Finalize once before freezing; vLLM never imports this settings file."""

KERNEL = "tpu_inference/kernels/ragged_paged_attention/v3/kernel.py"
BASE_REVISION = "1bd693780c5fda65d85cc35f44473bdfa4a98f59"
BASE_SHA256 = "bcd21fdaff6b333903a09991d0ad57f3683fef18bf8abef449c0f10d6879609c"
MODEL = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TOKENIZER = "/opt/tpu-research/models/qwen3-4b"
HARDWARE = "v5e-1"  # GCP accelerator type v5litepod-1.
ENV = {"USE_BATCHED_RPA_KERNEL": "0", "JAX_PLATFORMS": "tpu,cpu",
       "PYTHONDONTWRITEBYTECODE": "1"}
SERVER = [
    "vllm", "serve", TOKENIZER, "--served-model-name", MODEL,
    "--revision", MODEL_REVISION,
    "--tokenizer", TOKENIZER, "--dtype", "bfloat16",
    "--tensor-parallel-size", "1", "--max-model-len", "8192",
    "--max-num-seqs", "8", "--max-num-batched-tokens", "2048",
    "--no-enable-prefix-caching",
    "--host", "127.0.0.1", "--port", "8000",
]
BENCH = [
    "vllm", "bench", "serve", "--backend", "vllm", "--model", MODEL,
    "--tokenizer", TOKENIZER, "--base-url", "http://127.0.0.1:8000",
    "--endpoint", "/v1/completions", "--dataset-name", "random",
    "--random-range-ratio", "1", "--num-prompts", "16",
    "--random-output-len", "128", "--request-rate", "inf",
    "--seed", "42", "--temperature", "0",
    "--ignore-eos", "--num-warmups", "8", "--save-result", "--save-detailed",
    "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,99",
]
WORKLOADS = {
    f"context{length}-c{concurrency}": ["--random-input-len", str(length),
                                     "--max-concurrency", str(concurrency)]
    for length in (256, 2048, 4096) for concurrency in (1, 8)
}
SCREEN_WORKLOADS = ["context256-c8", "context2048-c8"]
# Set to the focused real-DECODE pytest command before freeze. The driver appends
# --junitxml=PATH. Its source is frozen even when located outside tpu-inference.
CORRECTNESS = ["python", "-m", "pytest", "-q", "-o", "junit_family=xunit1",
               "autoresearch/verify_rpa.py"]
METRICS = {"output_throughput": 1} | {
    f"{stat}_{metric}_ms": -1
    for metric in ("ttft", "tpot", "itl", "e2el")
    for stat in ("mean", "median", "p99")
}
PAIRS = 4  # AB, BA, AB, BA; every run uses a fresh server.
MIN_GAIN = 0.01
MAX_CHANGED_LINES = 40  # Added plus deleted lines against the fixed base.
TIMEOUT_SECONDS = 1800
