from __future__ import annotations

import difflib
import hashlib
import importlib.util
from importlib.metadata import distributions
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import statistics
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from autoresearch import candidate as cfg


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for directory, folders, files in os.walk(root):
        folders[:] = sorted(f for f in folders if f not in
                            (".git", ".venv", "__pycache__", ".pytest_cache"))
        for name in sorted(files):
            path = Path(directory) / name
            if (path not in (root / cfg.KERNEL, root / "autoresearch/results.tsv")
                    and path.suffix != ".pyc"):
                hashes[str(path.relative_to(root))] = digest(path)
    return hashes


def runtime_hashes() -> dict:
    vllm = Path(importlib.util.find_spec("vllm").origin).parent
    tokenizer = Path(cfg.TOKENIZER)
    if not tokenizer.is_dir():
        raise ValueError("TOKENIZER must name the pinned local tokenizer snapshot")
    probe = [sys.executable, "-c", "import jax\nd=jax.devices()\nprint(d)\n"
             "print([x.device_kind for x in d])\n"
             "if len(d)!=1 or d[0].platform!='tpu':\n"
             "    raise RuntimeError('one real TPU required')"]
    device = subprocess.check_output(probe, env=os.environ | cfg.ENV).decode()
    return {
        "device": device,
        "python": sys.executable,
        "vllm_entrypoint": digest(Path(shutil.which("vllm"))),
        "packages": sorted([d.metadata["Name"], d.version] for d in distributions()),
        "vllm": {str(p.relative_to(vllm)): digest(p)
                 for p in sorted(vllm.rglob("*.py"))},
        "tokenizer": {p.name: digest(p) for p in tokenizer.iterdir()
                      if p.is_file() and p.suffix in (".json", ".model", ".jinja")},
        "weights": {p.name: digest(p) for p in tokenizer.glob("*.safetensors")},
        "environment": {k: hashlib.sha256(v.encode()).hexdigest()
                        for k, v in sorted(os.environ.items())
                        if k.startswith(("JAX_", "XLA_", "LIBTPU_", "VLLM_",
                                         "TPU_", "USE_", "MODEL_IMPL_"))
                        or k in ("PATH", "PYTEST_ADDOPTS", "PYTHONOPTIMIZE",
                                 "LD_LIBRARY_PATH", "OMP_NUM_THREADS")},
    }


def run_logged(command: list[str], root: Path, env: dict, log: Path) -> None:
    log.with_suffix(".command.json").write_text(json.dumps(command, indent=2))
    with log.open("w") as output:
        subprocess.run(command, cwd=root, env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True,
                       timeout=cfg.TIMEOUT_SECONDS)


def correctness_report(path: Path) -> None:
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    if not suites or sum(int(s.attrib["tests"]) for s in suites) == 0:
        raise ValueError("correctness suite ran no tests")
    if any(int(s.attrib[k]) for s in suites for k in ("failures", "errors", "skipped")):
        raise ValueError("correctness suite failed or skipped cases")


def serving_run(root: Path, env: dict, directory: Path, screen: bool) -> None:
    # Never connect to a pre-existing server with unknown source/settings.
    with socket.socket() as port:
        port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port.bind(("127.0.0.1", 8000))
    (directory / "server.command.json").write_text(json.dumps(cfg.SERVER, indent=2))
    with (directory / "server.log").open("w") as log:
        server = subprocess.Popen(cfg.SERVER, cwd=root, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + cfg.TIMEOUT_SECONDS
            while True:
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("server failed to become ready; inspect server.log")
                try:
                    with urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except URLError:
                    time.sleep(1)
            # Reuse CLI imports; each official benchmark call still gets fresh args
            # and its own warmup/measurement. Import time is never a serving metric.
            command = [sys.executable, "-m", "autoresearch.experiment", "client",
                       str(directory), "screen" if screen else "full"]
            if screen:
                command.append(str(directory.parent.parent / "reference"))
            run_logged(command,
                       Path(__file__).resolve().parent.parent, env, directory / "client.log")
            if server.poll() is not None:
                raise RuntimeError("server exited during measurement")
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            finally:
                # Reap workers even if the server leader failed first.
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server.wait()



def client(directory: Path, screen: bool, reference: Path | None = None) -> None:
    from vllm.entrypoints.cli.main import main as vllm_main

    for workload in (cfg.SCREEN_WORKLOADS if screen else cfg.WORKLOADS):
        for phase in ("warmup", "measured"):
            name = f"{workload}-{phase}"
            command = cfg.BENCH + cfg.WORKLOADS[workload] + [
                "--result-dir", str(directory), "--result-filename", name + ".json"]
            (directory / (name + ".command.json")).write_text(json.dumps(command, indent=2))
            print("Benchmark command:", json.dumps(command), flush=True)
            sys.argv = command
            vllm_main()
        if screen:
            outcome = screen_result(directory, reference)
            (directory / "screen.json").write_text(json.dumps(outcome, indent=2))
            print(json.dumps(outcome), flush=True)
            if outcome["verdict"] != "screen_pending":
                return


def benchmark(path: Path) -> dict:
    result = json.loads(path.read_text())
    expected = int(cfg.BENCH[cfg.BENCH.index("--num-prompts") + 1])
    output_len = int(cfg.BENCH[cfg.BENCH.index("--random-output-len") + 1])
    if result["model_id"] != cfg.MODEL or result["num_prompts"] != expected:
        raise ValueError(f"{path}: model or request count changed")
    if result["completed"] != expected or result["failed"] != 0:
        raise ValueError(f"{path}: incomplete or failed requests")
    for field in ("input_lens", "output_lens", "errors", "generated_texts", "ttfts", "itls"):
        if len(result[field]) != expected:
            raise ValueError(f"{path}: wrong {field} count")
    if any(result["errors"]) or result["output_lens"] != [output_len] * expected:
        raise ValueError(f"{path}: errors or changed output lengths")
    if any(type(n) is not int or n <= 0 for n in result["input_lens"]):
        raise ValueError(f"{path}: invalid input lengths")
    for field, count in (("input_lens", "total_input_tokens"),
                         ("output_lens", "total_output_tokens")):
        if sum(result[field]) != result[count]:
            raise ValueError(f"{path}: inconsistent {count}")
    for metric in cfg.METRICS:
        if not math.isfinite(result[metric]) or result[metric] <= 0:
            raise ValueError(f"{path}: invalid/missing metric {metric}")
    return result


def workload_signature(result: dict) -> list:
    return [result[k] for k in ("input_lens", "output_lens", "model_id",
            "tokenizer_id", "backend", "request_rate", "max_concurrency", "burstiness")]


def screen_result(directory: Path, reference: Path) -> dict:
    report = {}
    verdict = "screen_pending"
    for workload in cfg.SCREEN_WORKLOADS:
        path = directory / f"{workload}-measured.json"
        if not path.exists():
            break
        baseline = benchmark(reference / path.name)
        candidate = benchmark(path)
        warmup = benchmark(directory / f"{workload}-warmup.json")
        if not (workload_signature(baseline) == workload_signature(candidate)
                == workload_signature(warmup)):
            raise ValueError(f"{workload}: workload or settings changed")
        gains = {metric: (candidate[metric] / baseline[metric] if direction == 1
                         else baseline[metric] / candidate[metric]) - 1
                 for metric, direction in cfg.METRICS.items()}
        report[workload] = gains
        if min(gains.values()) < 0:
            verdict = "screen_rejected"
            break
        if len(report) == 1 and max(gains.values()) < cfg.MIN_GAIN:
            verdict = "screen_inconclusive"
            break
    if len(report) == len(cfg.SCREEN_WORKLOADS) and verdict == "screen_pending":
        verdict = "screen_pass"
    return {"verdict": verdict, "metrics": report,
            "rule": "saved baseline screen only; fresh balanced pairs still required"}


def compare(directory: Path, screen: bool = False) -> dict:
    if screen:
        return screen_result(directory / "00-candidate", directory.parent / "reference")
    if not screen:
        for variant in ("baseline", "candidate"):
            correctness_report(directory / f"00-{variant}" / "correctness.xml")
    report, signatures = {}, {}
    for workload in (cfg.SCREEN_WORKLOADS if screen else cfg.WORKLOADS):
        gains = {metric: [] for metric in cfg.METRICS}
        for pair in range(1 if screen else cfg.PAIRS):
            runs = {}
            for variant in ("baseline", "candidate"):
                run = directory / f"{pair:02d}-{variant}"
                result = benchmark(run / f"{workload}-measured.json")
                warmup = benchmark(run / f"{workload}-warmup.json")
                signature = [result[k] for k in (
                    "input_lens", "output_lens", "model_id", "tokenizer_id",
                    "backend", "request_rate", "max_concurrency", "burstiness")]
                if workload not in signatures:
                    signatures[workload] = signature
                if signature != signatures[workload] or warmup["input_lens"] != result["input_lens"]:
                    raise ValueError(f"{run}: workload, settings, or token shapes changed")
                runs[variant] = result
            for metric, direction in cfg.METRICS.items():
                ratio = runs["candidate"][metric] / runs["baseline"][metric]
                gains[metric].append((ratio if direction == 1 else 1 / ratio) - 1)
        report[workload] = {
            metric: {"median_gain": statistics.median(values),
                     "min_pair_gain": min(values), "max_pair_gain": max(values)}
            for metric, values in gains.items()}
    metrics = [m for workload in report.values() for m in workload.values()]
    if screen:
        verdict = "screen_only"
    elif any(m["max_pair_gain"] < 0 for m in metrics):
        verdict = "rejected"
    elif any(m["min_pair_gain"] < 0 for m in metrics):
        verdict = "inconclusive"
    elif any(m["min_pair_gain"] >= cfg.MIN_GAIN for m in metrics):
        verdict = "winner"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "metrics": report,
            "rule": "observed paired ranges; descriptive evidence, not a confidence interval"}


def main(arguments: list[str]) -> None:
    if len(arguments) in (3, 4) and arguments[0] == "client":
        if arguments[2] not in ("screen", "full"):
            raise ValueError("unknown client workload selection")
        if (arguments[2] == "screen") != (len(arguments) == 4):
            raise ValueError("screen client requires its frozen reference directory")
        client(Path(arguments[1]), arguments[2] == "screen",
               Path(arguments[3]) if len(arguments) == 4 else None)
        return
    if not arguments or len(arguments) != (2 if arguments[0] == "freeze" else 3):
        raise ValueError("use: freeze SESSION | screen/run/compare SESSION ATTEMPT")
    mode, session = arguments[0], Path(arguments[1]).resolve()
    if mode not in ("freeze", "screen", "run", "compare"):
        raise ValueError("unknown mode")
    control = {p.name: digest(p) for p in Path(__file__).parent.iterdir()
               if p.name in ("candidate.py", "experiment.py", "problem.md", "instructions.md")}
    if cfg.HARDWARE not in ("v5e-1", "v6e-1") or len(cfg.MODEL_REVISION) != 40:
        raise ValueError("finalize TPU hardware and immutable model revision before freezing")
    if cfg.CORRECTNESS == ["FINALIZE_BEFORE_FREEZE"]:
        raise ValueError("finalize the focused real-DECODE correctness command before freezing")
    if cfg.PAIRS < 4 or cfg.PAIRS % 2:
        raise ValueError("use at least four pairs with equal AB/BA ordering")
    if mode == "compare":
        frozen = json.loads((session / "frozen.json").read_text())
        if frozen["control"] != control:
            raise ValueError("frozen experiment/contract changed")
        if any(digest(session / "reference" / p) != h
               for p, h in frozen["reference"].items()):
            raise ValueError("saved baseline reference changed")
        attempt = session / arguments[2]
        hashes = json.loads((attempt / "artifacts.json").read_text())
        if any(digest(attempt / p) != h for p, h in hashes.items()):
            raise ValueError("saved measurement artifact changed")
        saved = json.loads((attempt / "result.json").read_text())
        if "detail" in saved:
            print(json.dumps(saved, indent=2))
        else:
            print(json.dumps(compare(attempt, screen=saved["stage"] == "screen"), indent=2))
        return
    root = Path(os.environ["TPU_INFERENCE_ROOT"]).resolve()
    kernel = root / cfg.KERNEL
    env = os.environ | cfg.ENV | {"PYTHONPATH": str(root)}
    state = {"control": control, "source": source_hashes(root), "runtime": runtime_hashes(),
             "correctness": {str(root / a): digest(root / a) for a in cfg.CORRECTNESS
                             if (root / a).is_file()},
             "settings": {k: v for k, v in vars(cfg).items() if k.isupper()}}
    reference = Path(cfg.SCREEN_REFERENCE)
    reference_files = sorted(reference.glob("*.json")) + [reference / "kernel.sha256"]
    state["reference"] = {p.name: digest(p) for p in reference_files}
    if mode == "freeze":
        if digest(kernel) != cfg.BASE_SHA256:
            raise ValueError("freeze requires the fixed upstream baseline kernel")
        previous = json.loads((reference.parent.parent / "frozen.json").read_text())
        if previous["runtime"] != state["runtime"]:
            raise ValueError("saved baseline runtime differs from this environment")
        old_source = {p: h for p, h in previous["source"].items()
                      if not p.startswith("autoresearch/")}
        new_source = {p: h for p, h in state["source"].items()
                      if not p.startswith("autoresearch/")}
        if old_source != new_source:
            raise ValueError("saved baseline inference source differs")
        for key in ("SERVER", "BENCH", "WORKLOADS", "METRICS", "ENV", "MODEL_REVISION"):
            if previous["settings"][key] != state["settings"][key]:
                raise ValueError(f"saved baseline {key} differs")
        if (reference / "kernel.sha256").read_text().strip() != cfg.BASE_SHA256:
            raise ValueError("saved serving baseline used a different kernel")
        for workload in cfg.SCREEN_WORKLOADS:
            benchmark(reference / f"{workload}-measured.json")
        session.mkdir(parents=True, exist_ok=False)
        (session / "reference").mkdir()
        for path in reference_files:
            shutil.copyfile(path, session / "reference" / path.name)
        (session / "baseline.py").write_bytes(kernel.read_bytes())
        (session / "frozen.json").write_text(json.dumps(state, indent=2))
        print(f"Frozen {session} with saved baseline; candidate unvalidated.")
        return
    if state != json.loads((session / "frozen.json").read_text()):
        raise ValueError("source, dependencies, environment, or frozen contract changed")
    if any(digest(session / "reference" / p) != h
           for p, h in state["reference"].items()):
        raise ValueError("saved baseline reference changed")
    baseline, candidate = (session / "baseline.py").read_bytes(), kernel.read_bytes()
    if hashlib.sha256(baseline).hexdigest() != cfg.BASE_SHA256:
        raise ValueError("baseline snapshot changed")
    delta = difflib.ndiff(baseline.decode().splitlines(), candidate.decode().splitlines())
    changed = sum(line.startswith(("+ ", "- ")) for line in delta)
    if not 0 < changed <= cfg.MAX_CHANGED_LINES:
        raise ValueError(f"kernel diff must contain 1..{cfg.MAX_CHANGED_LINES} added/deleted lines")
    directory = session / arguments[2]
    directory.mkdir(exist_ok=False)
    (directory / "candidate.py").write_bytes(candidate)
    screen = mode == "screen"
    outcome = {"verdict": "inconclusive", "detail": "measurement interrupted"}
    try:
        (directory / "device.log").write_text(state["runtime"]["device"])
        for pair in range(1 if screen else cfg.PAIRS):
            order = (("candidate",) if screen else
                     ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline"))
            for variant in order:
                kernel.write_bytes(baseline if variant == "baseline" else candidate)
                # Equal-size edits within one second can otherwise reuse old bytecode.
                for cached in (kernel.parent / "__pycache__").glob("kernel.*.pyc"):
                    cached.unlink()
                run = directory / f"{pair:02d}-{variant}"
                run.mkdir()
                (run / "kernel.sha256").write_text(digest(kernel))
                if pair == 0 and not screen:
                    report = run / "correctness.xml"
                    run_logged(cfg.CORRECTNESS + [f"--junitxml={report}"], root, env,
                               run / "correctness.log")
                    correctness_report(report)
                serving_run(root, env, run, screen)
                if digest(kernel) != (run / "kernel.sha256").read_text():
                    raise ValueError("kernel changed during measurement")
        if source_hashes(root) != state["source"]:
            raise ValueError("source changed during measurement")
        outcome = compare(directory, screen)
    except subprocess.TimeoutExpired as error:
        outcome = {"verdict": "inconclusive", "detail": str(error)}
    except (ValueError, KeyError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        outcome = {"verdict": "invalid", "detail": str(error)}
    finally:
        if kernel.exists() and kernel.read_bytes() in (baseline, candidate):
            kernel.write_bytes(candidate)
        else:
            outcome = {"verdict": "invalid", "detail": "kernel changed concurrently; left untouched"}
        outcome["stage"] = mode
        (directory / "result.json").write_text(json.dumps(outcome, indent=2))
        artifacts = {str(p.relative_to(directory)): digest(p)
                     for p in sorted(directory.rglob("*")) if p.is_file()}
        (directory / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
