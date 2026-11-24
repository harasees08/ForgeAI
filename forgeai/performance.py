"""Collect hardware samples and run bounded, reproducible CPU/GPU experiments."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import platform
from statistics import mean, stdev
from time import perf_counter, sleep

from forgeai.ollama import Ollama
from forgeai.retrieval import EMBEDDING_MODEL, EMBEDDING_REVISION, SentenceEncoder
from scripts.smoke import telemetry_check


CORPUS = [
    f"Validation example {index + 1}: this bracket uses a uniform cross section "
    "under an axial tensile load. Record the material condition, density, yield strength, "
    "sample geometry, load direction, and factor of safety for the engineering report."
    for index in range(72)
]
BATCH_SIZES = (1, 8, 32)
WARMUPS = 1
TRIALS = 3
BENCHMARK_PROMPT = "In two short sentences, explain tensile stress and give its formula."


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_telemetry(samples: int = 3, interval_seconds: float = 0.5) -> dict:
    if not 1 <= samples <= 10 or not 0 <= interval_seconds <= 2:
        raise ValueError("Request 1 to 10 samples with an interval of 0 to 2 seconds.")
    readings = []
    for index in range(samples):
        before = timestamp()
        gpus = telemetry_check()
        readings.append({"utc": before, "gpus": gpus})
        if index + 1 < samples:
            sleep(interval_seconds)
    return {"requested_interval_seconds": interval_seconds, "samples": readings}


def summarize(values: list[float]) -> dict:
    return {"mean": mean(values), "sample_stddev": stdev(values) if len(values) > 1 else None}


def _embedding_trials(encoder, inputs: list[str], batch_size: int, sync) -> tuple[dict, object]:
    for _ in range(WARMUPS):
        encoder.encode(inputs, batch_size=batch_size)
    times = []
    for _ in range(TRIALS):
        sync()
        start = perf_counter()
        output = encoder.encode(inputs, batch_size=batch_size)
        sync()
        times.append(perf_counter() - start)
    rates = [len(inputs) / elapsed for elapsed in times]
    return {"seconds": times, "seconds_summary": summarize(times),
            "texts_per_second": rates, "rate_summary": summarize(rates)}, output


def _parity(cpu, gpu) -> dict:
    import numpy as np

    if cpu.shape != gpu.shape or not np.isfinite(cpu).all() or not np.isfinite(gpu).all():
        raise RuntimeError("CPU and CUDA outputs must have the same shape and contain finite values.")
    cpu_norms = np.linalg.norm(cpu, axis=1)
    gpu_norms = np.linalg.norm(gpu, axis=1)
    if (cpu_norms == 0).any() or (gpu_norms == 0).any():
        raise RuntimeError("Embedding parity requires nonzero vectors.")
    cosine = (cpu * gpu).sum(axis=1) / (cpu_norms * gpu_norms)
    return {
        "shape": list(cpu.shape), "max_absolute_error": float(np.max(np.abs(cpu - gpu))),
        "min_cosine": float(np.min(cosine)), "mean_cosine": float(np.mean(cosine)),
        "rtol": 1e-4, "atol": 1e-5,
        "allclose": bool(np.allclose(gpu, cpu, rtol=1e-4, atol=1e-5)),
    }


def benchmark_embeddings() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build and accessible GPU are required.")
    before = telemetry_check()
    cpu = SentenceEncoder("cpu")
    gpu = SentenceEncoder("cuda:0")

    def cpu_sync():
        return None

    runs = []
    for batch in BATCH_SIZES:
        cpu_stats, cpu_array = _embedding_trials(cpu, CORPUS, batch, cpu_sync)
        gpu_stats, gpu_array = _embedding_trials(gpu, CORPUS, batch, torch.cuda.synchronize)
        runs.append({
            "batch_size": batch, "cpu": cpu_stats, "cuda": gpu_stats,
            "throughput_ratio_cuda_to_cpu": gpu_stats["rate_summary"]["mean"] / cpu_stats["rate_summary"]["mean"],
            "numerical_parity": _parity(cpu_array, gpu_array),
        })
    return {
        "utc": timestamp(), "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "sentence_transformers": metadata.version("sentence-transformers"), "numpy": metadata.version("numpy"),
        "cpu_threads": torch.get_num_threads(), "cpu_arithmetic": str(cpu_array.dtype),
        "gpu_arithmetic": str(gpu_array.dtype), "telemetry_before": before,
        "telemetry_after": telemetry_check(), "model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION,
        "corpus": CORPUS, "warmups_per_device_batch": WARMUPS, "trials_per_device_batch": TRIALS,
        "timing_scope": "Tokenization, model execution, normalization and transfer to CPU; excludes model load.",
        "run_order": "Batch sizes ascending, CPU then CUDA at each batch size.",
        "runs": runs,
    }


def benchmark_ollama(model: str) -> dict:
    ollama = Ollama(model)
    messages = [{"role": "user", "content": BENCHMARK_PROMPT}]
    warmup = ollama.chat(messages)
    runs = []
    for _ in range(TRIALS):
        response = ollama.chat(messages)
        placement = ollama.placement()
        metrics = asdict(response)
        metrics.pop("message")
        runs.append({"metrics": metrics, "placement": placement})
    rates = [
        item["metrics"]["output_tokens_per_second"] for item in runs
        if item["metrics"]["output_tokens_per_second"] is not None
    ]
    return {
        "utc": timestamp(), "model": model, "prompt": BENCHMARK_PROMPT,
        "telemetry_after": telemetry_check(),
        "warmup_seconds": warmup.wall_seconds,
        "seconds_summary": summarize([item["metrics"]["wall_seconds"] for item in runs]),
        "output_rate_summary": summarize(rates) if rates else None,
        "timing_scope": "Wall time includes the HTTP request; generation throughput uses Ollama eval counters.",
        "runs": runs,
    }
