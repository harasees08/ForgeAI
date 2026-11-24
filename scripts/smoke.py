"""Check ForgeAI's environment and emit a machine-readable result."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from urllib.request import ProxyHandler, Request, build_opener

from scripts.cuda_driver import run_driver_check


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
OLLAMA_MODEL = "qwen3:4b"
TELEMETRY_FIELDS = (
    "name", "driver_version", "compute_cap", "memory.total", "memory.free",
    "utilization.gpu", "temperature.gpu", "clocks.sm", "power.draw",
)
TELEMETRY_KEYS = (
    "gpu", "driver", "compute_capability", "vram_total_mib", "vram_free_mib",
    "gpu_utilization_pct", "temperature_c", "sm_clock_mhz", "power_w",
)


def command(*args):
    return subprocess.run(
        args, check=True, capture_output=True, text=True, cwd=ROOT, timeout=15,
    ).stdout.strip()


def python_check():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this check with the repository's virtual environment.")
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError(f"Expected Python 3.14, found {platform.python_version()}.")
    return {"version": platform.python_version(), "isolated": True}


def git_check():
    version = command("git", "--version")
    root = Path(command("git", "rev-parse", "--show-toplevel"))
    if root.resolve() != ROOT:
        raise RuntimeError("Git did not resolve the ForgeAI repository.")
    command("git", "status", "--porcelain")
    command("git", "check-ignore", ".venv/pyvenv.cfg", "uploads/private.pdf")
    return {"version": version, "repository_accessible": True}


def pdf_check():
    import pymupdf

    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "ForgeAI page one")
        document.new_page().insert_text((72, 72), "ForgeAI page two")
        payload = document.tobytes()
    with pymupdf.open(stream=payload, filetype="pdf") as document:
        extracted = [page.get_text().strip() for page in document]
    if extracted != ["ForgeAI page one", "ForgeAI page two"]:
        raise RuntimeError("PDF text extraction did not preserve page order.")
    return {"version": pymupdf.VersionBind, "pages": len(extracted)}


def torch_check():
    import torch

    result = torch.arange(4, dtype=torch.int64).square().tolist()
    if result != [0, 1, 4, 9]:
        raise RuntimeError("Unexpected PyTorch CPU tensor result.")
    return {
        "version": torch.__version__,
        "build_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_architectures": torch.cuda.get_arch_list(),
    }


def torch_cuda_check():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch does not detect a CUDA device.")
    cpu_input = torch.arange(1024, dtype=torch.float32).reshape(32, 32) / 16
    gpu_input = cpu_input.to("cuda:0")
    gpu_output = gpu_input @ gpu_input.T
    torch.cuda.synchronize(gpu_input.device)
    cpu_output = cpu_input @ cpu_input.T
    actual = gpu_output.cpu()
    torch.testing.assert_close(actual, cpu_output, rtol=1e-5, atol=1e-5)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "device": str(gpu_output.device),
        "shape": list(actual.shape),
        "max_absolute_error": (actual - cpu_output).abs().max().item(),
    }


def embedding_check():
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cuda:0",
        model_kwargs={"use_safetensors": True},
    )
    vectors = model.encode(
        [
            "The aluminum bracket failed a tensile test.",
            "A tensile test broke the aluminum bracket.",
            "The chef prepared a vegetable soup.",
        ],
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    if vectors.shape != (3, 384) or vectors.device.type != "cuda":
        raise RuntimeError(f"Unexpected embedding shape or device: {vectors.shape}, {vectors.device}")
    if not torch.isfinite(vectors).all().item():
        raise RuntimeError("Embeddings contain non-finite values.")
    related, unrelated = (vectors[1:] @ vectors[0]).tolist()
    if related <= unrelated:
        raise RuntimeError("The example related pair ranked below the unrelated pair.")
    return {
        "model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION,
        "shape": list(vectors.shape), "device": str(vectors.device),
        "related_cosine": related, "unrelated_cosine": unrelated,
    }


def ollama_json(path, payload=None):
    request = Request(
        f"http://127.0.0.1:11434/api/{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=180 if payload is not None else 5) as response:
        return json.load(response)


def ollama_inventory():
    version = ollama_json("version")["version"]
    installed = ollama_json("tags")["models"]
    return {"version": version, "models": [model["name"] for model in installed]}


def ollama_check(model):
    result = ollama_json(
        "generate",
        {
            "model": model,
            "prompt": "Reply with exactly the word FORGEAI.",
            "stream": False,
            "think": False,
            "keep_alive": "1m",
            "options": {"num_ctx": 2048, "num_predict": 48, "temperature": 0, "seed": 42},
        },
    )
    if not result.get("done") or not result.get("response", "").strip():
        raise RuntimeError("Ollama did not return a complete, nonempty response.")
    loaded = next(
        (item for item in ollama_json("ps")["models"] if item["name"] == result["model"]),
        None,
    )
    if not loaded or loaded.get("size_vram", 0) <= 0:
        raise RuntimeError("Ollama generated text but reported no model allocation in VRAM.")
    return {
        "model": result["model"],
        "digest": loaded.get("digest"),
        "vram_bytes": loaded["size_vram"],
        "loaded_bytes": loaded.get("size"),
        "context_length": loaded.get("context_length"),
        "output_tokens": result.get("eval_count"),
        "output_eval_ns": result.get("eval_duration"),
    }


def parse_telemetry(output):
    rows = []
    for cells in csv.reader(output.splitlines(), skipinitialspace=True):
        gpu, driver, capability, *numbers = (cell.strip() for cell in cells)
        numeric = [None if value.startswith("[") or value == "N/A" else float(value)
                   for value in numbers]
        rows.append(dict(zip(TELEMETRY_KEYS, (gpu, driver, capability, *numeric), strict=True)))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU telemetry rows.")
    return rows


def telemetry_check():
    output = command(
        "nvidia-smi", f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )
    return parse_telemetry(output)


def main():
    checks = {
        "python": python_check,
        "git": git_check,
        "cuda-driver": run_driver_check,
        "telemetry": telemetry_check,
        "pdf": pdf_check,
        "torch": torch_check,
        "torch-cuda": torch_cuda_check,
        "embeddings": embedding_check,
        "ollama-inventory": ollama_inventory,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks", nargs="*", metavar="CHECK", help="Defaults to all checks.")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--output", type=Path, help="Save a copy of the JSON report.")
    args = parser.parse_args()
    checks["ollama"] = lambda: ollama_check(args.ollama_model)
    selected = args.checks or list(checks)
    unknown = set(selected) - checks.keys()
    if unknown:
        parser.error(f"unknown checks: {sorted(unknown)}. Available: {', '.join(checks)}")

    results = {}
    for name in selected:
        try:
            results[name] = {"status": "pass", "details": checks[name]()}
        except Exception as error:
            results[name] = {"status": "fail", "error": f"{type(error).__name__}: {error}"}
    report = json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "checks": results}, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    return int(any(result["status"] == "fail" for result in results.values()))


if __name__ == "__main__":
    sys.exit(main())
