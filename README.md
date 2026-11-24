# ForgeAI

ForgeAI is a local engineering AI and GPU validation project for manufacturing workflows. ForgeAI Intelligence will retrieve evidence from technical PDFs, run deterministic engineering calculations, and produce material comparison reports with page citations. ForgeAI Performance Lab will test the hardware running those workflows and export reproducible measurements.

The flagship demo will screen supplied datasheets for a lightweight component under a 15 kN static tensile load with a factor of safety of 2. It will calculate the minimum net area for each eligible material, compare options under stated assumptions, and flag missing or inapplicable properties. This is a simplified design-screening demonstration.

## Current status

Phase 0 includes a Python virtual environment on the development laptop, Git history, a CUDA driver compute probe, an NVIDIA telemetry probe, and integration smoke checks. The application workflow and performance benchmarks are planned. The [Phase 0 results](docs/phase-0.md) describe what ran on the laptop and what still needs verification.

## Setup

The first target is 64-bit Windows with Python 3.14 and an NVIDIA driver that supports CUDA 12.8. From the repository root in a PowerShell session with package registry access:

```powershell
.\scripts\bootstrap.ps1
```

The script creates or reuses `.venv`, installs the CUDA PyTorch wheel before the other dependencies, runs pytest, and writes a smoke report. The first embedding run downloads the public model from Hugging Face. If the Ollama server is running and its CLI is on PATH, add `-WithOllama` to pull and test the default 4B model or use `-OllamaModel` to select a model already in use.

The checks that require only the existing NVIDIA driver, Git, and the virtual environment can run offline:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.smoke python git cuda-driver telemetry
```

Run `python -m scripts.smoke --help` with the project Python for options. The default run checks every component and returns a nonzero exit code if any check fails. Git metadata is included in the local checkout. An extracted source archive can run the bootstrap without Git.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Windows bootstrap and environment checks |
| `tests/` | Offline smoke-test assertions |
| `docs/` | Architecture, roadmap, and hardware validation status |
| `sample_documents/` | Local inputs and future fixture provenance |
| `experiments/raw_results/` | Ignored local smoke and benchmark output |

See the [architecture](docs/architecture.md) and [development roadmap](docs/roadmap.md).
