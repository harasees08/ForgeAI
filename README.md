# ForgeAI

ForgeAI is a local engineering AI and GPU validation project for manufacturing workflows. The MVP combines page-aware document retrieval, an Ollama adapter, traceable material screening, and a Performance Lab that measures the hardware running the workflow.

The flagship demo screens three fictional materials for a lightweight component under a 15 kN static tensile load with a factor of safety of 2. It calculates minimum net area, compares mass per unit length, and cites the source record for each result. This is a simplified design-screening demonstration.

## Current status

The offline CLI demo, engineering calculations, page and citation handling, and RTX 5070 CUDA driver probe and telemetry have run on the development laptop. The PDF, semantic retrieval, live Ollama, CUDA/PyTorch benchmark, and Streamlit paths require dependency and service access. See the [validation record](docs/validation.md).

## Run the offline demo

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m forgeai demo
.\.venv\Scripts\python.exe -m forgeai demo --length-mm 120 --output experiments/raw_results/demo-120mm.json
.\.venv\Scripts\python.exe -m forgeai telemetry --samples 3 --output experiments/raw_results/telemetry.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The offline demo uses keyword retrieval and deterministic tool selection. Its source records are built-in synthetic page text. The optional PDF round trip is enabled with --through-pdf.

## Setup

The first target is 64-bit Windows with Python 3.14 and an NVIDIA driver that supports CUDA 12.8. From the repository root in a PowerShell session with package registry access:

```powershell
.\scripts\bootstrap.ps1
```

The script creates or reuses `.venv`, installs the CUDA PyTorch wheel before the other dependencies, runs pytest, and writes a smoke report. The first embedding smoke run downloads the public model from Hugging Face. If the Ollama server is running and its CLI is on PATH, add `-WithOllama` to download and test the default 4B model or use `-OllamaModel` to select an existing model. Application embedding calls then load cached model files locally.

The checks that require only the existing NVIDIA driver, Git, and the virtual environment can run offline:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.smoke python git cuda-driver telemetry
```

Run `python -m scripts.smoke --help` with the project Python for options. The default run checks every component and returns a nonzero exit code if any check fails. Git metadata is included in the local checkout. An extracted source archive can run the bootstrap without Git.

## Run with PDFs, a local model, and the interface

After the bootstrap and Ollama smoke tests pass:

```powershell
.\.venv\Scripts\python.exe -m forgeai demo --through-pdf --retrieval cuda --model qwen3:4b
.\.venv\Scripts\python.exe -m forgeai workspace --pdf .\sample_documents\manual.pdf --question "What is the rated load?" --retrieval cuda --model qwen3:4b
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

Open [the local interface](http://127.0.0.1:8501). In Knowledge Base, index uploaded PDFs or load the synthetic demo. In Engineering Workspace, run the prefilled tensile screening prompt or choose Document question. In Performance Lab, choose a short experiment and download the result JSON.

For material screening, the first version reads explicit labeled records on the same PDF page. A suitable record has lines for `Material: name and condition`, `Yield strength: 250 MPa`, and `Density: 2700 kg/m3`. Free-form PDFs can still be searched. Scans, ambiguous property tables, untreated conditions, and conflicting sources require review before use.

## Repository layout

| Path | Contents |
|---|---|
| `forgeai/` | Document retrieval, Ollama, engineering tools, reports, and benchmarks |
| `app.py` | Streamlit workflow and performance views |
| `scripts/` | Windows bootstrap and environment checks |
| `tests/` | Offline unit and protocol tests, plus optional PDF and semantic tests |
| `docs/` | Architecture, roadmap, and hardware validation status |
| `sample_documents/` | Local inputs and future fixture provenance |
| `experiments/raw_results/` | Ignored local smoke and benchmark output |

See the [architecture](docs/architecture.md), [benchmark method](docs/benchmarking.md), and [development roadmap](docs/roadmap.md).
