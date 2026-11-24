# Phase 0 validation

The checks ran on the GIGABYTE laptop on November 24, 2026. BIOS product identification, a local GPU driver query, and a CUDA kernel all agreed on the target machine. Raw smoke output is kept under `experiments/raw_results/` in the local checkout.

## Hardware and tools

| Item | Observed |
|---|---|
| Laptop and processor | GIGABYTE AERO X16 1WH, AMD Ryzen AI 7 350 with Radeon 860M |
| Windows | Windows 11 Home 25H2, build 26200.9974 |
| System RAM | 31.12 GiB visible to Windows |
| Storage | GIGABYTE AG474E1TB-SI NVMe SSD, about 636 GiB free on C: at inspection |
| GPU | GeForce RTX 5070 Laptop GPU, compute capability 12.0, 8,151 MiB reported VRAM |
| Driver | NVIDIA 576.76, driver-reported CUDA support through 12.9 |
| Project Python | Existing Python 3.14.7 x64, isolated in `.venv` with pip 26.2.1 |
| Other Python | Python 3.15.1 on the terminal PATH |
| Development tools | Git 2.56.0 for Windows, VS Code 1.143.1, Node.js 24.22.1 |
| WSL | WSL runtime 2.7.12.0 with kernel 6.18.33.2-2. Listing distributions and launching Ubuntu returned access denied. |
| GitHub | GitHub CLI absent from PATH and standard install locations checked. No authenticated GitHub session was verified. |
| Ollama | Executable and model manifests absent from the paths checked. Access to the local API was denied, leaving the server version and model inventory unknown. |
| Other tools | Docker, `nvcc`, and uv absent from PATH. The Windows Package Manager alias was present, but execution failed inside this session. |

Microsoft documents [Windows 11 build 26200](https://learn.microsoft.com/en-us/windows/release-health/windows11-release-information). NVIDIA explains [the CUDA version reported by the driver](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html).

## Test results

| Test | Result | Evidence |
|---|---|---|
| Python and Git | Passed | Python 3.14.7 ran in the isolated environment. Git resolved the project and verified ignore rules. |
| CUDA device and GPU computation | Passed via the NVIDIA driver | A 256-element CUDA vector addition on the RTX 5070 matched the CPU reference exactly. Output checksum was 5,592,320. |
| GPU telemetry | Passed | `nvidia-smi` returned utilization, total/free VRAM, temperature, SM clock and power. One diagnostic sample showed 23% utilization, 7,400 MiB free, 50 °C, 300 MHz and 12.5 W. |
| Offline unit tests | Passed | Three standard-library tests covered telemetry values, missing fields, and malformed rows. |
| PyTorch, PyTorch CUDA tensor and GPU embeddings | Blocked | The session could not resolve package hosts. The selected Windows PyTorch wheel exists in the official index. |
| PDF extraction and pytest | Blocked | Dependencies could not be downloaded from PyPI. |
| Ollama inference and GPU placement | Blocked | Loopback returned Windows socket error 10013. |
| WSL2 distribution and Ubuntu CUDA | Blocked | `wsl --list --verbose` and the Ubuntu launch attempt returned `Wsl/E_ACCESSDENIED`. |

The telemetry values are a single diagnostic sample. Phase 4 will use timed runs and a repeatable workload.

## Configuration decisions

- **Native Windows first.** The driver and telemetry already work here, and the existing Python 3.14 installation can use the CUDA 12.8 PyTorch wheel. Pin `torch==2.11.0+cu128`. NVIDIA lists 570.65 as the Windows driver version corresponding to CUDA 12.8 GA. The installed driver is 576.76. The wheel includes the runtime libraries needed for this workflow. See [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/), the [official CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128/), and the [NVIDIA CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/).
- **Hugging Face MiniLM for the first retrieval baseline.** It produces 384-dimensional vectors and the check pins the model revision. See the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- **Ollama Qwen3 4B for the first inference baseline.** The default Ollama package is about 2.5 GB, leaving headroom on this device for context and other processes. The smoke test uses a 2,048-token context and checks reported VRAM allocation. Compare to Llama 3.1 8B if it is already present. See the [Ollama model listing](https://ollama.com/library/qwen3) and [running-models API](https://docs.ollama.com/api/ps).
- **WSL2 as a second platform.** Check distributions in a native Windows terminal first. If Ubuntu is already installed with WSL 2, run `wsl -d Ubuntu -- /usr/lib/wsl/lib/nvidia-smi` and build a separate Linux virtual environment there. If Ubuntu or Windows features need installation or changes, confirm first. The Windows host supplies the GPU driver to WSL. See the [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

## Resume on the laptop

Open a regular PowerShell window in the repository and run `.\scripts\bootstrap.ps1`. The existing `.venv` can be reused. If the script cannot be executed in that window, the equivalent package and check commands are:

```powershell
$forgePython = ".\.venv\Scripts\python.exe"
& $forgePython -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
& $forgePython -m pip install -r requirements.txt
& $forgePython -m pip check
& $forgePython -m pytest -q
& $forgePython -m scripts.smoke python cuda-driver telemetry pdf torch torch-cuda embeddings
```

In that same terminal, use `ollama --version` and `ollama list` to identify existing software and models. The [official Windows download](https://ollama.com/download/windows) is an option if the executable is missing. With the server running, use `ollama pull qwen3:4b` and run `& $forgePython -m scripts.smoke ollama-inventory ollama`. To write a combined report, run the smoke command with `--output experiments/raw_results/phase0-smoke.json` and omit the individual check names.

Git was initialized on `main`. The first commit uses a setup identity local to that command. GitHub account and repository visibility must be confirmed before a remote is created or code is pushed.
