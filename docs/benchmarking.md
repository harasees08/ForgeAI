# Benchmark method

Run these in a PowerShell window with a verified CUDA PyTorch installation and cached embedding model:

```powershell
.\.venv\Scripts\python.exe -m forgeai benchmark-embeddings
.\.venv\Scripts\python.exe -m forgeai benchmark-ollama --model qwen3:4b
.\.venv\Scripts\python.exe -m forgeai telemetry --samples 3
```

## Embedding experiment

- Use 72 fixed synthetic snippets, an identical pinned MiniLM revision on CPU and CUDA, and batches of 1, 8, and 32. The snippets have closely matched lengths. A corpus with mixed lengths is a separate experiment.
- Load both models first. For each device and batch, run one warmup and three timed trials. Time tokenization, execution, normalization and transfer to host. Synchronize CUDA immediately before starting and after each measured call so wall time includes completed work.
- Export trial times, throughput, mean and sample standard deviation, CPU thread count, model revision, driver and GPU snapshots, Python and library versions. Compare the final batch outputs using cosine similarity, maximum absolute error, and all-close with relative tolerance 1e-4 and absolute tolerance 1e-5.
- Batch sizes run in ascending order and CPU runs first in each pair. Treat short trials as a first comparison. For a shareable benchmark, repeat with the laptop plugged in, record the power profile and background load, and counterbalance device order across separate runs.

## Local inference experiment

Send a fixed non-streaming prompt once as a warmup and three more times. Export client wall time and Ollama's server generation throughput computed from output tokens divided by evaluation duration. Prompt reuse can benefit from the server cache. Record model digest, placement and context length where the API reports them. First-token latency requires streaming and is outside this experiment.

## Telemetry

NVIDIA-SMI snapshots contain free and total VRAM, utilization, temperature, power and SM clocks where supported. The three-sample run is a short hardware check on a laptop in use. Sustained workloads, throughput comparisons and conclusions about thermal behavior require dedicated experiments.
