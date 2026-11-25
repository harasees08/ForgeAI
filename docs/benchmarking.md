# ForgeAI benchmark methodology

## Reproduce the experiments

Run from the repository root in PowerShell with the existing CUDA-enabled environment and the cached MiniLM model. Start Ollama with `qwen3:4b` available for the inference experiment.

```powershell
.\.venv\Scripts\python.exe -m forgeai benchmark-embeddings
.\.venv\Scripts\python.exe -m forgeai benchmark-ollama --model qwen3:4b
.\.venv\Scripts\python.exe -m forgeai telemetry --samples 3 --output experiments/raw_results/telemetry.json
```

The benchmark commands write new run data to `experiments/raw_results/`. Performance Lab also exports its displayed run as JSON. [Audited original exports](../experiments/published/)

## CPU versus CUDA embeddings

- Both devices encoded the same 72 synthetic snippets in the same order at batch sizes 1, 8 and 32, using one pinned `all-MiniLM-L6-v2` revision and float32 arithmetic.
- The benchmark loaded both models before timing. It ran one warmup and three measured trials per device and batch. Timing includes tokenization, model execution, normalization and moving results to host memory. CUDA was synchronized at both timer boundaries.
- Throughput below is the mean and sample standard deviation of the three trials. Speedup uses the ratio of mean throughput. The final CPU and CUDA embedding arrays were checked with relative tolerance 1e-4 and absolute tolerance 1e-5.
- The saved run used an AMD Ryzen AI 7 350, an NVIDIA RTX 5070 Laptop GPU with 8,151 MiB VRAM, driver 576.76 and PyTorch 2.11.0+cu128.

| Batch | CPU texts/s, mean ± SD | CUDA texts/s, mean ± SD | CUDA speedup | Max absolute error |
|---:|---:|---:|---:|---:|
| 1 | 63.37 ± 1.69 | 84.87 ± 5.11 | 1.3391× | 1.90e-7 |
| 8 | 145.12 ± 0.47 | 627.73 ± 142.30 | 4.3254× | 1.90e-7 |
| 32 | 173.62 ± 0.85 | 1,497.98 ± 63.30 | 8.6278× | 1.98e-7 |

All three batches passed parity. CPU ran first in each pair. Batch 8 showed the largest relative CUDA variation in this small run. Device order, power profile, warm state and concurrent workload can affect throughput. [Embedding trial data](../experiments/published/embeddings.json)

## Local LLM inference

The benchmark ran one warmup and three measured non-streaming requests with a fixed prompt. Ollama generation throughput uses output tokens divided by server evaluation duration. Client wall time covers the full non-streaming request. The model was Qwen3:4b and Ollama reported GPU placement for every measured trial.

| Trial | Client wall time (s) | Output tokens | Generation tokens/s |
|---:|---:|---:|---:|
| 1 | 12.0515 | 500 | 41.7810 |
| 2 | 12.1064 | 500 | 41.5766 |
| 3 | 12.1240 | 500 | 41.4101 |

Mean generation throughput was **41.59 tokens/s**. Every original trial reached 500 output tokens. The original export did not log the finish reason, and a later replay logged `length`. The data measures generation throughput. First-token latency, response quality, and uncapped response completion require separate experiments. The repeated prompt can benefit from the server cache. [Inference trial data](../experiments/published/ollama.json)

## Diagnostic telemetry

NVIDIA-SMI snapshots record temperature, power, utilization, SM clocks and VRAM when supported. The three saved samples reported 43 °C and 15% GPU utilization in that collection window. Use a sustained controlled workload to characterize thermals or power. [Telemetry snapshots](../experiments/published/telemetry.json)
