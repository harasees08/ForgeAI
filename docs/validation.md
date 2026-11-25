# Validation record

## November 25 laptop run

| Check | Observed result |
|---|---|
| CUDA parity | Native 256-element reference matched exactly. The PyTorch smoke matrix had zero maximum absolute error. |
| MiniLM | 384-dimensional CUDA vectors. Related and unrelated smoke similarities were 0.93949 and 0.01295. |
| PDF RAG | CUDA indexed three actual ForgeAI reference PDFs. Live Qwen3 returned cited passages and reported GPU placement. The joining query surfaced the correct 6061 source and an extra 7075 result. An unrelated music question was marked unsupported. |
| Flagship tool selection | Native Qwen3 calls exhausted the response budget without emitting a tool call. A schema-constrained call selected compare_materials and stopped after 12 output tokens. Python bound the validated values of 15,000 N, 2 and 0.12 m. The resulting ranks were titanium, 7075-T6 and 6061-T6. |
| Automated suite | 31 passed, 0 failed and 0 skipped, plus 6 passing subtests. A Streamlit app test also rendered all three sections. |
| Retrieval | Six labeled paraphrases. CUDA semantic ranked 3/6 correct first and 4/6 in the top three (MRR 0.583). Keyword ranked 4/6 correct first and 5/6 in the top three (MRR 0.722). The semantic floor missed the two titanium paraphrases. |
| Embedding performance | Original laptop trials reached 1.3391×, 4.3254× and 8.6278× speedup at batches 1, 8 and 32. All passed all-close with maximum absolute difference below 2e-7. |
| Ollama performance | Three saved GPU trials averaged 41.59 output tokens/s. All produced exactly 500 output tokens. A live repeat of the benchmark prompt confirmed termination reason length. |

The audited original benchmark and telemetry files are in [experiments/published](../experiments/published/). The retrieval CLI preserves question-level ranks and scores in its local JSON. The product remains an educational material screen.

## Setup history

On November 24, 2026, the ForgeAI checks ran on the GIGABYTE AERO X16 with the RTX 5070 Laptop GPU. See the [Phase 0 hardware inventory](phase-0.md) for the original diagnostics.

| Check | Observed result |
|---|---|
| Python and Git | The isolated Python 3.14.7 and pip 26.2.1 ran normally. The repository was initialized and committed locally. |
| CUDA driver computation | All 256 integer outputs exactly matched the CPU reference. Checksum 5,592,320. |
| GPU telemetry | Three recorded samples showed 7,400 MiB free of 8,151 MiB, 23–25% utilization, 46 °C, SM clocks at 277–307 MHz, and 12.17–12.26 W. |
| Flagship CLI | Three fictional materials ranked by mass per length. Synthetic Alloy B required 60 mm² and 0.270 kg/m. With a 120 mm length, the result was 0.0324 kg. |
| Automated tests | 21 passed and 3 skipped. Skips required PyMuPDF, NumPy, or Streamlit. Ollama API metrics and tool/citation validation passed against local test doubles. |
| PyTorch, PDF, embeddings, Ollama, and Streamlit | Implemented integration paths. Live verification is pending package and service access. |

The bootstrap stopped at the CUDA wheel download. DNS resolution failed for the official PyTorch, PyPI and Hugging Face hosts from this session. The local Ollama API returned Windows socket error 10013. The GPU snapshots are diagnostics. CPU/CUDA embedding throughput, Tensor parity through PyTorch, and live LLM speed remain unmeasured.

Local evidence is in `experiments/raw_results/phase0-resume.json`, `telemetry.json`, `demo.json`, and `demo-120mm.json`. Raw results are ignored by Git.

## Remaining validation work

Expand the held-out corpus before drawing conclusions about semantic retrieval or answer quality. Repeat the performance runs with counterbalanced CPU/GPU order before generalizing across hardware or workloads.
