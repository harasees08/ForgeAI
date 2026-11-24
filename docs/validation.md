# Validation and remaining integration

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

## First follow-up

Run the bootstrap from a normal terminal with package access, then repeat the demo through an actual PDF, with CUDA retrieval and Ollama tool selection. Run the benchmark commands and start Streamlit only after the integration smoke tests pass. Add several public or redistributable datasheets and label a small retrieval set before drawing conclusions about citation or extraction quality.
