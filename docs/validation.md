# ForgeAI validation record

## Release verification

| Check | Result |
|---|---|
| Laptop runtime | NVIDIA-SMI reported an RTX 5070 Laptop GPU, driver 576.76 and 8,151 MiB VRAM. PyTorch 2.11.0+cu128 reported CUDA available. |
| Live Streamlit on port 8502 | Engineering Workspace, Knowledge Base and Performance Lab loaded. The manufacturer demo indexed six passages from three ForgeAI reference PDFs on CUDA. |
| Model tool selection | Live Qwen3 selected `compare_materials` through structured JSON while deterministic fallback was unchecked. The trace showed 15,000 N, safety factor 2 and length 0.12 m. Ollama reported GPU placement. |
| Engineering output | Live UI ranked annealed Ti-6Al-4V, 7075-T6 and 6061-T6 in ascending mass. Titanium required 36.23 mm² and weighed 19.22 g. The deterministic CLI reproduced the ranking and masses. |
| Report export | The live UI exposed and accepted the JSON download action. An earlier browser upload run saved a parseable JSON export containing the input, tool trace, material evidence and GPU placement. |
| Document Q&A | The live torch-and-filler-metal query returned the expected 6061 page and an extra 7075 result. An unrelated music question triggered the unsupported-answer message. |
| Retrieval replay | Six paraphrases on three reference PDFs. Semantic CUDA ranked 3/6 first and 4/6 in the top three, with MRR 0.583. Keyword ranked 4/6 first and 5/6 in the top three, with MRR 0.722. |
| Benchmark evidence | Three published JSON files matched the original downloaded exports byte-for-byte. The live Performance Lab loaded the saved measurements, showed 8.63× peak speedup, 41.6 tokens/s rounded and 3/3 parity batches. |
| Automated tests | Release replay completed 30 core tests and 6 subtests, all passing. The Streamlit AppTest stalled at Windows loopback socket initialization in the restricted shell. The earlier laptop record reported 31 tests and 6 subtests passing. |
| Release fix | Switching to Document Question could leave the last material calculation and export visible. The release patch displays the last result only for its original task and prompt. The targeted patched rendering paths passed an isolated check using a Streamlit test double. The AppTest regression is included for local execution. |

The manufacturer demonstration exercises PDF extraction on ForgeAI-generated reference cards. The supplier datasheets were checked for property values and pages. The six-query retrieval evaluation used the cards as its corpus.

## Known issues

- Quote and page validation does not guarantee that a cited passage answers the question. The joining query still returned an irrelevant 7075 entry.
- The calculator requires explicit, unambiguous material properties. Scanned PDFs and irregular property tables can require OCR or manual normalization.
- Benchmarks characterize one workload on one laptop. The embedding benchmark used a fixed CPU-first device order. The saved generation trials capped output at 500 tokens.

[Benchmark methodology](benchmarking.md) · [Published measurements](../experiments/published/) · [Initial hardware inventory](phase-0.md)
