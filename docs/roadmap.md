# Development roadmap

These are rough focused-work estimates for one developer familiar with the project. Learning, debugging, and sourcing usable datasheets can extend them.

| Phase | Estimate | Exit criterion |
|---|---:|---|
| 0. Setup | 1–3 hours after registry and loopback access | The laptop passes CUDA PyTorch, PDF, GPU embeddings and Ollama smoke tests. |
| 1. Ingestion and retrieval | 8–14 hours | Three to five vetted text PDFs load with stable page metadata. Hand-labeled queries retrieve the expected pages and identify weak cases. |
| 2. Grounded local inference | 6–12 hours | An Ollama answer cites retrieved document pages and abstains on an unanswerable example. |
| 3. Validated engineering tools | 10–18 hours | The 15 kN demo produces traceable results with unit tests and an independent hand-calculation reference. |
| 4. Performance Lab | 12–22 hours | Repeated CPU/GPU embedding, LLM inference, parity, and sustained runs export raw data and a reproducible summary. |
| 5. Integration and evaluation | 10–18 hours | Streamlit runs the full flow and displays evidence, calculations, and measured performance. A held-out test set covers successful and missing-data cases. |
| 6. Documentation and demo | 6–10 hours | A clean setup walkthrough, demo video, one-pager, and résumé bullets describe the implemented system and measured results. |
| 7. Optional optimization | 10–25+ hours | One documented Docker, quantization, or TensorRT experiment answers a measured bottleneck. |

Build the first vertical slice in this order. Retrieve a known datasheet page, answer from that page, calculate and verify one candidate, then compare the full set. Add measurements around the same entry points before building the UI.
