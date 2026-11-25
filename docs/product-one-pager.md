# ForgeAI

**Local material screening with traceable evidence and measured GPU acceleration**

### Manufacturing use case

An engineer comparing stock for a small part may need to find yield strength and density across multiple datasheets and keep the material condition and source attached to each result. ForgeAI brings that lookup, a preliminary sizing calculation and a shareable report into one workstation.

### What the product demonstrates

| Capability | Workflow |
|---|---|
| PDF knowledge base | Extract page text with PyMuPDF. Use keyword search or CUDA-accelerated MiniLM retrieval and inspect the supporting passage. |
| Local engineering assistant | Qwen3 selects an allowlisted tool through structured output. Python validates units, sizes each material under the same load and ranks the results. |
| Evidence and export | Review page references, manufacturer links and calculation assumptions, then export the result as JSON. |
| Performance Lab | Compare matched CPU/CUDA inputs, numerical parity, Qwen3 generation throughput and diagnostic GPU telemetry. |

### Flagship example

For 15 kN of static axial tension, a safety factor of 2 and a 120 mm length, ForgeAI compares 6061-T6, 7075-T6 and annealed Ti-6Al-4V. Titanium is the lightest candidate under these inputs at **19.22 g** and **36.23 mm²** minimum net area. The example retains each material's stock condition and distinguishes typical aluminum yield from the titanium specification minimum.

### Measured laptop performance

On an RTX 5070 Laptop GPU, MiniLM reached **8.63×** the CPU throughput at batch 32, with CPU/CUDA numerical parity across all three batch sizes. Three capped Qwen3:4b runs averaged **41.59 output tokens/s**. Both measurements are specific to the tested laptop and workload. [Benchmark method and data](benchmarking.md)

### Scope

This is an educational screen for uniform static axial tension. Evaluate actual stock allowables, joints, fatigue and other failure modes before design decisions. The knowledge base retrieves and verifies citation text. Ambiguous questions can still surface an extra source.
