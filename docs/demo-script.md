# ForgeAI demo script

**Target length** 75–85 seconds. Before recording, open ForgeAI, load the manufacturer demo with **Semantic on CUDA**, and confirm Ollama is serving Qwen3:4b. Leave deterministic fallback unchecked. Use the saved reference results in Performance Lab so the recording shows the documented benchmark.

| Time | On screen | Narration |
|---:|---|---|
| 0–12 s | Show Engineering Workspace and the prefilled question. | Manufacturing engineers often start a material comparison by pulling strength and density values from separate datasheets. ForgeAI brings the documents, calculations and evidence into one local workspace. |
| 12–30 s | Select **Run**, then open **Agent execution trace**. | I'm screening a 120 millimeter part for 15 kilonewtons in tension with a safety factor of two. Qwen3 selects the comparison tool. Python verifies the inputs and units, then calculates the minimum net area and estimated mass. |
| 30–49 s | Show the ranking, expand **Property evidence** and point to a manufacturer page. | For this simplified load case, annealed titanium comes out at 19.22 grams. The calculation preserves the stock condition and links back to the source pages. The aluminum inputs are typical values, while the titanium input is a specification minimum. |
| 49–69 s | Switch to **Performance Lab** and show the grouped CPU and CUDA bars. | I benchmarked the same MiniLM workload on the CPU and RTX 5070 Laptop GPU. At batch size 32, CUDA reached 8.63 times the CPU throughput. The embeddings passed numerical parity at all three batch sizes. |
| 69–80 s | Return to **Engineering Workspace** and point to **Download report JSON**. | The report includes the inputs, tool trace, page evidence and assumptions. The workflow runs on the workstation, and the underlying benchmark runs are available for inspection. |
