# ForgeAI

**Local material screening with traceable evidence and measured GPU acceleration**

## Use case

A manufacturing engineer needs a preliminary material shortlist for a small part. Relevant values sit in separate datasheets and depend on temper, stock, size and test basis. ForgeAI brings the source pages, calculations, assumptions, and hardware measurements into one local workspace.

## The demo

Compare rod and bar options for a 15 kN tensile load, a factor of safety of 2, and a component length of 120 mm. The reference cards document typical tensile yields from Kaiser Aluminum and the specified minimum for an annealed titanium product from TIMET. Each result includes a link to the original publisher.

| Capability | What it shows |
|---|---|
| PDF retrieval | MiniLM embeddings on CUDA rank page passages, preserving physical page numbers and evidence. |
| Local tool selection | Ollama selects an allowlisted tool with structured output. ForgeAI checks arguments and units against the request. |
| Calculation and export | Python computes net area and mass and exports source references and assumptions as JSON. |
| Performance Lab | Compare CPU and CUDA throughput at matched batch sizes, inspect numerical parity, and review GPU telemetry and LLM generation. |

## Measured on the RTX 5070 laptop

The saved run measured 1.34×, 4.33× and 8.63× CPU-to-CUDA embedding speedups at batches 1, 8 and 32. All three batches passed the stated parity tolerance. The three capped Qwen3 trials averaged 41.59 output tokens/s and each reported GPU placement. This is a single laptop configuration with a fixed synthetic benchmark corpus.

## Decision boundary

The material result is an educational static axial-tension screen. Check the actual stock specification and allowables before selecting material for a design. Connections, fatigue, stress concentration, corrosion and manufacturing yield are separate engineering checks.
