# ForgeAI demo script

**Run time** About 75 seconds. Index the manufacturer cards on CUDA and open the saved Performance Lab results before recording.

| Time | Screen | Narration |
|---:|---|---|
| 0–12 s | Engineering Workspace and flagship prompt | Manufacturing engineers often start material selection by comparing scattered datasheets. ForgeAI brings those sources and the screening calculation into one local workspace. |
| 12–30 s | Run the 15 kN prompt and expand the tool trace | I’m screening a 120 millimeter component under 15 kilonewtons of tension with a safety factor of two. Qwen selects an engineering tool, then Python verifies the force, units and length and computes the minimum net area and mass. |
| 30–49 s | Compare candidates, expand evidence, show the publisher links | Each row preserves the material condition and links back to the manufacturer’s yield and density pages. The screen keeps typical aluminum values distinguishable from the titanium minimum. These results support an early comparison under a static axial-load assumption. |
| 49–69 s | Performance Lab with grouped bars and speedup chart | The same laptop ran MiniLM on both CPU and CUDA. Across three measured trials per configuration, speedup reached 8.63 times at batch size 32. All three batch sizes passed numerical parity. The raw timing and telemetry are available to inspect. |
| 69–77 s | Return to the report and show JSON export | The report travels with its inputs, sources and calculation trace. The PDFs, model and measurements are processed on the workstation. |
