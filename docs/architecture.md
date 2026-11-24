# Architecture

## Intelligence

1. **Document ingestion.** Start with text-based PDFs. Use PyMuPDF to extract text, preserving a stable document ID and one-based physical PDF page for every chunk. Store printed page labels when available. Detect empty/scanned pages and report them in the ingestion summary. Manufacturer property tables need text extraction checks because their column order can shift during extraction.
2. **Retrieval.** Encode page-aware chunks with the Hugging Face all-MiniLM-L6-v2 model through Sentence Transformers on PyTorch. An embedding is a vector representing a passage or query. Normalize the vectors so a dot product measures cosine similarity, then use NumPy search for the first small corpus. Rank pages on hand-labeled example questions before judging the LLM's answers.
3. **Inference.** Send the query and selected excerpts to Ollama's local API. Give each excerpt an identifier and attach its document and PDF page to the final citation. Validate cited IDs against retrieved sources. Mark answers unsupported when the required evidence is absent or conflicting.
4. **Engineering tools.** Use typed, allowlisted Python functions for mechanics and materials screening. The model may propose tool arguments. Python checks units, value ranges, evidence provenance, and applicability before calculating. Return structured tool outputs so the report can show the input, formula, result, and source.
5. **Report.** Use one report schema for findings, evidence, calculations, assumptions, and missing inputs. Keep the computation and retrieval interfaces independent of Streamlit.

For the flagship demo, use static, uniaxial tension and a stated yield-strength basis for eligible ductile candidates. With force in N and yield strength in MPa, minimum net area in mm² is `F × factor_of_safety / yield_strength`. An estimated mass comparison requires a common length and prismatic geometry assumption, plus density. Preserve product form, condition and other source qualifiers. Reject guessed strength, mixed units, and unsupported material cases explicitly. Validate with hand calculations, unit conversion cases, and missing-data examples. Joint design, fatigue, stress concentrations, and certification are outside the demo model.

## Performance Lab

Collect device metadata and telemetry via `nvidia-smi`. Sample with timestamps and track unsupported metrics as missing. Wrap the same embedding and inference entry points used by Intelligence so the measurements describe the demo workload.

- For embedding CPU vs. CUDA, fix the corpus, batch sizes, model revision, dtype, preprocessing, and thread policy. Separate loading, warmups, and timed runs. Synchronize CUDA around wall-clock timing and label whether transfers and tokenization are included.
- For numerical validation, use the same inputs and document tolerances and the largest observed error. Check ranking agreement separately from numerical agreement.
- For Ollama, record the model digest, context length, prompt, placement and token counts. Distinguish time to first token, prompt evaluation, token generation, and end-to-end time. Use streaming for time to first token and server counters for generation throughput.
- For sustained workload tests, record repetitions, errors, thermal and power traces, and AC/power mode. Capture background load, test duration, and warmup with each run. Keep raw results local and publish selected, reviewed summaries with reproduction commands.

## How the MVP is organized

| File | Responsibility |
|---|---|
| `forgeai/documents.py` | Decode PDF text, preserve physical pages and printed labels, and make bounded excerpts. |
| `forgeai/retrieval.py` | Index page passages in memory and score them by cosine similarity. The keyword mode runs without numerical packages. |
| `forgeai/ollama.py` | Call the local loopback API with a timeout, and read inference counters and model residency. |
| `forgeai/agent.py` | Ask Ollama for cited claims or a tool call. Resolve citations to exact retrieved text and compare tool inputs with the units and values provided by the user. |
| `forgeai/engineering.py` and `forgeai/materials.py` | Validate SI inputs, screen eligible source records, and show calculations, assumptions, and conflicts. |
| `forgeai/performance.py` | Time the same embedding and inference code and export raw measurements. |
| `forgeai/__main__.py` and `app.py` | Expose the same backend in the terminal and Streamlit. |

Native tool calls select one registered calculation. Python checks the model's proposed name and values before it runs, then returns a structured report directly. The offline path selects tools using local rules. Quote validation proves the cited text exists on the given page. Evaluating whether each generated paraphrase follows from that quote is a separate test.
