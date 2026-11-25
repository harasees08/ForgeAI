"""Local Streamlit front end for ForgeAI."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import altair as alt
import streamlit as st

from forgeai.agent import grounded_answer, run_engineering
from forgeai.demo import DEMO_QUESTION, demo_pages
from forgeai.documents import passages_from_pages, pdf_pages
from forgeai.materials import Comparison
from forgeai.ollama import DEFAULT_MODEL, Ollama
from forgeai.performance import benchmark_embeddings, benchmark_ollama, sample_telemetry
from forgeai.references import FLAGSHIP_QUESTION, identify_reference_pages, reference_pages
from forgeai.retrieval import Index, SentenceEncoder


ERRORS = (OSError, RuntimeError, ValueError, ImportError, subprocess.SubprocessError)
RETRIEVAL_OPTIONS = ("Keyword", "Semantic on CUDA", "Semantic on CPU")
PUBLISHED_EXPERIMENTS = Path(__file__).resolve().parent / "experiments" / "published"
STYLES = """
<style>
.stMainBlockContainer {max-width: 1220px; padding-top: 3.1rem; padding-bottom: 4rem;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding: 1.25rem 0.45rem;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {min-height: 2.35rem;}
[data-testid="stHeading"] h1 {font-size: clamp(2rem, 4vw, 3.15rem); letter-spacing: -.045em;}
[data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {letter-spacing: -.025em;}
.stAppDeployButton {display: none;}
.forge-brand {font-size: 1.7rem; font-weight: 720; letter-spacing: -.055em; padding-bottom: .55rem;}
.forge-brand span {color: #B6D76A;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; letter-spacing: -.035em;}
</style>
"""


def export_json(label, filename, payload):
    st.download_button(label, json.dumps(payload, indent=2, allow_nan=False), filename, "application/json")


def build_index(pages, backend: str, collection: str) -> None:
    device = {"Semantic on CUDA": "cuda:0", "Semantic on CPU": "cpu"}.get(backend)
    with st.spinner("Indexing document passages"):
        index = Index(passages_from_pages(pages), SentenceEncoder(device) if device else None)
    if not index.passages:
        raise ValueError("No extractable text. Use text-based PDFs.")
    st.session_state.pages = pages
    st.session_state.index = index
    st.session_state.collection = collection
    for key in ("report", "question_Engineering calculation", "question_Document question"):
        st.session_state.pop(key, None)


def index_controls() -> None:
    files = st.file_uploader("Technical PDFs", type=["pdf"], accept_multiple_files=True)
    backend = st.selectbox("Retrieval", RETRIEVAL_OPTIONS)
    col_upload, col_real, col_demo = st.columns(3)
    upload = col_upload.button("Index uploaded PDFs", type="primary")
    real = col_real.button("Load manufacturer demo")
    demo = col_demo.button("Load synthetic demo")
    if not (upload or demo or real):
        return
    try:
        if upload:
            if not files:
                raise ValueError("Select at least one PDF.")
            if len({file.name.casefold() for file in files}) != len(files):
                raise ValueError("Give the PDFs distinct filenames.")
            pages = identify_reference_pages([
                page for file in files for page in pdf_pages(file.name, file.getvalue())
            ])
            collection = "Uploaded PDFs"
        elif real:
            pages = reference_pages()
            collection = "Manufacturer reference cards"
        else:
            pages = demo_pages()
            collection = "Synthetic fixtures"
        build_index(pages, backend, collection)
    except ERRORS as error:
        st.error(str(error))


def publisher_links(source) -> None:
    if source:
        st.markdown(
            f"[{source.publisher} yield, p. {source.yield_page}]({source.url}#page={source.yield_page})"
            f" · [Density, p. {source.density_page}]({source.url}#page={source.density_page})"
        )


def display_comparison(result: Comparison) -> None:
    st.subheader("Material screening results")
    inputs = f"{result.force_n / 1000:g} kN tension · Factor of safety {result.safety_factor:g}"
    if result.length_m is not None:
        inputs += f" · Length {result.length_m * 1000:g} mm"
    st.caption(inputs)
    if result.candidates:
        lightest = result.candidates[0]
        best, area, mass = st.columns((1.8, 1, 1))
        best.metric("Lightest screened candidate", lightest.material.name, border=True)
        area.metric("Minimum net area", f"{lightest.area_mm2:.2f} mm²", border=True)
        mass.metric(
            "Estimated mass" if lightest.mass_kg is not None else "Mass per unit length",
            f"{lightest.mass_kg * 1000:.2f} g" if lightest.mass_kg is not None else f"{lightest.kg_per_m:.4f} kg/m",
            border=True,
        )
        table = []
        for candidate in result.candidates:
            material = candidate.material
            row = {
                "Material": material.name,
                "Yield (MPa)": round(material.yield_mpa, 3),
                "Density (kg/m³)": round(material.density_kg_m3, 3),
                "Minimum area (mm²)": round(candidate.area_mm2, 3),
                "Mass per length (kg/m)": round(candidate.kg_per_m, 4),
                "Source": f"{material.source.document}, p. {material.source.page}",
            }
            if candidate.mass_kg is not None:
                row["Mass at specified length (g)"] = round(candidate.mass_kg * 1000, 2)
            table.append(row)
        st.dataframe(table, hide_index=True)
        with st.expander("Property evidence"):
            for candidate in result.candidates:
                source = candidate.material.source
                st.markdown(f"**{candidate.material.name}**")
                st.caption(f"{source.document} · p. {source.page}")
                if source.publisher:
                    st.caption(source.publisher.yield_basis)
                    publisher_links(source.publisher)
                st.text(source.quote)
    with st.expander("Method and limitations"):
        st.latex(r"A_\mathrm{min} = F\,n/\sigma_y,\qquad m=\rho A_\mathrm{min} L")
        for item in result.limitations:
            st.write(item)


def workspace(model: str) -> None:
    indexed = st.session_state.get("index")
    if indexed is None:
        with st.container(border=True):
            st.subheader("Screen material options for a tensile load")
            st.write(
                "Compare 6061-T6, 7075-T6, and annealed Ti-6Al-4V for a 15 kN load. "
                "Review the manufacturer sources and the calculation behind each result."
            )
            backend = st.selectbox("Demo retrieval", RETRIEVAL_OPTIONS, index=1)
            if st.button("Load manufacturer demo", type="primary"):
                try:
                    build_index(reference_pages(), backend, "Manufacturer reference cards")
                except ERRORS as error:
                    st.error(str(error))
                else:
                    st.rerun()
            st.caption("For your own PDFs or the synthetic fixture, open Knowledge Base.")
        return
    collection = st.session_state.get("collection", "Indexed PDFs")
    st.caption(f"{collection} · {len(indexed.passages)} indexed passages · {indexed.mode} retrieval ({indexed.device})")
    action = st.radio("Task", ("Engineering calculation", "Document question"), horizontal=True, key="workspace_task")
    default = DEMO_QUESTION if collection == "Synthetic fixtures" else FLAGSHIP_QUESTION
    question = st.text_area("Question", value=default if action == "Engineering calculation" else "",
                            key=f"question_{action}")
    use_model = action == "Document question" or st.checkbox(
        "Have Ollama select the calculation tool", value=collection != "Synthetic fixtures",
    )
    fallback = action == "Engineering calculation" and use_model and st.checkbox(
        "Allow deterministic fallback if Ollama returns no usable selection", value=False,
    )
    if st.button("Run", type="primary"):
        st.session_state.pop("report", None)
        try:
            if not question.strip():
                raise ValueError("Enter a question.")
            client = Ollama(model) if use_model else None
            with st.spinner("Processing locally"):
                if action == "Engineering calculation":
                    report = {"tool": run_engineering(question, indexed, client, allow_fallback=fallback)}
                else:
                    report = {"answer": grounded_answer(question, indexed, client)}
            if client and next(iter(report.values())).inference:
                try:
                    report["placement"] = client.placement()
                except RuntimeError as error:
                    report["placement_error"] = str(error)
            report["input"] = {"question": question, "retrieval": indexed.mode, "embedding_device": indexed.device}
            st.session_state.report = report
        except ERRORS as error:
            st.error(str(error))
    report = st.session_state.get("report")
    result_key = "tool" if action == "Engineering calculation" else "answer"
    if not report or result_key not in report or report["input"]["question"] != question:
        return
    payload = {"input": report.get("input", {"retrieval": indexed.mode, "embedding_device": indexed.device})}
    if "tool" in report:
        tool = report["tool"]
        inference = tool.inference
        if isinstance(tool.result, Comparison):
            display_comparison(tool.result)
        else:
            st.metric("Tensile stress (MPa)", f"{tool.result:.3f}")
        if tool.selected_by == "deterministic fallback":
            st.warning("Deterministic fallback executed after Ollama could not choose a tool.")
        with st.expander("Agent execution trace"):
            st.write(f"{tool.name} · {tool.selected_by} · {tool.selection_method}")
            for step in tool.selection_trace:
                st.write(step)
            st.json(tool.arguments)
        payload["tool"] = asdict(tool)
    else:
        answer = report["answer"]
        inference = answer.inference
        if answer.unsupported:
            st.info("The retrieved passages do not support an answer.")
        for claim in answer.claims:
            citation = claim.citation
            st.write(claim.text)
            st.caption(f"{citation.document} · p. {citation.page}")
            publisher_links(citation.publisher)
            st.text(citation.quote)
        payload["answer"] = asdict(answer)
    if inference:
        st.caption(f"Ollama wall time {inference.wall_seconds:.2f} s")
        if inference.output_tokens_per_second is not None:
            st.caption(f"Ollama generation {inference.output_tokens_per_second:.1f} tokens/s")
    if report.get("placement"):
        payload["placement"] = report["placement"]
        st.caption(f"Ollama placement: {report['placement']['status']}")
    if report.get("placement_error"):
        st.warning(report["placement_error"])
    export_json("Download report JSON", "forgeai-report.json", payload)


def knowledge_base() -> None:
    index_controls()
    pages = st.session_state.get("pages", [])
    if not pages:
        return
    index = st.session_state.index
    st.caption(f"{len(index.passages)} passages · {index.mode} retrieval ({index.device})")
    st.dataframe([
        {"Document": page.document, "PDF page": page.number, "Printed label": page.label,
         "Text characters": len(page.text.strip()),
         "Source": page.source.publisher if page.source else ("Synthetic" if page.synthetic else "User PDF")}
        for page in pages
    ], hide_index=True)
    empty = [f"{page.document} p. {page.number}" for page in pages if not page.text.strip()]
    if empty:
        st.warning(f"No extractable text on {', '.join(empty)}.")
    query = st.text_input("Search indexed passages")
    if query.strip():
        try:
            hits = st.session_state.index.search(query)
            if not hits:
                st.info("No matching passages.")
            for hit in hits:
                passage = hit.passage
                st.caption(f"{passage.page.document} · p. {passage.page.number} · score {hit.score:.3f}")
                publisher_links(passage.page.source)
                st.text(passage.text)
        except ERRORS as error:
            st.error(str(error))


def embedding_charts(result: dict) -> None:
    runs = sorted(result["runs"], key=lambda run: run["batch_size"])
    order = [run["batch_size"] for run in runs]
    grouped = [
        {"Batch": run["batch_size"], "Device": device, "Throughput": run[key]["rate_summary"]["mean"]}
        for run in runs for key, device in (("cpu", "CPU"), ("cuda", "CUDA"))
    ]
    speedup = [{"Batch": run["batch_size"], "Speedup": run["throughput_ratio_cuda_to_cpu"]} for run in runs]
    x = alt.X("Batch:O", sort=order, title="Texts per batch", axis=alt.Axis(labelAngle=0))
    bars = alt.Chart(alt.Data(values=grouped)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=x, xOffset=alt.XOffset("Device:N", sort=["CPU", "CUDA"]),
        y=alt.Y("Throughput:Q", title="Texts per second", stack=None, axis=alt.Axis(format=",.0f")),
        color=alt.Color("Device:N", scale=alt.Scale(domain=["CPU", "CUDA"], range=["#7588A0", "#B6D76A"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Batch:O", "Device:N", alt.Tooltip("Throughput:Q", format=",.1f")],
    ).properties(height=270)
    ratios = alt.Chart(alt.Data(values=speedup)).mark_line(
        color="#B6D76A", strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=70),
    ).encode(
        x=x, y=alt.Y("Speedup:Q", title="CUDA / CPU", axis=alt.Axis(format=".1f"), scale=alt.Scale(zero=True)),
        tooltip=["Batch:O", alt.Tooltip("Speedup:Q", format=".2f")],
    ).properties(height=270)
    throughput_panel, speed_panel = st.columns((1.4, 1))
    with throughput_panel:
        st.markdown("**Embedding throughput**")
        st.altair_chart(bars, width="stretch")
    with speed_panel:
        st.markdown("**GPU speedup**")
        st.altair_chart(ratios, width="stretch")


def lab_summary(results: dict) -> None:
    embeddings = results.get("CPU vs. CUDA embeddings")
    llm = results.get("Local Ollama inference")
    telemetry = results.get("GPU telemetry")
    if embeddings and llm:
        peak, average, parity = st.columns(3)
        runs = embeddings["runs"]
        peak.metric("Peak measured speedup", f"{max(run['throughput_ratio_cuda_to_cpu'] for run in runs):.2f}×", border=True)
        rate = llm["output_rate_summary"]
        average.metric("Mean Ollama generation", f"{rate['mean']:.1f} tok/s" if rate else "Unavailable", border=True)
        parity_count = sum(run["numerical_parity"]["allclose"] for run in runs)
        parity.metric("Numerical parity", f"{parity_count}/{len(runs)} batches", border=True)
    if telemetry:
        sample = telemetry["samples"][-1]
        gpu = sample["gpus"][0]
        st.caption(f"{gpu['gpu']} · Driver {gpu['driver']} · Compute {gpu['compute_capability']} · {sample['utc']}")
        temperature, memory, power = st.columns(3)
        temp_value = gpu.get("temperature_c")
        free, total = gpu.get("vram_free_mib"), gpu.get("vram_total_mib")
        power_value = gpu.get("power_w")
        temperature.metric("GPU temperature", f"{temp_value:g} °C" if temp_value is not None else "Unavailable", border=True)
        memory.metric("VRAM in use", f"{(total-free)/1024:.2f} / {total/1024:.2f} GiB"
                      if total and free is not None else "Unavailable", border=True)
        power.metric("GPU power", f"{power_value:.1f} W" if power_value is not None else "Unavailable", border=True)


def performance_lab(model: str) -> None:
    actions = {
        "CPU vs. CUDA embeddings": benchmark_embeddings,
        "Local Ollama inference": lambda: benchmark_ollama(model),
        "GPU telemetry": sample_telemetry,
    }
    saved = {}
    for label, filename in zip(actions, ("embeddings.json", "ollama.json", "telemetry.json"), strict=True):
        path = PUBLISHED_EXPERIMENTS / filename
        if path.is_file():
            saved[label] = json.loads(path.read_text(encoding="utf-8-sig"))
    session = st.session_state.setdefault("lab_results", {})
    st.caption("Compare embedding throughput, inspect local generation, or sample GPU telemetry.")
    choice = st.selectbox("Experiment", list(actions))
    if st.button("Run experiment", type="primary"):
        try:
            with st.spinner("Collecting measurements"):
                session[choice] = actions[choice]()
        except ERRORS as error:
            st.error(str(error))
    results = saved | session
    if choice not in results:
        return
    st.subheader("Measurement summary")
    st.caption("Session measurements take precedence over the saved RTX 5070 reference results.")
    lab_summary(results)
    st.subheader(choice)
    st.caption("Measured in this session" if choice in session else "Saved laptop measurements")
    result = results[choice]
    if "samples" in result:
        st.dataframe([{"UTC": reading["utc"], **gpu} for reading in result["samples"] for gpu in reading["gpus"]],
                     hide_index=True)
    elif "corpus" in result:
        embedding_charts(result)
        st.dataframe([
            {"Batch size": run["batch_size"],
             "CPU ± SD (texts/s)": f"{run['cpu']['rate_summary']['mean']:,.1f} ± {run['cpu']['rate_summary']['sample_stddev']:,.1f}",
             "CUDA ± SD (texts/s)": f"{run['cuda']['rate_summary']['mean']:,.1f} ± {run['cuda']['rate_summary']['sample_stddev']:,.1f}",
             "Parity": "Pass" if run["numerical_parity"]["allclose"] else "Fail",
             "Max absolute error": f"{run['numerical_parity']['max_absolute_error']:.2e}"}
            for run in result["runs"]
        ], hide_index=True)
        st.caption(
            f"{len(result['corpus'])} fixed texts · {result['warmups_per_device_batch']} warmup · "
            f"{result['trials_per_device_batch']} timed trials per device and batch · "
            f"{result['cpu_threads']} CPU threads · {result['gpu_arithmetic']} · PyTorch {result['torch']}"
        )
        st.caption(result["timing_scope"])
    else:
        rows = [
            {"Trial": number, "Wall time (s)": run["metrics"]["wall_seconds"],
             "Output tokens": run["metrics"]["output_tokens"],
             "Tokens/s": run["metrics"]["output_tokens_per_second"],
             "Placement": (run["placement"] or {}).get("status", "unknown"),
             "Finish": run["metrics"].get("done_reason") or "Not logged"}
            for number, run in enumerate(result["runs"], 1)
        ]
        st.dataframe(rows, hide_index=True)
        if result.get("output_limit_trials"):
            st.warning("The model reached the output cap. Use these trials for throughput measurement only.")
        elif all(row["Output tokens"] == 500 for row in rows):
            st.warning("Each trial produced the 500-token cap. This saved export did not record finish reasons.")
        st.caption(f"{result['model']} · 1 warmup · {len(rows)} measured trials")
        st.caption(result["timing_scope"])
    export_json("Download raw experiment JSON", "forgeai-experiment.json", result)


def main() -> None:
    st.set_page_config(page_title="ForgeAI", layout="wide")
    st.html(STYLES)
    st.sidebar.html('<div class="forge-brand">Forge<span>AI</span></div>')
    section = st.sidebar.radio("Section", ("Engineering Workspace", "Knowledge Base", "Performance Lab"))
    with st.sidebar.expander("Local model"):
        model = st.text_input("Local Ollama model", value=DEFAULT_MODEL)
    st.title(section)
    if section == "Engineering Workspace":
        workspace(model)
    elif section == "Knowledge Base":
        knowledge_base()
    else:
        performance_lab(model)


if __name__ == "__main__":
    main()
