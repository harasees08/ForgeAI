"""Local Streamlit front end for ForgeAI."""

from dataclasses import asdict
import json
import subprocess

import streamlit as st

from forgeai.agent import grounded_answer, run_engineering
from forgeai.demo import DEMO_QUESTION, demo_pages
from forgeai.documents import passages_from_pages, pdf_pages
from forgeai.materials import Comparison
from forgeai.ollama import DEFAULT_MODEL, Ollama
from forgeai.performance import benchmark_embeddings, benchmark_ollama, sample_telemetry
from forgeai.retrieval import Index, SentenceEncoder


ERRORS = (OSError, RuntimeError, ValueError, ImportError, subprocess.SubprocessError)


def export_json(label, filename, payload):
    st.download_button(label, json.dumps(payload, indent=2, allow_nan=False), filename, "application/json")


def index_controls() -> None:
    files = st.file_uploader("Technical PDFs", type=["pdf"], accept_multiple_files=True)
    backend = st.selectbox("Retrieval", ("Keyword", "Semantic on CUDA", "Semantic on CPU"))
    col_upload, col_demo = st.columns(2)
    upload = col_upload.button("Index uploaded PDFs")
    demo = col_demo.button("Load synthetic demo")
    if not (upload or demo):
        return
    try:
        if upload:
            if not files:
                raise ValueError("Select at least one PDF.")
            if len({file.name.casefold() for file in files}) != len(files):
                raise ValueError("Give the PDFs distinct filenames.")
            pages = [page for file in files for page in pdf_pages(file.name, file.getvalue())]
        else:
            pages = demo_pages()
        device = {"Semantic on CUDA": "cuda:0", "Semantic on CPU": "cpu"}.get(backend)
        with st.spinner("Indexing document passages"):
            index = Index(passages_from_pages(pages), SentenceEncoder(device) if device else None)
        if not index.passages:
            raise ValueError("No extractable text. Use text-based PDFs.")
        st.session_state.pages = pages
        st.session_state.index = index
        st.session_state.pop("report", None)
    except ERRORS as error:
        st.error(str(error))


def display_comparison(result: Comparison) -> None:
    st.write(f"Load {result.force_n:,.0f} N · Factor of safety {result.safety_factor:g}")
    if result.candidates:
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
                row["Mass (kg)"] = round(candidate.mass_kg, 5)
            table.append(row)
        st.dataframe(table, hide_index=True)
        with st.expander("Property evidence"):
            for candidate in result.candidates:
                source = candidate.material.source
                st.write(f"{source.document} · p. {source.page}")
                st.text(source.quote)
    for item in result.limitations:
        st.write(item)


def workspace(model: str) -> None:
    indexed = st.session_state.get("index")
    if indexed is None:
        st.info("Index PDFs or load the synthetic demo from Knowledge Base.")
        return
    st.caption(f"{len(indexed.passages)} indexed passages · {indexed.mode} retrieval")
    action = st.radio("Task", ("Engineering calculation", "Document question"), horizontal=True)
    question = st.text_area("Question", value=DEMO_QUESTION if action == "Engineering calculation" else "",
                            key=f"question_{action}")
    use_model = action == "Document question" or st.checkbox("Have Ollama select the calculation tool")
    if st.button("Run", type="primary"):
        st.session_state.pop("report", None)
        try:
            if not question.strip():
                raise ValueError("Enter a question.")
            client = Ollama(model) if use_model else None
            with st.spinner("Processing locally"):
                if action == "Engineering calculation":
                    report = {"tool": run_engineering(question, indexed, client)}
                else:
                    report = {"answer": grounded_answer(question, indexed, client)}
            if client and next(iter(report.values())).inference:
                try:
                    report["placement"] = client.placement()
                except RuntimeError as error:
                    report["placement_error"] = str(error)
            st.session_state.report = report
        except ERRORS as error:
            st.error(str(error))
    report = st.session_state.get("report")
    if not report:
        return
    payload = {}
    if "tool" in report:
        tool = report["tool"]
        inference = tool.inference
        if isinstance(tool.result, Comparison):
            display_comparison(tool.result)
        else:
            st.metric("Tensile stress (MPa)", f"{tool.result:.3f}")
        with st.expander("Tool execution"):
            st.write(f"{tool.name} · selected by {tool.selected_by}")
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
    st.dataframe([
        {"Document": page.document, "PDF page": page.number, "Printed label": page.label,
         "Text characters": len(page.text.strip()), "Synthetic": page.synthetic}
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
                st.text(passage.text)
        except ERRORS as error:
            st.error(str(error))


def performance_lab(model: str) -> None:
    actions = {
        "GPU telemetry": sample_telemetry,
        "CPU vs. CUDA embeddings": benchmark_embeddings,
        "Local Ollama inference": lambda: benchmark_ollama(model),
    }
    choice = st.selectbox("Experiment", list(actions))
    if st.button("Run experiment"):
        st.session_state.pop("experiment", None)
        try:
            with st.spinner("Collecting measurements"):
                st.session_state.experiment = (choice, actions[choice]())
        except ERRORS as error:
            st.error(str(error))
    if "experiment" not in st.session_state:
        return
    completed_choice, result = st.session_state.experiment
    st.write(completed_choice)
    if "samples" in result:
        st.dataframe([{"UTC": reading["utc"], **gpu} for reading in result["samples"] for gpu in reading["gpus"]],
                     hide_index=True)
    elif "corpus" in result:
        st.bar_chart([
            {"Batch size": str(run["batch_size"]), "CPU": run["cpu"]["rate_summary"]["mean"],
             "CUDA": run["cuda"]["rate_summary"]["mean"]}
            for run in result["runs"]
        ], x="Batch size", y=["CPU", "CUDA"], y_label="Texts per second")
        st.dataframe([
            {"Batch size": run["batch_size"], "CUDA / CPU": run["throughput_ratio_cuda_to_cpu"],
             "Parity": run["numerical_parity"]["allclose"], "Max error": run["numerical_parity"]["max_absolute_error"]}
            for run in result["runs"]
        ], hide_index=True)
    else:
        st.dataframe([{**run["metrics"], "Placement": (run["placement"] or {}).get("status", "unknown")}
                      for run in result["runs"]], hide_index=True)
    export_json("Download experiment JSON", "forgeai-experiment.json", result)


def main() -> None:
    st.set_page_config(page_title="ForgeAI", layout="wide")
    st.title("ForgeAI")
    model = st.sidebar.text_input("Local Ollama model", value=DEFAULT_MODEL)
    section = st.sidebar.radio("Section", ("Engineering Workspace", "Knowledge Base", "Performance Lab"))
    st.header(section)
    if section == "Engineering Workspace":
        workspace(model)
    elif section == "Knowledge Base":
        knowledge_base()
    else:
        performance_lab(model)


if __name__ == "__main__":
    main()
