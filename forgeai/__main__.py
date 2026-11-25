"""Run ForgeAI from the terminal."""

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

from forgeai.agent import grounded_answer, run_engineering
from forgeai.demo import DEMO_QUESTION, demo_pages
from forgeai.documents import load_pdfs, passages_from_pages
from forgeai.engineering import positive
from forgeai.ollama import DEFAULT_MODEL, Ollama, OllamaError
from forgeai.references import FLAGSHIP_QUESTION, identify_reference_pages, reference_pages, write_reference_pdfs
from forgeai.retrieval import Index, SentenceEncoder


RAW_RESULTS = Path("experiments/raw_results")


def emit(data, output: Path | None = None) -> None:
    json_text = json.dumps(data, indent=2, allow_nan=False)
    print(json_text)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Compare three labeled fictional materials.")
    demo.add_argument("--through-pdf", action="store_true", help="Round-trip the fixtures through PyMuPDF.")
    demo.add_argument("--length-mm", type=float)
    flagship = subparsers.add_parser("flagship", help="Compare the three manufacturer-sourced material cards.")
    reference = subparsers.add_parser("reference-pdfs", help="Generate the three PDF cards for manual upload.")
    reference.add_argument("--directory", type=Path, default=Path("sample_documents/references"))
    evaluation = subparsers.add_parser("evaluate-retrieval", help="Evaluate paraphrase retrieval against labeled cards.")
    evaluation.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    evaluation.add_argument("--output", type=Path, default=RAW_RESULTS / "retrieval-eval.json")
    workspace = subparsers.add_parser("workspace", help="Index local PDFs and ask a question.")
    workspace.add_argument("--pdf", type=Path, required=True, action="append")
    workspace.add_argument("--question", required=True)
    workspace.add_argument("--engineering", action="store_true", help="Run a supported engineering tool.")
    for command in (demo, flagship, workspace):
        command.add_argument("--retrieval", choices=("keyword", "cpu", "cuda"), default="keyword")
        command.add_argument("--model", help=f"Use a local Ollama model, for example {DEFAULT_MODEL}.")
        command.add_argument("--output", type=Path)

    telemetry = subparsers.add_parser("telemetry", help="Collect a short series of NVIDIA telemetry samples.")
    telemetry.add_argument("--samples", type=int, default=3)
    telemetry.add_argument("--output", type=Path)
    embeddings = subparsers.add_parser("benchmark-embeddings", help="Compare CPU and CUDA embeddings.")
    embeddings.add_argument("--output", type=Path, default=RAW_RESULTS / "embeddings.json")
    inference = subparsers.add_parser("benchmark-ollama", help="Measure a short series of local chat requests.")
    inference.add_argument("--model", default=DEFAULT_MODEL)
    inference.add_argument("--output", type=Path, default=RAW_RESULTS / "ollama.json")
    args = parser.parse_args(argv)

    try:
        if args.command == "reference-pdfs":
            emit({"files": [str(path) for path in write_reference_pdfs(args.directory)]})
            return 0
        if args.command == "evaluate-retrieval":
            from forgeai.evaluation import evaluate_retrieval

            emit(evaluate_retrieval(args.device), args.output)
            return 0
        if args.command in {"telemetry", "benchmark-embeddings", "benchmark-ollama"}:
            from forgeai.performance import benchmark_embeddings, benchmark_ollama, sample_telemetry

            if args.command == "telemetry":
                report = sample_telemetry(args.samples)
            elif args.command == "benchmark-embeddings":
                report = benchmark_embeddings()
            else:
                report = benchmark_ollama(args.model)
            emit(report, args.output)
            return 0
        return _workspace(args)
    except (ValueError, RuntimeError, OSError, ImportError, subprocess.SubprocessError) as error:
        print(f"ForgeAI: {error}", file=sys.stderr)
        return 1


def _workspace(args) -> int:
    if args.command == "demo":
        pages = demo_pages(through_pdf=args.through_pdf)
        question = DEMO_QUESTION
        if args.length_mm is not None:
            length = format(Decimal(str(positive(args.length_mm, "Length"))), "f")
            question += f" Length: {length} mm."
    elif args.command == "flagship":
        pages = reference_pages()
        question = FLAGSHIP_QUESTION
    else:
        pages = identify_reference_pages(load_pdfs(args.pdf))
        question = args.question
    device = {"cpu": "cpu", "cuda": "cuda:0"}.get(args.retrieval)
    index = Index(passages_from_pages(pages), SentenceEncoder(device) if device else None)
    client = Ollama(args.model) if args.model else None
    if args.command in {"demo", "flagship"} or args.engineering:
        run = run_engineering(question, index, client)
        result = asdict(run)
    elif client:
        result = asdict(grounded_answer(question, index, client))
    else:
        result = {
            "search_only": True,
            "matches": [asdict(hit) for hit in index.search(question)],
        }
    warnings = [
        f"{page.document} p. {page.number} has no extractable text."
        for page in pages if not page.text.strip()
    ]
    report = {"retrieval": index.mode, "embedding_device": index.device, "warnings": warnings, "result": result}
    if client and result.get("inference"):
        try:
            report["ollama_placement"] = client.placement()
        except OllamaError as error:
            report["placement_error"] = str(error)
    emit(report, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
