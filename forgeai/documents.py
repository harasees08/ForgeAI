"""Extract page-aware text from PDFs and build bounded retrieval passages."""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class PublisherSource:
    publisher: str
    url: str
    yield_page: int
    density_page: int
    yield_basis: str


@dataclass(frozen=True)
class Page:
    document: str
    number: int
    text: str
    label: str = ""
    synthetic: bool = False
    source: PublisherSource | None = None


@dataclass(frozen=True)
class Passage:
    id: str
    page: Page
    text: str


def pdf_pages(name: str, data: bytes, *, synthetic: bool = False) -> list[Page]:
    """Keep empty pages in the inventory and use physical, one-based page numbers."""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("The PDF is encrypted. Supply an unlocked copy.")
        return [
            Page(name, page.number + 1, page.get_text(sort=True), page.get_label(), synthetic)
            for page in document
        ]


def load_pdfs(paths: list[Path]) -> list[Page]:
    names = [path.name.casefold() for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Give each input PDF a distinct filename.")
    return [page for path in paths for page in pdf_pages(path.name, path.read_bytes())]


def passages_from_pages(
    pages: list[Page], *, max_words: int = 120, overlap: int = 20,
) -> list[Passage]:
    """Prefer paragraph boundaries and avoid combining text from different pages."""
    if max_words < 1 or not 0 <= overlap < max_words:
        raise ValueError("Choose an overlap between zero and the passage size.")
    passages = []
    for page in pages:
        paragraphs = re.split(r"\n[ \t]*\n", page.text.strip())
        for paragraph in paragraphs:
            words = list(re.finditer(r"\S+", paragraph))
            for start in range(0, len(words), max_words - overlap):
                end = min(start + max_words, len(words))
                passage = paragraph[words[start].start():words[end - 1].end()]
                passages.append(Passage(f"P{len(passages) + 1}", page, passage))
                if end == len(words):
                    break
    return passages


def normalized_text(value: str) -> str:
    return " ".join(value.split())
