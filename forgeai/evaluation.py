"""A small, labeled retrieval check against the three material reference cards."""

from statistics import mean

from forgeai.documents import passages_from_pages
from forgeai.references import reference_pages
from forgeai.retrieval import Index, SentenceEncoder


CASES = (
    ("Which stock is suggested if the fabrication process joins parts using a filler metal and a torch?",
     "6061-T6-reference.pdf"),
    ("Which rod is appropriate if the shop has to fuse several pieces into one assembly?",
     "6061-T6-reference.pdf"),
    ("Which alloy has a heat-treated alternative that better resists cracking from a corrosive environment under stress?",
     "7075-T6-reference.pdf"),
    ("Which grade has an overaged alternative for better resistance to environmentally assisted cracking?",
     "7075-T6-reference.pdf"),
    ("Which bar has a guaranteed lower bound based on the stress that produces a small permanent strain?",
     "Ti-6Al-4V-reference.pdf"),
    ("Which softened stock has a proof-stress minimum drawing on a longitudinal test?",
     "Ti-6Al-4V-reference.pdf"),
)


def evaluate(index: Index) -> dict:
    rows = []
    for question, target in CASES:
        seen = {}
        for hit in index.search(question, limit=len(index.passages)):
            page = hit.passage.page
            seen.setdefault(page.document, {"document": page.document, "page": page.number, "score": hit.score})
        ranked = list(seen.values())
        rank = next((i for i, match in enumerate(ranked, 1) if match["document"] == target), None)
        rows.append({"query": question, "expected": target, "expected_page": 1, "rank": rank, "retrieved": ranked})
    return {
        "retrieval": index.mode, "device": index.device, "queries": len(rows),
        "hit_at_1": sum(item["rank"] == 1 for item in rows),
        "hit_at_3": sum(item["rank"] is not None and item["rank"] <= 3 for item in rows),
        "mean_reciprocal_rank": mean(1 / item["rank"] if item["rank"] else 0 for item in rows),
        "results": rows,
    }


def evaluate_retrieval(device: str) -> dict:
    passages = passages_from_pages(reference_pages())
    return {
        "corpus": "Three ForgeAI PDF reference cards sourced from Kaiser Aluminum and TIMET",
        "keyword": evaluate(Index(passages)),
        "semantic": evaluate(Index(passages, SentenceEncoder(device))),
    }
