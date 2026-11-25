"""Original ForgeAI reference cards derived from cited manufacturer data."""

from dataclasses import dataclass, replace
from pathlib import Path

from forgeai.documents import Page, PublisherSource, normalized_text, pdf_pages


FLAGSHIP_QUESTION = (
    "I need a lightweight component capable of withstanding a tensile load of 15 kN with a factor of safety of 2. "
    "Compare the available materials, calculate the minimum cross-sectional area, and estimate the mass for a "
    "component length of 120 mm."
)


@dataclass(frozen=True)
class ReferenceCard:
    filename: str
    material: str
    yield_mpa: int
    density_kg_m3: int
    stock: str
    guidance: str
    source: PublisherSource

    def body(self) -> str:
        return (
            "ForgeAI | Manufacturer data reference card\n"
            f"Material: {self.material}\nYield strength: {self.yield_mpa} MPa\n"
            f"Density: {self.density_kg_m3} kg/m3\n"
            f"Stock and condition: {self.stock}\n"
            f"Strength basis: {self.source.yield_basis}\n"
            f"Selection notes: {self.guidance}\n"
            f"Original publisher: {self.source.publisher}. PDF p. {self.source.yield_page} for yield, "
            f"p. {self.source.density_page} for density."
        )


ALUMINUM_SOURCE = "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/"
CARDS = (
    ReferenceCard(
        "6061-T6-reference.pdf", "Aluminum 6061-T6", 276, 2700,
        "Rod and bar. T6 / T651 table row; tensile sample diameter 0.500 in.",
        "A common choice for welded or brazed assemblies and service requiring good corrosion resistance.",
        PublisherSource("Kaiser Aluminum", ALUMINUM_SOURCE + "1025/Kaiser_Aluminum_6061_Rod_and_Bar.pdf",
                        1, 2, "Typical tensile yield"),
    ),
    ReferenceCard(
        "7075-T6-reference.pdf", "Aluminum 7075-T6", 503, 2800,
        "Rod and bar. T6 / T651 table row; tensile sample diameter 0.500 in.",
        "High-strength aluminum for machined stock. The overaged T73 condition offers better stress corrosion resistance.",
        PublisherSource("Kaiser Aluminum", ALUMINUM_SOURCE + "1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
                        1, 2, "Typical tensile yield"),
    ),
    ReferenceCard(
        "Ti-6Al-4V-reference.pdf", "Titanium Ti-6Al-4V, annealed", 828, 4420,
        "ASTM B348 annealed stock, diameter or thickness up to 3.00 in, longitudinal at 20 C.",
        "For this size and direction the requirement is a minimum at the 0.2 percent offset.",
        PublisherSource("TIMET", "https://www.timet.com/documents/datasheets/alpha-and-beta-alloys/timetal-6-4.pdf",
                        2, 1, "Minimum 0.2% offset yield"),
    ),
)


def reference_pdf(card: ReferenceCard) -> bytes:
    import pymupdf

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        text = card.body()
        if page.insert_textbox((55, 55, 540, 455), text, fontsize=11) < 0:
            raise RuntimeError("The material reference card did not fit on one page.")
        page.insert_text((55, 440), card.source.url, fontsize=6)
        page.insert_link({
            "kind": pymupdf.LINK_URI, "from": pymupdf.Rect(55, 430, 540, 448), "uri": card.source.url,
        })
        return pdf.tobytes(no_new_id=True)


def identify_reference_pages(pages: list[Page]) -> list[Page]:
    expected = {
        card.filename: (normalized_text(card.body() + " " + card.source.url), card.source)
        for card in CARDS
    }
    verified = []
    for page in pages:
        match = expected.get(page.document)
        if page.number == 1 and match and normalized_text(page.text) == match[0]:
            page = replace(page, source=match[1])
        verified.append(page)
    return verified


def reference_pages() -> list[Page]:
    return identify_reference_pages([
        page for card in CARDS for page in pdf_pages(card.filename, reference_pdf(card))
    ])


def write_reference_pdfs(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for card in CARDS:
        path = directory / card.filename
        data = reference_pdf(card)
        if path.exists():
            existing = identify_reference_pages(pdf_pages(card.filename, path.read_bytes()))
            if len(existing) != 1 or existing[0].source != card.source:
                raise FileExistsError(f"{path} already exists and differs from the generated reference card.")
        else:
            path.write_bytes(data)
        files.append(path)
    return files
