"""Fictional datasheet records that can be reproduced without external files."""

from forgeai.documents import Page, pdf_pages


DEMO_QUESTION = (
    "I need to design a lightweight component capable of withstanding a tensile load "
    "of 15 kN with a factor of safety of 2. Compare the available materials."
)

DEMO_PAGES = [
    Page("Synthetic Alloy A datasheet.pdf", 1,
         "SYNTHETIC TEST DATA. Fictional properties for the ForgeAI demo.\n"
         "Material: Synthetic Alloy A\nYield strength: 250 MPa\nDensity: 2700 kg/m3\n"
         "Test basis: illustrative ductile material under static tension.", synthetic=True),
    Page("Synthetic Alloy B datasheet.pdf", 1,
         "SYNTHETIC TEST DATA. Fictional properties for the ForgeAI demo.\n"
         "Material: Synthetic Alloy B\nYield strength: 500 MPa\nDensity: 4500 kg/m3\n"
         "Test basis: illustrative ductile material under static tension.", synthetic=True),
    Page("Synthetic Alloy C datasheet.pdf", 1,
         "SYNTHETIC TEST DATA. Fictional properties for the ForgeAI demo.\n"
         "Material: Synthetic Alloy C\nYield strength: 350 MPa\nDensity: 7800 kg/m3\n"
         "Test basis: illustrative ductile material under static tension.", synthetic=True),
]


def demo_pages(*, through_pdf: bool = False) -> list[Page]:
    if not through_pdf:
        return DEMO_PAGES.copy()
    import pymupdf

    pages = []
    for record in DEMO_PAGES:
        with pymupdf.open() as document:
            document.new_page().insert_text((60, 72), record.text)
            pages.extend(pdf_pages(record.document, document.tobytes(), synthetic=True))
    return pages
