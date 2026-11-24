"""Evidence-backed axial screening for text records with explicit material units."""

from dataclasses import dataclass
import re

from forgeai.documents import Page
from forgeai.engineering import (
    component_mass_kg, component_volume_m3, minimum_safety_factor, positive,
    required_area_mm2, specific_yield_strength_j_kg,
)


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
MATERIAL = re.compile(r"^[ \t]*Material[ \t]*:[ \t]*(\S[^\r\n]*)", re.I | re.M)
YIELD = re.compile(rf"^[ \t]*Yield strength[ \t]*:[ \t]*({NUMBER})[ \t]*(MPa|ksi)[ \t]*\r?$", re.I | re.M)
DENSITY = re.compile(rf"^[ \t]*Density[ \t]*:[ \t]*({NUMBER})[ \t]*(kg/m[³3]|g/cm[³3])[ \t]*\r?$", re.I | re.M)


@dataclass(frozen=True)
class PropertySource:
    document: str
    page: int
    quote: str
    synthetic: bool


@dataclass(frozen=True)
class Material:
    name: str
    yield_mpa: float
    density_kg_m3: float
    source: PropertySource


@dataclass(frozen=True)
class Candidate:
    material: Material
    area_mm2: float
    kg_per_m: float
    specific_yield_j_kg: float
    mass_kg: float | None


@dataclass(frozen=True)
class Comparison:
    force_n: float
    safety_factor: float
    length_m: float | None
    candidates: list[Candidate]
    limitations: list[str]


def extract_materials(pages: list[Page]) -> tuple[list[Material], list[str]]:
    materials: dict[str, Material] = {}
    conflicts = set()
    issues = []
    for page in pages:
        headers = list(MATERIAL.finditer(page.text))
        for index, header in enumerate(headers):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(page.text)
            block = page.text[header.end():end]
            strength = YIELD.findall(block)
            density = DENSITY.findall(block)
            name = header[1].strip()
            if len(strength) != 1 or len(density) != 1:
                issues.append(f"{name} ({page.document}, p. {page.number}): expected one unambiguous yield strength and density.")
                continue
            try:
                strength_value, strength_unit = strength[0]
                density_value, density_unit = density[0]
                strength_scale = 6.894757293 if strength_unit.lower() == "ksi" else 1
                density_scale = 1000 if density_unit.lower().startswith("g/") else 1
                yield_mpa = positive(float(strength_value) * strength_scale, "Yield strength")
                density_kg_m3 = positive(float(density_value) * density_scale, "Density")
            except ValueError as error:
                issues.append(f"{name} ({page.document}, p. {page.number}): {error}")
                continue
            key = name.casefold()
            record = Material(
                name, yield_mpa, density_kg_m3,
                PropertySource(page.document, page.number, page.text[header.start():end].strip(), page.synthetic),
            )
            prior = materials.setdefault(key, record)
            if (prior.yield_mpa, prior.density_kg_m3, prior.source.synthetic) != (yield_mpa, density_kg_m3, page.synthetic):
                if key not in conflicts:
                    issues.append(f"Excluded {name} because its sources disagree.")
                conflicts.add(key)
    return [record for key, record in materials.items() if key not in conflicts], issues


def compare_materials(
    materials: list[Material], force_n: float, safety_factor: float, length_m: float | None = None,
    *, extraction_issues: list[str] | None = None,
) -> Comparison:
    force_n = positive(force_n, "Force (N)")
    safety_factor = minimum_safety_factor(safety_factor)
    if length_m is not None:
        length_m = positive(length_m, "Length (m)")
    candidates = []
    for material in materials:
        area = required_area_mm2(force_n, safety_factor, material.yield_mpa)
        mass_per_m = component_mass_kg(material.density_kg_m3, component_volume_m3(area, 1))
        mass = (
            component_mass_kg(material.density_kg_m3, component_volume_m3(area, length_m))
            if length_m is not None else None
        )
        candidates.append(Candidate(
            material, area, mass_per_m,
            specific_yield_strength_j_kg(material.yield_mpa, material.density_kg_m3), mass,
        ))
    assumptions = [
        "Static uniaxial tension, uniform net area and a yield-strength limit for ductile material.",
        "Connection geometry, fatigue, stress concentrations and environmental effects require separate analysis.",
        "Educational screening only. Review source applicability and allowable properties before real design.",
    ]
    if length_m is None:
        assumptions.append("Length was unspecified, so the mass comparison uses kilograms per metre.")
    if any(candidate.material.source.synthetic for candidate in candidates):
        assumptions.append("Synthetic records use fictional values for demonstration.")
    if not candidates:
        assumptions.append("No eligible material record supplied both labeled yield strength and density.")
    assumptions.extend(extraction_issues or [])
    return Comparison(force_n, safety_factor, length_m, sorted(candidates, key=lambda item: item.kg_per_m), assumptions)
