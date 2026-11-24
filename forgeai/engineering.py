"""Deterministic screening calculations for uniform members in static tension."""

import math


def positive(value: int | float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is too large.") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return number


def minimum_safety_factor(value: int | float) -> float:
    factor = positive(value, "Factor of safety")
    if factor < 1:
        raise ValueError("The screening factor of safety must be at least 1.")
    return factor


def tensile_stress_mpa(force_n: float, area_mm2: float) -> float:
    """N per mm² is numerically equal to MPa."""
    return positive(positive(force_n, "Force (N)") / positive(area_mm2, "Area (mm²)"), "Stress (MPa)")


def factor_of_safety(yield_strength_mpa: float, stress_mpa: float) -> float:
    return positive(
        positive(yield_strength_mpa, "Yield strength (MPa)") / positive(stress_mpa, "Stress (MPa)"),
        "Factor of safety",
    )


def required_area_mm2(force_n: float, safety_factor: float, yield_strength_mpa: float) -> float:
    yield_strength = positive(yield_strength_mpa, "Yield strength (MPa)")
    area = positive(force_n, "Force (N)") / yield_strength * minimum_safety_factor(safety_factor)
    return positive(area, "Required area (mm²)")


def component_volume_m3(area_mm2: float, length_m: float) -> float:
    volume = positive(area_mm2, "Area (mm²)") * 1e-6 * positive(length_m, "Length (m)")
    return positive(volume, "Volume (m³)")


def component_mass_kg(density_kg_m3: float, volume_m3: float) -> float:
    mass = positive(density_kg_m3, "Density (kg/m³)") * positive(volume_m3, "Volume (m³)")
    return positive(mass, "Mass (kg)")


def specific_yield_strength_j_kg(yield_strength_mpa: float, density_kg_m3: float) -> float:
    """Yield strength per unit density, in Pa/(kg/m³), equivalent to J/kg."""
    return positive(
        positive(yield_strength_mpa, "Yield strength (MPa)")
        / positive(density_kg_m3, "Density (kg/m³)") * 1e6,
        "Specific yield strength (J/kg)",
    )
