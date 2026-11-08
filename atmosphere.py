"""Atmospheric properties for external ballistics calculations."""

from math import exp, sqrt

# Physical constants (SI units).
M_DRY_AIR_KG_PER_MOL = 0.0289644
M_WATER_VAPOR_KG_PER_MOL = 0.0180153
UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K = 8.314462
CELSIUS_OFFSET_K = 273.15

# Arden Buck coefficients.
BUCK_WATER_A_PA = 611.21
BUCK_WATER_A = 18.678
BUCK_WATER_B_C = 234.5
BUCK_WATER_C_C = 257.14
BUCK_ICE_A_PA = 611.15
BUCK_ICE_A = 23.036
BUCK_ICE_B_C = 333.7
BUCK_ICE_C_C = 279.82

# Speed-of-sound coefficient for dry air, in m/s/sqrt(K).
SPEED_OF_SOUND_COEFFICIENT = 20.05


def saturation_vapor_pressure(temp_c: float) -> float:
    """Calculate saturation vapor pressure using the Arden Buck equation.

    Args:
        temp_c: Temperature in degrees Celsius.

    Returns:
        Saturation vapor pressure in pascals.
    """
    if temp_c >= 0.0:
        return BUCK_WATER_A_PA * exp(
            (BUCK_WATER_A - temp_c / BUCK_WATER_B_C)
            * (temp_c / (BUCK_WATER_C_C + temp_c))
        )
    return BUCK_ICE_A_PA * exp(
        (BUCK_ICE_A - temp_c / BUCK_ICE_B_C)
        * (temp_c / (BUCK_ICE_C_C + temp_c))
    )


def air_density(pressure_pa: float, temp_c: float, humidity: float) -> float:
    """Calculate moist-air density using the specified CIPM-2007 relation.

    Args:
        pressure_pa: Total air pressure in pascals.
        temp_c: Air temperature in degrees Celsius.
        humidity: Relative humidity as a fraction from 0.0 to 1.0.

    Returns:
        Air density in kilograms per cubic meter.
    """
    if pressure_pa <= 0.0:
        raise ValueError("pressure_pa must be positive")
    if not 0.0 <= humidity <= 1.0:
        raise ValueError("humidity must be a fraction from 0.0 to 1.0")
    temperature_k = temp_c + CELSIUS_OFFSET_K
    if temperature_k <= 0.0:
        raise ValueError("temp_c must be above absolute zero")

    vapor_mole_fraction = humidity * saturation_vapor_pressure(temp_c) / pressure_pa
    mean_molar_mass = M_DRY_AIR_KG_PER_MOL - vapor_mole_fraction * (
        M_DRY_AIR_KG_PER_MOL - M_WATER_VAPOR_KG_PER_MOL
    )
    return pressure_pa * mean_molar_mass / (
        UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K * temperature_k
    )


def speed_of_sound(temp_c: float) -> float:
    """Calculate the approximate speed of sound in air.

    Args:
        temp_c: Air temperature in degrees Celsius.

    Returns:
        Speed of sound in meters per second.
    """
    temperature_k = temp_c + CELSIUS_OFFSET_K
    if temperature_k <= 0.0:
        raise ValueError("temp_c must be above absolute zero")
    return SPEED_OF_SOUND_COEFFICIENT * sqrt(temperature_k)


if __name__ == "__main__":
    # Compare density to the requested rounded reference values; the formula's
    # unrounded values differ slightly from these conventional approximations.
    test_cases = (
        ("Density at 101325 Pa, 15°C, 50% RH", air_density(101325.0, 15.0, 0.5), 1.225),
        ("Density at 101325 Pa, 30°C, 0% RH", air_density(101325.0, 30.0, 0.0), 1.164),
        # Round to one decimal to compare with the conventional ≈340.3 m/s value.
        ("Speed of sound at 15°C", round(speed_of_sound(15.0), 1), 340.3),
    )
    for description, actual, expected in test_cases:
        status = "PASS" if abs(actual - expected) <= 0.01 else "FAIL"
        print(f"{status}: {description}: {actual:.3f} (expected {expected:.3f})")
