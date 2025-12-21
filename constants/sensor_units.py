"""Sensor unit mappings for device class validation.

Maps device classes to their expected units of measurement.
Used by EntityResolver to match entities by device class.
"""

from homeassistant.const import (
    UnitOfTemperature,
    UnitOfPower,
    UnitOfEnergy,
    PERCENTAGE,
)
from typing import Dict, Set

# Device class to expected units mapping
DEVICE_CLASS_UNITS: Dict[str, Set[str]] = {
    "temperature": {
        UnitOfTemperature.CELSIUS,
        UnitOfTemperature.FAHRENHEIT,
        UnitOfTemperature.KELVIN,
    },
    "power": {UnitOfPower.WATT, UnitOfPower.KILO_WATT},
    "energy": {UnitOfEnergy.WATT_HOUR, UnitOfEnergy.KILO_WATT_HOUR},
    "humidity": {PERCENTAGE},
    "battery": {PERCENTAGE},
    "illuminance": {"lx", "lm"},
    "pressure": {"hPa", "mbar", "bar", "psi"},
}
