import sys
from unittest.mock import MagicMock

# --- PRE-IMPORT MOCKING ---
# We need to mock Home Assistant modules before they're imported anywhere.
if "homeassistant" not in sys.modules:
    ha_mock = MagicMock()
    ha_mock.__path__ = [] # Mark as package
    sys.modules["homeassistant"] = ha_mock
    sys.modules["homeassistant.core"] = MagicMock()
    sys.modules["homeassistant.util"] = MagicMock()
    sys.modules["homeassistant.config_entries"] = MagicMock()
    sys.modules["homeassistant.const"] = MagicMock()
    sys.modules["homeassistant.helpers"] = MagicMock()
    sys.modules["homeassistant.helpers.typing"] = MagicMock()

import _pytest.python as _pyt

_orig_setup = _pyt.Package.setup

def _safe_package_setup(self):
    """Allow Package.setup() to fail silently for HA integration __init__.py."""
    try:
        _orig_setup(self)
    except Exception:
        pass

_pyt.Package.setup = _safe_package_setup
