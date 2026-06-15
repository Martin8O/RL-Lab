"""System / hardware-capability contract (G4a). Mirrors the TS ``SystemInfo`` type."""

from pydantic import BaseModel


class SystemInfo(BaseModel):
    """Runtime capabilities the UI needs to gate features (e.g. GPU-only training)."""

    gpu_available: bool
