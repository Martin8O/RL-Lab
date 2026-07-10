"""System / hardware-capability contract (G4a). Mirrors the TS ``SystemInfo`` type."""

from pydantic import BaseModel


class SystemInfo(BaseModel):
    """Runtime capabilities the UI needs to gate features (e.g. GPU-only training)."""

    gpu_available: bool
    # R1: is the optional ``ale-py`` package installed? False on a default install (ADR-101 made it
    # opt-in) → the UI gates the Atari family instead of crashing on select/run.
    atari_available: bool
