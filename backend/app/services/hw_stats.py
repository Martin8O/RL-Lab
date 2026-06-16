"""Live hardware telemetry for the training HW-stats panel (G4b).

Sampled once per 1 Hz progress frame from the trainer's ticker thread (never per-step) and carried
on :class:`~app.schemas.training.TrainingProgress`. CPU + RAM come from ``psutil`` (already a dep);
GPU metrics come from NVIDIA's ``pynvml`` (the ``nvidia-ml-py`` package), imported **optionally** so
a machine with no NVIDIA GPU — the old laptop, a CI box — degrades to GPU fields ``None`` instead of
crashing. Any NVML error at runtime is swallowed the same way: the panel shows ``—`` for the GPU.

State is module-level and lazy: the per-process CPU meter is primed once (its first reading is
always 0), and NVML is initialised at most once.
"""

from __future__ import annotations

import psutil

from app.core.logging import get_logger
from app.schemas.training import HwStats

logger = get_logger(__name__)

_proc = psutil.Process()
_proc_primed = False
_ncpu = psutil.cpu_count() or 1
_nvml_ready: bool | None = None  # None = not yet tried; True/False after the one init attempt


def _ensure_nvml() -> bool:
    """Initialise NVML once; return whether GPU telemetry is available. Never raises."""
    global _nvml_ready
    if _nvml_ready is None:
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_ready = True
        except Exception:  # noqa: BLE001 — no NVIDIA GPU / no driver / no pynvml → GPU stats off
            logger.debug("NVML unavailable; GPU stats disabled", exc_info=True)
            _nvml_ready = False
    return _nvml_ready


def _gpu_fields() -> dict[str, float]:
    """GPU 0's util / VRAM / temp / power, or ``{}`` if NVML is unavailable or errors."""
    if not _ensure_nvml():
        return {}
    try:
        import pynvml

        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "gpu_util_pct": float(util.gpu),
            "gpu_vram_used_mb": float(mem.used) / 1e6,
            "gpu_vram_total_mb": float(mem.total) / 1e6,
            "gpu_temp_c": float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
            "gpu_power_w": float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0,
        }
    except Exception:  # noqa: BLE001 — a transient NVML hiccup must not break the progress frame
        logger.debug("NVML sample failed", exc_info=True)
        return {}


def sample() -> HwStats:
    """One telemetry snapshot. Cheap (~0.5 ms); safe to call from the ticker thread each tick."""
    global _proc_primed
    if not _proc_primed:
        _proc.cpu_percent(None)  # prime — the first per-process reading is always 0.0
        _proc_primed = True
    # Normalise per-process % to 0–100 of the whole machine; clamp the transient >100 psutil can
    # report across a long sampling gap so the panel never shows e.g. 103 %.
    cpu = min(100.0, _proc.cpu_percent(None) / _ncpu)
    vm = psutil.virtual_memory()
    return HwStats(
        cpu_process_pct=cpu,
        ram_used_mb=float(vm.used) / 1e6,
        ram_total_mb=float(vm.total) / 1e6,
        **_gpu_fields(),
    )
