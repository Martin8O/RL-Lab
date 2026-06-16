"""G4b — HW-stats telemetry: psutil CPU/RAM always present, optional pynvml GPU fields, and the
field riding the 1 Hz TrainingProgress frame. NVML may or may not be available on the test box, so
the GPU assertions are conditional (present-and-sane, or cleanly all-None)."""

from app.schemas.training import HwStats, HwStatsFrame
from app.services import hw_stats


def test_sample_returns_cpu_and_ram() -> None:
    s = hw_stats.sample()
    assert isinstance(s, HwStats)
    assert s.cpu_process_pct >= 0.0  # normalised 0–100 % of the machine (0 right after priming is fine)
    assert s.ram_total_mb > 0.0
    assert 0.0 <= s.ram_used_mb <= s.ram_total_mb


def test_gpu_fields_are_all_present_or_all_none() -> None:
    """A non-NVIDIA machine degrades to GPU=None (panel shows '—'); an NVIDIA one fills sane numbers.
    Either way it must be consistent — never a half-populated mix that would render garbage."""
    s = hw_stats.sample()
    gpu = [s.gpu_util_pct, s.gpu_vram_used_mb, s.gpu_vram_total_mb, s.gpu_temp_c, s.gpu_power_w]
    if s.gpu_util_pct is None:
        assert all(v is None for v in gpu)
    else:
        assert all(v is not None for v in gpu)
        assert 0.0 <= s.gpu_util_pct <= 100.0
        assert s.gpu_vram_total_mb and s.gpu_vram_used_mb <= s.gpu_vram_total_mb
        assert s.gpu_temp_c > 0.0


def test_sample_is_stable_across_calls() -> None:
    """Priming + NVML init are one-shot, so repeated sampling never raises."""
    for _ in range(3):
        assert isinstance(hw_stats.sample(), HwStats)


def test_hwstats_frame_is_algo_independent() -> None:
    """Telemetry rides its own {type:"hwstats"} frame (broadcast by the manager for any run, not the
    PPO-only progress ticker) so neuroevolution + Q-learning light up the panel too."""
    frame = HwStatsFrame(stats=hw_stats.sample())
    dumped = frame.model_dump()
    assert dumped["type"] == "hwstats"
    assert dumped["stats"]["ram_total_mb"] > 0.0
