"""Enlarge the *visible* checker floor of the locomotion MuJoCo models — cosmetic only.

Gymnasium's bundled MuJoCo locomotion XMLs render a **finite** checker plane (Ant/HalfCheetah/
Walker2d ±40, Hopper/Humanoid ±20, Swimmer ±40 world units). A trained runner outruns that patch
and then appears to sprint over a grey void — because a ``type="plane"`` geom **collides** as an
infinite plane but only **renders** its checker out to the geom's ``size``. Reacher (a fixed top-down
arm) never travels, so it is left alone.

We patch a *copy* of the model XML with two coupled edits:

1. **Bigger floor** — scale the floor geom's X/Y half-extent by :data:`_FLOOR_SCALE` and the
   material's ``texrepeat`` by the same factor, so each checker square keeps its physical size (the
   runway just gets longer). The plane collides as an infinite plane regardless of ``size``, so this
   is **purely visual** — physics, observations and reproducibility are byte-for-byte unchanged.

2. **Pinned ``<statistic>``** — MuJoCo derives the camera clip planes (``znear``/``zfar``) and the
   directional-light **shadow frustum** from ``model.stat.extent``, which is auto-computed from the
   *floor size*. A naively enlarged floor inflated ``extent`` (Ant 8 → 80 at ×10), pushing ``znear``
   up 10× → depth-buffer precision collapsed → the ant's shadow z-fought the floor into flickering
   dark blotches (shadow acne, worst mid-jump). We therefore pin ``<statistic>`` back to the **stock**
   model's ``extent``/``center`` so all render scaling matches the unpatched model regardless of how
   big we draw the floor. (The rgb camera is the model's ``trackcom`` camera, not the free camera, so
   pinning ``center`` does not affect how it follows the robot.)

The patched file is generated once per process into the writable data dir and cached, then handed to
``gym.make(..., xml_file=...)`` by :func:`app.envs.factory.make_env` for the whole ``mujoco`` family.
Any failure (asset moved by an upstream upgrade, unwritable dir) returns ``None`` so the caller falls
back to the stock model — a cosmetic patch must never break env construction.
"""

from __future__ import annotations

import logging
import os
import re
from functools import cache
from pathlib import Path

from app.core.paths import data_dir

logger = logging.getLogger(__name__)

# How much bigger to make the rendered floor. ×4 turns Ant/HalfCheetah's ±40 into ±160 (a 320-unit
# runway, 4× the stock) — enough that a runner stays on the checker for a watched preview — while
# keeping the visible horizon far nearer than the first attempt's ×10 (which reached "to the horizon").
# The matching texrepeat bump keeps the checker squares the same physical size as the stock model;
# the pinned <statistic> (below) means this size no longer drives clip/shadow scaling, so it is free
# to grow without reintroducing artifacts.
_FLOOR_SCALE = 4

# gym_id (version-stripped) → the bundled asset filename under gymnasium/envs/mujoco/assets/. Only the
# locomotion robots that actually run off their floor are listed; Reacher/Pusher/InvertedPendulum etc.
# are intentionally absent (a None lookup → the caller keeps the stock model).
_ASSET_BY_ENV = {
    "Ant": "ant.xml",
    "HalfCheetah": "half_cheetah.xml",
    "Hopper": "hopper.xml",
    "Walker2d": "walker2d.xml",
    "Humanoid": "humanoid.xml",
    "Swimmer": "swimmer.xml",
}


def _fmt(value: float) -> str:
    """Format a scaled number back into the XML: drop the ``.0`` on whole values (40 → ``40``)."""
    return str(int(value)) if value == int(value) else str(value)


def _scale_attr_pair(attr: str, text: str, scale: int, *, only_first_two: bool) -> str:
    """Scale the numeric components of one ``attr="a b c"`` occurrence in ``text``.

    ``only_first_two`` scales just the X/Y components (the floor geom's size — its third component is
    a plane render-spacing / thickness we leave untouched); otherwise every component is scaled (the
    material's ``texrepeat``, so the checker density tracks the larger plane).
    """

    def repl(m: re.Match[str]) -> str:
        parts = m.group(1).split()
        limit = 2 if only_first_two else len(parts)
        scaled = [_fmt(float(p) * scale) if i < limit else p for i, p in enumerate(parts)]
        return f'{attr}="{" ".join(scaled)}"'

    return re.sub(rf'{attr}="([^"]+)"', repl, text)


def _enlarge_floor(xml: str, scale: int) -> str:
    """Return ``xml`` with the floor geom's footprint and the checker density scaled by ``scale``."""
    # Only the floor geom (name="floor") — never the robot's body geoms, which also carry a size attr.
    xml = re.sub(
        r'<geom[^>]*\bname="floor"[^>]*?/>',
        lambda m: _scale_attr_pair("size", m.group(0), scale, only_first_two=True),
        xml,
    )
    # texrepeat lives once on the MatPlane material; scale every component to keep checker cells square.
    return _scale_attr_pair("texrepeat", xml, scale, only_first_two=False)


def _pin_statistic(xml: str, extent: float, center: tuple[float, float, float]) -> str:
    """Inject a ``<statistic>`` with the stock ``extent``/``center`` so render scaling is floor-size
    independent. Inserted right after the always-present ``<compiler …/>`` element (a valid position;
    MuJoCo resolves top-level sections by name). No-op if the model already declares ``<statistic>``."""
    if "<statistic" in xml:
        return xml
    cx, cy, cz = center
    stat = f'\n  <statistic extent="{_fmt(extent)}" center="{_fmt(cx)} {_fmt(cy)} {_fmt(cz)}"/>'
    return re.sub(r"(<compiler\b[^>]*/>)", lambda m: m.group(1) + stat, xml, count=1)


def _stock_statistic(xml: str) -> tuple[float, tuple[float, float, float]]:
    """Compile the stock XML to read the ``extent``/``center`` MuJoCo auto-derives from it."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    c = m.stat.center
    return float(m.stat.extent), (float(c[0]), float(c[1]), float(c[2]))


@cache
def floored_xml_path(gym_id: str) -> str | None:
    """Path to a floor-enlarged copy of ``gym_id``'s model XML, or ``None`` if it has no big-floor
    variant (non-locomotion env) or generation failed (the caller then uses the stock model).

    Generated once per process and cached; written atomically so a concurrent trainer + preview that
    both build the same env never read a half-written file.
    """
    base = gym_id.split("-")[0]  # "Ant-v5" → "Ant"
    asset = _ASSET_BY_ENV.get(base)
    if asset is None:
        return None
    try:
        import gymnasium.envs.mujoco as mj

        src = Path(mj.__file__).resolve().parent / "assets" / asset
        stock = src.read_text(encoding="utf-8")
        # Pin clip/shadow scaling to the stock model *before* enlarging the floor, so the bigger plane
        # cannot inflate extent (the shadow-acne fix); then enlarge the floor for the longer runway.
        extent, center = _stock_statistic(stock)
        patched = _enlarge_floor(_pin_statistic(stock, extent, center), _FLOOR_SCALE)

        out_dir = data_dir() / "cache" / "mujoco_floors"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / asset
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(patched, encoding="utf-8")
        os.replace(tmp, out)  # atomic on both Windows and POSIX
        return str(out)
    except Exception:  # noqa: BLE001 — a cosmetic patch must never break env construction
        logger.debug("MuJoCo floor enlargement failed for %s; using stock model", gym_id, exc_info=True)
        return None
