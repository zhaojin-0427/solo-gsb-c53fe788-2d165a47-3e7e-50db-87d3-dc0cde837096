"""Visibility-window detection.

Strategy
--------
Two nested adaptive scans over the search interval, treated as the half-open
range ``[t_start, t_end)``:

1. **Coarse geometric gate.** The Earth-central angle between satellite and
   station is cheap to evaluate. A rigorous angular-rate bound
   ``v_perigee / r_perigee + omega_E`` limits the angle change per coarse
   step, so the uniform coarse grid provably cannot skip any geometry that
   could be above the local horizon (gate built at apogee, the loosest
   possible visibility geometry, plus an 8 deg buffer).

2. **Dense elevation scan** inside candidate intervals. The step adapts to
   the elevation margin above/below the threshold and a global Lipschitz
   bound ``K1`` on |d(el)/dt|. Between two *below-threshold* samples the
   curve can then provably not cross the threshold (apart from windows
   smaller than the 0.02 s step floor, i.e. peaks < ~0.02 deg and lasting
   < ~0.1 s).

Crossings are refined with bisection; the peak with golden-section search
bracketed by the adjacent samples. Numerical failures never disappear: the
affected time ranges come back as ``uncovered_intervals`` with reasons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from .geometry import Station, _rz, observe
from .mechanics import EARTH_OMEGA, MU_EARTH_KM, OrbitError, TwoBodyOrbit
from .time_utils import gmst_rad

# ----- tunable numerical knobs ------------------------------------------
GEOMETRY_BUFFER_DEG = 8.0
COARSE_MARGIN_DEG = 5.0
DENSE_FLOOR_S = 0.02
DENSE_CAP_S = 20.0
BISECTION_ITERS = 45
GOLDEN_ITERS = 32
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_NUMERIC_ERRORS = (FloatingPointError, ValueError, OverflowError, OrbitError)


@dataclass
class PassWindow:
    start: datetime
    end: datetime
    peak_time: datetime
    peak_elevation_deg: float
    peak_azimuth_deg: float
    rise_azimuth_deg: float
    set_azimuth_deg: float
    max_range_km: float
    duration_s: float
    start_open: bool = False
    end_open: bool = False
    truncated: bool = False


@dataclass
class UncoveredInterval:
    start: datetime
    end: datetime
    reason: str


@dataclass
class PassResult:
    windows: list[PassWindow] = field(default_factory=list)
    uncovered: list[UncoveredInterval] = field(default_factory=list)
    status: str = "complete"  # complete | partial | failed
    coarse_samples: int = 0
    dense_samples: int = 0


# ------------------------------------------------------------------ bounds
class _Bounds:
    def __init__(self, orbit: TwoBodyOrbit, station: Station):
        rp, ra = orbit.periapsis, orbit.apoapsis
        self.rp, self.ra = rp, ra
        r_stat = float(np.linalg.norm(station.ecef))
        self.r_stat = r_stat

        self.v_max = math.sqrt(MU_EARTH_KM * (2.0 / rp - 1.0 / orbit.a))
        # Earth-fixed satellite speed bound.
        self.s_max = self.v_max + EARTH_OMEGA * ra
        self.rho_min = max(rp - r_stat, 1.0)
        # |d(el)/dt| <= K1 (rad/s): station-fixed frame, range >= rho_min.
        self.k_el = self.s_max / self.rho_min

        # Visible above horizon => cos(alpha) >= r_station / r_sat, so the
        # loosest gate is at apogee (largest r).
        gate_ratio = min(1.0, max(-1.0, r_stat / ra))
        self.psi_horizon = math.acos(gate_ratio)
        self.psi_flag = min(
            math.pi - 1.0e-6,
            self.psi_horizon + math.radians(GEOMETRY_BUFFER_DEG),
        )
        # Angle between two moving unit vectors changes no faster than the
        # sum of their angular rates.
        self.k_alpha = self.v_max / rp + EARTH_OMEGA
        self.h_coarse = math.radians(COARSE_MARGIN_DEG) / self.k_alpha


# ------------------------------------------------------------- raw samplers
def _when(epoch: datetime, t: float) -> datetime:
    return epoch + timedelta(seconds=t)


def _central_angle(orbit: TwoBodyOrbit, t: float, station: Station,
                   epoch: datetime) -> float:
    pos, _ = orbit.state_at(t)
    theta = gmst_rad(_when(epoch, t))
    stat_eci = _rz(theta) @ station.ecef
    d = float(np.dot(pos, stat_eci)) / (
        float(np.linalg.norm(pos)) * float(np.linalg.norm(stat_eci))
    )
    return math.acos(min(1.0, max(-1.0, d)))


def _obs(orbit: TwoBodyOrbit, t: float, station: Station, epoch: datetime):
    return observe(orbit, t, _when(epoch, t), station)


def _elevation(orbit, t, station, epoch) -> float:
    return _obs(orbit, t, station, epoch).elevation_deg


# --------------------------------------------------------- interval helpers
def _merge(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _subtract(base, holes):
    pieces = []
    for a, b in base:
        cur = a
        for c, d in holes:
            if d <= cur or c >= b:
                continue
            if c > cur:
                pieces.append((cur, min(c, b)))
            cur = max(cur, d)
            if cur >= b:
                break
        if cur < b:
            pieces.append((cur, b))
    return pieces


# ---------------------------------------------------------------- crossing
def _bisect_crossing(orbit, station, epoch, lo, hi, el_lo, el_hi, threshold):
    a, b = lo, hi
    fa, fb = el_lo - threshold, el_hi - threshold
    for _ in range(BISECTION_ITERS):
        mid = 0.5 * (a + b)
        try:
            om = _obs(orbit, mid, station, epoch)
        except _NUMERIC_ERRORS:
            break
        fm = om.elevation_deg - threshold
        if (b - a) < 1.0e-8:
            return mid, om
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    t_root = 0.5 * (a + b)
    try:
        return t_root, _obs(orbit, t_root, station, epoch)
    except _NUMERIC_ERRORS:
        return None


# -------------------------------------------------------------------- peak
def _refine_peak(orbit, station, epoch, a, b, fallback):
    lo, hi = a, b
    if hi - lo < 1.0e-9:
        t, el = fallback
        try:
            return t, el, _obs(orbit, t, station, epoch)
        except _NUMERIC_ERRORS:
            return t, el, None
    x1 = hi - _GOLDEN * (hi - lo)
    x2 = lo + _GOLDEN * (hi - lo)
    try:
        f1 = _elevation(orbit, x1, station, epoch)
        f2 = _elevation(orbit, x2, station, epoch)
    except _NUMERIC_ERRORS:
        t, el = fallback
        try:
            return t, el, _obs(orbit, t, station, epoch)
        except _NUMERIC_ERRORS:
            return t, el, None
    for _ in range(GOLDEN_ITERS):
        if f1 < f2:
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + _GOLDEN * (hi - lo)
            try:
                f2 = _elevation(orbit, x2, station, epoch)
            except _NUMERIC_ERRORS:
                break
        else:
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - _GOLDEN * (hi - lo)
            try:
                f1 = _elevation(orbit, x1, station, epoch)
            except _NUMERIC_ERRORS:
                break
    t_best = 0.5 * (lo + hi)
    try:
        ob = _obs(orbit, t_best, station, epoch)
        el_best = ob.elevation_deg
    except _NUMERIC_ERRORS:
        t_best, el_best, ob = fallback[0], fallback[1], None
    if el_best < fallback[1]:
        t_best, el_best = fallback
        try:
            ob = _obs(orbit, t_best, station, epoch)
        except _NUMERIC_ERRORS:
            ob = None
    return t_best, el_best, ob


# ------------------------------------------------------------- dense scan
@dataclass
class _RawWindow:
    rise_t: float
    rise_az: float
    set_t: Optional[float] = None
    set_az: Optional[float] = None
    start_open: bool = False
    end_open: bool = False
    truncated: bool = False
    peak_t: float = 0.0
    peak_el: float = -1.0e30
    peak_left: float = 0.0
    peak_right: float = 0.0
    max_range: float = 0.0


def _scan_segment(orbit, station, epoch, a, b, threshold, k1,
                  domain_start, domain_end, result, fail_intervals):
    windows: list[_RawWindow] = []
    cur: Optional[_RawWindow] = None

    def safe(t):
        try:
            return _obs(orbit, t, station, epoch)
        except _NUMERIC_ERRORS as exc:
            return ("FAIL", str(exc))

    o_prev = safe(a)
    result.dense_samples += 1
    t_prev = a
    if isinstance(o_prev, tuple):
        fail_intervals.append(
            (a, min(b, a + DENSE_CAP_S), f"evaluation failed: {o_prev[1]}")
        )
        return windows

    if o_prev.elevation_deg >= threshold and abs(a - domain_start) < 1.0e-9:
        cur = _RawWindow(rise_t=a, rise_az=o_prev.azimuth_deg, start_open=True,
                         peak_t=a, peak_el=o_prev.elevation_deg,
                         peak_left=a, peak_right=a,
                         max_range=o_prev.range_km)

    def close_open_window(t_close, az_close):
        nonlocal cur
        if cur is not None:
            cur.set_t, cur.set_az = t_close, az_close
            cur.truncated = True
            cur.peak_right = min(cur.peak_right, t_close)
            windows.append(cur)
            cur = None

    def recover(t_from, t_to, why):
        """Mark [t_from, t_to) failed and walk to the next good sample.
        Returns (t_good, obs_good) or None if the piece end is unreachable."""
        close_open_window(t_from, o_prev.azimuth_deg)
        fail_intervals.append((t_from, t_to, why))
        walk = t_to
        for _ in range(8):
            if walk >= b:
                return None
            walk = min(b, walk + max(DENSE_CAP_S, (t_to - t_from)))
            probe = safe(walk)
            result.dense_samples += 1
            if not isinstance(probe, tuple):
                return walk, probe
        # extend the failed interval to the piece end
        fail_intervals[-1] = (t_from, b, why)
        return None

    while t_prev < b - 1.0e-12:
        margin = max(abs(math.radians(o_prev.elevation_deg - threshold)), 1.0e-12)
        h = min(DENSE_CAP_S, max(DENSE_FLOOR_S, 0.5 * margin / k1))
        t_new = min(t_prev + h, b)
        o_new = safe(t_new)
        result.dense_samples += 1

        # ---- failure at the new endpoint: mark bracket, walk to recovery ----
        if isinstance(o_new, tuple):
            rec = recover(t_prev, t_new,
                          f"elevation evaluation failed: {o_new[1]}")
            if rec is None:
                return windows
            t_prev, o_prev = rec
            continue

        # ---- refine a below-below bracket until the Lipschitz gap holds ----
        midpoint_failed = False
        if o_prev.elevation_deg < threshold and o_new.elevation_deg < threshold:
            while (t_new - t_prev) > DENSE_FLOOR_S * 1.5:
                m_lo = abs(math.radians(o_prev.elevation_deg - threshold))
                m_hi = abs(math.radians(o_new.elevation_deg - threshold))
                if k1 * (t_new - t_prev) <= 0.5 * min(m_lo, m_hi):
                    break
                t_mid = 0.5 * (t_prev + t_new)
                o_mid = safe(t_mid)
                result.dense_samples += 1
                if isinstance(o_mid, tuple):
                    rec = recover(t_prev, t_new,
                                  f"elevation evaluation failed: {o_mid[1]}")
                    if rec is None:
                        return windows
                    t_prev, o_prev = rec
                    midpoint_failed = True
                    break
                t_new, o_new = t_mid, o_mid
        if midpoint_failed:
            continue

        below_prev = o_prev.elevation_deg < threshold
        below_new = o_new.elevation_deg < threshold

        if not below_prev and not below_new:
            if cur is not None:
                cur.peak_right = t_new
                cur.max_range = max(cur.max_range, o_new.range_km)
                if o_new.elevation_deg > cur.peak_el:
                    cur.peak_t, cur.peak_el = t_new, o_new.elevation_deg
                    cur.peak_left = t_prev
                    cur.peak_right = t_new
        elif below_prev and not below_new:
            cross = _bisect_crossing(orbit, station, epoch, t_prev, t_new,
                                     o_prev.elevation_deg,
                                     o_new.elevation_deg, threshold)
            if cross is None:
                fail_intervals.append((t_prev, t_new,
                                       "bisection failed on rise crossing"))
            else:
                tc, oc = cross
                cur = _RawWindow(rise_t=tc, rise_az=oc.azimuth_deg,
                                 peak_t=tc, peak_el=threshold,
                                 peak_left=tc, peak_right=t_new,
                                 max_range=oc.range_km)
        elif not below_prev and below_new:
            cross = _bisect_crossing(orbit, station, epoch, t_prev, t_new,
                                     o_prev.elevation_deg,
                                     o_new.elevation_deg, threshold)
            if cross is None:
                fail_intervals.append((t_prev, t_new,
                                       "bisection failed on set crossing"))
                close_open_window(t_prev, o_prev.azimuth_deg)
            else:
                tc, oc = cross
                if cur is None:
                    cur = _RawWindow(rise_t=t_prev, rise_az=o_prev.azimuth_deg,
                                     start_open=True, peak_t=t_prev,
                                     peak_el=o_prev.elevation_deg,
                                     peak_left=t_prev, peak_right=t_prev,
                                     max_range=o_prev.range_km)
                cur.set_t, cur.set_az = tc, oc.azimuth_deg
                cur.peak_right = tc
                windows.append(cur)
                cur = None

        t_prev, o_prev = t_new, o_new

    if cur is not None:
        if abs(b - domain_end) < 1.0e-6:
            cur.set_t, cur.set_az = domain_end, o_prev.azimuth_deg
            cur.end_open = True
        else:
            cur.set_t, cur.set_az = b, o_prev.azimuth_deg
            cur.truncated = True
        windows.append(cur)

    return windows


# ------------------------------------------------------------------- public
def find_passes(orbit, station, epoch, t_start, t_end, min_elevation_deg):
    epoch = epoch.astimezone(timezone.utc)
    t_start = t_start.astimezone(timezone.utc)
    t_end = t_end.astimezone(timezone.utc)

    s0 = (t_start - epoch).total_seconds()
    s1 = (t_end - epoch).total_seconds()
    threshold = float(min_elevation_deg)
    result = PassResult()

    bounds = _Bounds(orbit, station)
    hc = bounds.h_coarse
    k1 = bounds.k_el

    # ---------------- coarse geometric gate ----------------
    flagged = []
    coarse_fail = []
    t = s0
    while t < s1 - 1.0e-9:
        try:
            if _central_angle(orbit, t, station, epoch) <= bounds.psi_flag:
                flagged.append((t - hc, t + hc))
        except _NUMERIC_ERRORS as exc:
            coarse_fail.append((t - hc, t + hc, f"coarse propagation failed: {exc}"))
        t += hc
        result.coarse_samples += 1

    candidates = _merge([(max(s0, a - hc), min(s1, b + hc))
                         for a, b in flagged])
    holes = _merge([(max(s0, a), min(s1, b)) for a, b, _ in coarse_fail])
    pieces = _subtract(candidates, holes)

    # ---------------- dense elevation scan ----------------
    dense_fail = []
    raw_windows = []
    for a, b in pieces:
        if b - a <= 1.0e-9:
            continue
        raw_windows.extend(
            _scan_segment(orbit, station, epoch, a, b, threshold, k1,
                          s0, s1, result, dense_fail)
        )

    all_fail = [(max(s0, a), min(s1, b), why) for a, b, why in coarse_fail]
    all_fail.extend(dense_fail)

    # ---------------- assemble windows ----------------
    for w in raw_windows:
        if w.set_t is None:
            w.set_t, w.set_az = s1, w.rise_az
            w.end_open = True
        if w.set_t - w.rise_t < 1.0e-6:
            continue  # zero-duration tangent contact
        pa = max(w.rise_t, w.peak_left)
        pb = min(w.set_t, w.peak_right)
        tp, elp, obp = _refine_peak(
            orbit, station, epoch, pa, pb,
            (w.peak_t, max(w.peak_el, threshold)),
        )
        azp = obp.azimuth_deg if obp is not None else float("nan")
        result.windows.append(
            PassWindow(
                start=_when(epoch, w.rise_t),
                end=_when(epoch, w.set_t),
                peak_time=_when(epoch, tp),
                peak_elevation_deg=elp,
                peak_azimuth_deg=azp,
                rise_azimuth_deg=w.rise_az,
                set_azimuth_deg=w.set_az if w.set_az is not None else float("nan"),
                max_range_km=w.max_range,
                duration_s=max(0.0, w.set_t - w.rise_t),
                start_open=w.start_open,
                end_open=w.end_open,
                truncated=w.truncated,
            )
        )

    result.windows.sort(key=lambda x: x.start)

    # ---------------- uncovered intervals ----------------
    for a, b in _merge([(x, y) for x, y, _ in all_fail]):
        if b - a <= 1.0e-9:
            continue
        why = next((reason for x, y, reason in all_fail
                    if abs(x - a) < 1.0e-6 and abs(y - b) < 1.0e-6),
                   "numerical failure")
        result.uncovered.append(
            UncoveredInterval(_when(epoch, a), _when(epoch, b), why)
        )

    covered_ok = not result.uncovered and not any(w.truncated for w in result.windows)
    if result.windows:
        result.status = "complete" if covered_ok else "partial"
    elif result.uncovered:
        result.status = "failed"
    else:
        result.status = "complete"
    return result
