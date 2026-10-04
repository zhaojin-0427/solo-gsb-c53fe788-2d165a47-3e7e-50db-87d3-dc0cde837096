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
   bound ``K1`` on |d(el)/dt|.

   A bracket whose *both* endpoints are below the threshold is **never**
   declared clear merely because it is short. It is certified clear only
   when a bound proves the whole bracket stays below:

   * first order (``K1`` tent): ``K1 * w <= (m_a + m_b)/2``;
   * second order: a rigorous per-bracket bound on |el''| together with the
     analytic endpoint elevation rates gives two Taylor upper envelopes;
     if both stay below the threshold the bracket cannot cross it.

   Otherwise the bracket is bisected recursively. A midpoint at/above the
   threshold reveals a (possibly very short) pass: both crossings are
   refined by bisection and the shoulders are certified in turn, so a
   sub-0.02 s pass wedged between two below-threshold samples is found
   rather than skipped. Recursion stops on a proof, or when the residue is
   below the numerical resolution:

   * the 1e-7 s float64 time resolution, or
   * the ~1e-6 deg elevation evaluation resolution (a contact that never
     rises more than that above the threshold cannot be told apart from the
     GMST/Kepler evaluation noise).

   Such residues -- plus propagation/root-finding failures -- come back as
   ``uncovered_intervals`` with reasons, never as a silent ``complete``.

Crossings are refined with bisection; the peak with golden-section search
bracketed by the adjacent samples.
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
# Initial dense step is at least this long for efficiency; it is ONLY a
# sampling convenience, never a completeness guarantee: below-below brackets
# are recursively certified from the rate bound regardless of this floor.
DENSE_FLOOR_S = 0.02
DENSE_CAP_S = 20.0
# Below this bracket width, float64 time arithmetic and elevation evaluation
# can no longer order the interior meaningfully; uncertified residues of
# this scale are reported as uncovered instead of guessed.
TIME_RESOLUTION_S = 1.0e-7
# Absolute numerical resolution of an evaluated elevation (degrees). Below
# this margin from the threshold the float64 GMST/Kepler chain (~3e-7 deg
# rms at second-300 epochs) cannot order crossings reliably. A bracket whose
# reachable elevation never rises more than this above the threshold is
# reported as an unclassifiable residue instead of a (noise-driven) window.
ELEVATION_EPS_DEG = 1.0e-6
# Contacts shorter than this are below the crossing output resolution and
# are reported as uncovered instead of as zero-length "windows".
CONTACT_MIN_S = 1.0e-6
# Backstops for the explicit certification stack (purely defensive; normal
# brackets certify in a few dozen nodes). Exceeding them is never treated
# as coverage: the residue is reported uncovered.
CERTIFY_MAX_NODES = 50_000
CERT_MAX_REACH_DEG = 89.0  # second-order proof is used only below this el
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

        # ECEF satellite acceleration bound (radial gravity + centrifugal
        # + Coriolis 2 w x v_ecef).
        a_ecef = (
            MU_EARTH_KM / (rp * rp)
            + EARTH_OMEGA * EARTH_OMEGA * ra
            + 2.0 * EARTH_OMEGA * self.s_max
        )
        self.a_ecef = a_ecef

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


def _obs(orbit, t, station, epoch):
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
    fa = el_lo - threshold
    for _ in range(BISECTION_ITERS):
        if (b - a) <= max(TIME_RESOLUTION_S,
                          1.0e-14 * max(1.0, abs(a), abs(b))):
            break
        mid = 0.5 * (a + b)
        if mid == a or mid == b:
            break
        try:
            om = _obs(orbit, mid, station, epoch)
        except _NUMERIC_ERRORS:
            break
        fm = om.elevation_deg - threshold
        if fa * fm <= 0.0:
            b = mid
        else:
            a, fa = mid, fm
    t_root = 0.5 * (a + b)
    try:
        return t_root, _obs(orbit, t_root, station, epoch)
    except _NUMERIC_ERRORS:
        return None


# ------------------------------------------------------------- rate proofs
def _tent_clear(margin_lo_rad: float, margin_hi_rad: float,
                width: float, k1: float) -> bool:
    """True iff a function bounded by ``|f'| <= k1`` (rad/s) cannot cross
    zero inside the bracket: both samples are ``margin`` below zero and the
    reachable rise over the whole width stays below zero.
    """
    return k1 * width <= 0.5 * (margin_lo_rad + margin_hi_rad)


def _local_k2(ox, oy, width: float, bounds: "_Bounds"):
    """Rigorous bound ``|el''(t)| <= K2`` (rad/s^2) valid throughout the
    bracket, or ``None`` when the geometry inside the bracket cannot be
    bounded tightly (the caller then subdivides instead).

    Derivation (ECEF, station fixed). With ``rho = sat - station``,
    ``R = |rho|``, ``u = sin(el) = rho_up / R`` and unit up-vector e_z::

        R'  = rho' . rho / R
        R'' = |rho'|^2/R + rho'' . rho/R
        u'  = rho'_up/R - u R'/R
        u'' = rho''_up/R - 2 rho'_up R'/R^2
              + 2 u (R')^2/R^2 - u R''/R
        cos(el) el'' = u'' - sin(el) (el')^2

    Bounding every term with |rho'| <= s_max, |rho''| <= a_ecef,
    |rho_up| <= R*sin(el_bar), |rho'_up| <= s_max, |R'| <= s_max and
    per-bracket lower bounds on R and cos(el) gives the expression below.
    """
    s_max, k1 = bounds.s_max, bounds.k_el
    # |elevation| reachable inside the bracket (K1 cone from both ends).
    reach = max(abs(ox.elevation_deg), abs(oy.elevation_deg)) + math.degrees(
        k1 * width)
    if reach >= CERT_MAX_REACH_DEG:
        return None  # 1/cos(el) would be unbounded; skip the 2nd-order proof
    c_min = math.cos(math.radians(reach))
    s_bar = math.sin(math.radians(reach))
    # Range can shrink at speed s_max from either observed endpoint.
    rho_lo = max(bounds.rho_min,
                 min(ox.range_km, oy.range_km) - s_max * width)
    if rho_lo <= 0.0:
        return None
    a_e = bounds.a_ecef
    k2 = (
        # |rho''_up|/R
        a_e / rho_lo
        # 2 |rho'_up| |R'| / R^2  (cross term of the quotient rule)
        + 2.0 * s_max * s_max / (rho_lo * rho_lo)
        # 2 |u| (R')^2 / R^2  (second quotient-rule term on u)
        + 2.0 * s_bar * s_max * s_max / (rho_lo * rho_lo)
        # |u| (|rho'|^2/R + |rho''|) / R  (the R'' term)
        + s_bar * (s_max * s_max / rho_lo + a_e) / rho_lo
        # sin(el) (el')^2 / cos(el)
        + s_bar * k1 * k1 / c_min
    ) / c_min
    return k2 if math.isfinite(k2) and k2 > 0.0 else None


def _parabola_max_below(f0, fp0, f1, fp1, w: float, k2: float) -> bool:
    """Second-order clearance test. Given f(0)=f0<=0, f(w)=f1<=0, endpoint
    derivatives fp0/fp1 (rad/s) and |f''| <= k2 over [0, w], return True iff
    f(t) <= 0 for every t in [0, w].

    Upper envelopes (Taylor with the second-derivative bound)::

        eL(t) = f0 + fp0 t   + 0.5 k2 t^2
        eR(t) = f1 - fp1(w-t) + 0.5 k2 (w-t)^2

    f is bounded by min(eL, eR); that min of two upward parabolas attains
    its maximum only at a vertex of one or at an intersection of the two.
    """
    cand = [0.0, w]
    # Vertices.
    tv = -fp0 / k2
    if 0.0 <= tv <= w:
        cand.append(tv)
    tv = w + fp1 / k2
    if 0.0 <= tv <= w:
        cand.append(tv)
    # Intersections: the t^2 terms of eL and eR are identical, so eL - eR
    # is linear; collect its root in [0, w].
    A = 0.5 * k2
    eL = lambda t: f0 + fp0 * t + A * t * t
    # eR(t) = f1 - fp1*(w-t) + A*(w-t)**2
    #        = f1 - fp1*w + fp1*t + A*(w*w - 2wt + t*t)
    c0 = f1 - fp1 * w + A * w * w
    c1 = fp1 - 2.0 * A * w
    # eL - eR = (f0 - c0) + (fp0 - c1) t  (t^2 terms cancel)
    d0 = f0 - c0
    d1 = fp0 - c1
    if abs(d1) > 1.0e-300:
        tr = -d0 / d1
        if 0.0 <= tr <= w:
            cand.append(tr)
    for t in cand:
        if min(eL(t), c0 + c1 * t + A * t * t) > 0.0:
            return False
    return True


def _cone_gap(x, m_lo, y, m_hi, k1):
    """Residue not excluded by the one-sided K1 cones: from x the function
    is provably below zero for m_lo/K1 seconds, symmetrically at y."""
    if k1 <= 0.0:
        return x, y
    ux = x + min(y - x, m_lo / k1)
    uy = y - min(y - x, m_hi / k1)
    return ux, uy


# ------------------------------------------------------- below-below scan
# Events emitted by the certifier / segment scanner:
#   ("rise",  t, obs)            threshold crossing upwards
#   ("above", t, obs)            above-threshold interior sample
#   ("set",   t, obs)            threshold crossing downwards
#   ("uncov", a, b, reason)      interval that could not be certified
def _certify_below(safe, result, a, oa, b, ob, threshold, k1, bounds,
                   root_side=None):
    """Certify the half-open bracket [a, b) whose endpoint elevations are
    both strictly below ``threshold``.

    Returns a chronologically sortable list of events:

    * the whole bracket is proven below the threshold -- either by the
      first-order (rate) tent test or the second-order Taylor envelope test
      using the analytic endpoint elevation rates and a rigorous local
      bound on |el''|;
    * an interior point at/above the threshold reveals a (possibly very
      short) pass: both crossings are refined by bisection, the above
      samples become ``rise``/``above``/``set`` events and the shoulders
      are re-certified;
    * anything that still cannot be proven at the numerical time resolution
      (or after the node budget is spent) is returned as an ``uncov`` event
      carrying the exact reason -- never silently skipped.

    ``root_side`` ("lo"/"hi"/None) marks an endpoint that coincides with an
    already resolved crossing: a sub-``TIME_RESOLUTION_S`` residue touching
    that endpoint is absorbed into the crossing uncertainty.
    """
    events: list = []
    nodes = [0]
    # Entries: (lo, obs_lo, hi, obs_hi, root_side)
    stack = [(a, oa, b, ob, root_side)]
    while stack:
        x, ox, y, oy, rside = stack.pop()
        m_lo = max(0.0, math.radians(threshold - ox.elevation_deg))
        m_hi = max(0.0, math.radians(threshold - oy.elevation_deg))
        width = y - x
        if width <= 0.0:
            continue

        # ---- first-order rate-bound proof ----
        if _tent_clear(m_lo, m_hi, width, k1):
            continue
        # Highest elevation the K1 cones from both endpoints allow anywhere
        # inside the bracket (the peak of the two-rate tent).
        reach_max = 0.5 * (ox.elevation_deg + oy.elevation_deg
                           ) + 0.5 * math.degrees(k1 * width)

        nodes[0] += 1
        if nodes[0] > CERTIFY_MAX_NODES:
            ux, uy = _cone_gap(x, m_lo, y, m_hi, k1)
            if uy - ux > 0.0 and not (
                uy - ux <= TIME_RESOLUTION_S and rside is not None
            ):
                events.append((
                    "uncov", ux, uy,
                    "below-threshold bracket certification spent its "
                    f"{CERTIFY_MAX_NODES}-sample budget without the rate/"
                    "curvature bounds excluding an above-threshold peak "
                    "inside",
                ))
            continue

        # ---- second-order proof (analytic endpoint rates + |el''| bound) ----
        k2 = _local_k2(ox, oy, width, bounds)
        if k2 is not None and _parabola_max_below(
            -m_lo, math.radians(ox.elevation_rate_deg_s),
            -m_hi, math.radians(oy.elevation_rate_deg_s),
            width, k2,
        ):
            continue

        # ---- numerical time resolution reached: report only the residue ----
        if width <= 1.5 * TIME_RESOLUTION_S:
            ux, uy = _cone_gap(x, m_lo, y, m_hi, k1)
            if uy - ux <= 0.0:
                continue  # one-sided cones cover the bracket
            if uy - ux <= TIME_RESOLUTION_S and rside is not None:
                # Inside the uncertainty of the crossing already resolved.
                continue
            events.append((
                "uncov", ux, uy,
                "below-threshold bracket could not be proven clear: reached "
                f"the {TIME_RESOLUTION_S:g} s numerical time resolution "
                "without the elevation rate/curvature bounds excluding an "
                "above-threshold peak inside",
            ))
            continue

        # ---- elevation evaluation resolution ----
        # The tightest first-order bound lets the curve rise at most
        # ELEVATION_EPS_DEG above the threshold, yet it is not proven clear:
        # a crossing at that scale cannot be told apart from the evaluation
        # noise. Report the cone residue as unclassifiable rather than emit
        # a noise-driven window or claim coverage. A genuinely large interior
        # hump makes reach_max >> eps, so it keeps subdividing and is found.
        if reach_max <= threshold + ELEVATION_EPS_DEG:
            ux, uy = _cone_gap(x, m_lo, y, m_hi, k1)
            if uy - ux <= 0.0:
                continue
            if uy - ux <= TIME_RESOLUTION_S and rside is not None:
                continue
            events.append((
                "uncov", ux, uy,
                "possible grazing contact cannot exceed "
                f"{ELEVATION_EPS_DEG:g} deg above the threshold, which is at "
                "the elevation evaluation resolution: visibility cannot be "
                "classified reliably",
            ))
            continue

        # ---- subdivide ----
        t_mid = 0.5 * (x + y)
        o_mid = safe(t_mid)
        result.dense_samples += 1
        if isinstance(o_mid, tuple):
            ux, uy = _cone_gap(x, m_lo, y, m_hi, k1)
            events.append(("uncov", ux, uy,
                           f"elevation evaluation failed: {o_mid[1]}"))
            continue

        if o_mid.elevation_deg < threshold:
            stack.append((t_mid, o_mid, y, oy,
                          "hi" if rside == "hi" else None))
            stack.append((x, ox, t_mid, o_mid,
                          "lo" if rside == "lo" else None))
            continue

        # The midpoint is above the threshold by more than the evaluation
        # resolution: a genuine contact to resolve.
        if o_mid.elevation_deg <= threshold + ELEVATION_EPS_DEG:
            ux, uy = _cone_gap(x, m_lo, y, m_hi, k1)
            events.append((
                "uncov", ux, uy,
                "possible grazing contact never rises more than "
                f"{ELEVATION_EPS_DEG:g} deg above the threshold, which is at "
                "the elevation evaluation resolution: visibility cannot be "
                "classified reliably",
            ))
            continue

        # Interior point at/above the threshold: resolve the hump.
        rise = _bisect_crossing_safe(safe, x, t_mid,
                                     ox.elevation_deg,
                                     o_mid.elevation_deg, threshold)
        if rise is None:
            ux, uy = _cone_gap(x, m_lo, y, m_hi, k1)
            events.append(("uncov", ux, uy,
                           "bisection failed on rise crossing of an "
                           "above-threshold hump inside a below-threshold "
                           "bracket"))
            continue
        t_rise, o_rise = rise

        setc = _bisect_crossing_safe(safe, t_mid, y,
                                     o_mid.elevation_deg,
                                     oy.elevation_deg, threshold)
        if setc is None:
            events.append(("rise", t_rise, o_rise))
            events.append(("above", t_mid, o_mid))
            events.append(("uncov", t_rise, y,
                           "bisection failed on set crossing of an "
                           "above-threshold hump inside a below-threshold "
                           "bracket"))
            continue
        t_set, o_set = setc

        events.append(("rise", t_rise, o_rise))
        events.append(("above", t_mid, o_mid))
        events.append(("set", t_set, o_set))

        # Re-certify the shoulders; root-side residues are absorbed into
        # the crossing uncertainty.
        if t_rise - x > 0.0:
            events.extend(_certify_below(safe, result, x, ox, t_rise, o_rise,
                                         threshold, k1, bounds, "hi"))
        if y - t_set > 0.0:
            events.extend(_certify_below(safe, result, t_set, o_set, y, oy,
                                         threshold, k1, bounds, "lo"))

    return events


def _bisect_crossing_safe(safe, lo, hi, el_lo, el_hi, threshold):
    """Bisection on a sign-changing bracket using a certifier-style safe
    sampler. Returns (t_root, obs) or None."""
    a, b = lo, hi
    fa, fb = el_lo - threshold, el_hi - threshold
    if fa * fb > 0.0:
        return None  # caller promises a sign change
    om = None
    for _ in range(BISECTION_ITERS):
        if (b - a) <= max(TIME_RESOLUTION_S, 1.0e-14 * max(1.0, abs(a), abs(b))):
            break
        mid = 0.5 * (a + b)
        if mid == a or mid == b:
            break
        om = safe(mid)
        if isinstance(om, tuple):
            return None
        fm = om.elevation_deg - threshold
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    t_root = 0.5 * (a + b)
    probe = safe(t_root)
    if isinstance(probe, tuple):
        return None
    return t_root, probe


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


def _scan_segment(orbit, station, epoch, a, b, threshold, k1, bounds,
                  domain_start, domain_end, result, fail_intervals):
    windows: list[_RawWindow] = []
    cur: Optional[_RawWindow] = None

    def safe(t):
        try:
            return _obs(orbit, t, station, epoch)
        except _NUMERIC_ERRORS as exc:
            return ("FAIL", str(exc))

    # ------------------------------------------------- window bookkeeping
    def open_window(tc, oc, start_open=False):
        nonlocal cur
        cur = _RawWindow(rise_t=tc, rise_az=oc.azimuth_deg,
                         start_open=start_open,
                         peak_t=tc, peak_el=max(oc.elevation_deg, threshold),
                         peak_left=tc, peak_right=tc,
                         max_range=oc.range_km)

    def update_above(t, o):
        if cur is not None:
            cur.peak_right = t
            cur.max_range = max(cur.max_range, o.range_km)
            if o.elevation_deg > cur.peak_el:
                cur.peak_t, cur.peak_el = t, o.elevation_deg
                cur.peak_left = t_prev_global[0]
                cur.peak_right = t

    def close_window(tc, oc):
        nonlocal cur
        if cur is not None:
            cur.set_t, cur.set_az = tc, oc.azimuth_deg
            cur.peak_right = min(cur.peak_right, tc)
            windows.append(cur)
            cur = None

    def truncate_at(t_close, az_close):
        nonlocal cur
        if cur is not None:
            cur.set_t, cur.set_az = t_close, az_close
            cur.truncated = True
            cur.peak_right = min(cur.peak_right, t_close)
            windows.append(cur)
            cur = None

    # Track the last known-good time for peak bracketing across replayed
    # certifier events.
    t_prev_global = [a]

    def mark_failure(ux, uy, why):
        if cur is not None:
            truncate_at(ux, cur.rise_az)
        fail_intervals.append((ux, uy, why))

    def replay(events):
        for ev in sorted(events, key=lambda e: e[1]):
            kind = ev[0]
            if kind == "rise":
                _, tc, oc = ev
                if cur is None:
                    open_window(tc, oc)
                t_prev_global[0] = tc
            elif kind == "above":
                _, t, o = ev
                left = t_prev_global[0]
                t_prev_global[0] = t
                if cur is not None:
                    cur.peak_right = t
                    cur.max_range = max(cur.max_range, o.range_km)
                    if o.elevation_deg > cur.peak_el:
                        cur.peak_t, cur.peak_el = t, o.elevation_deg
                        cur.peak_left = left
                        cur.peak_right = t
            elif kind == "set":
                _, tc, oc = ev
                close_window(tc, oc)
                t_prev_global[0] = tc
            else:
                _, ux, uy, why = ev
                mark_failure(ux, uy, why)
                t_prev_global[0] = uy

    # ------------------------------------------------- segment walk
    o_prev = safe(a)
    result.dense_samples += 1
    t_prev = a
    t_prev_global[0] = a
    if isinstance(o_prev, tuple):
        fail_intervals.append(
            (a, min(b, a + DENSE_CAP_S), f"evaluation failed: {o_prev[1]}")
        )
        return windows

    if o_prev.elevation_deg >= threshold and abs(a - domain_start) < 1.0e-9:
        open_window(a, o_prev, start_open=True)

    def recover(t_from, t_to, why):
        """Mark [t_from, t_to) failed and walk to the next good sample.
        Returns (t_good, obs_good) or None if the piece end is unreachable."""
        mark_failure(t_from, t_to, why)
        walk = t_to
        for _ in range(8):
            if walk >= b:
                return None
            walk = min(b, walk + max(DENSE_CAP_S, (t_to - t_from)))
            probe = safe(walk)
            result.dense_samples += 1
            if not isinstance(probe, tuple):
                return walk, probe
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
            t_prev_global[0] = t_prev
            continue

        below_prev = o_prev.elevation_deg < threshold
        below_new = o_new.elevation_deg < threshold

        if below_prev and below_new:
            # Certify the bracket from the rate bound; never trust its
            # width. The certifier returns crossing events for any hump it
            # finds and uncov events for anything it cannot prove.
            events = _certify_below(safe, result, t_prev, o_prev,
                                    t_new, o_new, threshold, k1, bounds)
            replay(events)
        elif below_prev and not below_new:
            if o_new.elevation_deg <= threshold + ELEVATION_EPS_DEG:
                # At the noise floor: do not claim a crossing; let the
                # certifier classify the bracket (it reports uncovered).
                events = _certify_below(safe, result, t_prev, o_prev,
                                        t_new, o_new, threshold, k1, bounds)
                replay(events)
            else:
                cross = _bisect_crossing(orbit, station, epoch, t_prev, t_new,
                                         o_prev.elevation_deg,
                                         o_new.elevation_deg, threshold)
                if cross is None:
                    mark_failure(t_prev, t_new,
                                 "bisection failed on rise crossing")
                else:
                    tc, oc = cross
                    open_window(tc, oc)
        elif not below_prev and below_new:
            if o_prev.elevation_deg <= threshold + ELEVATION_EPS_DEG:
                events = _certify_below(safe, result, t_prev, o_prev,
                                        t_new, o_new, threshold, k1, bounds)
                replay(events)
            else:
                cross = _bisect_crossing(orbit, station, epoch, t_prev, t_new,
                                         o_prev.elevation_deg,
                                         o_new.elevation_deg, threshold)
                if cross is None:
                    mark_failure(t_prev, t_new,
                                 "bisection failed on set crossing")
                else:
                    tc, oc = cross
                    if cur is None:
                        # Above at the previous sample without an open window
                        # (e.g. segment start): synthesize an open start.
                        open_window(t_prev, o_prev, start_open=True)
                    close_window(tc, oc)
        else:
            # Both above: sample belongs to the current window.
            update_above(t_new, o_new)

        t_prev, o_prev = t_new, o_new
        t_prev_global[0] = t_prev

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
            _scan_segment(orbit, station, epoch, a, b, threshold, k1, bounds,
                          s0, s1, result, dense_fail)
        )

    all_fail = [(max(s0, a), min(s1, b), why) for a, b, why in coarse_fail]
    all_fail.extend(dense_fail)

    # ---------------- assemble windows ----------------
    unresolved_contacts = []
    for w in raw_windows:
        if w.set_t is None:
            w.set_t, w.set_az = s1, w.rise_az
            w.end_open = True
        if w.set_t - w.rise_t < CONTACT_MIN_S:
            # Sub-resolution contact: do not emit a zero-length window and
            # do not claim it clear; register the time range as unclassified.
            unresolved_contacts.append((
                w.rise_t, w.set_t,
                f"contact at the {threshold:g} deg threshold is shorter than "
                f"the {CONTACT_MIN_S:g} s output resolution "
                f"(peak {max(w.peak_el, threshold):.6g} deg): visibility "
                "could not be classified",
            ))
            continue
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
    unresolved = [(x, y, reason) for x, y, reason in all_fail]
    unresolved.extend(unresolved_contacts)
    for a, b in _merge([(x, y) for x, y, _ in unresolved]):
        if b - a <= 1.0e-12:
            continue
        why = next((reason for x, y, reason in unresolved
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
