"""Unit tests for find_passes certification semantics."""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import numpy as np

from app import passes as P
from app.geometry import Station
from app.mechanics import OrbitError, TwoBodyOrbit
from conftest import circular_leo_state


def _setup():
    epoch, pos, vel = circular_leo_state()
    return (epoch, TwoBodyOrbit(np.array(pos), np.array(vel)),
            Station(39.9, 116.4, 0.05))


def test_evaluation_failure_is_reported_not_silenced():
    """Propagation failures inside a pass must downgrade status to partial,
    mark the open window truncated and register uncovered time."""
    epoch, orb, sta = _setup()
    base = P.find_passes(orb, sta, epoch, epoch,
                         epoch + timedelta(days=1), 10.0)
    t_bad = (base.windows[0].peak_time - epoch).total_seconds()
    real_observe = P.observe

    def fake_observe(o, t, when, st):
        if abs(t - t_bad) < 12.0:
            raise OrbitError("simulated Kepler failure")
        return real_observe(o, t, when, st)

    with patch.object(P, "observe", fake_observe):
        res = P.find_passes(orb, sta, epoch, epoch,
                            epoch + timedelta(days=1), 10.0)

    assert res.status == "partial"
    assert any(w.truncated for w in res.windows)
    assert len(res.uncovered) >= 1
    total = sum((u.end - u.start).total_seconds()
                for u in res.uncovered)
    assert total > 0.0
    assert all("failed" in u.reason for u in res.uncovered)


def test_below_below_bracket_with_gap_must_not_be_complete():
    """A below-threshold bracket that cannot be certified (node budget
    spent) comes back as an uncovered interval instead of 'complete'."""
    epoch, orb, sta = _setup()
    real_certify = P._certify_below

    def force_budget(safe, result, a, oa, b, ob, threshold, k1, bounds,
                     root_side=None):
        evs = real_certify(safe, result, a, oa, b, ob, threshold, k1, bounds,
                           root_side)
        width = b - a
        if width > 1.0:  # only poison the first large dense brackets
            evs = list(evs) + [
                ("uncov", a + 0.3 * width, a + 0.4 * width,
                 "injected certification budget exhaustion")
            ]
        return evs

    with patch.object(P, "_certify_below", force_budget):
        res = P.find_passes(orb, sta, epoch, epoch,
                            epoch + timedelta(hours=2), 10.0)
    assert res.status in ("partial", "failed")
    assert len(res.uncovered) >= 1
