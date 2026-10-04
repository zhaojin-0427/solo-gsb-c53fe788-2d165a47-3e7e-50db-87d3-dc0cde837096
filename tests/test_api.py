"""HTTP-level tests for POST /passes."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.mechanics import TwoBodyOrbit
from app.geometry import Station
from app.passes import _obs
from conftest import EPOCH, make_body, circular_leo_state

client = TestClient(app)


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


# ------------------------------------------------------------------- happy
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_basic_request_returns_sorted_windows():
    r = client.post("/passes", json=make_body())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "complete"
    assert data["count"] == len(data["windows"]) >= 4
    times = [w["start"] for w in data["windows"]]
    assert times == sorted(times)
    w = data["windows"][0]
    assert w["peak_elevation_deg"] >= data["min_elevation_deg"]
    assert w["duration_s"] > 0
    assert w["end"] > w["start"]
    assert w["peak_time"] >= w["start"] and w["peak_time"] <= w["end"]
    for key in ("rise_azimuth_deg", "set_azimuth_deg", "peak_azimuth_deg"):
        assert 0.0 <= w[key] < 360.0
    assert data["uncovered_intervals"] == []
    assert data["diagnostics"]["coarse_samples"] > 0


def test_windows_match_dense_truth():
    """Endpoints within ~0.05 s of a 0.25 s independent grid; count equal."""
    body = make_body()
    r = client.post("/passes", json=body)
    assert r.status_code == 200
    windows = r.json()["windows"]

    epoch, pos, vel = circular_leo_state()
    orb = TwoBodyOrbit(np.array(pos), np.array(vel))
    sta = Station(39.9, 116.4, 0.05)
    N = 86400 * 4
    el = np.array([_obs(orb, k * 0.25, sta, epoch).elevation_deg
                   for k in range(N)])
    above = el >= 10.0
    d = np.diff(above.astype(np.int8))
    rises = list(np.where(d == 1)[0] + 1)
    sets = list(np.where(d == -1)[0])
    if above[0]:
        rises = [0] + rises
    assert len(windows) == len(rises) == len(sets)
    for w, ir, is_ in zip(windows, rises, sets):
        rt = ((ir - 1) + (10.0 - el[ir - 1]) / (el[ir] - el[ir - 1])) * 0.25
        st = (is_ + (10.0 - el[is_]) / (el[is_ + 1] - el[is_])) * 0.25
        got_r = datetime.fromisoformat(w["start"].replace("Z", "+00:00"))
        got_s = datetime.fromisoformat(w["end"].replace("Z", "+00:00"))
        assert abs((got_r - epoch).total_seconds() - rt) < 0.05
        assert abs((got_s - epoch).total_seconds() - st) < 0.05
        seg = el[ir:is_ + 1]
        assert abs(w["peak_elevation_deg"] - seg.max()) < 0.001


def test_peak_refinement_improves_on_grid():
    body = make_body()
    data = client.post("/passes", json=body).json()
    # every endpoint sits at the threshold within tight tolerance
    epoch, pos, vel = circular_leo_state()
    orb = TwoBodyOrbit(np.array(pos), np.array(vel))
    sta = Station(39.9, 116.4, 0.05)
    w0 = data["windows"][0]
    for key in ("start", "end"):
        t = datetime.fromisoformat(w0[key].replace("Z", "+00:00"))
        ob = _obs(orb, (t - epoch).total_seconds(), sta, epoch)
        assert abs(ob.elevation_deg - 10.0) < 1.0e-4


# ------------------------------------------------------------- semantics
def test_half_open_domain_end_at_crossing_is_not_open():
    """Domain boundaries placed on the crossings: rise at start -> open start;
    set at (numerically exactly) end -> closed end, window still returned."""
    data = client.post("/passes", json=make_body()).json()
    w0 = data["windows"][0]
    rise = datetime.fromisoformat(w0["start"].replace("Z", "+00:00"))
    set_ = datetime.fromisoformat(w0["end"].replace("Z", "+00:00"))
    # Serialized endpoints are truncated to microseconds: put the domain end a
    # few ms after the reported set crossing (still the same crossing by far).
    body = make_body(
        start=iso(rise),
        end=iso(set_ + timedelta(milliseconds=5)),
    )
    data2 = client.post("/passes", json=body).json()
    assert data2["count"] == 1
    w = data2["windows"][0]
    assert w["start_open"] is True
    assert w["end_open"] is False
    assert w["truncated"] is False
    assert abs((datetime.fromisoformat(w["end"].replace("Z", "+00:00"))
                - set_).total_seconds()) < 0.01


def test_window_fully_contains_domain_clips_open_both_ends():
    data = client.post("/passes", json=make_body()).json()
    w0 = data["windows"][0]
    body = make_body(
        start=(datetime.fromisoformat(w0["start"].replace("Z", "+00:00"))
               + timedelta(seconds=60)).isoformat().replace("+00:00", "Z"),
        end=(datetime.fromisoformat(w0["end"].replace("Z", "+00:00"))
             - timedelta(seconds=60)).isoformat().replace("+00:00", "Z"),
    )
    data2 = client.post("/passes", json=body).json()
    assert data2["count"] == 1
    w = data2["windows"][0]
    assert w["start_open"] and w["end_open"]


def test_no_windows_returns_complete():
    body = make_body(
        end=iso(EPOCH + timedelta(hours=1)),
        min_elevation_deg=89.0,
    )
    data = client.post("/passes", json=body).json()
    assert data["status"] == "complete"
    assert data["count"] == 0
    assert data["windows"] == []


def test_short_window_not_missed():
    """Threshold just below the peak yields a window of ~1-2 s; must appear."""
    data = client.post("/passes", json=make_body()).json()
    peak = data["windows"][0]["peak_elevation_deg"]
    body = make_body(min_elevation_deg=peak - 0.05)
    data2 = client.post("/passes", json=body).json()
    first = data2["windows"][0]
    assert first["duration_s"] < 5.0
    assert first["peak_elevation_deg"] > peak - 0.05


def test_grazing_contact_below_step_floor_not_missed():
    """A pass peak ~5e-6 deg above the threshold gives a contact only ~14 ms
    wide -- narrower than the old 0.02 s dense step floor, with both
    adjacent samples below the threshold. It must be detected, and its
    endpoints must match an independent dense grid (regression for the bug
    that returned empty windows + empty uncovered + complete)."""
    from datetime import datetime, timedelta

    data = client.post("/passes", json=make_body()).json()
    w0 = data["windows"][0]
    peak = w0["peak_elevation_deg"]
    peak_t = datetime.fromisoformat(w0["peak_time"].replace("Z", "+00:00"))
    epoch, pos, vel = circular_leo_state()
    orb = TwoBodyOrbit(np.array(pos), np.array(vel))
    sta = Station(39.9, 116.4, 0.05)
    tp = (peak_t - epoch).total_seconds()

    delta = 5.0e-6  # deg above threshold -> ~14 ms contact at this apex
    body = make_body(
        start=iso(epoch + timedelta(seconds=tp - 0.05)),
        end=iso(epoch + timedelta(seconds=tp + 0.05)),
        min_elevation_deg=peak - delta,
    )
    r = client.post("/passes", json=body)
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["status"] == "complete"
    assert got["count"] == 1
    win = got["windows"][0]
    assert 0.0 < win["duration_s"] < 0.02  # narrower than the old step floor
    assert win["peak_elevation_deg"] >= peak - delta

    # Independent 0.5 ms truth grid inside the search window.
    t0 = tp - 0.05
    t1 = tp + 0.05
    thr = peak - delta
    ts = np.arange(t0, t1, 0.0005)
    els = np.array([_obs(orb, t, sta, epoch).elevation_deg for t in ts])
    above = np.where(els >= thr)[0]
    assert len(above) > 0
    truth_rise = ts[above[0]]
    truth_set = ts[above[-1]]
    got_rise = (datetime.fromisoformat(
        win["start"].replace("Z", "+00:00")) - epoch).total_seconds()
    got_set = (datetime.fromisoformat(
        win["end"].replace("Z", "+00:00")) - epoch).total_seconds()
    assert abs(got_rise - truth_rise) < 0.002
    assert abs(got_set - truth_set) < 0.002
    assert abs(win["peak_elevation_deg"] - els.max()) < 1.0e-5
    assert got["uncovered_intervals"] == []
    assert got["diagnostics"]["coverage_fraction"] == 1.0


def test_grazing_contact_at_noise_floor_is_reported_uncovered():
    """When the threshold is so close to the peak (1e-6 deg) that whether a
    crossing exists is below the elevation evaluation resolution, the API
    must NOT return a noise-driven window nor claim complete: the residue
    goes to uncovered_intervals with status failed and a reduced
    coverage_fraction."""
    from datetime import datetime, timedelta

    data = client.post("/passes", json=make_body()).json()
    w0 = data["windows"][0]
    peak = w0["peak_elevation_deg"]
    peak_t = datetime.fromisoformat(w0["peak_time"].replace("Z", "+00:00"))
    epoch, _, _ = circular_leo_state()
    tp = (peak_t - epoch).total_seconds()

    body = make_body(
        start=iso(epoch + timedelta(seconds=tp - 0.011)),
        end=iso(epoch + timedelta(seconds=tp + 0.049)),
        min_elevation_deg=peak - 1.0e-6,
    )
    got = client.post("/passes", json=body).json()
    assert got["status"] == "failed"
    assert got["count"] == 0
    assert got["windows"] == []
    assert len(got["uncovered_intervals"]) >= 1
    total = sum(
        (datetime.fromisoformat(u["end"].replace("Z", "+00:00"))
         - datetime.fromisoformat(u["start"].replace("Z", "+00:00"))
         ).total_seconds()
        for u in got["uncovered_intervals"]
    )
    assert 0.0 < total <= 0.1
    assert all("resolution" in u["reason"] for u in got["uncovered_intervals"])
    assert got["diagnostics"]["uncovered_seconds"] == total
    assert got["diagnostics"]["coverage_fraction"] < 1.0



def test_empty_result_stays_complete():
    """A search range nowhere near the station geometry reports complete
    with zero windows and zero uncovered intervals."""
    epoch, pos, vel = circular_leo_state()
    body = make_body(
        start=iso(EPOCH + timedelta(hours=2)),
        end=iso(EPOCH + timedelta(hours=3)),
    )
    got = client.post("/passes", json=body).json()
    assert got["status"] == "complete"
    assert got["count"] == 0
    assert got["windows"] == []
    assert got["uncovered_intervals"] == []
    assert got["diagnostics"]["coverage_fraction"] == 1.0


# ------------------------------------------------------------------- 422
@pytest.mark.parametrize("field,value", [
    ("latitude_deg", 90.1),
    ("latitude_deg", -91),
    ("longitude_deg", 200),
    ("altitude_km", -13.0),
])
def test_bad_station_ranges_422(field, value):
    body = make_body()
    body["station"][field] = value
    r = client.post("/passes", json=body)
    assert r.status_code == 422


def test_bad_elevation_ranges_422():
    for v in (-90.0, 90.0, 100.0):
        body = make_body(min_elevation_deg=v)
        assert client.post("/passes", json=body).status_code == 422


def test_time_order_422():
    body = make_body(
        end=iso(EPOCH - timedelta(hours=1)),
        start=iso(EPOCH),
    )
    assert client.post("/passes", json=body).status_code == 422


def test_span_over_seven_days_422():
    body = make_body(end=iso(EPOCH + timedelta(days=7, seconds=1)))
    assert client.post("/passes", json=body).status_code == 422


def test_naive_datetime_422():
    body = make_body(epoch="2025-06-01T00:00:00")
    assert client.post("/passes", json=body).status_code == 422


def test_wrong_units_position_422():
    # Position in meters instead of km -> orbit perigee inside Earth.
    body = make_body()
    body["satellite"]["position_km"] = [
        x * 1000.0 for x in body["satellite"]["position_km"]
    ]
    r = client.post("/passes", json=body)
    assert r.status_code == 422


def test_wrong_units_velocity_422():
    # m/s velocities -> hyperbolic / degenerate energy.
    body = make_body()
    body["satellite"]["velocity_km_s"] = [
        x * 1000.0 for x in body["satellite"]["velocity_km_s"]
    ]
    r = client.post("/passes", json=body)
    assert r.status_code == 422


def test_collinear_state_422():
    body = make_body()
    p = body["satellite"]["position_km"]
    body["satellite"]["velocity_km_s"] = [p[0] * 0.001,
                                          p[1] * 0.001, p[2] * 0.001]
    r = client.post("/passes", json=body)
    assert r.status_code == 422


def test_non_finite_vector_422():
    """NaN/Inf can't round-trip through strict JSON, so validate the model
    directly (the same validator FastAPI runs)."""
    from pydantic import ValidationError
    from app.schemas import PassRequest

    body = make_body()
    body["satellite"]["position_km"] = [7000.0, float("nan"), 0.0]
    with pytest.raises(ValidationError):
        PassRequest.model_validate(body)
    body["satellite"]["position_km"] = [7000.0, float("inf"), 0.0]
    with pytest.raises(ValidationError):
        PassRequest.model_validate(body)


def test_wrong_vector_length_422():
    body = make_body()
    body["satellite"]["position_km"] = [7000.0, 0.0]
    assert client.post("/passes", json=body).status_code == 422


def test_extra_field_422():
    body = make_body()
    body["unknown"] = 1
    assert client.post("/passes", json=body).status_code == 422
