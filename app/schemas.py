"""Request/response models with strict unit and range validation.

Anything outside the documented physical ranges (units, time order, search
span, geodetic coordinates, elevation, finite vectors) is rejected with
HTTP 422 by FastAPI. Orbital consistency is checked separately in the
endpoint and also returned as 422.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_SPAN = timedelta(days=7)


def _check_finite_vec3(v, name: str):
    if len(v) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in v):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(x) for x in v]


class SatelliteState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_km: list[float] = Field(
        ..., description="ECI position [x, y, z] in kilometers at 'epoch'."
    )
    velocity_km_s: list[float] = Field(
        ..., description="ECI velocity [vx, vy, vz] in km/s at 'epoch'."
    )

    @field_validator("position_km", "velocity_km_s")
    @classmethod
    def _vec(cls, v, info):
        return _check_finite_vec3(v, info.field_name)


class GroundStation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude_deg: float = Field(..., ge=-90.0, le=90.0,
                                description="WGS84 geodetic latitude, degrees.")
    longitude_deg: float = Field(..., ge=-180.0, le=180.0,
                                 description="WGS84 longitude, degrees.")
    altitude_km: float = Field(..., ge=-12.0, le=100000.0,
                               description="WGS84 ellipsoidal height, km.")


class PassRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: datetime = Field(
        ..., description="UTC epoch of the satellite state (timezone aware)."
    )
    start: datetime = Field(..., description="UTC search start (timezone aware).")
    end: datetime = Field(..., description="UTC search end (timezone aware).")
    min_elevation_deg: float = Field(
        ..., gt=-90.0, lt=90.0,
        description="Minimum elevation considered visible, degrees.",
    )
    satellite: SatelliteState
    station: GroundStation

    @field_validator("epoch", "start", "end")
    @classmethod
    def _aware(cls, v: datetime, info):
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError(f"{info.field_name} must be timezone-aware UTC (use e.g. 'Z')")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _check_times(self):
        if self.end <= self.start:
            raise ValueError("'end' must be strictly after 'start'")
        if self.end - self.start > MAX_SPAN:
            raise ValueError("search span (end - start) must not exceed 7 days")
        # Reject absurd offsets of the search window from the epoch.
        for name, t in (("start", self.start), ("end", self.end)):
            if abs((t - self.epoch)) > timedelta(days=3650):
                raise ValueError(f"'{name}' is more than 10 years from 'epoch'")
        return self


class PassWindowOut(BaseModel):
    start: datetime
    end: datetime
    peak_time: datetime
    peak_elevation_deg: float
    peak_azimuth_deg: Optional[float] = None
    rise_azimuth_deg: Optional[float] = None
    set_azimuth_deg: Optional[float] = None
    max_range_km: float
    duration_s: float
    start_open: bool
    end_open: bool
    truncated: bool


class UncoveredOut(BaseModel):
    start: datetime
    end: datetime
    reason: str


class PassResponse(BaseModel):
    status: Literal["complete", "partial", "failed"]
    count: int
    windows: list[PassWindowOut]
    uncovered_intervals: list[UncoveredOut]
    search_start: datetime
    search_end: datetime
    min_elevation_deg: float
    diagnostics: dict
