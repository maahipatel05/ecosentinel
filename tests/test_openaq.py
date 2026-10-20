"""
Offline tests for EcoSentinel's OpenAQ v3 calls, using a fake OpenAQ
(httpx.MockTransport) so no network is needed.

Regression tests for a real bug: /sensors/{id}/measurements only accepts
datetime_from, datetime_to, limit and page. The old code sent date_from,
date_to, order_by and sort_order, which OpenAQ silently ignored, so it got
each sensor's OLDEST readings (e.g. a CO value from 2018 shown as "current").
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import cache
import server
from anomaly import fetch_pm25_history, fetch_current_pm25
from causal import fetch_pm25_daily

MEASUREMENT_PARAMS = {"datetime_from", "datetime_to", "limit", "page"}


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def measurement(value: float, when: datetime) -> dict:
    return {"value": value, "period": {"datetimeFrom": {"utc": iso(when)}}}


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()
    yield
    cache.clear()


# ── anomaly.py ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_uses_valid_params_and_drops_old_rows():
    now = datetime.now(timezone.utc)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = set(request.url.params.keys())
        return httpx.Response(
            200,
            json={
                "results": [
                    measurement(40.0, now - timedelta(days=400)),  # outside window
                    measurement(50.0, now - timedelta(days=2)),
                    measurement(60.0, now - timedelta(days=1)),
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        values = await fetch_pm25_history(client, sensor_id=1, days=30)

    assert seen["params"] <= MEASUREMENT_PARAMS
    assert {"datetime_from", "datetime_to"} <= seen["params"]
    assert values == [50.0, 60.0]


@pytest.mark.asyncio
async def test_history_pages_past_the_1000_row_limit():
    # Some sensors report several times an hour, so 30 days exceeds one page
    now = datetime.now(timezone.utc)
    pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        pages.append(page)
        rows = 1000 if page == 1 else 5
        return httpx.Response(
            200,
            json={"results": [measurement(10.0, now - timedelta(hours=1))] * rows},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        values = await fetch_pm25_history(client, sensor_id=1, days=30)

    assert pages == [1, 2]
    assert len(values) == 1005


@pytest.mark.asyncio
async def test_current_pm25_reads_sensor_latest_field():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v3/sensors/7"
        return httpx.Response(200, json={"results": [{"latest": {"value": 46.27}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await fetch_current_pm25(client, sensor_id=7) == 46.27


# ── causal.py ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pm25_daily_uses_valid_params_and_groups_by_day():
    now = datetime.now(timezone.utc)
    day1 = now - timedelta(days=3)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = set(request.url.params.keys())
        return httpx.Response(
            200,
            json={
                "results": [
                    measurement(30.0, now - timedelta(days=500)),  # outside window
                    measurement(20.0, day1),
                    measurement(40.0, day1 + timedelta(minutes=1)),
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        by_date = await fetch_pm25_daily(client, sensor_id=1, window_days=20)

    assert seen["params"] <= MEASUREMENT_PARAMS
    assert sum(len(v) for v in by_date.values()) == 2
    assert all(v in (20.0, 40.0) for vals in by_date.values() for v in vals)


# ── server.py get_air_quality ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_air_quality_uses_latest_and_skips_stale_readings(monkeypatch):
    now = datetime.now(timezone.utc)
    fresh = now - timedelta(hours=2)
    stale = datetime(2018, 3, 9, 5, 15, tzinfo=timezone.utc)

    locations = [
        {  # decommissioned station: must be ignored entirely
            "id": 1,
            "name": "Old Station",
            "datetimeLast": {"utc": iso(stale)},
            "sensors": [{"id": 10, "parameter": {"name": "co", "units": "µg/m³"}}],
        },
        {
            "id": 2,
            "name": "Active Station",
            "datetimeLast": {"utc": iso(fresh)},
            "sensors": [
                {"id": 20, "parameter": {"name": "pm25", "units": "µg/m³"}},
                {"id": 21, "parameter": {"name": "co", "units": "ppb"}},
                {"id": 22, "parameter": {"name": "so2", "units": "µg/m³"}},
            ],
        },
    ]
    latest = {
        2: [
            {"sensorsId": 20, "value": 46.27, "datetime": {"utc": iso(fresh)}},
            {"sensorsId": 21, "value": 410.0, "datetime": {"utc": iso(fresh)}},
            {"sensorsId": 22, "value": 104.0, "datetime": {"utc": iso(stale)}},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v3/locations":
            return httpx.Response(200, json={"results": locations})
        if path.endswith("/latest"):
            loc_id = int(path.split("/")[3])
            assert loc_id == 2, "stale station's readings should not be requested"
            return httpx.Response(200, json={"results": latest[loc_id]})
        raise AssertionError(f"unexpected request: {request.url}")

    real_client = httpx.AsyncClient

    def fake_client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    async def fake_geocode(city):
        return 23.0, 72.6, "Ahmedabad, India"

    monkeypatch.setattr(server.httpx, "AsyncClient", fake_client)
    monkeypatch.setattr(server, "geocode", fake_geocode)

    report = await server.get_air_quality("Ahmedabad")

    assert "46.3 µg/m³" in report  # fresh PM2.5
    assert "410.0 ppb" in report  # CO with its own sensor's unit
    assert "SO₂" not in report  # 2018 reading dropped
    assert "Old Station" not in report
    assert "Active Station" in report


def test_co_plausibility_ranges():
    assert server.is_plausible("co", 0.7, "mg/m³")  # normal city air
    assert server.is_plausible("co", 700.0, "ppb")
    assert not server.is_plausible("co", 0.7, "ppb")  # real Ahmedabad case
    assert server.is_plausible("pm25", 0.7, "µg/m³")  # only CO is checked


@pytest.mark.asyncio
async def test_air_quality_hides_implausible_co(monkeypatch):
    fresh = datetime.now(timezone.utc) - timedelta(hours=1)
    location = {
        "id": 3,
        "name": "GPCB Station",
        "datetimeLast": {"utc": iso(fresh)},
        "sensors": [
            {"id": 30, "parameter": {"name": "pm25", "units": "µg/m³"}},
            {"id": 31, "parameter": {"name": "co", "units": "ppb"}},
        ],
    }
    readings = [
        {"sensorsId": 30, "value": 46.0, "datetime": {"utc": iso(fresh)}},
        {"sensorsId": 31, "value": 0.7, "datetime": {"utc": iso(fresh)}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/locations":
            return httpx.Response(200, json={"results": [location]})
        return httpx.Response(200, json={"results": readings})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )

    async def fake_geocode(_city):
        return 23.0, 72.6, "Ahmedabad, India"

    monkeypatch.setattr(server, "geocode", fake_geocode)

    report = await server.get_air_quality("Ahmedabad")

    assert "46.0 µg/m³" in report
    assert "**0.7 ppb**" not in report
    assert "reported 0.7 ppb, which is physically implausible" in report
