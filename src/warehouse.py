"""
EcoSentinel BigQuery Warehouse
==============================
Stores PM2.5 sensor history in BigQuery so trends can be answered with SQL
instead of re-downloading everything from OpenAQ on every question.

Table: <GCP_PROJECT_ID>.ecosentinel.pm25_readings
  city STRING, sensor_id INT64, measured_at TIMESTAMP, pm25 FLOAT64,
  loaded_at TIMESTAMP
  Partitioned by day of measured_at, clustered by city.

Built for the free BigQuery sandbox, which has no streaming inserts and no
DML (INSERT/DELETE), and a 10 GiB storage limit that is lifetime and not
refunded on deletion. So rows are added with batch load jobs, and only rows
newer than what the table already holds for a sensor are ever loaded.

Auth: Application Default Credentials, created once with
    gcloud auth application-default login
"""

import os
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.cloud import bigquery

from anomaly import geocode, find_pm25_sensor, fetch_pm25_measurements

load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
DATASET    = "ecosentinel"
TABLE      = "pm25_readings"
LOCATION   = "US"
LOAD_DAYS  = 30  # days of OpenAQ history fetched per refresh

SCHEMA = [
    bigquery.SchemaField("city",        "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("sensor_id",   "INT64",     mode="REQUIRED"),
    bigquery.SchemaField("measured_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("pm25",        "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("loaded_at",   "TIMESTAMP", mode="REQUIRED"),
]

_client: bigquery.Client | None = None


def table_id() -> str:
    return f"{PROJECT_ID}.{DATASET}.{TABLE}"


def get_client() -> bigquery.Client:
    """Create the BigQuery client once and reuse it."""
    global _client
    if _client is None:
        if not PROJECT_ID:
            raise RuntimeError("GCP_PROJECT_ID is not set in .env")
        # google-auth looks for this while loading credentials and warns
        # "No project ID could be determined" without it
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
        _client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    return _client


def ensure_table(client: bigquery.Client) -> None:
    """Create the dataset and partitioned, clustered table if missing."""
    dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET}")
    dataset.location = LOCATION
    client.create_dataset(dataset, exists_ok=True)

    table = bigquery.Table(table_id(), schema=SCHEMA)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="measured_at",
    )
    table.clustering_fields = ["city"]
    client.create_table(table, exists_ok=True)


def latest_loaded(client: bigquery.Client, sensor_id: int) -> datetime | None:
    """Newest measured_at already stored for this sensor, or None."""
    job = client.query(
        f"SELECT MAX(measured_at) AS latest FROM `{table_id()}` "
        "WHERE sensor_id = @sensor_id",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sensor_id", "INT64", sensor_id)
            ]
        ),
    )
    return next(iter(job.result())).latest


def new_rows(
    city: str,
    sensor_id: int,
    measurements: list[tuple[str, float]],
    latest: datetime | None,
    loaded_at: datetime,
) -> list[dict]:
    """
    Turn (timestamp, value) pairs into table rows, keeping only readings
    newer than `latest` and dropping duplicate timestamps within the batch.
    """
    rows = []
    seen = set()
    for ts, value in measurements:
        measured_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if latest is not None and measured_at <= latest:
            continue
        if measured_at in seen:
            continue
        seen.add(measured_at)
        rows.append(
            {
                "city": city,
                "sensor_id": sensor_id,
                "measured_at": measured_at.isoformat(),
                "pm25": value,
                "loaded_at": loaded_at.isoformat(),
            }
        )
    return rows


def _store(city: str, sensor_id: int, measurements: list[tuple[str, float]]) -> dict:
    """Blocking BigQuery work: create table if needed, append only new rows."""
    client = get_client()
    ensure_table(client)
    latest = latest_loaded(client, sensor_id)
    rows = new_rows(city, sensor_id, measurements, latest, datetime.now(timezone.utc))

    if rows:
        job = client.load_table_from_json(
            rows,
            table_id(),
            job_config=bigquery.LoadJobConfig(
                schema=SCHEMA,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            ),
        )
        job.result()  # wait for the load job to finish

    return {"loaded": len(rows), "latest_before": latest}


def query_daily(city: str, days: int) -> list[dict]:
    """
    One row per day for a city over the last `days` days: reading count,
    average and peak PM2.5. The partition filter on measured_at means
    BigQuery only scans those days' partitions.
    """
    job = get_client().query(
        f"""
        SELECT
          DATE(measured_at) AS day,
          COUNT(*) AS readings,
          ROUND(AVG(pm25), 1) AS avg_pm25,
          ROUND(MAX(pm25), 1) AS max_pm25
        FROM `{table_id()}`
        WHERE city = @city
          AND measured_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        GROUP BY day
        ORDER BY day
        """,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("city", "STRING", city),
                bigquery.ScalarQueryParameter("days", "INT64", days),
            ]
        ),
    )
    return [dict(row.items()) for row in job.result()]


def summarize_trends(daily: list[dict]) -> dict | None:
    """
    Summarize daily rows: overall average, worst and cleanest day, and the
    latest 7 days of data vs the 7 days before. Windows are anchored on the
    newest day WITH data, not today, because some stations publish late.
    """
    if not daily:
        return None

    total = sum(d["readings"] for d in daily)
    overall = sum(d["avg_pm25"] * d["readings"] for d in daily) / total
    latest_day = daily[-1]["day"]

    # Partial days (a window edge, a sensor outage) can't be named worst or
    # cleanest: e.g. 15 readings of a 4-hour spike is not a "worst day".
    # "Full enough" is relative to the median, so it works for sensors that
    # report hourly (~24/day) and every 15 minutes (~96/day) alike.
    counts = sorted(d["readings"] for d in daily)
    min_readings = counts[len(counts) // 2] / 2
    full_days = [d for d in daily if d["readings"] >= min_readings]

    def window_avg(start_offset: int, end_offset: int) -> float | None:
        rows = [
            d for d in daily
            if start_offset <= (latest_day - d["day"]).days <= end_offset
        ]
        n = sum(d["readings"] for d in rows)
        return sum(d["avg_pm25"] * d["readings"] for d in rows) / n if n else None

    recent, previous = window_avg(0, 6), window_avg(7, 13)
    change_pct = (recent - previous) / previous * 100 if recent and previous else None

    return {
        "days_with_data": len(daily),
        "readings": total,
        "overall_avg": round(overall, 1),
        "worst_day": max(full_days, key=lambda d: d["avg_pm25"]),
        "cleanest_day": min(full_days, key=lambda d: d["avg_pm25"]),
        "peak_reading": max(daily, key=lambda d: d["max_pm25"]),
        "recent_7_avg": round(recent, 1) if recent is not None else None,
        "previous_7_avg": round(previous, 1) if previous is not None else None,
        "change_pct": round(change_pct, 1) if change_pct is not None else None,
        "latest_day": latest_day,
    }


async def get_daily_trends(city: str, days: int = LOAD_DAYS) -> dict:
    """Refresh the city's data in BigQuery, then return its daily rows."""
    load = await load_city_history(city)
    rows = await asyncio.to_thread(query_daily, load["city"], days)
    return {"load": load, "daily": rows}


async def load_city_history(city: str, days: int = LOAD_DAYS) -> dict:
    """
    Fetch a city's recent PM2.5 history from OpenAQ and append any readings
    BigQuery doesn't have yet. Safe to run repeatedly: a second run right
    after the first loads 0 rows.
    """
    lat, lon, display_name = await geocode(city)

    async with httpx.AsyncClient(timeout=30) as http:
        sensor_id = await find_pm25_sensor(http, lat, lon)
        if sensor_id is None:
            return {"city": display_name, "fetched": 0, "loaded": 0,
                    "message": "No PM2.5 sensor found near this city."}
        measurements = await fetch_pm25_measurements(http, sensor_id, days)

    # The BigQuery client is synchronous; run it in a worker thread so it
    # doesn't block the MCP server's event loop.
    stored = await asyncio.to_thread(_store, display_name, sensor_id, measurements)

    return {
        "city": display_name,
        "sensor_id": sensor_id,
        "fetched": len(measurements),
        **stored,
    }
