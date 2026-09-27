# resolveiq/scripts/pull_heat_2023_2024_season.py
"""Download the verified 2023-2024 NYC 311 Heat/Hot Water heating season cohort (235,654 rows).

This script performs deterministic offset paging over the Socrata API, verifies
row counts, unique keys, and date boundaries, and saves the raw snapshot along
with its SHA-256 provenance sidecar into data/raw/.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

from resolveiq.config import REPO_ROOT

ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
EXPECTED_ROWS = 235654
PAGE_SIZE = 20000

WHERE_CLAUSE = (
    "lower(complaint_type) = 'heat/hot water' "
    "and created_date >= '2023-10-01T00:00:00' "
    "and created_date < '2024-06-01T00:00:00'"
)

SELECT_FIELDS = (
    "unique_key, created_date, complaint_type, descriptor, descriptor_2, "
    "location_type, borough, open_data_channel_type, agency, closed_date, status"
)

RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "heat_2023_10_2024_05_season.json"
RAW_META_PATH = REPO_ROOT / "data" / "raw" / "heat_2023_10_2024_05_season.meta.json"


def fetch_page_with_retry(client: httpx.Client, params: dict, max_retries: int = 3) -> list[dict]:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.get(ENDPOINT, params=params, timeout=60.0)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_time = 2 ** attempt
                print(f"  [WARN] HTTP {resp.status_code}. Retrying in {sleep_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(sleep_time)
                continue
            resp.raise_for_status()
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            if attempt == max_retries:
                raise
            sleep_time = 2 ** attempt
            print(f"  [WARN] Network error: {exc}. Retrying in {sleep_time}s (attempt {attempt}/{max_retries})...")
            time.sleep(sleep_time)
    raise RuntimeError(f"Failed to fetch page after {max_retries} retries.")


def main() -> None:
    print(f"=== ResolveIQ 2023-2024 Heating Season Snapshot Pull ===")
    print(f"Source endpoint : {ENDPOINT}")
    print(f"Expected rows   : {EXPECTED_ROWS:,}")
    print(f"Target file     : {RAW_DATA_PATH}")

    # Rerun guard: check if already downloaded with verified sidecar
    if RAW_DATA_PATH.exists() and RAW_META_PATH.exists():
        sidecar = json.loads(RAW_META_PATH.read_text(encoding="utf-8"))
        actual_sha = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
        if actual_sha == sidecar.get("sha256") and sidecar.get("rows") == EXPECTED_ROWS:
            print(f"\n[RERUN GUARD] Verified snapshot already exists on disk ({EXPECTED_ROWS:,} rows).")
            print(f"SHA-256: {actual_sha}")
            print("To re-download, delete or move the existing file first.")
            return

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    page_index = 0
    t_start = time.perf_counter()

    with httpx.Client() as client:
        while True:
            offset = page_index * PAGE_SIZE
            params = {
                "$select": SELECT_FIELDS,
                "$where": WHERE_CLAUSE,
                "$order": "unique_key ASC",
                "$limit": str(PAGE_SIZE),
                "$offset": str(offset),
            }

            t_page = time.perf_counter()
            page_data = fetch_page_with_retry(client, params)
            elapsed_page = time.perf_counter() - t_page

            records.extend(page_data)
            print(
                f"[Page {page_index + 1:2d}] offset={offset:7d} -> fetched {len(page_data):5d} rows "
                f"({len(records):7d}/{EXPECTED_ROWS:,}) in {elapsed_page:.2f}s"
            )

            if len(page_data) < PAGE_SIZE or len(records) >= EXPECTED_ROWS:
                break

            page_index += 1

    total_time = time.perf_counter() - t_start
    print(f"\nCompleted fetch of {len(records):,} records in {total_time:.1f}s.")

    # Integrity Assertions
    print("\nRunning integrity checks...")
    assert len(records) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(records)}"
    print(f"  [PASS] Row count matches expected total exactly: {len(records):,}")

    unique_keys = set(r["unique_key"] for r in records)
    assert len(unique_keys) == EXPECTED_ROWS, f"Duplicate keys detected: {len(unique_keys)} distinct vs {EXPECTED_ROWS} total"
    print(f"  [PASS] Primary key uniqueness verified: {len(unique_keys):,} unique keys")

    min_date = min(r["created_date"] for r in records)
    max_date = max(r["created_date"] for r in records)
    assert min_date >= "2023-10-01T00:00:00", f"Earliest date {min_date} precedes 2023-10-01"
    assert max_date < "2024-06-01T00:00:00", f"Latest date {max_date} exceeds 2024-05-31"
    print(f"  [PASS] Date bounds verified: {min_date} to {max_date}")

    # Status distribution check
    status_counts: dict[str, int] = {}
    null_closed_count = 0
    for r in records:
        st = r.get("status", "Unknown")
        status_counts[st] = status_counts.get(st, 0) + 1
        if r.get("closed_date") is None:
            null_closed_count += 1

    expected_statuses = {"Closed": 235331, "Open": 211, "In Progress": 111, "Unspecified": 1}
    assert status_counts == expected_statuses, f"Status mismatch: expected {expected_statuses}, got {status_counts}"
    assert null_closed_count == 322, f"Expected 322 null closed dates, got {null_closed_count}"
    print(f"  [PASS] Status breakdown matches pre-audit exactly: {status_counts}")
    print(f"  [PASS] Null closed_date count matches pre-audit: {null_closed_count}")

    # Write raw JSON snapshot
    print(f"\nWriting raw snapshot to {RAW_DATA_PATH}...")
    content_bytes = json.dumps(records, indent=2).encode("utf-8")
    RAW_DATA_PATH.write_bytes(content_bytes)
    sha256_digest = hashlib.sha256(content_bytes).hexdigest()
    file_size_mb = len(content_bytes) / (1024 * 1024)
    print(f"  Wrote {file_size_mb:.2f} MB. SHA-256: {sha256_digest}")

    # Write provenance sidecar
    sidecar_data = {
        "url": ENDPOINT,
        "where": WHERE_CLAUSE,
        "rows": len(records),
        "sha256": sha256_digest,
        "size_bytes": len(content_bytes),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "select": SELECT_FIELDS,
        "order": "unique_key ASC",
        "paging": f"$limit {PAGE_SIZE}, $offset until {EXPECTED_ROWS}",
        "population": "ALL HEAT/HOT WATER tickets created in the 2023-2024 heating season (2023-10-01 to 2024-05-31)",
        "status_distribution": status_counts,
        "null_closed_date_count": null_closed_count,
        "open_tickets_included": True,
        "earliest_created_date": min_date,
        "latest_created_date": max_date,
    }

    RAW_META_PATH.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")
    print(f"  Wrote sidecar to {RAW_META_PATH}")
    print(f"\nSnapshot pull and verification successfully finished.")


if __name__ == "__main__":
    main()
