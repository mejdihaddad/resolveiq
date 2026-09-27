# resolveiq\scripts\check_nyc311.py
"""Ask the dataset one cheap question: how many rows match?"""
import time

import httpx

from resolveiq.config import get_settings

settings = get_settings()
url = f"{settings.nyc311_base_url}/{settings.nyc311_dataset_id}.json"

params = {
    "$select": "count(*) as n",
    "$where": (
        "lower(complaint_type) = 'heat/hot water' "
        "and created_date >= '2024-01-01T00:00:00' "
        "and created_date < '2024-02-01T00:00:00'"
    ),
}

started = time.perf_counter()
response = httpx.get(url, params=params, timeout=30.0)
elapsed = time.perf_counter() - started

response.raise_for_status()
print(f"status {response.status_code} in {elapsed:.2f}s")
print("rows :", response.json()[0]["n"])