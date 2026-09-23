"""
extract.py

AI Job Market Intelligence Platform
------------------------------------

Purpose:
    Collect job-market data from Adzuna API.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw" / "adzuna"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(BASE_DIR / ".env")

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")


if not APP_ID or not APP_KEY:
    raise RuntimeError(
        "\nAdzuna credentials are missing.\n"
        "Please create a .env file containing:\n\n"
        "ADZUNA_APP_ID=your_app_id\n"
        "ADZUNA_APP_KEY=your_app_key\n"
    )


# ============================================================
# 3. CONFIGURATION
# ============================================================

@dataclass
class Config:
    country: str = "gb"
    results_per_page: int = 20
    max_pages_per_keyword: int = 10
    max_requests_per_minute: int = 20
    request_delay_seconds: float = 3.0
    timeout_seconds: int = 30
    max_retries: int = 4

    keywords: tuple = (
        "data analyst",
        "data scientist",
        "data engineer",
        "machine learning engineer",
        "business analyst",
        "business intelligence analyst",
        "software engineer",
        "cloud engineer",
        "cybersecurity engineer",
        "AI engineer",
    )


CONFIG = Config()


# ============================================================
# 4. LOGGING
# ============================================================

LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "extract.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# 5. API CONFIGURATION
# ============================================================

BASE_API_URL = "https://api.adzuna.com/v1/api"

SESSION = requests.Session()

SESSION.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": "AI-Job-Market-Intelligence-Platform/1.0",
    }
)


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collection_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = " ".join(text.split())
    return text


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# 7. RATE LIMITER
# ============================================================

class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 20):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times: List[float] = []

    def wait_if_needed(self) -> None:
        now = time.time()

        self.request_times = [
            t for t in self.request_times if now - t < 60
        ]

        if len(self.request_times) >= self.max_requests_per_minute:
            oldest_request = self.request_times[0]
            wait_time = 60 - (now - oldest_request) + 1

            logger.info("Rate limit protection: sleeping %.1f seconds", wait_time)

            time.sleep(max(wait_time, 0))

            now = time.time()
            self.request_times = [
                t for t in self.request_times if now - t < 60
            ]

        self.request_times.append(time.time())


RATE_LIMITER = RateLimiter(max_requests_per_minute=CONFIG.max_requests_per_minute)


# ============================================================
# 8. API REQUEST
# ============================================================

def request_with_retry(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for attempt in range(1, CONFIG.max_retries + 1):

        RATE_LIMITER.wait_if_needed()

        try:
            logger.info("API request attempt %d/%d", attempt, CONFIG.max_retries)

            response = SESSION.get(url, params=params, timeout=CONFIG.timeout_seconds)

            logger.info("HTTP status: %s", response.status_code)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_seconds = int(retry_after)
                    except ValueError:
                        wait_seconds = 30
                else:
                    wait_seconds = min(60, 2 ** attempt * 5)

                logger.warning("Rate limited by API. Waiting %d seconds.", wait_seconds)
                time.sleep(wait_seconds)
                continue

            if response.status_code in {500, 502, 503, 504}:
                wait_seconds = min(60, 2 ** attempt)
                logger.warning(
                    "Temporary API error %s. Retrying in %d seconds.",
                    response.status_code, wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            logger.error("API request failed: HTTP %s", response.status_code)
            logger.error("Response: %s", response.text[:1000])
            return None

        except requests.exceptions.Timeout:
            wait_seconds = min(60, 2 ** attempt)
            logger.warning("Request timeout. Retrying in %d seconds.", wait_seconds)
            time.sleep(wait_seconds)

        except requests.exceptions.ConnectionError:
            wait_seconds = min(60, 2 ** attempt)
            logger.warning("Connection error. Retrying in %d seconds.", wait_seconds)
            time.sleep(wait_seconds)

        except requests.exceptions.RequestException as exc:
            logger.error("Unexpected request error: %s", exc)
            return None

        except ValueError as exc:
            logger.error("Invalid JSON response: %s", exc)
            return None

    logger.error("Request failed after %d attempts.", CONFIG.max_retries)
    return None


# ============================================================
# 9. BUILD SEARCH URL
# ============================================================

def build_search_url(country: str, page: int) -> str:
    return f"{BASE_API_URL}/jobs/{country}/search/{page}"


# ============================================================
# 10. FETCH ONE PAGE
# ============================================================

def fetch_page(keyword: str, page: int) -> Optional[Dict[str, Any]]:
    url = build_search_url(CONFIG.country, page)

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": CONFIG.results_per_page,
        "what": keyword,
        "content-type": "application/json",
    }

    logger.info("Fetching keyword='%s' page=%d", keyword, page)

    return request_with_retry(url=url, params=params)


# ============================================================
# 11. NORMALIZE JOB
# ============================================================

def normalize_job(job: Dict[str, Any], keyword: str) -> Dict[str, Any]:
    location = job.get("location") or {}
    company = job.get("company") or {}
    category = job.get("category") or {}

    job_id = clean_text(job.get("id"))

    return {
        "job_id": job_id,
        "source": "adzuna",
        "title": clean_text(job.get("title")),
        "description": clean_text(job.get("description")),
        "company": clean_text(company.get("display_name")),
        "location": clean_text(location.get("display_name")),
        "location_area": json.dumps(location.get("area"), ensure_ascii=False) if location.get("area") else None,
        "latitude": safe_float(job.get("latitude")),
        "longitude": safe_float(job.get("longitude")),
        "salary_min": safe_float(job.get("salary_min")),
        "salary_max": safe_float(job.get("salary_max")),
        "salary_is_predicted": safe_int(job.get("salary_is_predicted")),
        "contract_type": clean_text(job.get("contract_type")),
        "contract_time": clean_text(job.get("contract_time")),
        "category": clean_text(category.get("label")),
        "category_tag": clean_text(category.get("tag")),
        "created": clean_text(job.get("created")),
        "redirect_url": clean_text(job.get("redirect_url")),
        "search_keyword": keyword,
        "country": CONFIG.country,
        "collected_at": utc_now_iso(),
    }


# ============================================================
# 12. FETCH ALL PAGES FOR ONE KEYWORD
# ============================================================

def collect_keyword(keyword: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    logger.info("=" * 70)
    logger.info("START KEYWORD: %s", keyword)
    logger.info("=" * 70)

    for page in range(1, CONFIG.max_pages_per_keyword + 1):

        data = fetch_page(keyword=keyword, page=page)

        if data is None:
            logger.warning("Stopping keyword '%s' because the API request failed.", keyword)
            break

        results = data.get("results", [])

        if not results:
            logger.info("No more jobs found for keyword '%s' at page %d.", keyword, page)
            break

        logger.info("Received %d jobs.", len(results))

        for job in results:
            normalized = normalize_job(job=job, keyword=keyword)
            jobs.append(normalized)

        time.sleep(CONFIG.request_delay_seconds)

        if len(results) < CONFIG.results_per_page:
            logger.info("Page %d returned fewer than %d jobs. End of results.", page, CONFIG.results_per_page)
            break

    logger.info("Finished keyword '%s': %d jobs", keyword, len(jobs))

    return jobs


# ============================================================
# 13. COLLECT ALL KEYWORDS
# ============================================================

def collect_all_jobs() -> List[Dict[str, Any]]:
    all_jobs: List[Dict[str, Any]] = []

    logger.info("Starting collection for %d keywords.", len(CONFIG.keywords))

    for keyword in CONFIG.keywords:
        keyword_jobs = collect_keyword(keyword=keyword)
        all_jobs.extend(keyword_jobs)

    logger.info("Total raw jobs collected: %d", len(all_jobs))

    return all_jobs


# ============================================================
# 14. DEDUPLICATE
# ============================================================

def deduplicate_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not jobs:
        return []

    df = pd.DataFrame(jobs)

    logger.info("Rows before deduplication: %d", len(df))

    df["unique_job_key"] = (
        df["source"].fillna("").astype(str) + "_" + df["job_id"].fillna("").astype(str)
    )

    missing_id = df["job_id"].isna() | (df["job_id"].astype(str).str.strip() == "")

    fallback_key = (
        df["title"].fillna("").astype(str) + "|" +
        df["company"].fillna("").astype(str) + "|" +
        df["location"].fillna("").astype(str)
    )

    df.loc[missing_id, "unique_job_key"] = fallback_key[missing_id]

    df = df.drop_duplicates(subset=["unique_job_key"], keep="first")
    df = df.drop(columns=["unique_job_key"], errors="ignore")

    logger.info("Rows after deduplication: %d", len(df))

    return df.to_dict(orient="records")


# ============================================================
# 15. SAVE RAW JSON
# ============================================================

def save_raw_json(jobs: List[Dict[str, Any]], timestamp: str) -> Path:
    output_file = RAW_DIR / f"jobs_{timestamp}.json"

    payload = {
        "metadata": {
            "source": "adzuna",
            "country": CONFIG.country,
            "keywords": list(CONFIG.keywords),
            "collected_at": utc_now_iso(),
            "records": len(jobs),
        },
        "jobs": jobs,
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    logger.info("Raw JSON saved: %s", output_file)

    return output_file


# ============================================================
# 16. SAVE CSV
# ============================================================

def save_csv(jobs: List[Dict[str, Any]], timestamp: str) -> Path:
    output_file = PROCESSED_DIR / f"jobs_{timestamp}.csv"

    df = pd.DataFrame(jobs)

    columns = [
        "job_id", "source", "title", "company", "location", "location_area",
        "latitude", "longitude", "description", "salary_min", "salary_max",
        "salary_is_predicted", "contract_type", "contract_time", "category",
        "category_tag", "created", "redirect_url", "search_keyword",
        "country", "collected_at",
    ]

    existing_columns = [c for c in columns if c in df.columns]

    df = df[existing_columns]

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    logger.info("CSV saved: %s", output_file)

    return output_file


# ============================================================
# 17. SAVE MASTER CSV
# ============================================================

def update_master_dataset(jobs: List[Dict[str, Any]]) -> Path:
    master_file = PROCESSED_DIR / "jobs_master.csv"

    new_df = pd.DataFrame(jobs)

    if master_file.exists():
        logger.info("Existing master dataset found.")
        old_df = pd.read_csv(master_file)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        logger.info("No master dataset found. Creating a new one.")
        combined_df = new_df

    if "job_id" in combined_df.columns:
        combined_df["unique_job_key"] = (
            combined_df["source"].fillna("").astype(str) + "_" +
            combined_df["job_id"].fillna("").astype(str)
        )

        missing_id = (
            combined_df["job_id"].isna() |
            (combined_df["job_id"].astype(str).str.strip() == "")
        )

        fallback_key = (
            combined_df["title"].fillna("").astype(str) + "|" +
            combined_df["company"].fillna("").astype(str) + "|" +
            combined_df["location"].fillna("").astype(str)
        )

        combined_df.loc[missing_id, "unique_job_key"] = fallback_key[missing_id]

        combined_df = combined_df.drop_duplicates(subset=["unique_job_key"], keep="first")
        combined_df = combined_df.drop(columns=["unique_job_key"], errors="ignore")

    combined_df.to_csv(master_file, index=False, encoding="utf-8-sig")

    logger.info("Master dataset updated: %s", master_file)
    logger.info("Master dataset rows: %d", len(combined_df))

    return master_file


# ============================================================
# 18. SAVE COLLECTION METADATA
# ============================================================

def save_collection_metadata(timestamp: str, jobs_count: int, unique_count: int) -> Path:
    metadata_file = RAW_DIR / f"collection_{timestamp}.json"

    metadata = {
        "collection_id": timestamp,
        "source": "adzuna",
        "country": CONFIG.country,
        "keywords": list(CONFIG.keywords),
        "results_per_page": CONFIG.results_per_page,
        "max_pages_per_keyword": CONFIG.max_pages_per_keyword,
        "raw_records": jobs_count,
        "unique_records": unique_count,
        "collected_at": utc_now_iso(),
    }

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    logger.info("Collection metadata saved: %s", metadata_file)

    return metadata_file


# ============================================================
# 19. PRINT SUMMARY
# ============================================================

def print_summary(jobs: List[Dict[str, Any]]) -> None:
    if not jobs:
        logger.warning("No jobs were collected.")
        return

    df = pd.DataFrame(jobs)

    print()
    print("=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)

    print(f"Total unique jobs: {len(df):,}")

    if "company" in df.columns:
        print(f"Unique companies: {df['company'].nunique():,}")

    if "location" in df.columns:
        print(f"Unique locations: {df['location'].nunique():,}")

    if "category" in df.columns:
        print(f"Unique categories: {df['category'].nunique():,}")

    if "salary_min" in df.columns:
        print(f"Jobs with salary_min: {df['salary_min'].notna().sum():,}")

    if "salary_max" in df.columns:
        print(f"Jobs with salary_max: {df['salary_max'].notna().sum():,}")

    if "contract_time" in df.columns:
        print()
        print("Contract time:")
        print(df["contract_time"].value_counts(dropna=False).head(10).to_string())

    if "category" in df.columns:
        print()
        print("Top categories:")
        print(df["category"].value_counts(dropna=False).head(10).to_string())

    print("=" * 70)
    print()


# ============================================================
# 20. MAIN
# ============================================================

def main() -> None:
    start_time = time.time()
    timestamp = collection_timestamp()

    logger.info("=" * 70)
    logger.info("AI JOB MARKET INTELLIGENCE PLATFORM")
    logger.info("ADZUNA EXTRACTION PIPELINE")
    logger.info("=" * 70)

    logger.info("Country: %s", CONFIG.country)
    logger.info("Keywords: %d", len(CONFIG.keywords))
    logger.info("Results/page: %d", CONFIG.results_per_page)
    logger.info("Max pages/keyword: %d", CONFIG.max_pages_per_keyword)

    raw_jobs = collect_all_jobs()

    if not raw_jobs:
        logger.error("Extraction returned zero jobs.")
        sys.exit(1)

    unique_jobs = deduplicate_jobs(raw_jobs)

    save_raw_json(jobs=raw_jobs, timestamp=timestamp)
    save_csv(jobs=unique_jobs, timestamp=timestamp)
    update_master_dataset(jobs=unique_jobs)
    save_collection_metadata(timestamp=timestamp, jobs_count=len(raw_jobs), unique_count=len(unique_jobs))

    print_summary(unique_jobs)

    elapsed = time.time() - start_time

    logger.info("Extraction completed in %.2f seconds.", elapsed)
    logger.info("DONE.")


# ============================================================
# 21. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()