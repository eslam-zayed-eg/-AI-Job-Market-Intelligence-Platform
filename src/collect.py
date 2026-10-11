"""
collect.py - Incremental, quota-aware job collector
AI Job Market Intelligence Platform
---------------------------------------------------

Every run:
  1. Resumes from where the previous run stopped (data/collection_state.json)
     and walks through (country, keyword) combinations until the request
     budget for the run is used up.
  2. Normalizes postings, extracts skills, tags country / currency / timestamps.
  3. Merges into data/processed/jobs_master.csv without duplicates:
     new jobs are added, jobs seen again only get a fresh `last_seen`.
  4. Refreshes data/processed/jobs_for_dashboard.csv.

Safe to run daily, locally or from GitHub Actions.

Environment variables:
  ADZUNA_APP_ID, ADZUNA_APP_KEY   required (from .env locally, Secrets in CI)
  MAX_REQUESTS_PER_RUN            optional, default 80
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = DATA_DIR / "logs"

STATE_FILE = DATA_DIR / "collection_state.json"
MASTER_FILE = PROCESSED_DIR / "jobs_master.csv"
DASHBOARD_FILE = PROCESSED_DIR / "jobs_for_dashboard.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")
APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

# ============================================================
# Configuration  (edit freely)
# ============================================================

# country code -> salary currency. Salaries are NOT comparable across
# currencies, so analyze/filter by `country` or `currency`.
COUNTRIES: Dict[str, str] = {
    "gb": "GBP",
    "us": "USD",
    "ca": "CAD",
    "au": "AUD",
}

KEYWORDS: List[str] = [
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
    "devops engineer",
    "full stack developer",
    "backend developer",
    "frontend developer",
    "data architect",
    "analytics engineer",
]

RESULTS_PER_PAGE = 50          # Adzuna allows up to 50
PAGES_PER_COMBO = 3            # pages fetched per (country, keyword) per visit
SLEEP_BETWEEN_REQUESTS = 3.0   # seconds (limit is 25 requests/minute)
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
MAX_CONSECUTIVE_FAILURES = 5   # stop the run early if the API keeps failing

# Default Adzuna limits are 250/day, 1000/week, 2500/month. Staying near
# 80/day keeps a daily schedule safely under the monthly cap.
MAX_REQUESTS_PER_RUN = int(os.getenv("MAX_REQUESTS_PER_RUN", "80"))

API_BASE = "https://api.adzuna.com/v1/api"

SKILLS: List[str] = [
    "Python", "SQL", "Java", "Scala",
    "Excel", "Power BI", "Tableau", "Looker",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP",
    "Spark", "Hadoop", "Airflow", "ETL",
    "Git", "Linux", "REST API",
    "PostgreSQL", "MySQL", "MongoDB", "Snowflake",
    "Django", "Flask", "React", "Node.js",
    "Agile", "Scrum",
]

COLUMNS: List[str] = [
    "uid", "job_id", "title", "company", "location",
    "salary_min", "salary_max", "currency", "contract_type", "category",
    "description", "search_keyword", "created", "redirect_url",
    "country", "skills", "first_seen", "last_seen",
]

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "collect.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("collect")

# ============================================================
# Skill extraction
# ============================================================

_SKILL_PATTERNS = [
    (skill, re.compile(r"\b" + re.escape(skill.lower()) + r"\b"))
    for skill in SKILLS
]


def extract_skills(text: Any) -> str:
    """Return matched skills as a comma-separated string ('' if none)."""
    if not isinstance(text, str) or not text:
        return ""
    lowered = text.lower()
    return ", ".join(s for s, pattern in _SKILL_PATTERNS if pattern.search(lowered))


# ============================================================
# State (resume position between runs)
# ============================================================

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("State file unreadable - starting from the beginning.")
    return {"cursor": 0}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ============================================================
# API access
# ============================================================

class StopRun(Exception):
    """Raised to end the run early (budget reached, quota hit, bad credentials)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def fetch_page(
    session: requests.Session,
    counter: Dict[str, int],
    country: str,
    keyword: str,
    page: int,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch one page. Returns a list of jobs, or None if it failed."""
    url = f"{API_BASE}/jobs/{country}/search/{page}"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "what": keyword,
        "content-type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        if counter["used"] >= MAX_REQUESTS_PER_RUN:
            raise StopRun("budget", f"Request budget reached ({MAX_REQUESTS_PER_RUN}).")
        counter["used"] += 1

        try:
            response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        except (requests.Timeout, requests.ConnectionError) as exc:
            logger.warning("Network problem (%s). Retry %d/%d", exc.__class__.__name__, attempt, MAX_RETRIES)
            time.sleep(5 * attempt)
            continue
        except requests.RequestException as exc:
            logger.error("Unexpected request error: %s", exc)
            return None

        status = response.status_code

        if status == 200:
            try:
                return response.json().get("results", [])
            except ValueError:
                logger.error("Invalid JSON from API.")
                return None

        if status in (401, 403):
            raise StopRun("auth", f"Adzuna rejected the credentials (HTTP {status}).")

        if status == 429:
            raise StopRun("quota", "Adzuna rate/usage limit hit (HTTP 429). Stopping for today.")

        if status in (500, 502, 503, 504):
            logger.warning("Temporary API error %s. Retry %d/%d", status, attempt, MAX_RETRIES)
            time.sleep(5 * attempt)
            continue

        logger.error("API error %s for %s/%s p%d: %s", status, country, keyword, page, response.text[:200])
        return None

    logger.error("Giving up on %s / %s page %d after %d attempts.", country, keyword, page, MAX_RETRIES)
    return None


# ============================================================
# Normalization
# ============================================================

def normalize_job(job: Dict[str, Any], country: str, keyword: str, now: str) -> Optional[Dict[str, Any]]:
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        return None

    location = job.get("location") or {}
    company = job.get("company") or {}
    category = job.get("category") or {}

    description = " ".join(str(job.get("description") or "").split()) or None

    return {
        "uid": f"{country}_{job_id}",
        "job_id": job_id,
        "title": job.get("title"),
        "company": company.get("display_name"),
        "location": location.get("display_name"),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "currency": COUNTRIES[country],
        "contract_type": job.get("contract_type"),
        "category": category.get("label"),
        "description": description,
        "search_keyword": keyword,
        "created": job.get("created"),
        "redirect_url": job.get("redirect_url"),
        "country": country,
        "skills": extract_skills(description),
        "first_seen": now,
        "last_seen": now,
    }


# ============================================================
# Collection loop
# ============================================================

def collect(state: Dict[str, Any], now: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Walk (country, keyword) combos from the saved cursor until the budget ends."""
    combos = [(c, k) for c in COUNTRIES for k in KEYWORDS]
    total = len(combos)
    index = state.get("cursor", 0) % total

    rows: List[Dict[str, Any]] = []
    counter = {"used": 0}
    stop_reason: Optional[str] = None
    consecutive_failures = 0
    streak_start = index

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "AI-Job-Market-Intelligence-Platform/2.0",
    })

    try:
        for _ in range(total):
            country, keyword = combos[index]
            logger.info("[%d/%d] %s | %s", index + 1, total, country, keyword)

            for page in range(1, PAGES_PER_COMBO + 1):
                results = fetch_page(session, counter, country, keyword, page)

                if results is None:  # request failed (not just "no results")
                    if consecutive_failures == 0:
                        streak_start = index
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        index = streak_start  # retry the failed combos next run
                        raise StopRun(
                            "api_down",
                            f"{MAX_CONSECUTIVE_FAILURES} requests failed in a row - API looks unavailable.",
                        )
                    break

                consecutive_failures = 0
                if not results:
                    break

                for job in results:
                    row = normalize_job(job, country, keyword, now)
                    if row:
                        rows.append(row)
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                if len(results) < RESULTS_PER_PAGE:
                    break

            index = (index + 1) % total
    except StopRun as stop:
        stop_reason = stop.reason
        logger.info("Run stopped: %s", stop)

    # If we stopped mid-combo, `index` still points at it, so the next run retries it.
    state["cursor"] = index
    logger.info("Requests used: %d / %d | postings fetched: %d", counter["used"], MAX_REQUESTS_PER_RUN, len(rows))
    return rows, stop_reason


# ============================================================
# Merge into master dataset
# ============================================================

def _upgrade_legacy(old: pd.DataFrame) -> pd.DataFrame:
    """Bring a master file from an older version up to the current schema."""
    file_time = datetime.fromtimestamp(MASTER_FILE.stat().st_mtime, tz=timezone.utc).isoformat()

    old["job_id"] = old["job_id"].astype(str)

    if "country" not in old.columns:
        old["country"] = "gb"
    old["country"] = old["country"].fillna("gb")

    if "currency" not in old.columns:
        old["currency"] = old["country"].map(COUNTRIES)

    if "redirect_url" not in old.columns:
        old["redirect_url"] = None

    for column in ("first_seen", "last_seen"):
        if column not in old.columns:
            old[column] = file_time

    if "skills" not in old.columns:
        old["skills"] = old["description"].apply(extract_skills)
    old["skills"] = old["skills"].fillna("")

    old["uid"] = old["country"] + "_" + old["job_id"]
    return old


def merge_into_master(new_rows: List[Dict[str, Any]], now: str) -> Tuple[int, int]:
    """Merge new postings. Returns (added, total_after)."""
    new_df = pd.DataFrame(new_rows).drop_duplicates(subset="uid")

    if MASTER_FILE.exists():
        old = pd.read_csv(MASTER_FILE, dtype={"job_id": str}, encoding="utf-8-sig")
        old = _upgrade_legacy(old)
    else:
        old = pd.DataFrame(columns=COLUMNS)

    seen_again = old["uid"].isin(new_df["uid"])
    old.loc[seen_again, "last_seen"] = now

    added = new_df[~new_df["uid"].isin(old["uid"])]

    combined = pd.concat([old, added], ignore_index=True).reindex(columns=COLUMNS)

    combined.to_csv(MASTER_FILE, index=False, encoding="utf-8-sig")
    combined.to_csv(DASHBOARD_FILE, index=False, encoding="utf-8-sig")

    return len(added), len(combined)


# ============================================================
# Main
# ============================================================

def main() -> int:
    if not APP_ID or not APP_KEY:
        logger.error("ADZUNA_APP_ID / ADZUNA_APP_KEY are missing (.env locally, Secrets in CI).")
        return 1

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = load_state()

    logger.info("=" * 60)
    logger.info("Collection run started | budget=%d requests | cursor=%s", MAX_REQUESTS_PER_RUN, state.get("cursor", 0))

    rows, stop_reason = collect(state, now)
    save_state(state)

    if rows:
        added, total = merge_into_master(rows, now)
        logger.info("New jobs added: %d | master dataset total: %d", added, total)
    else:
        logger.info("No postings fetched this run.")

    if stop_reason == "auth":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
