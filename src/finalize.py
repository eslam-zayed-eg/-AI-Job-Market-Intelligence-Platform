"""
finalize.py - Refresh EVERYTHING before publishing, with one command.
AI Job Market Intelligence Platform
-------------------------------------------------------------------

    python src/finalize.py

What it does, in order:
  1. SQL Server   rebuilds the 4 tables (companies, skills, jobs_normalized, job_skills)
  2. Model        retrains the job-family classifier (5-fold CV), saves model + metrics
  3. Statistics   skill demand and salary-by-skill, collection overview
  4. Saves        data/processed/project_stats.json  (every number in one place)
  5. README       rewrites the "Results snapshot" block of README.md
                  (the text between the STATS:START / STATS:END markers)

Because the README numbers are generated from the same data as the SQL tables
and the model, they can never disagree with each other.

Options:
    --country gb     country used for SQL, model and statistics (default: gb)
    --skip-sql       do not touch SQL Server
    --skip-model     do not retrain (reuses data/processed/model_metrics.json)

Local-only dependencies:
    python -m pip install scikit-learn sqlalchemy pyodbc joblib
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_all  # noqa: E402
from collect import BASE_DIR, COUNTRIES, KEYWORDS, MASTER_FILE, PROCESSED_DIR, SKILLS  # noqa: E402
from skill_stats import MIN_SALARY_SAMPLES, compute as compute_skill_stats  # noqa: E402

README_FILE = BASE_DIR / "README.md"
STATS_FILE = PROCESSED_DIR / "project_stats.json"
METRICS_FILE = update_all.METRICS_FILE

START_MARK = "<!-- STATS:START -->"
END_MARK = "<!-- STATS:END -->"

COUNTRY_NAMES = {"gb": "United Kingdom", "us": "United States", "ca": "Canada", "au": "Australia"}
CURRENCY_SYMBOL = {"GBP": "\u00a3", "USD": "$", "CAD": "CA$", "AUD": "A$"}


# ============================================================
# Statistics
# ============================================================

def parse_timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    """
    Parse ISO timestamps that may mix formats (with and without fractional
    seconds). Plain pandas learns the format from the first row and turns every
    row that does not match into NaT, which silently shrinks the date range.
    """
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    try:
        return pd.to_datetime(frame[column], errors="coerce", utc=True, format="ISO8601")
    except (TypeError, ValueError):  # very old pandas without format="ISO8601"
        return pd.to_datetime(frame[column], errors="coerce", utc=True)


def collection_overview() -> Dict[str, Any]:
    """Numbers about the whole master dataset (all countries)."""
    raw = pd.read_csv(MASTER_FILE, dtype={"job_id": str}, encoding="utf-8-sig")
    if "country" not in raw.columns:
        raw["country"] = "gb"
    raw["country"] = raw["country"].fillna("gb")

    counts = raw["country"].value_counts()
    per_country = {c: int(counts.get(c, 0)) for c in COUNTRIES}
    for extra in counts.index:
        per_country.setdefault(extra, int(counts[extra]))

    first = parse_timestamps(raw, "first_seen")
    last = parse_timestamps(raw, "last_seen")

    def day(value) -> Optional[str]:
        return None if pd.isna(value) else str(value.date())

    return {
        "postings_total": int(len(raw)),
        "postings_per_country": per_country,
        "first_collected": day(first.min()),
        "last_seen": day(last.max()),
        "collection_days": int(first.dt.date.nunique()) if first.notna().any() else 0,
    }


def load_metrics() -> Optional[Dict[str, Any]]:
    if METRICS_FILE.exists():
        try:
            return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


# ============================================================
# README block
# ============================================================

def render_block(stats: Dict[str, Any]) -> str:
    n = lambda v: f"{v:,}"  # noqa: E731
    country = stats["analysis_country"]
    cname = COUNTRY_NAMES.get(country, country.upper())
    symbol = CURRENCY_SYMBOL.get(stats["currency"], "")

    per_country = " \u00b7 ".join(
        f"{COUNTRY_NAMES.get(c, c.upper())} {n(v)}" for c, v in stats["postings_per_country"].items()
    )
    window = "n/a"
    if stats["first_collected"] and stats["last_seen"]:
        window = f"{stats['first_collected']} to {stats['last_seen']} ({stats['collection_days']} collection day(s))"

    sql = stats["sql_tables"]
    sql_text = (
        f"companies {n(sql['companies'])} \u00b7 jobs_normalized {n(sql['jobs_normalized'])} \u00b7 "
        f"skills {n(sql['skills'])} \u00b7 job_skills {n(sql['job_skills'])}"
    )
    if stats["sql_status"] != "updated":
        sql_text += f"  _(SQL status: {stats['sql_status']})_"

    lines = [
        START_MARK,
        f"_Generated automatically by `python src/finalize.py` on {stats['generated_at'][:10]}. "
        "Do not edit this block by hand._",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Postings collected (all countries) | **{n(stats['postings_total'])}** |",
        f"| Postings per country | {per_country} |",
        f"| Collection window | {window} |",
        f"| Job families \u00b7 countries \u00b7 tracked skills | {stats['job_families']} \u00b7 {stats['countries']} \u00b7 {stats['skills_tracked']} |",
        f"| {cname}: postings in the SQL database | {n(stats['analysis_postings'])} from {n(stats['employers'])} employers |",
        f"| {cname}: unique postings used for analysis | {n(stats['analysis_unique_postings'])} (repeated adverts counted once) |",
        f"| SQL tables (JobMarketDB) | {sql_text} |",
    ]

    m = stats.get("model")
    if m:
        lines.append(
            f"| Classifier ({m['features']}) | **{m['cv_accuracy'] * 100:.1f}% \u00b1 {m['cv_accuracy_std'] * 100:.1f}%** "
            f"5-fold CV accuracy \u00b7 macro-F1 {m['cv_macro_f1']:.3f} \u00b7 {m['n_classes']} classes \u00b7 "
            f"{n(m['n_postings'])} postings |"
        )
    else:
        lines.append("| Classifier | not trained yet |")

    lines += [
        "",
        f"**Most requested skills ({cname}, unique postings).** "
        f"{n(stats['postings_with_skill'])} of {n(stats['analysis_unique_postings'])} postings "
        f"({stats['postings_with_skill_pct']:.0f}%) mention at least one tracked skill.",
        "",
        "| Skill | Postings | Share of postings |",
        "|---|---:|---:|",
    ]
    for row in stats["top_skills"]:
        lines.append(f"| {row['skill']} | {n(row['postings'])} | {row['share_pct']}% |")

    lines += [
        "",
        f"**Salary by skill ({cname}, advertised minimum, {stats['currency']}).** "
        f"Only skills with at least {MIN_SALARY_SAMPLES} salaried postings are shown.",
        "",
        "| Skill | Average | Median | Salaried postings |",
        "|---|---:|---:|---:|",
    ]
    for row in stats["salary_by_skill"]:
        lines.append(
            f"| {row['skill']} | {symbol}{n(row['avg'])} | {symbol}{n(row['median'])} | {n(row['postings'])} |"
        )
    if not stats["salary_by_skill"]:
        lines.append("| _not enough salaried postings yet_ | | | |")

    lines.append(END_MARK)
    return "\n".join(lines)


def update_readme(block: str) -> str:
    """Replace the block between the markers. Returns a status string."""
    if not README_FILE.exists():
        return "README.md not found - skipped"
    text = README_FILE.read_text(encoding="utf-8")
    if START_MARK not in text or END_MARK not in text:
        return f"markers {START_MARK} / {END_MARK} not found in README.md - skipped"
    pattern = re.compile(re.escape(START_MARK) + r".*?" + re.escape(END_MARK), re.DOTALL)
    README_FILE.write_text(pattern.sub(lambda _m: block, text, count=1), encoding="utf-8")
    return "README.md results block updated"


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh SQL, model, statistics and README.")
    parser.add_argument("--country", default="gb")
    parser.add_argument("--skip-sql", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("FINALIZE: refreshing everything")
    print("=" * 60)

    overview = collection_overview()
    df = update_all.load_jobs(args.country)
    exit_code = 0

    # ---- 1. SQL ----------------------------------------------------
    names = ("companies", "skills", "jobs_normalized", "job_skills")
    local_tables = update_all.build_tables(df)
    sql_counts = {name: len(table) for name, table in zip(names, local_tables)}
    sql_status = "skipped"
    if not args.skip_sql:
        try:
            sql_counts = update_all.update_sql(df)
            sql_status = "updated"
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            sql_status = "FAILED"
            exit_code = 1
            print(f"\n!! SQL update failed: {exc}\n   Continuing with the model and statistics.")

    # ---- 2. Model --------------------------------------------------
    if not args.skip_model:
        update_all.train_model(df, args.country)
    metrics = load_metrics()

    # ---- 3. Statistics ---------------------------------------------
    unique = df.drop_duplicates(subset=["company", "title", "description"])
    demand, pay, total, with_skill = compute_skill_stats(unique)

    currency = "?"
    if "currency" in df.columns and df["currency"].notna().any():
        currency = str(df["currency"].dropna().iloc[0])

    stats: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_country": args.country,
        "currency": currency,
        **overview,
        "job_families": len(KEYWORDS),
        "countries": len(COUNTRIES),
        "skills_tracked": len(SKILLS),
        "analysis_postings": int(len(df)),
        "analysis_unique_postings": int(total),
        "employers": int(df["company"].nunique()),
        "postings_with_skill": int(with_skill),
        "postings_with_skill_pct": round(with_skill / max(total, 1) * 100, 1),
        "sql_tables": sql_counts,
        "sql_status": sql_status,
        "model": metrics,
        "top_skills": [
            {"skill": skill, "postings": int(row["postings"]), "share_pct": float(row["share_of_postings_%"])}
            for skill, row in demand.head(10).iterrows()
        ],
        "salary_by_skill": [
            {
                "skill": skill,
                "avg": int(row["avg_salary_min"]),
                "median": int(row["median_salary_min"]),
                "postings": int(row["postings_with_salary"]),
            }
            for skill, row in pay.iterrows()
        ],
    }

    # ---- 4 + 5. Save and update README ------------------------------
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    readme_status = update_readme(render_block(stats))

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  SQL Server : {sql_status}")
    print(f"  Model      : {'retrained' if not args.skip_model else 'kept as is'}")
    print(f"  Stats file : {STATS_FILE}")
    print(f"  README     : {readme_status}")
    if exit_code:
        print("\n  Something failed (see above). Fix it and run again before publishing.")
    else:
        print("\n  Next:  git pull  ->  git add .  ->  git commit  ->  git push")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
