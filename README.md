# AI Job Market Intelligence Platform

An end-to-end data pipeline that collects, cleans, analyzes, and models real UK job market data to uncover in-demand skills, salary patterns, and job trends across 10 tech and data-related career families.

## Overview

This project answers a simple question — **what does the current job market actually want?** — by building a complete data pipeline on top of real job postings, rather than relying on assumptions or outdated reports.

It covers the full data lifecycle:

```
Adzuna API → Data Cleaning → NLP Skill Extraction → SQL Database → Dashboard → ML Model
```

## Key Results

- **1,348** unique job postings collected across **10 job families**
- **35** technical skills tracked and extracted from job descriptions using NLP
- **83.33%** accuracy on a machine learning model that classifies a job's category purely from its description
- A normalized, relational SQL database (4 linked tables) instead of a flat spreadsheet
- An interactive, filterable dashboard built on the real dataset (not mock data)

## Job Families Covered

Data Analyst · Data Scientist · Data Engineer · Machine Learning Engineer · Business Analyst · Business Intelligence Analyst · Software Engineer · Cloud Engineer · Cybersecurity Engineer · AI Engineer

## Tech Stack

| Layer | Tools |
|---|---|
| Data Collection | Python, Adzuna Job Search API, `requests` |
| Data Processing | `pandas`, regex-based NLP for skill extraction |
| Database | Microsoft SQL Server, `sqlalchemy`, `pyodbc` |
| Machine Learning | `scikit-learn` (TF-IDF + Logistic Regression) |
| Visualization | Power BI (relational model) & a web dashboard |
| Environment | Python, Jupyter Notebook, VS Code |

## Project Structure

```
job-market-intelligence/
├── .env                          # API credentials (not committed)
├── .gitignore
├── requirements.txt
├── src/
│   └── job_market_analysis.ipynb # Main analysis notebook
└── data/
    ├── raw/adzuna/                # Raw JSON responses per collection run
    └── processed/
        ├── jobs_master.csv        # Cleaned, deduplicated dataset
        ├── jobs_for_dashboard.csv # Flattened export for the dashboard
        ├── job_classifier_model.pkl
        └── tfidf_vectorizer.pkl
```

## Pipeline Walkthrough

### 1. Data Collection
Jobs are pulled from the [Adzuna Job Search API](https://developer.adzuna.com/) across 10 search keywords, with pagination, retry logic, and rate-limit handling to stay within the API's usage limits.

### 2. Data Cleaning
Raw API responses are flattened into a tabular format and deduplicated on `job_id`, since the same posting can appear under multiple search keywords.

### 3. Skill Extraction (NLP)
A curated list of 35 technical skills (Python, SQL, AWS, Power BI, etc.) is matched against each job description using word-boundary regex, producing a `skills` column per job.

### 4. Relational Database
Instead of one flat table, the data is normalized into four linked SQL Server tables:

- `companies` — unique employers
- `skills` — the master skill list
- `jobs_normalized` — job postings, linked to `companies`
- `job_skills` — a many-to-many link table between jobs and skills

### 5. Dashboard
A filterable, chart-driven dashboard built on the real exported dataset, covering:
- Top requested skills overall and per job family
- Salary distribution and averages by job family and skill
- Contract type breakdown
- Top hiring companies and locations

### 6. Machine Learning Model
A `TfidfVectorizer` + `LogisticRegression` pipeline predicts which of the 10 job families a posting belongs to, based only on its description text — achieving **83.33% accuracy** on a held-out test set of 270 postings.

### 7. Job Matcher (Employment Feature)
An interactive "find your best-fit jobs" tool built into the dashboard. The user enters the skills they have, and every job posting is scored against them:

```
coverage      = (matched skills) / (skills required by the job)
userCoverage  = (matched skills) / (skills the user entered)
matchScore    = coverage * 0.5 + userCoverage * 0.5
```

Jobs are ranked by `matchScore`, so a posting only rises to the top if it both fits the user's skill set well **and** makes good use of the skills they actually have — rather than favoring low-requirement postings by default. Each result shows the matched skills and an **Apply Now** button linking directly to the original posting (`redirect_url`).

## Notable Findings

- **Machine Learning**, **Python**, **AWS**, and **SQL** are the most frequently requested skills across the dataset.
- Skills with small sample sizes showed unstable salary averages that shifted significantly once the dataset was expanded — a reminder that sample size matters before drawing conclusions from job market data.
- The model's most common confusion is between closely related roles (e.g. Data Analyst ↔ Business Analyst, Data Scientist ↔ Data Engineer), which reflects genuine overlap in how these roles are described in real job postings.

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Adzuna API credentials to .env
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key

# 3. Open and run the notebook
jupyter notebook src/job_market_analysis.ipynb
```

## Data Source & Usage Notes

Data is collected via the Adzuna API under its published rate limits (25 requests/minute, 250/day by default). This project is intended for personal/academic research; any broader or commercial use should be verified against Adzuna's current terms of service.

## Presentation

A 12-slide project presentation (`AI Job Market Intelligence Platform.pptx`) summarizes the problem, pipeline, key findings, and results for graduation project defense — covering the collection stats, top skills and salary charts, database design, dashboard walkthrough, ML model accuracy, the Job Matcher feature, and an honest assessment of the project's current limitations and next steps.

## Future Improvements

- Expand data collection to additional countries and job boards for broader coverage
- Add time-series collection to track how skill demand changes over weeks/months
- Improve skill extraction with a trained NLP model instead of keyword matching
- Extend the ML model to salary prediction (regression) as the dataset grows
