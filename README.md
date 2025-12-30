# Jira Issues Dashboard (Streamlit)

Streamlit app for analyzing Jira issues for the OBS project (activity, aging/staleness, trends, and persona insights).

## For users

### What you get

- Activity + aging/staleness views over Jira issues
- Trend charts and top ticket tables
- Persona insights (Assignee / Reporter)
- Optional OBS hierarchy drilldowns (OBS-only custom field)

### Running locally

1) Set up Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

2) Configure Jira credentials

Create `.streamlit/secrets.toml` in the repo root:

```toml
[jira]
JIRA_SERVER = "https://your-jira-instance"
JIRA_EMAIL = "user@example.com"
JIRA_API_TOKEN = "<api_token>"
```

Notes:
- The launcher also supports top-level `JIRA_SERVER`, `JIRA_EMAIL`, `JIRA_API_TOKEN`.
- If secrets are missing, use **Setup / Connection** (interactive; not persisted to disk).
- Never commit secrets.

3) Start the app

```bash
streamlit run run_dashboard.py
```

`run_dashboard.py` auto-imports every module in `jira_app/pages/` so each page decorated with `@register_page(...)` registers itself.

### Pages

- **Activity Overview** (`jira_app/pages/activity_overview.py`)
  - Main view: activity + aging + trends + top ticket tables
- **Assignee | Reporter Insights** (`jira_app/pages/persona_insights.py`)
  - Unified page with two tabs:
    - Assignee tab renderer: `jira_app/pages/assignees.py` (`render_assignee_tab`)
    - Reporter tab renderer: `jira_app/pages/reporter.py` (`render_reporter_tab`)
- **Stale Tickets** (`jira_app/pages/stale.py`)
  - Open tickets exceeding an inactivity threshold
- **Setup / Connection** (`jira_app/pages/setup.py`)
  - Initializes `IssueService` and stores it in Streamlit session state

## For developers (adapting the app)

### High-level architecture

The codebase is split into:
- **Pages**: Streamlit UI and user interactions (`jira_app/pages/`)
- **Core**: Jira API + fetching/enrichment pipeline (`jira_app/core/`)
- **Analytics**: aggregations/metrics/segments used by pages (`jira_app/analytics/`)
- **Visual**: chart/table helpers and column metadata (`jira_app/visual/`)
- **Features**: feature-owned glue that keeps pages thin and testable (`jira_app/features/`)

### Repository structure

- `jira_app/app.py`
  - Page registry + sidebar router (`register_page`, `main`)
- `jira_app/core/`
  - `jira_client.py`: `JiraAPI` wrapper + caching
  - `service.py`: `IssueService` fetch/enrich pipeline (dataset orchestration)
  - `mappers.py`: Jira issue JSON → normalized DataFrame columns
  - `config.py`: constants, field IDs, feature flags (timezone, hydration tuning)
- `jira_app/analytics/`
  - `aggregations/`: grouping helpers (OBS hierarchy, assignee, etc.)
  - `metrics/`: metric computation (activity, aging)
  - `segments/`: segment builders for top-N and drilldown views
  - `persona/metrics.py`: shared persona metric builders
- `jira_app/visual/`
  - charts, column metadata, tables, progress reporting
- `jira_app/features/`
  - `activity_overview/context.py`: builds a testable dashboard context object
- `tests/`
  - pytest suite for core behaviors and page import sanity

### Dataset model (DataFrame columns)

Most analysis in the app is done on a single Pandas DataFrame produced by `IssueService`.
Conceptually, the pipeline is:

1. **Fetch** raw Jira issue JSON (search + optional per-issue hydration)
2. **Map** to a normalized DataFrame (`jira_app/core/mappers.py`)
3. **Enrich** with computed metrics (`jira_app/analytics/metrics/*` + `jira_app/core/service.py`)

### Core mapped columns (from Jira)

These are produced in `jira_app/core/mappers.py` and are expected to exist for most views:

- Identity: `key`, `summary`, `issuetype`
- Dates: `created`, `updated`, `resolution_date`
- People: `assignee`, `reporter`
- Status: `status`, `resolution`, `priority`, `priority_value`
- Text tags: `labels` (normalized to a comma-separated string for display)
- Structures:
	- `comments` (list of dicts)
	- `histories` (list of dicts from changelog)

### Derived/enriched columns (computed)

Commonly added by the enrichment pipeline:

- Aging: `created_dt`, `updated_dt`, `days_open`, `days_since_update`
- Time-to-resolution (when resolution is available): `resolution_dt`, `time_to_resolution_days`
- Activity window metrics (for a selected date range):
	- `comments_in_range`, `status_changes`, `other_changes`, `activity_score_weighted`
	- Unified counters: `filtered_comments_count`, `filtered_histories_count`
	- OLE bot-specific signals: `ole_comments_count`, `ole_last_comment`

The source of truth for “what gets computed when” is `jira_app/core/service.py`.

## OBS project-specific custom fields

This app is currently optimized for the OBS Jira project and uses two OBS-specific custom fields.
They are defined by numeric field IDs in `jira_app/core/config.py` under `FIELD_IDS`:

- **OBS hierarchy** (`FIELD_IDS["obs_hierarchy"]`, currently `customfield_10476`)
	- Parsed into: `obs_system`, `obs_subsystem`, `obs_component`
	- Used for hierarchy aggregations and drilldowns
- **Time lost** (`FIELD_IDS["time_lost"]`, currently `customfield_10106`)
	- Mapped into: `time_lost`

If you point the tool at a Jira project that does not have these fields:
- The mapping should safely produce empty/None values for these columns.
- Views that depend on these fields may show “Unknown”/empty groupings or reduced insights.

## Portability (adapting to another project)

Today, adapting this repo to a different Jira project typically requires code/config changes because:
- The project key is assumed to be `OBS` in some pages.
- Some analyses rely on OBS-only custom fields.

Roadmap for making this more “drop-in” for external users:
- Move project key + custom field IDs to a user-facing config layer (e.g., `.streamlit/secrets.toml`).
- Provide feature toggles (enable/disable hierarchy and time-lost features).
- Make segmentation dimensions robust when optional columns are missing.

## Configuration knobs

Most behavior lives in `jira_app/core/config.py`, including:
- `TIMEZONE` (default: `America/Santiago`)
- `FULL_COMMENT_HYDRATION` and parallel hydration tuning
- Jira base field list (`JIRA_FETCH_BASE_FIELDS`)

Column ordering defaults are in `jira_app/columns.yaml` and shared display constants in `jira_app/core/config.py`.

## Development

### Tests

```bash
pytest
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## License

Internal / TBD.
