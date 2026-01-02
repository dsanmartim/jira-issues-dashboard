# Jira Issues Dashboard

A Streamlit dashboard for analyzing Jira project issues (activity, aging/staleness, trends, persona insights, etc).

## For users

### What you get

- Activity + aging/staleness views over Jira issues
- Trend charts and top ticket tables
- Persona insights (Assignee / Reporter)
- Optional OBS hierarchy drilldowns

#### Running the app

Quickstart:
- Fastest: run with Docker (no Python needed) — see [Run with Docker](#run-with-docker).
- Tinker/dev: run locally from source with Python 3.13 — see [Running locally (clone the repo)](#running-locally-clone-the-repo).

## Run with Docker

This option lets you run the dashboard by only pulling the Docker image.

### 1. One-time setup

Install Docker (Desktop or Engine): https://www.docker.com/get-started/
2) Create your secrets file and put your real Jira values in it:

```bash
mkdir -p ~/.jira-issues-dashboard/.streamlit
vim ~/.jira-issues-dashboard/.streamlit/secrets.toml
```

Paste this template, then replace the placeholders before saving:

```toml
[jira]
JIRA_SERVER = "https://your-jira-instance"   # replace with your Jira base URL
JIRA_EMAIL = "user@example.com"               # replace with your Jira email/login
JIRA_API_TOKEN = "<api_token>"               # replace with your API token
```

Notes:
- Tokens: create an API token from your Atlassian account settings (Profile → Account Settings → Security → API tokens) and paste it here.
- Never commit secrets to git.

### 2. Start/stop the app (anytime)

Pull (or refresh) the image when you want the latest build:

```bash
docker pull dsanmartim/jira-issues-dashboard:latest
```

Run it in the foreground:
```bash
docker run --rm -p 8501:8501 \
  -v "$HOME/.jira-issues-dashboard/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" \
  dsanmartim/jira-issues-dashboard:latest
```

To stop the application, just press Ctrl+C in that terminal.

Optionally, you can name and run the container in the background:

```bash
docker run -d --name jira-dashboard -p 8501:8501 \
  -v "$HOME/.jira-issues-dashboard/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" \
  dsanmartim/jira-issues-dashboard:latest
```

In this case, to stop/remove the container later:

```bash
docker stop jira-dashboard && docker rm jira-dashboard
```

**Making it easier:** add a helper alias to your shell (e.g., ~/.zshrc or ~/.bashrc) for the background run:

```bash
echo "alias jira-dashboard-start='docker run -d --name jira-dashboard -p 8501:8501 -v \"$HOME/.jira-issues-dashboard/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro\" dsanmartim/jira-issues-dashboard:latest'" >> ~/.zshrc
echo "alias jira-dashboard-stop='docker stop jira-dashboard && docker rm jira-dashboard'" >> ~/.zshrc
```
Reload your shell (`source ~/.zshrc`) and use `jira-dashboard-start` / `jira-dashboard-stop`.

### 3. Access the app

Open: http://localhost:8501

Notes:
- Port 8501 must be free on your machine; change the left side of `-p` if needed.
- You can place the secrets file anywhere; update the host path in `-v` to match.
- If you skip mounting secrets, you can enter them in the app via the **Setup / Connection** page (not persistent).

### 4. Updating the app (at a glance)

- Pull the newest image: `docker pull dsanmartim/jira-issues-dashboard:latest`
- If running in background: `docker stop jira-dashboard && docker rm jira-dashboard`
- Start again (background): `docker run -d --name jira-dashboard -p 8501:8501 -v "$HOME/.jira-issues-dashboard/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" dsanmartim/jira-issues-dashboard:latest`

---

## Running locally

### 1. Clone the repository

```bash
git clone https://github.com/dsanmartim/jira-issues-dashboard.git
cd jira-issues-dashboard
```

### 2. Set up Python (Python 3.13)

This project targets **Python 3.13.x**. Before creating the virtual environment, confirm you are using Python 3.13:

```bash
python3 --version
```

If needed, install Python 3.13 and make sure `python3` points to it.

Then create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .            # runtime only
# or, for local dev tools (pytest, ruff, black, pre-commit):
pip install -e ".[dev]"
```

### 3. Configure Jira credentials

Create `.streamlit/secrets.toml` in the repo root:

```toml
[jira]
JIRA_SERVER = "https://your-jira-instance"
JIRA_EMAIL = "user@example.com"
JIRA_API_TOKEN = "<api_token>"
```

Notes:
- Ensure the `.streamlit` folder exists: `mkdir -p .streamlit`.
- The launcher also supports top-level `JIRA_SERVER`, `JIRA_EMAIL`, `JIRA_API_TOKEN` if you prefer a flatter secrets file.
- Tokens: create an API token from your Atlassian account settings (Profile → Account Settings → Security → API tokens) and paste it here.
- If secrets are missing, use **Setup / Connection** (interactive; not persisted to disk).
- Never commit secrets; keep `.streamlit/secrets.toml` out of version control (it is gitignored).

### 4. Start the app

```bash
streamlit run run_dashboard.py
```

`run_dashboard.py` auto-imports every module in `jira_app/pages/` so each page decorated with `@register_page(...)` registers itself.

Quick recipe (all steps in one go):

```bash
git clone <your-repo-url> jira-issues-dashboard && cd jira-issues-dashboard
python3.13 -m venv .venv && source .venv/bin/activate
pip install .
mkdir -p .streamlit && cat > .streamlit/secrets.toml <<'EOF'
[jira]
JIRA_SERVER = "https://your-jira-instance"
JIRA_EMAIL = "user@example.com"
JIRA_API_TOKEN = "<api_token>"
EOF
```
Run it:

```bash
streamlit run run_dashboard.py
```

What it does: clones the repo, creates a Python 3.13 venv, installs the app, writes the secrets file, then launches Streamlit. Adjust the Jira values, and finally run the app.

---

## What to expect in the app

The app has the following main views:

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

---

## For developers (contributing / adapting the app)

### Workflow for updating the image

Every time you update your code locally and want to share the updated version with the latest tag, you should follow these steps:

1. **Rebuild the image**: Run the build command again. Docker will rebuild the image with your code changes and re-assign the latest tag to the new build.

    ```bash
    docker build -t dsanmartim/jira-issues-dashboard:latest .
    ```

2. **Push the updated image**: Upload the new version to Docker Hub.

    ```bash
    docker push dsanmartim/jira-issues-dashboard:latest
    ```

This command will upload the new layers to Docker Hub and update the latest tag in your repository to point to the new build.


**Alternative: Automated Builds**
If you have a Docker Pro, Team, or Business subscription, you can configure Automated Builds. This allows Docker Hub to automatically build and push a new image every time you push new code to your source provider (like GitHub or Bitbucket), eliminating the need to manually build and push from your local machine

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

---

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

---

## Portability (adapting to another project)

Today, adapting this repo to a different Jira project typically requires code/config changes because:
- The project key is assumed to be `OBS` in some pages.
- Some analyses rely on OBS-only custom fields.

Roadmap for making this more “drop-in” for external users:
- Move project key + custom field IDs to a user-facing config layer (e.g., `.streamlit/secrets.toml`).
- Provide feature toggles (enable/disable hierarchy and time-lost features).
- Make segmentation dimensions robust when optional columns are missing.

---

## Configuration knobs

Most behavior lives in `jira_app/core/config.py`, including:
- `TIMEZONE` (default: `America/Santiago`)
- `FULL_COMMENT_HYDRATION` and parallel hydration tuning
- Jira base field list (`JIRA_FETCH_BASE_FIELDS`)

Column ordering defaults are in `jira_app/columns.yaml` and shared display constants in `jira_app/core/config.py`.

---

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

---

## License

Internal / TBD.
