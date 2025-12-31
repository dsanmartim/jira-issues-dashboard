"""Streamlit launcher for the Jira dashboard.

This shim preserves the public entry point name while ensuring no real
credentials are baked into the repository.
"""

# API Credentials (placeholders only; actual secrets live in .streamlit/secrets.toml)
# These names remain for backward compatibility with docs pointing here; run_dashboard.py
# sources credentials from Streamlit secrets, so leave real values out of version control.
JIRA_SERVER = "https://your-jira-instance"
JIRA_EMAIL = "user@example.com"
JIRA_API_TOKEN = "<api_token>"


def main():
    """Delegate to the canonical dashboard entry point."""
    from run_dashboard import main as _run_dashboard_main

    _run_dashboard_main()


if __name__ == "__main__":
    main()
