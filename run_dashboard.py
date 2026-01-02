"""Convenience launcher for the Streamlit app.

Usage:
  streamlit run run_dashboard.py

Automatically imports every module in ``jira_app/pages`` so each page
decorated with ``@register_page`` registers itself without manual edits here.
"""

from importlib import import_module
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from jira_app.app import main

st.set_page_config(layout="wide")


def _auto_init_issue_service():
    """Initialize Jira service from Streamlit secrets if available."""
    if "issue_service" in st.session_state:
        return

    try:
        # Try to get secrets from a [jira] section, fall back to top-level
        secrets_obj = st.secrets
        jira_secrets = secrets_obj.get("jira", {})
        server = jira_secrets.get("JIRA_SERVER") or secrets_obj.get("JIRA_SERVER")
        email = jira_secrets.get("JIRA_EMAIL") or secrets_obj.get("JIRA_EMAIL")
        token = (
            jira_secrets.get("JIRA_API_TOKEN")
            or secrets_obj.get("JIRA_API_TOKEN")
            or jira_secrets.get("JIRA_TOKEN")
            or secrets_obj.get("JIRA_TOKEN")
        )
    except StreamlitSecretNotFoundError:
        st.sidebar.info("No secrets file found. Please use the Setup page.")
        return

    if server and email and token:
        st.sidebar.info("Secrets found, attempting to connect to Jira...")
        try:
            from jira_app.core.jira_client import JiraAPI
            from jira_app.core.service import IssueService

            api = JiraAPI(server, email, token)
            st.session_state["jira_server"] = server
            st.session_state["issue_service"] = IssueService(api)
            st.session_state["show_sidebar_connected"] = True
        except Exception as e:
            st.sidebar.error(f"Jira connection failed: {e}")
            # Clear any partial state to ensure user is directed to setup
            if "issue_service" in st.session_state:
                del st.session_state["issue_service"]
    else:
        st.sidebar.warning("Jira secrets not found. Please use the Setup page.")


_auto_init_issue_service()

# Show a connection success message after reruns triggered by setup actions
if st.session_state.pop("show_sidebar_connected", False):
    st.sidebar.success("Jira connection successful!")

PAGES_DIR = Path(__file__).parent / "jira_app" / "pages"
for py in sorted(PAGES_DIR.glob("[!_]*.py")):
    mod_name = f"jira_app.pages.{py.stem}"
    try:
        import_module(mod_name)
    except Exception as e:  # pragma: no cover - defensive
        print(f"Failed importing page {mod_name}: {e}")

if __name__ == "__main__":
    main()
