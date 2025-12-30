"""Connection setup page: collect Jira credentials and initialize IssueService."""

from __future__ import annotations

import streamlit as st

from jira_app.app import register_page
from jira_app.core.jira_client import JiraAPI
from jira_app.core.service import IssueService


@register_page("Setup / Connection")
def setup_page():
    st.title("Jira Connection Setup")
    st.caption("Enter credentials (use secrets manager in production).")

    # Pre-fill from secrets if available (user can override)
    jira_secrets = st.secrets.get("jira", {})
    secret_server = jira_secrets.get("JIRA_SERVER") or st.secrets.get("JIRA_SERVER")
    secret_email = jira_secrets.get("JIRA_EMAIL") or st.secrets.get("JIRA_EMAIL")
    secret_token = (
        jira_secrets.get("JIRA_API_TOKEN")
        or st.secrets.get("JIRA_API_TOKEN")
        or jira_secrets.get("JIRA_TOKEN")
        or st.secrets.get("JIRA_TOKEN")
    )

    server = st.text_input(
        "Jira Server URL",
        value=st.session_state.get("jira_server") or secret_server or "",
    )
    email = st.text_input(
        "Email / Username",
        value=st.session_state.get("jira_email") or secret_email or "",
    )
    token = st.text_input(
        "API Token",
        type="password",
        value=secret_token or "",
    )
    ttl = st.number_input("Client cache TTL (seconds)", min_value=60, max_value=3600, value=300)
    init_btn = st.button("Initialize Connection", type="primary")

    if init_btn:
        if not (server and email and token):
            st.error("All fields required.")
            return
        try:
            api = JiraAPI(server, email, token)
            # Adjust TTL if attribute present
            if hasattr(api, "_cache_ttl"):
                api._cache_ttl = float(ttl)
            st.session_state["jira_server"] = server
            st.session_state["jira_email"] = email
            st.session_state["issue_service"] = IssueService(api)
            st.success("Connection initialized.")
        except Exception as e:  # pragma: no cover
            st.error(f"Failed to initialize Jira client: {e}")

    if "issue_service" in st.session_state:
        st.info("IssueService ready.")
