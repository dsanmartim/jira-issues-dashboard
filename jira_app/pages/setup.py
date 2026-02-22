"""Connection setup page: collect Jira credentials and initialize IssueService securely."""

from __future__ import annotations

import logging
import tomllib

import streamlit as st
from jira import JIRAError
from streamlit.errors import StreamlitSecretNotFoundError

from jira_app.app import register_page
from jira_app.core.config import JIRA_DEFAULT_SERVER
from jira_app.core.jira_client import JiraAPI
from jira_app.core.service import IssueService

logger = logging.getLogger(__name__)

DEFAULT_TTL = 300


def _extract_secrets(source: object | None) -> dict[str, str]:
    """Extract Jira credentials from a secrets source (dict-like object).

    Canonical format (recommended):
        JIRA_SERVER = "https://your-instance.atlassian.net"
        JIRA_EMAIL = "user@example.com"
        JIRA_API_TOKEN = "your-api-token"
        JIRA_CACHE_TTL = 300  # optional, defaults to 300 seconds

    Alternative key names accepted for flexibility:
        - Server: JIRA_SERVER, jira_server
        - Email: JIRA_EMAIL, jira_email
        - Token: JIRA_API_TOKEN, JIRA_TOKEN, jira_api_token, jira_token
        - TTL: JIRA_CACHE_TTL, jira_cache_ttl

    Secrets can be provided at the top level or nested under a [jira] section.
    """
    if not source:
        return {}
    # Accept any mapping/dict-like object
    if hasattr(source, "keys"):
        src = {k: source.get(k) for k in source}  # type: ignore[arg-type]
    else:
        return {}

    def _as_str(val: object | None) -> str | None:
        if val is None:
            return None
        return str(val)

    def _as_float(val: object | None) -> float | None:
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "server": _as_str(src.get("JIRA_SERVER") or src.get("jira_server")) or "",
        "email": _as_str(src.get("JIRA_EMAIL") or src.get("jira_email")) or "",
        "token": _as_str(
            src.get("JIRA_API_TOKEN")
            or src.get("JIRA_TOKEN")
            or src.get("jira_api_token")
            or src.get("jira_token")
        )
        or "",
        "ttl": _as_float(src.get("JIRA_CACHE_TTL") or src.get("jira_cache_ttl")),
    }


def _mark_sidebar_connected() -> None:
    """Show a sidebar success when connection becomes ready."""
    st.session_state["show_sidebar_connected"] = True


def _init_issue_service(server: str, email: str, token: str, ttl: float) -> None:
    api = JiraAPI(server, email, token)
    if hasattr(api, "_cache_ttl"):
        api._cache_ttl = float(ttl)
    st.session_state["jira_server"] = server
    st.session_state["jira_email"] = email
    st.session_state["jira_client_ttl"] = float(ttl)
    st.session_state["issue_service"] = IssueService(api)


@register_page("Setup / Connection")
def setup_page():
    st.title("Jira Connection Setup")
    st.caption("Securely initialize the connection.")

    # If already initialized, we still allow overrides.
    ready = "issue_service" in st.session_state

    # Attempt automatic init from st.secrets (server-side only, no token shown in UI)
    if not ready:
        try:
            secrets_flat = _extract_secrets(dict(st.secrets))
            secrets_nested = _extract_secrets(dict(st.secrets.get("jira", {})))
        except StreamlitSecretNotFoundError:
            secrets_flat = {}
            secrets_nested = {}

        secrets_source = secrets_nested if secrets_nested.get("token") else secrets_flat
        if secrets_source.get("server") and secrets_source.get("email") and secrets_source.get("token"):
            try:
                ttl = float(
                    secrets_source.get("ttl") or st.session_state.get("jira_client_ttl") or DEFAULT_TTL
                )
                _init_issue_service(
                    secrets_source["server"], secrets_source["email"], secrets_source["token"], ttl
                )
                ready = True
                st.success("Connection initialized from Streamlit secrets.")
                _mark_sidebar_connected()
            except JIRAError as exc:
                error_detail = exc.text if hasattr(exc, "text") else str(exc)
                logger.warning("Jira connection failed from secrets: %s", error_detail)
                st.error("Failed to initialize from secrets. Please provide credentials below.")
            except (ValueError, TypeError) as exc:
                logger.warning("Invalid credentials in secrets: %s", type(exc).__name__)
                st.error("Failed to initialize from secrets. Please provide credentials below.")

    if ready:
        st.success("Jira connection is ready. You can proceed to the other pages.")
        st.caption("Want to switch accounts? Upload a new secrets.toml or enter credentials below.")

    tabs = st.tabs(["Upload secrets.toml", "Enter manually"])

    st.warning("Do not upload secrets on shared computers.")

    # Upload path: parse secrets.toml in-memory only
    with tabs[0]:
        uploaded = st.file_uploader(
            "Upload secrets.toml (not stored)", type=["toml"], accept_multiple_files=False
        )
        if uploaded is not None:
            try:
                data = uploaded.getvalue()
                parsed = tomllib.loads(data.decode("utf-8"))
                del data
                creds = _extract_secrets(parsed.get("jira", {})) or _extract_secrets(parsed)
                if not (creds.get("server") and creds.get("email") and creds.get("token")):
                    st.error("Missing required keys. Expect JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN.")
                else:
                    ttl_val = float(
                        creds.get("ttl") or st.session_state.get("jira_client_ttl") or DEFAULT_TTL
                    )
                    st.info("Secrets file parsed. Click Initialize Connection to proceed.")
                    if st.button("Initialize Connection", type="primary", key="init_from_upload"):
                        _init_issue_service(creds["server"], creds["email"], creds["token"], ttl_val)
                        st.success("Connection initialized from uploaded secrets (session only).")
                        _mark_sidebar_connected()
                        st.rerun()
            except tomllib.TOMLDecodeError as exc:
                logger.warning("TOML parsing failed: %s", exc)
                st.error("Could not read the uploaded secrets file. Check formatting and try again.")
            except (UnicodeDecodeError, ValueError) as exc:
                logger.warning("Secrets file decode error: %s", type(exc).__name__)
                st.error("Could not read the uploaded secrets file. Check formatting and try again.")

    # Manual entry path
    with tabs[1]:
        server = st.text_input(
            "Jira Server URL",
            value=st.session_state.get("jira_server", JIRA_DEFAULT_SERVER),
            placeholder=JIRA_DEFAULT_SERVER,
        )
        email = st.text_input(
            "Email / Username",
            value=st.session_state.get("jira_email", ""),
            placeholder="<your-user-name>@lsst.org",
        )
        token = st.text_input("API Token", type="password", value="")
        ttl = st.number_input(
            "Client cache TTL (seconds)",
            min_value=60,
            max_value=3600,
            value=int(st.session_state.get("jira_client_ttl", DEFAULT_TTL)),
            help="How long to cache Jira responses in the client (in seconds).",
        )
        init_btn = st.button("Initialize Connection", type="primary")

        if init_btn:
            if not (server and email and token):
                st.error("All fields required.")
            else:
                try:
                    _init_issue_service(server, email, token, float(ttl))
                    st.success("Connection initialized.")
                    _mark_sidebar_connected()
                    st.rerun()
                except JIRAError as exc:
                    logger.warning("Jira connection failed: %s", exc.text if hasattr(exc, "text") else exc)
                    st.error("Failed to initialize Jira client. Please verify the credentials.")
                except (ValueError, TypeError) as exc:
                    logger.warning("Invalid credential values: %s", type(exc).__name__)
                    st.error("Failed to initialize Jira client. Please verify the credentials.")

    # Reset connection
    if ready and st.button("Reset connection", type="secondary"):
        for key in ["issue_service", "jira_server", "jira_email", "jira_client_ttl"]:
            st.session_state.pop(key, None)
        st.rerun()
