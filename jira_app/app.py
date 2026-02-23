"""Application entry point: page registry and router."""

from __future__ import annotations

import streamlit as st

PAGES = {}


def register_page(label):
    def decorator(func):
        PAGES[label] = func
        return func

    return decorator


def main():
    st.sidebar.title("OBS Project Dashboard")
    pages = list(PAGES.keys())
    if not pages:
        st.write("No pages registered yet.")
        return
    preferred_order = [
        "Activity Overview",  # main holistic view
        "Personal View",  # user-specific triage
        "Assignee | Reporter Insights",  # persona metrics
        "Stale Tickets",  # focused remediation list
        "Setup / Connection",  # configuration
    ]

    # Build ordered list explicitly to avoid subtle sort mismatches if labels
    # differ by spacing
    ordered = [name for name in preferred_order if name in pages]
    trailing = [name for name in pages if name not in preferred_order]
    trailing.sort()  # stable alphabetical for unpreferred pages (e.g., Debug)
    pages = ordered + trailing

    # Surface any missing preferred pages (debug aid in sidebar footer)
    missing = [name for name in preferred_order if name not in ordered]
    if missing:
        st.sidebar.caption(f"(Info) Missing expected pages not yet registered: {', '.join(missing)}")
    # If setup exists and no issue_service yet, default to setup page
    if "Setup / Connection" in pages and "issue_service" not in st.session_state:
        default = pages.index("Setup / Connection")
    else:
        default = 0
    page = st.sidebar.selectbox("Page", pages, index=default)
    PAGES[page]()


if __name__ == "__main__":
    main()
