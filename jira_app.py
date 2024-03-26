import streamlit as st

from jira_dashboard import JiraDashboard
from modules import detailed_tickets

# Set the page to wide mode
st.set_page_config(layout="wide")


def main_page():
    dashboard = JiraDashboard()
    dashboard.display()


def detailed_tickets_page():
    # Legacy page retained if needed
    detailed_tickets.show_detailed_tickets()


def stale_tickets_page():
    dashboard = JiraDashboard()
    dashboard.display_stale_tickets()


def project_aggregations_page():
    dashboard = JiraDashboard()
    dashboard.display_project_aggregations()


# Dictionary to hold your pages
pages = {
    "Main Dashboard": main_page,
    "Stale Tickets": stale_tickets_page,
    "Assignees & OBS Hierarchy": project_aggregations_page,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()), key="nav_selection")

# Call the selected page function
page = pages[selection]
page()
