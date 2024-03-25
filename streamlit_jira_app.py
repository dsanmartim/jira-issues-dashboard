from datetime import date, timedelta

import pandas as pd
import streamlit as st
from jira import JIRA


class JiraDashboard:
    def __init__(self, email, api_token, server):
        self.jira = JIRA(
            basic_auth=(email, api_token),
            options={"server": server}
        )
        self.issues_df = None

    def fetch_issues_with_details(self, project_key, start_date, end_date):
        jql_query = (
            f"project = {project_key} AND updated >= '{start_date}' "
            f"AND updated <= '{end_date}'"
        )
        issues = self.jira.search_issues(
            jql_query, expand="renderedFields, changelog", maxResults=False
        )
        self.issues_df = pd.DataFrame(
            [
                {
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "created": issue.fields.created,
                    "updated": issue.fields.updated,
                    "assignee": issue.fields.assignee.displayName
                    if issue.fields.assignee
                    else "Unassigned",
                    "reporter": issue.fields.reporter.displayName
                    if issue.fields.reporter
                    else "Unknown",
                    "priority": issue.fields.priority.name
                    if issue.fields.priority
                    else "None",
                    "status": issue.fields.status.name,
                    "resolution": issue.fields.resolution.name
                    if issue.fields.resolution
                    else "Unresolved",
                    "comments": [
                        {
                            "author": comment.author.displayName,
                            "created": comment.created,
                            "body": comment.body,
                        }
                        for comment in issue.fields.comment.comments
                    ],
                    "histories": [
                        {
                            "author": history.author.displayName,
                            "created": history.created,
                            "items": history.items,
                        }
                        for history in issue.changelog.histories
                    ],
                }
                for issue in issues
            ]
        )

        self.issues_df["attention_score"] = (
            self.issues_df["comments"].apply(len)
            + self.issues_df["histories"].apply(len)
        )
        return self.issues_df.sort_values(
            by="attention_score",
            ascending=False
            )

    def get_most_attended_tickets(self):
        self.issues_df["attention_score"] = self.issues_df["comments"].apply(
            len
        ) + self.issues_df["histories"].apply(len)

        return self.issues_df.sort_values(
            by="attention_score",
            ascending=False
        )

    def get_most_commented_tickets(self):
        self.issues_df["comment_count"] = self.issues_df["comments"].apply(len)
        return self.issues_df.sort_values(by="comment_count", ascending=False)

    def get_tickets_with_most_user_comments(self, user_name):
        def count_user_comments(comments):
            return sum(
                1 for comment in comments if comment["author"] == user_name
            )

        self.issues_df["user_comment_count"] = (
            self.issues_df["comments"]
            .apply(lambda x: count_user_comments(x))
        )
        return self.issues_df.sort_values(
            by="user_comment_count",
            ascending=False
        )

    def visualize_tickets(self, df, column, title, explanation):
        st.write(f"## {title}")
        st.write(explanation)

        df_display = df[
            [
                "key",
                "summary",
                "created",
                "updated",
                "assignee",
                "reporter",
                "priority",
                "status",
                "resolution",
                column,
            ]
        ].head(10)

        df_display = df_display.rename(columns={column: 'Total Count'})

        st.dataframe(df_display)

    def display_all_tickets(self):
        
        if self.issues_df is None:
            
            if st.button("Show All Tickets"):
                st.write("## All Tickets")
                st.dataframe(
                    self.issues_df[
                        [
                            "key",
                            "summary",
                            "created",
                            "updated",
                            "assignee",
                            "reporter",
                            "priority",
                            "status",
                            "resolution",
                        ]
                    ]
                )
        else:
            st.write("No data to display. Please fetch the data first.")

    def display(self):
        st.sidebar.title("JIRA Issue Analysis Dashboard")
        project_key = st.sidebar.text_input("Project Key", value="OBS")

        default_end_date = date.today()
        default_start_date = default_end_date - timedelta(days=7)

        # Set default end_date to today and start_date to 7 days before
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start_date
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=default_end_date
        )
        user_for_comment_analysis = st.sidebar.text_input(
            "User for Comment Analysis", 
            value="Rubin Jira API Access"
        )

        if st.sidebar.button("Fetch and Analyze"):
            with st.spinner("Fetching and analyzing data..."):
                self.fetch_issues_with_details(
                    project_key,
                    start_date,
                    end_date
                )

                # Display the most attended tickets
                attended_df = self.get_most_attended_tickets()
                self.visualize_tickets(
                    attended_df,
                    'attention_score',
                    'Most Attended Tickets',
                    "This metric combines the total number of comments and "
                    "history entries to identify the most engaged tickets."
                )

                # Display the most commented tickets
                commented_df = self.get_most_commented_tickets()
                self.visualize_tickets(
                    commented_df,
                    'comment_count',
                    'Most Commented Tickets',
                    "This metric ranks tickets based on the total number "
                    "of comments they received."
                )

                # Display the tickets with most comments from a specific user
                user_commented_df = self.get_tickets_with_most_user_comments(
                    user_for_comment_analysis
                )
                self.visualize_tickets(
                    user_commented_df,
                    'user_comment_count',
                    'Tickets with Most Comments from Specific User',
                    "This metric shows tickets with the highest number of "
                    "comments from the specified user."
                )

            st.success("Analysis completed!")

        self.display_all_tickets()


# API credentials
email = "dsanmartim@lsst.org"
api_token = "ATATT3xFfGF0aDZ_zqSdqL_Zmd6Z7skvMZkwM6cj-tijPw5ankdXQ0ItXUKfuIkzfOqErBGtLNM_C0LI2Hg0w-utZC5yEH_-UK8KE5m-nVMqu7guXrhqAtVXNBnONPyunvHv37-S98ZeH99jp1Ze23JJFFOiXuA9mWI9N-nBcNuP0jJXE_B4_gY=4FCBB2F0"
server = "https://rubinobs.atlassian.net/"

# Create and display the JIRA Issue Analysis Dashboard
dashboard = JiraDashboard(email, api_token, server)
dashboard.display()
