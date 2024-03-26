import pandas as pd
import pytz
import streamlit as st


def show_detailed_tickets():
    st.title("Detailed Tickets")

    # Access the data from the session state
    if "issues_df" in st.session_state and st.session_state["issues_df"] is not None:
        df_display = st.session_state["issues_df"].copy()

        # Convert 'created' and 'updated' to Santiago time
        santiago_timezone = pytz.timezone("America/Santiago")

        df_display["created"] = pd.to_datetime(df_display["created"]).dt.tz_convert(
            santiago_timezone
        )

        df_display["updated"] = pd.to_datetime(df_display["updated"]).dt.tz_convert(
            santiago_timezone
        )

        df_display["comment_count"] = df_display["comments"].apply(len)
        df_display["history_count"] = df_display["histories"].apply(len)

        st.dataframe(
            df_display[
                [
                    "key",
                    "summary",
                    "priority",
                    "priority_value",
                    "status",
                    "time_lost",
                    "created",
                    "updated",
                    "assignee",
                    "reporter",
                    "resolution",
                    "comment_count",
                    "history_count",
                ]
            ],
            hide_index=True,
        )
    else:
        st.write(
            "No data available. Please fetch the data " "from the main page first."
        )


# Provide a link or button to navigate back to the dashboard
# st.markdown("[Back to Dashboard](/)")
