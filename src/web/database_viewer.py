# flake8: noqa: E501
"""
Streamlit application for viewing email analysis database contents.
"""

import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import sys
import os
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.database import Database
from web.visualizations import EmailVisualizations, create_dashboard_layout

# Role-based user dictionary
USERS = {
    "admin": {"password": "adminpass", "role": "admin"},
    "abhishek": {"password": "pass123", "role": "analyst"},
    "prasun": {"password": "pass123", "role": "viewer"},
    # Add more users as needed
}

def format_tags(tags):
    """Format tags for display in the table."""
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except:
            return tags
    if isinstance(tags, list):
        return ", ".join(tags)
    return str(tags)


def format_confidence(confidence):
    """Format confidence as percentage."""
    if isinstance(confidence, float):
        return f"{confidence:.1f}%"
    return f"{confidence}%"


def display_kpi_cards(kpi_metrics):
    """Display KPI cards in a grid layout."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📧 Total Emails",
            f"{kpi_metrics['total_emails']:,}",
            help="Total number of emails analyzed"
        )
    
    with col2:
        st.metric(
            "😊 Positive Sentiment",
            f"{kpi_metrics['positive_pct']:.1f}%",
            help="Percentage of emails with positive sentiment"
        )
    
    with col3:
        st.metric(
            "📊 Avg Confidence",
            f"{kpi_metrics['avg_confidence']:.1f}%",
            help="Average confidence score across all analyses"
        )
    
    with col4:
        st.metric(
            "📅 Emails/Day",
            f"{kpi_metrics['emails_per_day']:.1f}",
            help="Average emails processed per day"
        )


def display_analytics_dashboard(df):
    """Display the analytics dashboard with charts and visualizations."""
    st.header("📊 Analytics Dashboard")

    # Create visualizations
    viz = EmailVisualizations(df)

    # Get KPI metrics
    kpi_metrics = viz.create_kpi_cards()

    # Display KPI cards
    display_kpi_cards(kpi_metrics)

    st.markdown("---")

    # Charts section
    st.subheader("📈 Key Insights")
        
    # Row 1: Sentiment and Classification
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(viz.create_sentiment_pie_chart(), use_container_width=True)

    with col2:
        st.plotly_chart(viz.create_classification_sentiment_heatmap(), use_container_width=True)

    # Row 2: Trends
    st.subheader("📈 Temporal Trends")

    trend_period = st.selectbox(
        "Select Time Period",
        options=['month', 'week', 'date'],
        format_func=lambda x: x.capitalize(),
        help="Group trends by month, week, or individual date"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(viz.create_sentiment_trend_chart(trend_period), use_container_width=True)

    with col2:
        st.plotly_chart(viz.create_classification_trend_chart(trend_period), use_container_width=True)

    # Row 3: Heatmap and Confidence
    st.subheader("🔍 Detailed Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(viz.create_confidence_distribution(), use_container_width=True)
        
        

    with col2:
        st.plotly_chart(viz.create_confidence_by_classification(), use_container_width=True)

    # Row 4: Tags and Confidence by Classification
    st.subheader("🏷️ Tag Analysis & Quality Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(viz.create_top_tags_bar_chart(), use_container_width=True)

    with col2:
        st.plotly_chart(viz.create_classification_bar_chart(), use_container_width=True)

    # Word Cloud
    st.subheader("☁️ Tag Word Cloud")
    wordcloud_img = viz.create_tag_wordcloud()
    if wordcloud_img != "No tags available":
        st.image(f"data:image/png;base64,{wordcloud_img}", use_container_width=True)
    else:
        st.info("No tags available for word cloud generation")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Email Analysis Dashboard",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- LOGIN SCREEN ---
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "role" not in st.session_state:
        st.session_state["role"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None

    if not st.session_state["authenticated"]:
        st.title("🔒 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            user = USERS.get(username)
            if user and user["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["role"] = user["role"]
                st.session_state["username"] = username
                st.success(f"Login successful! Role: {user['role']}")
                st.rerun()
            else:
                st.error("Invalid username or password")
        return  # Stop here if not authenticated

    # --- REST OF DASHBOARD CODE BELOW ---

    st.title("📧 Email Analysis Dashboard")
    st.markdown("---")

    # Initialize database
    try:
        db = Database()
        st.success("✅ Database connection established successfully!")
    except Exception as e:
        st.error(f"❌ Error connecting to database: {str(e)}")
        return

    # Get all data
    try:
        all_data = db.get_all_analyses()
        if not all_data:
            st.warning("⚠️ No data found in the database.")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(all_data)

        # Add formatted columns for display
        df["tags_display"] = df["tags"].apply(format_tags)
        df["confidence_display"] = df["confidence"].apply(format_confidence)
        df["created_at_display"] = pd.to_datetime(df["created_at"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        df["email_date_display"] = pd.to_datetime(df["email_date"]).dt.strftime("%Y-%m-%d")

        # Ensure email_date is datetime (primary) or created_at (fallback)
        if 'email_date' in df.columns:
            df['email_date'] = pd.to_datetime(df['email_date'], errors='coerce')
            df = df.dropna(subset=['email_date'])
        else:
            df['email_date'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = df.dropna(subset=['email_date'])

        # Sidebar for filters (common to both tabs)
        with st.sidebar:
            st.header("🔍 Filters")

            # Date Range Slider
            min_date = df['email_date'].min().date()
            max_date = df['email_date'].max().date()
            date_range = st.slider(
                "Email Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD",
                help="Filter by the date when emails were sent/received"
            )

            # Sentiment Filter
            sentiment_filter = st.multiselect(
                "Select sentiments to display:",
                options=sorted(df["sentiment"].unique()),
                default=sorted(df["sentiment"].unique()),
                help="Choose which sentiment categories to include in the results"
            )

            # Classification Filter (replace multiselect with checkboxes)
            classification_options = list(Database.VALID_CLASSIFICATIONS)
            classification_options.sort()
            selected_classifications = []
            st.markdown("**Select Categories to display:**")
            for c in classification_options:
                if st.checkbox(c, value=True, key=f"class_{c}"):
                    selected_classifications.append(c)

            # Add a logout button below the clear filters button
            if st.button("🚪 Logout"):
                for key in ["authenticated", "role", "username"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Apply filters to DataFrame
        filtered_df = df[
            (df["sentiment"].isin(sentiment_filter))
            & (df["classification"].isin(selected_classifications))
            & (df["email_date"].dt.date >= date_range[0])
            & (df["email_date"].dt.date <= date_range[1])
        ]

        # Create tabs
        tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "📋 Data View"])

        with tab1:
            try:
                display_analytics_dashboard(filtered_df)
            except Exception as e:
                st.error(f"❌ Error displaying analytics dashboard: {str(e)}")
                st.info("💡 This might be due to data format issues. Please check your database data.")
                st.exception(e)

        with tab2:
            # Original data view functionality
            st.subheader("📋 Email Analysis Data")

            # Display statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", len(df))

            with col2:
                st.metric("Filtered Records", len(filtered_df))

            with col3:
                st.metric("Unique Sentiments", len(df["sentiment"].unique()))

            with col4:
                st.metric("Unique Classifications", len(df["classification"].unique()))

            st.markdown("---")

            # Display the data table
            if len(filtered_df) > 0:
                # Create display DataFrame with selected columns
                display_df = filtered_df[
                    [
                        "id",
                        "email_date_display",
                        "sentiment",
                        "classification",
                        "confidence_display",
                        "tags_display",
                        "created_at_display",
                    ]
                ].copy()

                # Rename columns for better display
                display_df.columns = [
                    "ID",
                    "Email Date",
                    "Sentiment",
                    "Classification",
                    "Confidence",
                    "Tags",
                    "Created At",
                ]

                # Display the table with scrollable view and better styling
                st.dataframe(
                    display_df, 
                    use_container_width=True, 
                    height=600,
                    hide_index=True
                )

                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download filtered data as CSV",
                    data=csv,
                    file_name=f"email_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                # Detailed view section
                st.markdown("---")
                st.subheader("🔍 Detailed Email View")

                # Select an email to view details
                selected_id = st.selectbox(
                    "Select an email ID to view details:",
                    options=filtered_df["id"].tolist(),
                    format_func=lambda x: f"ID {x}"
                )

                if selected_id:
                    selected_email = filtered_df[filtered_df["id"] == selected_id].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Email Text:**")
                        st.text_area(
                            "Email Content",
                            value=selected_email["email_text"],
                            height=300,
                            disabled=True,
                            key="email_text_display"
                        )
                    
                    with col2:
                        st.write("**Analysis Results:**")
                        st.json({
                            "Sentiment": selected_email["sentiment"],
                            "Classification": selected_email["classification"],
                            "Confidence": f"{selected_email['confidence']}%",
                            "Tags": selected_email["tags"],
                            "Email Date": selected_email["email_date_display"],
                            "Analysis Date": selected_email["created_at_display"]
                        })

            else:
                st.warning("⚠️ No emails match the selected filters.")

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()