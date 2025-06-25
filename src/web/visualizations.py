# flake8: noqa: E501
"""
Visualization module for email analysis dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import streamlit as st
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta


class EmailVisualizations:
    """Class for creating email analysis visualizations."""
    
    # Consistent color schemes
    SENTIMENT_COLORS = {
        'Positive': '#2E8B57',  # Sea Green
        'Neutral': '#FFD700',   # Gold
        'Negative': '#DC143C'   # Crimson
    }
    
    CLASSIFICATION_COLORS = {
        'Product/Stocking Requests': '#4682B4',      # Steel Blue
        'Admin/Coordination': '#32CD32',             # Lime Green
        'Feedback/Complaints': '#FF6347',            # Tomato
        'Maintenance/Repairs': '#FF8C00',            # Dark Orange
        'Billing/Invoices': '#9370DB',               # Medium Purple
        'General Follow-ups': '#20B2AA',             # Light Sea Green
        'Operational Logistics': '#FF69B4'           # Hot Pink
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame of email analysis data.
        
        Args:
            df: DataFrame with email analysis data
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for visualizations."""
        # Use email_date as the primary date for analysis
        if 'email_date' in self.df.columns:
            try:
                self.df['email_date'] = pd.to_datetime(self.df['email_date'], errors='coerce')
                # Remove rows where datetime conversion failed
                self.df = self.df.dropna(subset=['email_date'])
            except Exception as e:
                print(f"Warning: Error converting email_date to datetime: {e}")
                # If conversion fails, create a dummy datetime column
                self.df['email_date'] = pd.Timestamp.now()
        else:
            # Fallback to created_at if email_date doesn't exist
            try:
                self.df['email_date'] = pd.to_datetime(self.df['created_at'], errors='coerce')
                self.df = self.df.dropna(subset=['email_date'])
            except Exception as e:
                print(f"Warning: Error converting created_at to datetime: {e}")
                self.df['email_date'] = pd.Timestamp.now()
        
        # Extract date components from email_date
        if not self.df['email_date'].isna().all():
            self.df['date'] = self.df['email_date'].dt.date
            self.df['month'] = self.df['email_date'].dt.to_period('M')
            self.df['week'] = self.df['email_date'].dt.to_period('W')
        else:
            # Fallback if no valid dates
            self.df['date'] = pd.Timestamp.now().date()
            self.df['month'] = pd.Timestamp.now().to_period('M')
            self.df['week'] = pd.Timestamp.now().to_period('W')
        
        # Parse tags if they're stored as JSON strings
        if 'tags' in self.df.columns:
            self.df['tags_parsed'] = self.df['tags'].apply(
                lambda x: x if isinstance(x, list) else json.loads(x) if isinstance(x, str) else []
            )
    
    def create_sentiment_pie_chart(self) -> go.Figure:
        """Create a pie chart showing sentiment distribution."""
        sentiment_counts = self.df['sentiment'].value_counts()
        
        # Use consistent colors for sentiments
        colors = [self.SENTIMENT_COLORS.get(sentiment, '#808080') for sentiment in sentiment_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Overall Email Sentiment Distribution",
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_classification_bar_chart(self) -> go.Figure:
        """Create a horizontal bar chart showing email volume by categories."""
        classification_counts = self.df['classification'].value_counts()
        
        # Use consistent colors for classifications
        colors = [self.CLASSIFICATION_COLORS.get(classification, '#808080') for classification in classification_counts.index]
        
        fig = go.Figure(data=[go.Bar(
            x=classification_counts.values,
            y=classification_counts.index,
            orientation='h',
            marker_color=colors,
            text=classification_counts.values,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Email Volume by Categories",
            xaxis_title="Number of Emails",
            yaxis_title="Categories",
            height=500,
            margin=dict(t=50, b=50, l=200, r=50)
        )
        
        return fig
    
    def create_sentiment_trend_chart(self, period: str = 'month') -> go.Figure:
        """Create a line chart showing sentiment trends over time."""
        if period == 'month':
            time_col = 'month'
        elif period == 'week':
            time_col = 'week'
        else:
            time_col = 'date'
        
        # Group by time period and sentiment
        sentiment_trends = self.df.groupby([time_col, 'sentiment']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            if sentiment in sentiment_trends.columns:
                fig.add_trace(go.Scatter(
                    x=sentiment_trends.index.astype(str),
                    y=sentiment_trends[sentiment],
                    mode='lines+markers',
                    name=sentiment,
                    line=dict(color=self.SENTIMENT_COLORS.get(sentiment, '#808080'), width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title=f"Email Sentiment Trends Over Time ({period.capitalize()})",
            xaxis_title="Time Period",
            yaxis_title="Number of Emails",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_classification_sentiment_heatmap(self) -> go.Figure:
        """Create a heatmap showing sentiment distribution within classifications."""
        # Create cross-tabulation
        cross_tab = pd.crosstab(self.df['classification'], self.df['sentiment'])
        
        # Calculate percentages
        cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=cross_tab_pct.values,
            x=cross_tab_pct.columns,
            y=cross_tab_pct.index,
            colorscale=[[0.0, 'white'], [1.0, 'blue']],  # Steel Blue
            text=cross_tab_pct.values.round(1),
            texttemplate="%{text}%",
            textfont={"size": 12},
            colorbar=dict(title="Percentage")
        ))
        
        fig.update_layout(
            title="Heatmap of Email Sentiment Distribution by Categories (%)",
            xaxis_title="Sentiment",
            yaxis_title="Categories",
            height=500,
            margin=dict(t=50, b=50, l=200, r=50)
        )
        
        return fig
    
    def create_confidence_distribution(self) -> go.Figure:
        """Create a histogram showing confidence score distribution."""
        fig = go.Figure(data=[go.Histogram(
            x=self.df['confidence'],
            nbinsx=20,
            marker_color='#4682B4',
            opacity=0.7
        )])
        
        fig.update_layout(
            title="Category Classification Confidence Score Distribution",
            xaxis_title="Confidence Score (%)",
            yaxis_title="Number of Emails",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_tag_wordcloud(self) -> str:
        """Create a word cloud from tags."""
        # Flatten all tags
        all_tags = []
        for tags in self.df['tags_parsed']:
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        if not all_tags:
            return "No tags available"
        
        # Count tag frequencies
        tag_counts = Counter(all_tags)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate_from_frequencies(tag_counts)
        
        # Convert to base64 for display
        import base64
        import io
        
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return img_str
    
    def create_top_tags_bar_chart(self, top_n: int = 20) -> go.Figure:
        """Create a bar chart showing the most common tags."""
        # Flatten all tags
        all_tags = []
        for tags in self.df['tags_parsed']:
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        if not all_tags:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Top Tags",
                xaxis_title="Tag",
                yaxis_title="Frequency",
                height=400,
                annotations=[dict(
                    text="No tags available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )]
            )
            return fig
        
        # Count tag frequencies
        tag_counts = Counter(all_tags)
        top_tags = dict(tag_counts.most_common(top_n))
        
        fig = go.Figure(data=[go.Bar(
            x=list(top_tags.keys()),
            y=list(top_tags.values()),
            marker_color='#4682B4',
            text=list(top_tags.values()),
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f"Top {top_n} Most Common Tags",
            xaxis_title="Tag",
            yaxis_title="Frequency",
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_kpi_cards(self) -> Dict[str, float]:
        """Calculate key performance indicators."""
        total_emails = len(self.df)
        
        # Sentiment percentages
        sentiment_counts = self.df['sentiment'].value_counts(normalize=True) * 100
        positive_pct = sentiment_counts.get('Positive', 0)
        negative_pct = sentiment_counts.get('Negative', 0)
        
        # Average confidence
        avg_confidence = self.df['confidence'].mean()
        
        # Most common classification
        top_classification = self.df['classification'].mode().iloc[0] if len(self.df) > 0 else "N/A"
        
        # Emails per day based on email_date (more meaningful for business analysis)
        emails_per_day = 0
        if 'email_date' in self.df.columns and not self.df['email_date'].isna().all():
            try:
                # Calculate the date range from email_date
                min_date = self.df['email_date'].min()
                max_date = self.df['email_date'].max()
                date_range_days = (max_date - min_date).days + 1  # +1 to include both start and end dates
                emails_per_day = total_emails / date_range_days if date_range_days > 0 else 0
            except Exception:
                emails_per_day = 0
        
        return {
            'total_emails': total_emails,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'avg_confidence': avg_confidence,
            'top_classification': top_classification,
            'emails_per_day': emails_per_day
        }
    
    def create_classification_trend_chart(self, period: str = 'month') -> go.Figure:
        """Create a line chart showing classification trends over time."""
        if period == 'month':
            time_col = 'month'
        elif period == 'week':
            time_col = 'week'
        else:
            time_col = 'date'
        
        # Group by time period and classification
        classification_trends = self.df.groupby([time_col, 'classification']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        for classification in classification_trends.columns:
            fig.add_trace(go.Scatter(
                x=classification_trends.index.astype(str),
                y=classification_trends[classification],
                mode='lines+markers',
                name=classification,
                line=dict(color=self.CLASSIFICATION_COLORS.get(classification, '#808080'), width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"Email Categories Trends Over Time ({period.capitalize()})",
            xaxis_title="Time Period",
            yaxis_title="Number of Emails",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_confidence_by_classification(self) -> go.Figure:
        """Create a box plot showing confidence scores by classification."""
        fig = go.Figure()
        
        for classification in self.df['classification'].unique():
            data = self.df[self.df['classification'] == classification]['confidence']
            fig.add_trace(go.Box(
                y=data,
                name=classification,
                boxpoints='outliers',
                marker_color=self.CLASSIFICATION_COLORS.get(classification, '#808080')
            ))
        
        fig.update_layout(
            title="Confidence Scores by Categories",
            yaxis_title="Confidence Score (%)",
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_color_legend(self) -> str:
        """Create HTML legend showing the color coding for sentiments and classifications."""
        legend_html = """
        <div style="display: flex; justify-content: space-between; margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
            <div style="flex: 1; margin-right: 20px;">
                <h4 style="margin-bottom: 10px; color: #333;">Sentiment Colors</h4>
        """
        
        for sentiment, color in self.SENTIMENT_COLORS.items():
            legend_html += f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: {color}; border-radius: 3px; margin-right: 10px;"></div>
                    <span style="color: #555;">{sentiment}</span>
                </div>
            """
        
        legend_html += """
            </div>
            <div style="flex: 1;">
                <h4 style="margin-bottom: 10px; color: #333;">Classification Colors</h4>
        """
        
        for classification, color in self.CLASSIFICATION_COLORS.items():
            legend_html += f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: {color}; border-radius: 3px; margin-right: 10px;"></div>
                    <span style="color: #555; font-size: 12px;">{classification}</span>
                </div>
            """
        
        legend_html += """
            </div>
        </div>
        """
        
        return legend_html


def create_dashboard_layout(df: pd.DataFrame) -> Tuple[Dict[str, float], List[go.Figure]]:
    """
    Create all dashboard components.
    
    Args:
        df: DataFrame with email analysis data
        
    Returns:
        Tuple of (kpi_metrics, list_of_figures)
    """
    viz = EmailVisualizations(df)
    
    # Get KPI metrics
    kpi_metrics = viz.create_kpi_cards()
    
    # Create all figures
    figures = {
        'sentiment_pie': viz.create_sentiment_pie_chart(),
        'classification_bar': viz.create_classification_bar_chart(),
        'sentiment_trend': viz.create_sentiment_trend_chart(),
        'classification_heatmap': viz.create_classification_sentiment_heatmap(),
        'confidence_dist': viz.create_confidence_distribution(),
        'top_tags': viz.create_top_tags_bar_chart(),
        'classification_trend': viz.create_classification_trend_chart(),
        'confidence_by_classification': viz.create_confidence_by_classification()
    }
    
    return kpi_metrics, figures 