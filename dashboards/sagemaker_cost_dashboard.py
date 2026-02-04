"""
SageMaker Cost and Usage Dashboard

An interactive Streamlit dashboard for visualizing AWS SageMaker costs
with breakdowns by user/project tags and real-time resource monitoring.

Author: Collinson ML Team

Usage:
    streamlit run dashboards/sagemaker_cost_dashboard.py
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboards.data_fetcher import CostDataFetcher, get_summary_metrics
from scripts.sagemaker_resource_manager.manage_sagemaker_spaces import SageMakerSpaceManager

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="SageMaker Cost Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Custom Styling
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Main container */
    .main {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #0f3460;
    }
    
    .stMetric label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f3460 100%);
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
    }
    
    /* Running indicator */
    .running-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Custom container */
    .dashboard-container {
        padding: 1rem;
        background: rgba(15, 23, 42, 0.3);
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Configuration Loading
# =============================================================================

@st.cache_data(ttl=3600)
def load_config():
    """Load dashboard configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    # Default configuration
    return {
        "region_name": "eu-west-1",
        "domain_id": None,
        "default_months_back": 6,
        "default_tag_key": "Name",
        "available_tag_keys": ["Name", "Project", "Team", "CostCenter", "Environment"],
    }


# =============================================================================
# Data Fetching Functions (with caching)
# =============================================================================

@st.cache_resource
def get_cost_fetcher(region_name: str = None):
    """Get cached Cost Data Fetcher instance."""
    return CostDataFetcher(region_name=region_name)


@st.cache_resource
def get_space_manager(region_name: str = None):
    """Get cached SageMaker Space Manager instance."""
    return SageMakerSpaceManager(region_name=region_name)


@st.cache_data(ttl=300)
def fetch_monthly_costs_cached(start_date: str, end_date: str, region_name: str = None):
    """Fetch monthly costs with caching."""
    fetcher = get_cost_fetcher(region_name)
    return fetcher.fetch_monthly_costs(start_date, end_date)


@st.cache_data(ttl=300)
def fetch_costs_by_tag_cached(
    start_date: str, end_date: str, tag_key: str, region_name: str = None
):
    """Fetch costs by tag with caching."""
    fetcher = get_cost_fetcher(region_name)
    return fetcher.fetch_costs_by_tag(start_date, end_date, tag_key=tag_key)


@st.cache_data(ttl=300)
def fetch_costs_by_usage_type_cached(
    start_date: str, end_date: str, region_name: str = None
):
    """Fetch costs by usage type with caching."""
    fetcher = get_cost_fetcher(region_name)
    return fetcher.fetch_costs_by_usage_type(start_date, end_date)


@st.cache_data(ttl=300)
def fetch_forecast_cached(start_date: str, end_date: str, region_name: str = None):
    """Fetch cost forecast with caching."""
    fetcher = get_cost_fetcher(region_name)
    return fetcher.fetch_cost_forecast(start_date, end_date)


@st.cache_data(ttl=60)
def fetch_running_spaces_cached(domain_id: str, region_name: str = None):
    """Fetch running spaces with caching."""
    manager = get_space_manager(region_name)
    return manager.list_running_spaces(domain_id)


@st.cache_data(ttl=300)
def fetch_all_spaces_with_tags_cached(domain_id: str, region_name: str = None):
    """Fetch all spaces with tags."""
    manager = get_space_manager(region_name)
    return manager.list_spaces_with_tags(domain_id)


@st.cache_data(ttl=300)
def fetch_domains_cached(region_name: str = None):
    """Fetch SageMaker domains."""
    manager = get_space_manager(region_name)
    return manager.list_domains()


# =============================================================================
# Chart Building Functions
# =============================================================================

def create_monthly_cost_chart(
    cost_df: pd.DataFrame,
    forecast_df: pd.DataFrame = None,
    title: str = "Cost and Usage Graph",
) -> go.Figure:
    """
    Create a monthly cost bar chart similar to AWS Cost Explorer.
    
    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame with monthly cost data.
    forecast_df : pd.DataFrame, optional
        DataFrame with forecast data.
    title : str, optional
        Chart title.
    
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    if not cost_df.empty:
        # Aggregate costs by month
        monthly_costs = (
            cost_df.groupby(["time_period_start", "month"])["blended_cost"]
            .sum()
            .reset_index()
            .sort_values("time_period_start")
        )

        # Add actual costs bars
        fig.add_trace(
            go.Bar(
                x=monthly_costs["month"],
                y=monthly_costs["blended_cost"],
                name="Costs",
                marker_color="#6366f1",
                hovertemplate="<b>%{x}</b><br>Cost: $%{y:,.2f}<extra></extra>",
            )
        )

    if forecast_df is not None and not forecast_df.empty:
        # Add forecast bars with different styling
        fig.add_trace(
            go.Bar(
                x=forecast_df["month"],
                y=forecast_df["mean_value"],
                name="Forecast",
                marker_color="rgba(99, 102, 241, 0.3)",
                marker_line_color="#6366f1",
                marker_line_width=2,
                hovertemplate="<b>%{x}</b><br>Forecast: $%{y:,.2f}<extra></extra>",
            )
        )

        # Add prediction interval
        fig.add_trace(
            go.Scatter(
                x=forecast_df["month"].tolist() + forecast_df["month"].tolist()[::-1],
                y=forecast_df["prediction_interval_upper"].tolist()
                + forecast_df["prediction_interval_lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(99, 102, 241, 0.1)",
                line=dict(color="rgba(99, 102, 241, 0.5)", width=1),
                name="80% Prediction Interval",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Space Grotesk, sans-serif"),
        ),
        xaxis_title="",
        yaxis_title="Costs ($)",
        yaxis=dict(
            tickformat="$,.0f",
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=30, t=80, b=40),
        bargap=0.3,
        hovermode="x unified",
    )

    return fig


def create_cost_breakdown_chart(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "blended_cost",
    title: str = "Cost Breakdown",
    top_n: int = 10,
) -> go.Figure:
    """
    Create a horizontal bar chart for cost breakdown.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cost data.
    group_col : str
        Column to group by.
    value_col : str
        Column with values to sum.
    title : str
        Chart title.
    top_n : int
        Number of top items to show.
    
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Aggregate by group
    grouped = df.groupby(group_col)[value_col].sum().reset_index()
    grouped = grouped.sort_values(value_col, ascending=True).tail(top_n)

    # Create color gradient
    colors = px.colors.sequential.Viridis[::-1]
    n_colors = len(grouped)
    color_indices = [int(i * (len(colors) - 1) / max(n_colors - 1, 1)) for i in range(n_colors)]
    bar_colors = [colors[i] for i in color_indices]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=grouped[value_col],
            y=grouped[group_col],
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="<b>%{y}</b><br>Cost: $%{x:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Space Grotesk, sans-serif"),
        ),
        xaxis_title="Cost ($)",
        yaxis_title="",
        xaxis=dict(
            tickformat="$,.0f",
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=200, r=30, t=60, b=40),
        height=400,
    )

    return fig


def create_category_pie_chart(
    df: pd.DataFrame,
    category_col: str = "category",
    value_col: str = "blended_cost",
    title: str = "Cost by Category",
) -> go.Figure:
    """
    Create a pie/donut chart for category breakdown.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cost data.
    category_col : str
        Column with categories.
    value_col : str
        Column with values.
    title : str
        Chart title.
    
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Aggregate by category
    grouped = df.groupby(category_col)[value_col].sum().reset_index()
    grouped = grouped.sort_values(value_col, ascending=False)

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            labels=grouped[category_col],
            values=grouped[value_col],
            hole=0.4,
            marker=dict(
                colors=px.colors.qualitative.Set2,
            ),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Cost: $%{value:,.2f}<br>%{percent}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Space Grotesk, sans-serif"),
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=30, r=30, t=60, b=80),
        height=400,
    )

    return fig


def create_trend_chart(
    df: pd.DataFrame,
    group_col: str,
    title: str = "Cost Trend by Category",
    top_n: int = 5,
) -> go.Figure:
    """
    Create a line chart showing cost trends over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cost data.
    group_col : str
        Column to group by (for multiple lines).
    title : str
        Chart title.
    top_n : int
        Number of top categories to show.
    
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Get top N categories by total cost
    top_categories = (
        df.groupby(group_col)["blended_cost"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    # Filter and pivot
    filtered = df[df[group_col].isin(top_categories)]
    pivoted = filtered.pivot_table(
        index="time_period_start",
        columns=group_col,
        values="blended_cost",
        aggfunc="sum",
    ).fillna(0)

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, col in enumerate(pivoted.columns):
        fig.add_trace(
            go.Scatter(
                x=pivoted.index,
                y=pivoted[col],
                name=col,
                mode="lines+markers",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{col}</b><br>%{{x|%b %Y}}<br>Cost: $%{{y:,.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Space Grotesk, sans-serif"),
        ),
        xaxis_title="",
        yaxis_title="Cost ($)",
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.1)",
            tickformat="%b %Y",
        ),
        yaxis=dict(
            tickformat="$,.0f",
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=30, t=80, b=40),
        hovermode="x unified",
    )

    return fig


# =============================================================================
# Running Resources Display
# =============================================================================

def display_running_resources(running_df: pd.DataFrame):
    """Display running resources in a formatted table."""
    if running_df.empty:
        st.info("✅ No running SageMaker spaces found.")
        return

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span class="running-indicator"></span>
            <span style="font-weight: 600;">{len(running_df)} Active Space(s)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prepare display dataframe
    display_cols = [
        "space_name",
        "space_type",
        "instance_type",
        "owner_user_profile",
        "status",
        "running_app_types",
    ]
    
    available_cols = [col for col in display_cols if col in running_df.columns]
    display_df = running_df[available_cols].copy()

    # Rename columns for display
    column_names = {
        "space_name": "Space Name",
        "space_type": "Type",
        "instance_type": "Instance",
        "owner_user_profile": "Owner",
        "status": "Status",
        "running_app_types": "Running Apps",
    }
    display_df = display_df.rename(columns=column_names)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="Current status of the space",
            ),
        },
    )


# =============================================================================
# Tag Management Section
# =============================================================================

def render_tag_management_section(domain_id: str, region_name: str, config: dict):
    """
    Render the tag management section for SageMaker spaces.
    
    Parameters
    ----------
    domain_id : str
        The SageMaker domain ID.
    region_name : str
        AWS region name.
    config : dict
        Dashboard configuration.
    """
    st.markdown(
        """
        <div class="info-box">
        <strong>ℹ️ Tag Management</strong><br>
        Add or modify tags on your SageMaker JupyterLab and CodeEditor spaces.
        Tags help with cost allocation, organization, and resource management.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize session state for tag management
    if "tag_operation_pending" not in st.session_state:
        st.session_state.tag_operation_pending = False
    if "selected_spaces_for_tagging" not in st.session_state:
        st.session_state.selected_spaces_for_tagging = []
    if "tags_to_apply" not in st.session_state:
        st.session_state.tags_to_apply = {}
    
    # Fetch spaces for the domain
    try:
        all_spaces_df = fetch_all_spaces_with_tags_cached(domain_id, region_name)
        
        if all_spaces_df.empty:
            st.info("No spaces found in this domain.")
            return
            
    except Exception as e:
        st.error(f"Error fetching spaces: {e}")
        return
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        space_type_filter = st.selectbox(
            "Filter by Space Type",
            options=["All", "JupyterLab", "CodeEditor"],
            index=0,
            help="Filter spaces by type",
        )
    
    # Apply filter
    if space_type_filter != "All":
        filtered_spaces = all_spaces_df[
            all_spaces_df["space_type"].str.lower() == space_type_filter.lower()
        ]
    else:
        filtered_spaces = all_spaces_df
    
    if filtered_spaces.empty:
        st.info(f"No {space_type_filter} spaces found.")
        return
    
    # Space selection
    space_options = filtered_spaces["space_name"].tolist()
    
    selected_spaces = st.multiselect(
        "Select Spaces to Tag",
        options=space_options,
        default=[],
        help="Select one or more spaces to add or update tags",
        placeholder="Choose spaces...",
    )
    
    if not selected_spaces:
        st.info("👆 Select one or more spaces above to manage their tags.")
        return
    
    # Show currently selected spaces with their existing tags
    st.markdown("### Selected Spaces")
    
    selected_spaces_df = filtered_spaces[
        filtered_spaces["space_name"].isin(selected_spaces)
    ].copy()
    
    # Display selected spaces
    display_cols = ["space_name", "space_type", "owner_user_profile", "tags"]
    available_cols = [col for col in display_cols if col in selected_spaces_df.columns]
    
    # Format tags for display
    if "tags" in selected_spaces_df.columns:
        selected_spaces_df["tags_display"] = selected_spaces_df["tags"].apply(
            lambda x: ", ".join([f"{k}={v}" for k, v in x.items()]) if isinstance(x, dict) else str(x)
        )
        available_cols = ["space_name", "space_type", "owner_user_profile", "tags_display"]
    
    st.dataframe(
        selected_spaces_df[available_cols].rename(columns={
            "space_name": "Space Name",
            "space_type": "Type", 
            "owner_user_profile": "Owner",
            "tags_display": "Current Tags",
        }),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("---")
    
    # Tag input section
    st.markdown("### Tags to Add/Update")
    
    st.markdown(
        """
        <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 1rem;">
        Enter the tags you want to add or update. Existing tags with the same key will be overwritten.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Predefined tag keys from config
    available_tag_keys = config.get("available_tag_keys", ["Name", "Project", "Team", "CostCenter", "Environment"])
    
    # Dynamic tag input
    num_tags = st.number_input(
        "Number of tags to add",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="How many tags do you want to add?",
    )
    
    tags_to_apply = {}
    
    for i in range(int(num_tags)):
        col_key, col_value = st.columns([1, 2])
        
        with col_key:
            # Allow selection from predefined keys or custom input
            tag_key = st.selectbox(
                f"Tag Key {i+1}",
                options=available_tag_keys + ["Custom..."],
                key=f"tag_key_{i}",
                label_visibility="collapsed" if i > 0 else "visible",
            )
            
            if tag_key == "Custom...":
                tag_key = st.text_input(
                    f"Custom Key {i+1}",
                    key=f"custom_key_{i}",
                    placeholder="Enter custom key",
                )
        
        with col_value:
            tag_value = st.text_input(
                f"Tag Value {i+1}",
                key=f"tag_value_{i}",
                placeholder="Enter tag value",
                label_visibility="collapsed" if i > 0 else "visible",
            )
        
        if tag_key and tag_value and tag_key != "Custom...":
            tags_to_apply[tag_key] = tag_value
    
    if not tags_to_apply:
        st.warning("Please enter at least one tag key-value pair.")
        return
    
    # Preview tags to be applied
    st.markdown("### Preview")
    
    preview_data = [{"Key": k, "Value": v} for k, v in tags_to_apply.items()]
    st.dataframe(
        pd.DataFrame(preview_data),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown(
        f"""
        <div style="background: rgba(99, 102, 241, 0.1); border-radius: 8px; padding: 1rem; margin: 1rem 0;">
        <strong>Summary:</strong> Apply <strong>{len(tags_to_apply)}</strong> tag(s) to 
        <strong>{len(selected_spaces)}</strong> space(s)
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Store in session state for confirmation dialog
    st.session_state.selected_spaces_for_tagging = selected_spaces
    st.session_state.tags_to_apply = tags_to_apply
    
    # Apply button with confirmation
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("🏷️ Apply Tags", type="primary", use_container_width=True):
            st.session_state.tag_operation_pending = True
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Clear Selection", use_container_width=True):
            st.session_state.selected_spaces_for_tagging = []
            st.session_state.tags_to_apply = {}
            st.session_state.tag_operation_pending = False
            st.rerun()
    
    # Confirmation dialog
    if st.session_state.tag_operation_pending:
        render_tag_confirmation_dialog(domain_id, region_name)


@st.dialog("Confirm Tag Operation")
def render_tag_confirmation_dialog(domain_id: str, region_name: str):
    """
    Render confirmation dialog for tag operations.
    
    Parameters
    ----------
    domain_id : str
        The SageMaker domain ID.
    region_name : str
        AWS region name.
    """
    selected_spaces = st.session_state.selected_spaces_for_tagging
    tags_to_apply = st.session_state.tags_to_apply
    
    st.markdown(
        """
        <div style="font-size: 1.1rem; margin-bottom: 1rem;">
        ⚠️ <strong>Are you sure you want to apply these tags?</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("**Spaces to be tagged:**")
    for space in selected_spaces:
        st.markdown(f"- `{space}`")
    
    st.markdown("**Tags to be applied:**")
    for key, value in tags_to_apply.items():
        st.markdown(f"- **{key}**: `{value}`")
    
    st.warning(
        "This action will add or overwrite tags on the selected spaces. "
        "Existing tags with different keys will not be affected."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Confirm & Apply", type="primary", use_container_width=True):
            # Execute the tagging operation
            apply_tags_to_spaces(domain_id, region_name, selected_spaces, tags_to_apply)
            st.session_state.tag_operation_pending = False
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.tag_operation_pending = False
            st.rerun()


def apply_tags_to_spaces(
    domain_id: str,
    region_name: str,
    space_names: list,
    tags: dict,
):
    """
    Apply tags to selected SageMaker spaces.
    
    Parameters
    ----------
    domain_id : str
        The SageMaker domain ID.
    region_name : str
        AWS region name.
    space_names : list
        List of space names to tag.
    tags : dict
        Dictionary of tags to apply.
    """
    manager = get_space_manager(region_name)
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, space_name in enumerate(space_names):
        status_text.text(f"Tagging space: {space_name}...")
        
        try:
            # Get space ARN
            space_arn = manager.get_space_arn(domain_id, space_name)
            
            if space_arn:
                success = manager.add_or_update_tags(space_arn, tags)
                results.append({
                    "space_name": space_name,
                    "success": success,
                    "message": "Tags applied successfully" if success else "Failed to apply tags",
                })
            else:
                results.append({
                    "space_name": space_name,
                    "success": False,
                    "message": "Could not find space ARN",
                })
                
        except Exception as e:
            results.append({
                "space_name": space_name,
                "success": False,
                "message": str(e),
            })
        
        progress_bar.progress((i + 1) / len(space_names))
    
    status_text.empty()
    progress_bar.empty()
    
    # Clear cache to refresh data
    st.cache_data.clear()
    
    # Show results
    success_count = sum(1 for r in results if r["success"])
    
    if success_count == len(results):
        st.success(f"✅ Successfully applied tags to all {len(results)} space(s)!")
    elif success_count > 0:
        st.warning(f"⚠️ Applied tags to {success_count}/{len(results)} space(s). Some operations failed.")
    else:
        st.error("❌ Failed to apply tags to any spaces.")
    
    # Show detailed results
    with st.expander("📋 View Detailed Results"):
        results_df = pd.DataFrame(results)
        results_df["Status"] = results_df["success"].apply(lambda x: "✅ Success" if x else "❌ Failed")
        st.dataframe(
            results_df[["space_name", "Status", "message"]].rename(columns={
                "space_name": "Space Name",
                "message": "Message",
            }),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    """Main dashboard function."""
    
    # Load configuration
    config = load_config()
    
    # ==========================================================================
    # Sidebar
    # ==========================================================================
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # AWS Region
        region_name = st.text_input(
            "AWS Region",
            value=config.get("region_name", "eu-west-1"),
            help="AWS region for Cost Explorer and SageMaker APIs",
        )
        
        st.markdown("---")
        
        # Date Range Selection
        st.markdown("### 📅 Date Range")
        
        fetcher = get_cost_fetcher(region_name)
        default_start, default_end = fetcher.get_default_date_range(
            months_back=config.get("default_months_back", 6)
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.strptime(default_start, "%Y-%m-%d"),
                help="Start date for cost analysis",
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.strptime(default_end, "%Y-%m-%d"),
                help="End date for cost analysis (exclusive)",
            )
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        st.markdown("---")
        
        # Tag Selection for Breakdown
        st.markdown("### 🏷️ Tag Grouping")
        
        tag_key = st.selectbox(
            "Group costs by tag",
            options=config.get("available_tag_keys", ["Name", "Project", "Team"]),
            index=0,
            help="Select a tag key to group costs by",
        )
        
        st.markdown("---")
        
        # Domain Selection for Resources
        st.markdown("### 🖥️ SageMaker Domain")
        
        try:
            domains_df = fetch_domains_cached(region_name)
            if not domains_df.empty:
                domain_options = dict(
                    zip(
                        domains_df["domain_name"],
                        domains_df["domain_id"],
                    )
                )
                selected_domain_name = st.selectbox(
                    "Select Domain",
                    options=list(domain_options.keys()),
                    help="SageMaker Studio domain for resource monitoring",
                )
                domain_id = domain_options.get(selected_domain_name)
            else:
                st.warning("No SageMaker domains found")
                domain_id = st.text_input(
                    "Domain ID",
                    value=config.get("domain_id", ""),
                    help="Enter SageMaker domain ID manually",
                )
        except Exception as e:
            logger.error(f"Error fetching domains: {e}")
            st.warning("Could not fetch domains. Enter ID manually.")
            domain_id = st.text_input(
                "Domain ID",
                value=config.get("domain_id", ""),
                help="Enter SageMaker domain ID manually",
            )
        
        st.markdown("---")
        
        # Refresh Button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94a3b8;">
            <strong>SageMaker Cost Dashboard</strong><br>
            Data refreshes every 5 minutes.<br>
            Running resources refresh every 1 minute.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================================================================
    # Main Content
    # ==========================================================================
    
    st.markdown("# 📊 SageMaker Cost & Usage Dashboard")
    
    st.markdown(
        f"""
        <div style="color: #94a3b8; margin-bottom: 1.5rem;">
        Analyzing costs from <strong>{start_date_str}</strong> to <strong>{end_date_str}</strong> 
        in region <strong>{region_name}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # ==========================================================================
    # Fetch Data
    # ==========================================================================
    
    with st.spinner("Fetching cost data from AWS Cost Explorer..."):
        try:
            # Fetch all cost data
            monthly_costs_df = fetch_monthly_costs_cached(
                start_date_str, end_date_str, region_name
            )
            costs_by_tag_df = fetch_costs_by_tag_cached(
                start_date_str, end_date_str, tag_key, region_name
            )
            costs_by_usage_df = fetch_costs_by_usage_type_cached(
                start_date_str, end_date_str, region_name
            )
            
            # Try to fetch forecast
            forecast_start, forecast_end = fetcher.get_forecast_date_range(months_ahead=1)
            try:
                forecast_df = fetch_forecast_cached(forecast_start, forecast_end, region_name)
            except Exception:
                forecast_df = pd.DataFrame()
            
            data_loaded = True
            
        except Exception as e:
            st.error(f"Error fetching cost data: {e}")
            logger.error(f"Error fetching cost data: {e}")
            data_loaded = False
            monthly_costs_df = pd.DataFrame()
            costs_by_tag_df = pd.DataFrame()
            costs_by_usage_df = pd.DataFrame()
            forecast_df = pd.DataFrame()

    # ==========================================================================
    # Summary Metrics
    # ==========================================================================
    
    if data_loaded and not costs_by_tag_df.empty:
        metrics = get_summary_metrics(costs_by_tag_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Cost",
                value=f"${metrics['total_cost']:,.2f}",
                help="Total SageMaker cost for the selected period",
            )
        
        with col2:
            st.metric(
                label="Avg Monthly Cost",
                value=f"${metrics['avg_monthly_cost']:,.2f}",
                help="Average monthly cost",
            )
        
        with col3:
            st.metric(
                label="Peak Month",
                value=f"${metrics['max_monthly_cost']:,.2f}",
                help="Highest monthly cost in the period",
            )
        
        with col4:
            # Count unique tag values (users/projects)
            if "tag_value" in costs_by_tag_df.columns:
                unique_tags = costs_by_tag_df["tag_value"].nunique()
            else:
                unique_tags = 0
            st.metric(
                label=f"Unique {tag_key}s",
                value=unique_tags,
                help=f"Number of unique {tag_key} tags",
            )

    st.markdown("---")

    # ==========================================================================
    # Monthly Cost Chart
    # ==========================================================================
    
    st.markdown("## Cost and Usage Overview")
    
    if data_loaded and not costs_by_tag_df.empty:
        monthly_chart = create_monthly_cost_chart(
            costs_by_tag_df,
            forecast_df if not forecast_df.empty else None,
            title="Cost and Usage Graph",
        )
        st.plotly_chart(monthly_chart, use_container_width=True)
    else:
        st.info("No cost data available for the selected period.")

    st.markdown("---")

    # ==========================================================================
    # Cost Breakdowns (Two Columns)
    # ==========================================================================
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"## Cost by {tag_key}")
        
        if data_loaded and not costs_by_tag_df.empty:
            tag_chart = create_cost_breakdown_chart(
                costs_by_tag_df,
                group_col="tag_value",
                title=f"Top 10 by {tag_key}",
                top_n=10,
            )
            st.plotly_chart(tag_chart, use_container_width=True)
            
            # Show data table in expander
            with st.expander("📋 View Detailed Data"):
                tag_summary = (
                    costs_by_tag_df.groupby("tag_value")["blended_cost"]
                    .sum()
                    .reset_index()
                    .sort_values("blended_cost", ascending=False)
                    .rename(columns={"tag_value": tag_key, "blended_cost": "Total Cost ($)"})
                )
                tag_summary["Total Cost ($)"] = tag_summary["Total Cost ($)"].apply(
                    lambda x: f"${x:,.2f}"
                )
                st.dataframe(tag_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No tag data available.")
    
    with col_right:
        st.markdown("## Cost by Usage Type")
        
        if data_loaded and not costs_by_usage_df.empty:
            # Create category pie chart
            category_chart = create_category_pie_chart(
                costs_by_usage_df,
                category_col="category",
                title="Cost Distribution by Category",
            )
            st.plotly_chart(category_chart, use_container_width=True)
            
            # Show instance type breakdown in expander
            with st.expander("📋 View Instance Type Details"):
                instance_summary = (
                    costs_by_usage_df.groupby("instance_type")["blended_cost"]
                    .sum()
                    .reset_index()
                    .sort_values("blended_cost", ascending=False)
                    .rename(columns={"instance_type": "Instance Type", "blended_cost": "Total Cost ($)"})
                )
                instance_summary["Total Cost ($)"] = instance_summary["Total Cost ($)"].apply(
                    lambda x: f"${x:,.2f}"
                )
                st.dataframe(instance_summary.head(15), use_container_width=True, hide_index=True)
        else:
            st.info("No usage type data available.")

    st.markdown("---")

    # ==========================================================================
    # Cost Trends
    # ==========================================================================
    
    st.markdown("## Cost Trends Over Time")
    
    tab1, tab2 = st.tabs(["By Category", f"By {tag_key}"])
    
    with tab1:
        if data_loaded and not costs_by_usage_df.empty:
            category_trend = create_trend_chart(
                costs_by_usage_df,
                group_col="category",
                title="Monthly Cost Trend by Category",
                top_n=5,
            )
            st.plotly_chart(category_trend, use_container_width=True)
        else:
            st.info("No trend data available.")
    
    with tab2:
        if data_loaded and not costs_by_tag_df.empty:
            tag_trend = create_trend_chart(
                costs_by_tag_df,
                group_col="tag_value",
                title=f"Monthly Cost Trend by {tag_key}",
                top_n=5,
            )
            st.plotly_chart(tag_trend, use_container_width=True)
        else:
            st.info("No trend data available.")

    st.markdown("---")

    # ==========================================================================
    # Running Resources Panel
    # ==========================================================================
    
    st.markdown("## 🖥️ Running SageMaker Resources")
    
    if domain_id:
        with st.spinner("Fetching running resources..."):
            try:
                running_df = fetch_running_spaces_cached(domain_id, region_name)
                display_running_resources(running_df)
                
                # Additional: Show all spaces with tags
                with st.expander("📋 View All Spaces with Tags"):
                    try:
                        all_spaces_df = fetch_all_spaces_with_tags_cached(domain_id, region_name)
                        if not all_spaces_df.empty:
                            display_cols = [
                                "space_name",
                                "space_type",
                                "instance_type",
                                "owner_user_profile",
                                "status",
                                "tags",
                            ]
                            available_cols = [col for col in display_cols if col in all_spaces_df.columns]
                            st.dataframe(
                                all_spaces_df[available_cols],
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No spaces found in this domain.")
                    except Exception as e:
                        st.warning(f"Could not fetch all spaces: {e}")
                        
            except Exception as e:
                st.error(f"Error fetching running resources: {e}")
                logger.error(f"Error fetching running resources: {e}")
    else:
        st.warning("Please select or enter a SageMaker domain ID to view running resources.")

    # ==========================================================================
    # Tag Management Section
    # ==========================================================================
    
    st.markdown("---")
    st.markdown("## 🏷️ Space Tag Management")
    
    if domain_id:
        render_tag_management_section(domain_id, region_name, config)
    else:
        st.warning("Please select or enter a SageMaker domain ID to manage tags.")

    # ==========================================================================
    # Footer
    # ==========================================================================
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem;">
        SageMaker Cost Dashboard | Built with Streamlit | Data from AWS Cost Explorer
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

