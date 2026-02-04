"""
SageMaker Cost Dashboard Package

This package provides an interactive Streamlit dashboard for visualizing
AWS SageMaker costs and monitoring running resources.

Modules:
    - sagemaker_cost_dashboard: Main Streamlit application
    - data_fetcher: AWS Cost Explorer data fetching utilities

Usage:
    streamlit run dashboards/sagemaker_cost_dashboard.py
"""

from dashboards.data_fetcher import CostDataFetcher, get_summary_metrics

__all__ = ["CostDataFetcher", "get_summary_metrics"]

