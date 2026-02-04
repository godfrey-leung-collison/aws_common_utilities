"""
AWS Cost Explorer Data Fetcher for SageMaker Dashboard

This module provides functions to fetch cost and usage data from AWS Cost Explorer API
for SageMaker services with various grouping options.

Author: Collinson ML Team
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import boto3
from botocore.exceptions import ClientError
import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CostDataFetcher:
    """
    Fetcher class for AWS Cost Explorer data focused on SageMaker services.
    """

    def __init__(self, region_name: Optional[str] = None, verify_ssl: bool = False):
        """
        Initialize the Cost Data Fetcher.

        Parameters
        ----------
        region_name : str, optional
            AWS region name. If not provided, uses default from AWS config.
        verify_ssl : bool, optional
            Whether to verify SSL certificates. Default is False.
        """
        if region_name:
            self.cost_client = boto3.client(
                "ce", region_name=region_name, verify=verify_ssl
            )
        else:
            self.cost_client = boto3.client("ce", verify=verify_ssl)

        self.region_name = region_name or boto3.Session().region_name
        logger.info(f"Cost Explorer client initialized for region: {self.region_name}")

    def _format_cost_response_to_dataframe(self, response: dict) -> pd.DataFrame:
        """
        Convert AWS Cost Explorer response to pandas DataFrame.

        Parameters
        ----------
        response : dict
            The response from AWS Cost Explorer get_cost_and_usage API.

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with cost data.
        """
        rows = []

        for result in response.get("ResultsByTime", []):
            time_period_start = result["TimePeriod"]["Start"]
            time_period_end = result["TimePeriod"]["End"]

            for group in result.get("Groups", []):
                # Extract group key(s)
                keys = group.get("Keys", [])
                group_key = keys[0] if keys else "Ungrouped"

                # Extract metrics
                metrics = group.get("Metrics", {})
                blended_cost = float(metrics.get("BlendedCost", {}).get("Amount", 0))
                unblended_cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0))
                usage_quantity = float(metrics.get("UsageQuantity", {}).get("Amount", 0))
                currency = metrics.get("BlendedCost", {}).get("Unit", "USD")

                rows.append(
                    {
                        "time_period_start": time_period_start,
                        "time_period_end": time_period_end,
                        "group_key": group_key,
                        "blended_cost": blended_cost,
                        "unblended_cost": unblended_cost,
                        "usage_quantity": usage_quantity,
                        "currency": currency,
                    }
                )

            # Handle ungrouped totals if no groups
            if not result.get("Groups"):
                metrics = result.get("Total", {})
                if metrics:
                    rows.append(
                        {
                            "time_period_start": time_period_start,
                            "time_period_end": time_period_end,
                            "group_key": "Total",
                            "blended_cost": float(metrics.get("BlendedCost", {}).get("Amount", 0)),
                            "unblended_cost": float(metrics.get("UnblendedCost", {}).get("Amount", 0)),
                            "usage_quantity": float(metrics.get("UsageQuantity", {}).get("Amount", 0)),
                            "currency": metrics.get("BlendedCost", {}).get("Unit", "USD"),
                        }
                    )

        return pd.DataFrame(rows)

    def fetch_monthly_costs(
        self,
        start_date: str,
        end_date: str,
        service_filter: str = "Amazon SageMaker",
    ) -> pd.DataFrame:
        """
        Fetch monthly SageMaker costs aggregated by month.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format (inclusive).
        end_date : str
            End date in YYYY-MM-DD format (exclusive).
        service_filter : str, optional
            AWS service to filter by. Default is "Amazon SageMaker".

        Returns
        -------
        pd.DataFrame
            DataFrame with monthly cost data.
        """
        logger.info(f"Fetching monthly costs from {start_date} to {end_date}")

        try:
            response = self.cost_client.get_cost_and_usage(
                TimePeriod={"Start": start_date, "End": end_date},
                Granularity="MONTHLY",
                Filter={
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": [service_filter],
                        "MatchOptions": ["EQUALS"],
                    }
                },
                Metrics=["BlendedCost", "UnblendedCost", "UsageQuantity"],
            )

            df = self._format_cost_response_to_dataframe(response)
            
            # Convert dates
            if not df.empty:
                df["time_period_start"] = pd.to_datetime(df["time_period_start"])
                df["month"] = df["time_period_start"].dt.strftime("%b %Y")
            
            logger.info(f"Retrieved {len(df)} monthly cost records")
            return df

        except ClientError as e:
            logger.error(f"Error fetching monthly costs: {e}")
            raise

    def fetch_costs_by_tag(
        self,
        start_date: str,
        end_date: str,
        tag_key: str = "Name",
        service_filter: str = "Amazon SageMaker",
        granularity: str = "MONTHLY",
    ) -> pd.DataFrame:
        """
        Fetch SageMaker costs grouped by a specific tag.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format (inclusive).
        end_date : str
            End date in YYYY-MM-DD format (exclusive).
        tag_key : str, optional
            Tag key to group by. Default is "Name".
        service_filter : str, optional
            AWS service to filter by. Default is "Amazon SageMaker".
        granularity : str, optional
            Time granularity: "DAILY", "MONTHLY", "HOURLY". Default is "MONTHLY".

        Returns
        -------
        pd.DataFrame
            DataFrame with cost data grouped by tag.
        """
        logger.info(f"Fetching costs by tag '{tag_key}' from {start_date} to {end_date}")

        try:
            response = self.cost_client.get_cost_and_usage(
                TimePeriod={"Start": start_date, "End": end_date},
                Granularity=granularity,
                Filter={
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": [service_filter],
                        "MatchOptions": ["EQUALS"],
                    }
                },
                GroupBy=[{"Type": "TAG", "Key": tag_key}],
                Metrics=["BlendedCost", "UnblendedCost", "UsageQuantity"],
            )

            df = self._format_cost_response_to_dataframe(response)

            if not df.empty:
                df["time_period_start"] = pd.to_datetime(df["time_period_start"])
                df["month"] = df["time_period_start"].dt.strftime("%b %Y")
                
                # Clean up tag values (remove tag key prefix)
                df["tag_value"] = df["group_key"].apply(
                    lambda x: x.replace(f"{tag_key}$", "") if f"{tag_key}$" in x else x
                )
                # Replace empty tag values with descriptive name
                df["tag_value"] = df["tag_value"].replace("", "Untagged")

            logger.info(f"Retrieved {len(df)} cost records by tag")
            return df

        except ClientError as e:
            logger.error(f"Error fetching costs by tag: {e}")
            raise

    def fetch_costs_by_usage_type(
        self,
        start_date: str,
        end_date: str,
        service_filter: str = "Amazon SageMaker",
        granularity: str = "MONTHLY",
    ) -> pd.DataFrame:
        """
        Fetch SageMaker costs grouped by usage type (instance types, etc.).

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format (inclusive).
        end_date : str
            End date in YYYY-MM-DD format (exclusive).
        service_filter : str, optional
            AWS service to filter by. Default is "Amazon SageMaker".
        granularity : str, optional
            Time granularity: "DAILY", "MONTHLY", "HOURLY". Default is "MONTHLY".

        Returns
        -------
        pd.DataFrame
            DataFrame with cost data grouped by usage type.
        """
        logger.info(f"Fetching costs by usage type from {start_date} to {end_date}")

        try:
            response = self.cost_client.get_cost_and_usage(
                TimePeriod={"Start": start_date, "End": end_date},
                Granularity=granularity,
                Filter={
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": [service_filter],
                        "MatchOptions": ["EQUALS"],
                    }
                },
                GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
                Metrics=["BlendedCost", "UnblendedCost", "UsageQuantity"],
            )

            df = self._format_cost_response_to_dataframe(response)

            if not df.empty:
                df["time_period_start"] = pd.to_datetime(df["time_period_start"])
                df["month"] = df["time_period_start"].dt.strftime("%b %Y")
                df["usage_type"] = df["group_key"]
                
                # Parse usage type to extract category and instance type
                df["category"] = df["usage_type"].apply(self._extract_category)
                df["instance_type"] = df["usage_type"].apply(self._extract_instance_type)

            logger.info(f"Retrieved {len(df)} cost records by usage type")
            return df

        except ClientError as e:
            logger.error(f"Error fetching costs by usage type: {e}")
            raise

    def _extract_category(self, usage_type: str) -> str:
        """
        Extract the category from a usage type string.
        
        Examples:
        - "EU-Studio:JupyterLab-ml.t3.medium" -> "Studio"
        - "EU-Processing:ml.m5.large" -> "Processing"
        - "EU-Train:ml.m5.large" -> "Training"
        """
        if "Studio" in usage_type:
            return "Studio"
        elif "Processing" in usage_type:
            return "Processing"
        elif "Train" in usage_type:
            return "Training"
        elif "Notebk" in usage_type:
            return "Notebook"
        elif "TensorBoard" in usage_type:
            return "TensorBoard"
        else:
            return "Other"

    def _extract_instance_type(self, usage_type: str) -> str:
        """
        Extract the instance type from a usage type string.
        
        Examples:
        - "EU-Studio:JupyterLab-ml.t3.medium" -> "ml.t3.medium"
        - "EU-Studio:VolumeUsage.gp3" -> "VolumeUsage.gp3"
        """
        # Try to find ml.* pattern
        if "-ml." in usage_type:
            parts = usage_type.split("-ml.")
            if len(parts) > 1:
                return "ml." + parts[-1]
        
        # Try to find VolumeUsage pattern
        if "VolumeUsage" in usage_type:
            parts = usage_type.split(":")
            if len(parts) > 1:
                return parts[-1]
        
        # Return the part after the colon
        if ":" in usage_type:
            return usage_type.split(":")[-1]
        
        return usage_type

    def fetch_cost_forecast(
        self,
        start_date: str,
        end_date: str,
        service_filter: str = "Amazon SageMaker",
        granularity: str = "MONTHLY",
    ) -> pd.DataFrame:
        """
        Fetch cost forecast for SageMaker services.

        Parameters
        ----------
        start_date : str
            Start date for forecast in YYYY-MM-DD format.
        end_date : str
            End date for forecast in YYYY-MM-DD format.
        service_filter : str, optional
            AWS service to filter by. Default is "Amazon SageMaker".
        granularity : str, optional
            Time granularity: "DAILY", "MONTHLY". Default is "MONTHLY".

        Returns
        -------
        pd.DataFrame
            DataFrame with forecast data including prediction intervals.
        """
        logger.info(f"Fetching cost forecast from {start_date} to {end_date}")

        try:
            response = self.cost_client.get_cost_forecast(
                TimePeriod={"Start": start_date, "End": end_date},
                Granularity=granularity,
                Metric="UNBLENDED_COST",
                Filter={
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": [service_filter],
                        "MatchOptions": ["EQUALS"],
                    }
                },
                PredictionIntervalLevel=80,
            )

            rows = []
            for forecast in response.get("ForecastResultsByTime", []):
                rows.append(
                    {
                        "time_period_start": forecast["TimePeriod"]["Start"],
                        "time_period_end": forecast["TimePeriod"]["End"],
                        "mean_value": float(forecast.get("MeanValue", 0)),
                        "prediction_interval_lower": float(
                            forecast.get("PredictionIntervalLowerBound", 0)
                        ),
                        "prediction_interval_upper": float(
                            forecast.get("PredictionIntervalUpperBound", 0)
                        ),
                    }
                )

            df = pd.DataFrame(rows)
            
            if not df.empty:
                df["time_period_start"] = pd.to_datetime(df["time_period_start"])
                df["month"] = df["time_period_start"].dt.strftime("%b %Y")

            logger.info(f"Retrieved {len(df)} forecast records")
            return df

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "DataUnavailableException":
                logger.warning("Forecast data not available - insufficient historical data")
                return pd.DataFrame()
            logger.error(f"Error fetching cost forecast: {e}")
            raise

    def get_default_date_range(self, months_back: int = 6) -> tuple:
        """
        Get default date range for queries.

        Parameters
        ----------
        months_back : int, optional
            Number of months to look back. Default is 6.

        Returns
        -------
        tuple
            (start_date, end_date) as strings in YYYY-MM-DD format.
        """
        today = datetime.now()
        
        # End date is first of current month (exclusive)
        end_date = today.replace(day=1)
        
        # Start date is first of month N months ago
        start_date = end_date
        for _ in range(months_back):
            start_date = (start_date - timedelta(days=1)).replace(day=1)

        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def get_forecast_date_range(self, months_ahead: int = 1) -> tuple:
        """
        Get date range for forecast queries.

        Parameters
        ----------
        months_ahead : int, optional
            Number of months to forecast ahead. Default is 1.

        Returns
        -------
        tuple
            (start_date, end_date) as strings in YYYY-MM-DD format.
        """
        today = datetime.now()
        
        # Start date is first of current month
        start_date = today.replace(day=1)
        
        # End date is first of month N months ahead
        end_date = start_date
        for _ in range(months_ahead + 1):
            # Move to next month
            if end_date.month == 12:
                end_date = end_date.replace(year=end_date.year + 1, month=1)
            else:
                end_date = end_date.replace(month=end_date.month + 1)

        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_summary_metrics(cost_df: pd.DataFrame) -> dict:
    """
    Calculate summary metrics from cost data.

    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame with cost data.

    Returns
    -------
    dict
        Dictionary with summary metrics.
    """
    if cost_df.empty:
        return {
            "total_cost": 0,
            "avg_monthly_cost": 0,
            "max_monthly_cost": 0,
            "min_monthly_cost": 0,
            "total_months": 0,
        }

    monthly_totals = cost_df.groupby("month")["blended_cost"].sum()

    return {
        "total_cost": cost_df["blended_cost"].sum(),
        "avg_monthly_cost": monthly_totals.mean(),
        "max_monthly_cost": monthly_totals.max(),
        "min_monthly_cost": monthly_totals.min(),
        "total_months": len(monthly_totals),
    }

