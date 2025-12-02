import logging
from pathlib import Path
import time

import boto3
import pandas as pd
import yaml

project_root_directory = Path(__file__).parent.parent.parent

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def format_cost_response_to_dataframe(response: dict) -> pd.DataFrame:
    """
    Convert AWS Cost Explorer response to pandas DataFrame

    Parameters
    ----------
    response : dict
        The response from AWS Cost Explorer get_cost_and_usage API.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with columns for time period, tag value, costs, usage quantity, and currency.

    """
    rows = []

    for result in response["ResultsByTime"]:
        time_period_start = result["TimePeriod"]["Start"]
        time_period_end = result["TimePeriod"]["End"]

        for group in result["Groups"]:
            # Extract tag value (first key in the Keys list)
            tag_value = group["Keys"][0] if group["Keys"] else "No tag value"

            # Extract metrics
            blended_cost = float(group["Metrics"]["BlendedCost"]["Amount"])
            unblended_cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
            usage_quantity = float(group["Metrics"]["UsageQuantity"]["Amount"])
            currency = group["Metrics"]["BlendedCost"]["Unit"]

            rows.append(
                {
                    "time_period_start": time_period_start,
                    "time_period_end": time_period_end,
                    "tag_value": tag_value,
                    "blended_cost": blended_cost,
                    "unblended_cost": unblended_cost,
                    "usage_quantity": usage_quantity,
                    "currency": currency,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":

    start_time = time.time()

    logger.info("Load config for cost analysis and breakdown ...")

    with open(project_root_directory / "config/cost_monitor.yaml", "r") as f:
        config = yaml.safe_load(f)

    start_date = config["start_date"]
    end_date = config["end_date"]

    granularity = config["Granularity"]
    metrics_to_report = config["metrics"]
    filter_to_apply = config["Filter"]
    groupby_to_apply = config["GroupBy"]

    logger.info("Config loaded.")

    logger.info("Connecting to the AWS Cost Explorer API via boto3 ...")

    aws_cost_client = boto3.client("ce", verify=False)

    response = aws_cost_client.get_cost_and_usage(
        TimePeriod={
            # "Start": "2025-01-01", "End": "2025-12-01"
            "Start": start_date, "End": end_date,
        },
        Granularity=granularity,
        Filter=filter_to_apply,
        GroupBy=groupby_to_apply,
        Metrics=metrics_to_report,
    )

    # TO REMOVE: for inspection/debugging only
    print(response)
    input("Press to continue ......")

    cost_df = format_cost_response_to_dataframe(response)

    # TO REMOVE: for inspection/debugging only
    print(cost_df)
    input("Press to continue ......")

    logger.info("Cost breakdown report extracted.")

    logger.info("Exporting the cost breakdown report to local CSV file ...")

    export_relative_filepath = config["output_filepath_prefix"]

    relative_filepath = f"{export_relative_filepath}_{start_date}_to_{end_date}.csv"
    relative_file_dir = project_root_directory / Path(relative_filepath).parent

    if not relative_file_dir.exists():
        relative_file_dir.mkdir(parents=True, exist_ok=True)

    cost_df.to_csv(
        project_root_directory / relative_filepath,
        index=False,
    )

    end_time = time.time()

    logger.info(f"Finished.")
    logger.info(f"Time used = {end_time - start_time} seconds.")
