import logging
from pathlib import Path
import time

import boto3
import pandas as pd
import yaml

project_root_directory = Path(__file__).parent.parent

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

    logger.info("Load config ...")

    with open(project_root_directory / "config/dev.yaml", "r") as f:
        common_config = yaml.safe_load(f)

    # project_tags = common_config["project_tag"]
    # cg_colors = common_config["cg_colors"]

    # aws_billing_client = boto3.client("billing")
    aws_cost_client = boto3.client("ce", verify=False)

    response = aws_cost_client.get_cost_and_usage(
        TimePeriod={"Start": "2025-01-01", "End": "2025-12-01"},
        Granularity="MONTHLY",
        Filter={
            "Dimensions": {
                "Key": "SERVICE",
                "Values": [
                    "Amazon SageMaker",
                ],
                "MatchOptions": ["EQUALS"],
            }
        },
        GroupBy=[
            {"Type": "TAG", "Key": "Resource"},
        ],
        # GroupBy=[
        #     {
        #         "Type": "TAG",
        #         "Key": "Environment"  # Replace with your tag key
        #     },
        #     # You can add multiple tags
        #     # {
        #     #     "Type": "TAG",
        #     #     "Key": "Project"
        #     # },
        # ],
        Metrics=["BlendedCost", "UnblendedCost", "UsageQuantity"],
    )

    # Parse and display results
    for result in response["ResultsByTime"]:
        print(
            f"Time Period: {result['TimePeriod']['Start']} to {result['TimePeriod']['End']}"
        )

        for group in result["Groups"]:
            tag_value = group["Keys"][0] if group["Keys"] else "No tag value"
            blended_cost = group["Metrics"]["BlendedCost"]["Amount"]
            currency = group["Metrics"]["BlendedCost"]["Unit"]

            print(f"  Tag: {tag_value}")
            print(f"    Blended Cost: {blended_cost} {currency}")
            print()

    print(response)

    input("......")

    cost_df = format_cost_response_to_dataframe(response)
    print(cost_df)

    input("......")

    end_time = time.time()

    logger.info(f"Finished.")
    logger.info(f"Time used = {end_time - start_time} seconds.")
