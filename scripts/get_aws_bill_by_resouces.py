import logging
import time

import boto3
import pandas as pd


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    start_time = time.time()

    # aws_billing_client = boto3.client("billing")
    aws_cost_client = boto3.client("ce", verify=False)

    response = aws_cost_client.get_cost_and_usage(
        TimePeriod={
            'Start': "2025-01-01",
            'End': "2025-12-01"
        },
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
            {
                "Type": "TAG",
                "Key": "Resource"
            },
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
        Metrics=[
            "BlendedCost",
            "UnblendedCost",
            "UsageQuantity"
        ],
    )

    # Parse and display results
    for result in response['ResultsByTime']:
        print(f"Time Period: {result['TimePeriod']['Start']} to {result['TimePeriod']['End']}")

        for group in result['Groups']:
            tag_value = group['Keys'][0] if group['Keys'] else 'No tag value'
            blended_cost = group['Metrics']['BlendedCost']['Amount']
            currency = group['Metrics']['BlendedCost']['Unit']

            print(f"  Tag: {tag_value}")
            print(f"    Blended Cost: {blended_cost} {currency}")
            print()


    print(response)

    input("......")

    end_time = time.time()

    logger.info(f"Finished.")
    logger.info(f"Time used = {end_time - start_time} seconds.")
