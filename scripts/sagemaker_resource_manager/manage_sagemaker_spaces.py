"""
SageMaker Space Resource Manager

This module provides functionality to:
1. List SageMaker JupyterLab and Code Editor spaces/instances
2. Query and display current resource tags
3. Add or update resource tags on SageMaker spaces
4. Stop running SageMaker spaces/apps

Author: Collinson ML Team
"""

import logging
from pathlib import Path
from typing import Optional
import time

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import yaml

project_root_directory = Path(__file__).parent.parent.parent

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SageMakerSpaceManager:
    """
    Manager class for SageMaker Studio spaces (JupyterLab and Code Editor).

    This class provides methods to list, inspect, and tag SageMaker spaces
    within a specified domain.
    """

    def __init__(self, region_name: Optional[str] = None, verify_ssl: bool = False):
        """
        Initialize the SageMaker Space Manager.

        Parameters
        ----------
        region_name : str, optional
            AWS region name. If not provided, uses default from AWS config.
        verify_ssl : bool, optional
            Whether to verify SSL certificates. Default is False.
        """
        if region_name:
            self.sagemaker_client = boto3.client(
                "sagemaker", region_name=region_name, verify=verify_ssl
            )
        else:
            self.sagemaker_client = boto3.client("sagemaker", verify=verify_ssl)

        self.region_name = region_name or boto3.Session().region_name
        logger.info(f"SageMaker client initialized for region: {self.region_name}")

    def list_domains(self) -> pd.DataFrame:
        """
        List all SageMaker Studio domains in the account.

        Returns
        -------
        pd.DataFrame
            DataFrame containing domain information.
        """
        logger.info("Listing SageMaker Studio domains...")

        domains = []
        paginator = self.sagemaker_client.get_paginator("list_domains")

        for page in paginator.paginate():
            for domain in page.get("Domains", []):
                domains.append(
                    {
                        "domain_id": domain.get("DomainId"),
                        "domain_name": domain.get("DomainName"),
                        "status": domain.get("Status"),
                        "creation_time": domain.get("CreationTime"),
                        "last_modified_time": domain.get("LastModifiedTime"),
                        "url": domain.get("Url"),
                    }
                )

        logger.info(f"Found {len(domains)} domain(s).")
        return pd.DataFrame(domains)

    def list_spaces(
        self,
        domain_id: str,
        space_type_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        List all spaces in a SageMaker Studio domain.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_type_filter : str, optional
            Filter by space type: 'JupyterLab', 'CodeEditor', or None for all.

        Returns
        -------
        pd.DataFrame
            DataFrame containing space information.
        """
        logger.info(f"Listing spaces in domain: {domain_id}")

        spaces = []
        paginator = self.sagemaker_client.get_paginator("list_spaces")

        try:
            for page in paginator.paginate(DomainIdEquals=domain_id):
                for space in page.get("Spaces", []):
                    space_name = space.get("SpaceName")

                    # Get detailed space information
                    space_details = self._describe_space(domain_id, space_name)

                    space_info = {
                        "domain_id": domain_id,
                        "space_name": space_name,
                        "status": space.get("Status"),
                        "creation_time": space.get("CreationTime"),
                        "last_modified_time": space.get("LastModifiedTime"),
                        "space_type": space_details.get("space_type", "Unknown"),
                        "instance_type": space_details.get("instance_type"),
                        "space_arn": space_details.get("space_arn"),
                        "owner_user_profile": space_details.get("ownership_settings", {}).get(
                            "OwnerUserProfileName"
                        ),
                    }

                    # Apply space type filter if specified
                    if space_type_filter:
                        if space_info["space_type"].lower() == space_type_filter.lower():
                            spaces.append(space_info)
                    else:
                        spaces.append(space_info)

        except ClientError as e:
            logger.error(f"Error listing spaces: {e}")
            raise

        logger.info(f"Found {len(spaces)} space(s).")
        return pd.DataFrame(spaces)

    def _describe_space(self, domain_id: str, space_name: str) -> dict:
        """
        Get detailed information about a specific space.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_name : str
            The name of the space.

        Returns
        -------
        dict
            Dictionary containing space details.
        """
        try:
            response = self.sagemaker_client.describe_space(
                DomainId=domain_id, SpaceName=space_name
            )

            space_settings = response.get("SpaceSettings", {})

            # Determine space type based on settings
            space_type = "Unknown"
            instance_type = None

            if "JupyterLabAppSettings" in space_settings:
                space_type = "JupyterLab"
                app_settings = space_settings.get("JupyterLabAppSettings", {})
                default_resource = app_settings.get("DefaultResourceSpec", {})
                instance_type = default_resource.get("InstanceType")

            elif "CodeEditorAppSettings" in space_settings:
                space_type = "CodeEditor"
                app_settings = space_settings.get("CodeEditorAppSettings", {})
                default_resource = app_settings.get("DefaultResourceSpec", {})
                instance_type = default_resource.get("InstanceType")

            elif "JupyterServerAppSettings" in space_settings:
                space_type = "JupyterServer"
                app_settings = space_settings.get("JupyterServerAppSettings", {})
                default_resource = app_settings.get("DefaultResourceSpec", {})
                instance_type = default_resource.get("InstanceType")

            elif "KernelGatewayAppSettings" in space_settings:
                space_type = "KernelGateway"
                app_settings = space_settings.get("KernelGatewayAppSettings", {})
                default_resource = app_settings.get("DefaultResourceSpec", {})
                instance_type = default_resource.get("InstanceType")

            # Check SpaceDisplayName for additional hints
            space_display_name = response.get("SpaceDisplayName", "")
            if "jupyterlab" in space_display_name.lower():
                space_type = "JupyterLab"
            elif "code-editor" in space_display_name.lower() or "codeeditor" in space_display_name.lower():
                space_type = "CodeEditor"

            return {
                "space_arn": response.get("SpaceArn"),
                "space_type": space_type,
                "instance_type": instance_type,
                "ownership_settings": response.get("OwnershipSettings", {}),
                "space_sharing_settings": response.get("SpaceSharingSettings", {}),
                "space_display_name": space_display_name,
            }

        except ClientError as e:
            logger.warning(f"Could not describe space {space_name}: {e}")
            return {}

    def list_apps(
        self,
        domain_id: str,
        space_name: Optional[str] = None,
        app_type_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        List all apps (running instances) in a domain or space.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_name : str, optional
            Filter by specific space name.
        app_type_filter : str, optional
            Filter by app type: 'JupyterLab', 'CodeEditor', 'JupyterServer', 'KernelGateway'.

        Returns
        -------
        pd.DataFrame
            DataFrame containing app information.
        """
        logger.info(f"Listing apps in domain: {domain_id}")

        apps = []
        paginator = self.sagemaker_client.get_paginator("list_apps")

        paginate_kwargs = {"DomainIdEquals": domain_id}
        if space_name:
            paginate_kwargs["SpaceNameEquals"] = space_name

        try:
            for page in paginator.paginate(**paginate_kwargs):
                for app in page.get("Apps", []):
                    app_type = app.get("AppType", "")

                    # Apply app type filter
                    if app_type_filter and app_type.lower() != app_type_filter.lower():
                        continue

                    apps.append(
                        {
                            "domain_id": domain_id,
                            "space_name": app.get("SpaceName"),
                            "user_profile_name": app.get("UserProfileName"),
                            "app_type": app_type,
                            "app_name": app.get("AppName"),
                            "status": app.get("Status"),
                            "creation_time": app.get("CreationTime"),
                            "resource_spec": app.get("ResourceSpec", {}),
                        }
                    )

        except ClientError as e:
            logger.error(f"Error listing apps: {e}")
            raise

        logger.info(f"Found {len(apps)} app(s).")
        return pd.DataFrame(apps)

    def get_resource_tags(self, resource_arn: str) -> dict:
        """
        Get tags for a SageMaker resource.

        Parameters
        ----------
        resource_arn : str
            The ARN of the SageMaker resource.

        Returns
        -------
        dict
            Dictionary of tag key-value pairs.
        """
        try:
            response = self.sagemaker_client.list_tags(ResourceArn=resource_arn)
            tags = {tag["Key"]: tag["Value"] for tag in response.get("Tags", [])}
            logger.info(f"Retrieved {len(tags)} tag(s) for resource: {resource_arn}")
            return tags

        except ClientError as e:
            logger.error(f"Error getting tags for {resource_arn}: {e}")
            raise

    def add_or_update_tags(self, resource_arn: str, tags: dict) -> bool:
        """
        Add or update tags on a SageMaker resource.

        Parameters
        ----------
        resource_arn : str
            The ARN of the SageMaker resource.
        tags : dict
            Dictionary of tag key-value pairs to add/update.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]

            self.sagemaker_client.add_tags(ResourceArn=resource_arn, Tags=tag_list)

            logger.info(f"Successfully added/updated {len(tags)} tag(s) on: {resource_arn}")
            return True

        except ClientError as e:
            logger.error(f"Error adding tags to {resource_arn}: {e}")
            return False

    def delete_tags(self, resource_arn: str, tag_keys: list) -> bool:
        """
        Delete tags from a SageMaker resource.

        Parameters
        ----------
        resource_arn : str
            The ARN of the SageMaker resource.
        tag_keys : list
            List of tag keys to delete.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            self.sagemaker_client.delete_tags(ResourceArn=resource_arn, TagKeys=tag_keys)
            logger.info(f"Successfully deleted {len(tag_keys)} tag(s) from: {resource_arn}")
            return True

        except ClientError as e:
            logger.error(f"Error deleting tags from {resource_arn}: {e}")
            return False

    def bulk_tag_spaces(
        self,
        domain_id: str,
        tags: dict,
        space_type_filter: Optional[str] = None,
        space_names: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Add tags to multiple spaces in bulk.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        tags : dict
            Dictionary of tag key-value pairs to add.
        space_type_filter : str, optional
            Filter by space type: 'JupyterLab', 'CodeEditor', or None for all.
        space_names : list, optional
            Specific space names to tag. If None, tags all spaces (optionally filtered by type).

        Returns
        -------
        pd.DataFrame
            DataFrame with tagging results.
        """
        logger.info(f"Starting bulk tagging for domain: {domain_id}")

        # Get spaces to tag
        spaces_df = self.list_spaces(domain_id, space_type_filter)

        if space_names:
            spaces_df = spaces_df[spaces_df["space_name"].isin(space_names)]

        results = []
        for _, space in spaces_df.iterrows():
            space_arn = space["space_arn"]
            space_name = space["space_name"]

            if not space_arn:
                logger.warning(f"No ARN found for space: {space_name}, skipping...")
                results.append(
                    {
                        "space_name": space_name,
                        "space_arn": None,
                        "success": False,
                        "error": "No ARN found",
                    }
                )
                continue

            success = self.add_or_update_tags(space_arn, tags)
            results.append(
                {
                    "space_name": space_name,
                    "space_arn": space_arn,
                    "success": success,
                    "error": None if success else "Failed to add tags",
                }
            )

        logger.info(f"Bulk tagging complete. Tagged {sum(r['success'] for r in results)}/{len(results)} spaces.")
        return pd.DataFrame(results)

    def get_space_arn(self, domain_id: str, space_name: str) -> Optional[str]:
        """
        Get the ARN for a specific space.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_name : str
            The name of the space.

        Returns
        -------
        str or None
            The space ARN, or None if not found.
        """
        space_details = self._describe_space(domain_id, space_name)
        return space_details.get("space_arn")

    def list_spaces_with_tags(
        self,
        domain_id: str,
        space_type_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        List all spaces with their current tags.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_type_filter : str, optional
            Filter by space type.

        Returns
        -------
        pd.DataFrame
            DataFrame containing space information with tags.
        """
        spaces_df = self.list_spaces(domain_id, space_type_filter)

        tags_list = []
        for _, space in spaces_df.iterrows():
            space_arn = space["space_arn"]
            if space_arn:
                try:
                    tags = self.get_resource_tags(space_arn)
                    tags_list.append(tags)
                except Exception:
                    tags_list.append({})
            else:
                tags_list.append({})

        spaces_df["tags"] = tags_list
        return spaces_df

    # =========================================================================
    # Stop/Delete App Methods
    # =========================================================================

    def stop_app(
        self,
        domain_id: str,
        app_name: str,
        app_type: str,
        space_name: Optional[str] = None,
        user_profile_name: Optional[str] = None,
    ) -> bool:
        """
        Stop (delete) a running SageMaker app.

        Note: In SageMaker, stopping a space means deleting its running app.
        The space itself remains and can be restarted later.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        app_name : str
            The name of the app to stop.
        app_type : str
            The type of app: 'JupyterLab', 'CodeEditor', 'JupyterServer', 'KernelGateway'.
        space_name : str, optional
            The name of the space (for space-based apps).
        user_profile_name : str, optional
            The user profile name (for user-profile-based apps).

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            delete_kwargs = {
                "DomainId": domain_id,
                "AppType": app_type,
                "AppName": app_name,
            }

            if space_name:
                delete_kwargs["SpaceName"] = space_name
            elif user_profile_name:
                delete_kwargs["UserProfileName"] = user_profile_name
            else:
                logger.error("Either space_name or user_profile_name must be provided.")
                return False

            self.sagemaker_client.delete_app(**delete_kwargs)

            logger.info(
                f"Successfully initiated stop for app: {app_name} "
                f"(type: {app_type}, space: {space_name or user_profile_name})"
            )
            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFound":
                logger.warning(f"App {app_name} not found or already stopped.")
                return True  # Consider this a success
            logger.error(f"Error stopping app {app_name}: {e}")
            return False

    def stop_space(
        self,
        domain_id: str,
        space_name: str,
        wait_for_deletion: bool = False,
        max_wait_seconds: int = 300,
    ) -> dict:
        """
        Stop all running apps in a SageMaker space.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_name : str
            The name of the space to stop.
        wait_for_deletion : bool, optional
            Whether to wait for the apps to be fully deleted. Default is False.
        max_wait_seconds : int, optional
            Maximum time to wait for deletion (if wait_for_deletion is True). Default is 300.

        Returns
        -------
        dict
            Dictionary with stop results including success status and stopped apps.
        """
        logger.info(f"Stopping space: {space_name}")

        # Get all running apps for this space
        apps_df = self.list_apps(domain_id, space_name=space_name)

        if apps_df.empty:
            logger.info(f"No running apps found in space: {space_name}")
            return {
                "space_name": space_name,
                "success": True,
                "apps_stopped": 0,
                "message": "No running apps found",
            }

        # Filter for only running/pending apps
        running_statuses = ["InService", "Pending"]
        running_apps = apps_df[apps_df["status"].isin(running_statuses)]

        if running_apps.empty:
            logger.info(f"No apps in running state for space: {space_name}")
            return {
                "space_name": space_name,
                "success": True,
                "apps_stopped": 0,
                "message": "No apps in running state",
            }

        stopped_apps = []
        failed_apps = []

        for _, app in running_apps.iterrows():
            app_name = app["app_name"]
            app_type = app["app_type"]

            success = self.stop_app(
                domain_id=domain_id,
                app_name=app_name,
                app_type=app_type,
                space_name=space_name,
            )

            if success:
                stopped_apps.append({"app_name": app_name, "app_type": app_type})
            else:
                failed_apps.append({"app_name": app_name, "app_type": app_type})

        # Optionally wait for deletion
        if wait_for_deletion and stopped_apps:
            logger.info(f"Waiting for apps to stop (max {max_wait_seconds}s)...")
            self._wait_for_apps_deletion(
                domain_id, space_name, max_wait_seconds
            )

        return {
            "space_name": space_name,
            "success": len(failed_apps) == 0,
            "apps_stopped": len(stopped_apps),
            "apps_failed": len(failed_apps),
            "stopped_apps": stopped_apps,
            "failed_apps": failed_apps,
        }

    def _wait_for_apps_deletion(
        self,
        domain_id: str,
        space_name: str,
        max_wait_seconds: int = 300,
        poll_interval: int = 10,
    ) -> bool:
        """
        Wait for all apps in a space to be deleted.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_name : str
            The name of the space.
        max_wait_seconds : int, optional
            Maximum time to wait. Default is 300.
        poll_interval : int, optional
            Time between status checks in seconds. Default is 10.

        Returns
        -------
        bool
            True if all apps are deleted, False if timeout.
        """
        elapsed = 0

        while elapsed < max_wait_seconds:
            apps_df = self.list_apps(domain_id, space_name=space_name)

            # Check if any apps are still running or deleting
            active_statuses = ["InService", "Pending", "Deleting"]
            if apps_df.empty or not apps_df["status"].isin(active_statuses).any():
                logger.info(f"All apps in space {space_name} have stopped.")
                return True

            active_count = apps_df["status"].isin(active_statuses).sum()
            logger.info(f"Waiting... {active_count} app(s) still active. Elapsed: {elapsed}s")

            time.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning(f"Timeout waiting for apps to stop in space: {space_name}")
        return False

    def bulk_stop_spaces(
        self,
        domain_id: str,
        space_names: Optional[list] = None,
        space_type_filter: Optional[str] = None,
        stop_all_running: bool = False,
        wait_for_deletion: bool = False,
    ) -> pd.DataFrame:
        """
        Stop multiple SageMaker spaces in bulk.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_names : list, optional
            Specific space names to stop. If None and stop_all_running is True,
            stops all spaces with running apps.
        space_type_filter : str, optional
            Filter by space type: 'JupyterLab', 'CodeEditor', or None for all.
        stop_all_running : bool, optional
            If True and space_names is None, stops all spaces with running apps.
            Default is False for safety.
        wait_for_deletion : bool, optional
            Whether to wait for each space's apps to be fully deleted. Default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame with stop results for each space.
        """
        logger.info(f"Starting bulk stop for domain: {domain_id}")

        # Get spaces to stop
        if space_names:
            spaces_to_stop = space_names
        elif stop_all_running:
            # Get all running apps and extract unique space names
            apps_df = self.list_apps(domain_id)

            if space_type_filter:
                # Filter by app type which corresponds to space type
                app_type_map = {
                    "jupyterlab": "JupyterLab",
                    "codeeditor": "CodeEditor",
                }
                filter_app_type = app_type_map.get(space_type_filter.lower(), space_type_filter)
                apps_df = apps_df[apps_df["app_type"] == filter_app_type]

            running_statuses = ["InService", "Pending"]
            running_apps = apps_df[apps_df["status"].isin(running_statuses)]

            # Get unique space names (excluding user profile apps)
            spaces_to_stop = running_apps[running_apps["space_name"].notna()]["space_name"].unique().tolist()
        else:
            logger.error(
                "Please specify space_names or set stop_all_running=True. "
                "This is a safety measure to prevent accidental bulk stops."
            )
            return pd.DataFrame()

        if not spaces_to_stop:
            logger.info("No spaces to stop.")
            return pd.DataFrame()

        logger.info(f"Will stop {len(spaces_to_stop)} space(s): {spaces_to_stop}")

        results = []
        for space_name in spaces_to_stop:
            result = self.stop_space(
                domain_id=domain_id,
                space_name=space_name,
                wait_for_deletion=wait_for_deletion,
            )
            results.append(result)

        success_count = sum(1 for r in results if r["success"])
        logger.info(f"Bulk stop complete. Successfully stopped {success_count}/{len(results)} spaces.")

        return pd.DataFrame(results)

    def list_running_spaces(
        self,
        domain_id: str,
        space_type_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        List only spaces that have running apps.

        Parameters
        ----------
        domain_id : str
            The ID of the SageMaker Studio domain.
        space_type_filter : str, optional
            Filter by space type.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only spaces with running apps.
        """
        logger.info(f"Listing running spaces in domain: {domain_id}")

        # Get all apps
        apps_df = self.list_apps(domain_id)

        if apps_df.empty:
            return pd.DataFrame()

        # Filter for running apps in spaces (not user profiles)
        running_statuses = ["InService", "Pending"]
        running_space_apps = apps_df[
            (apps_df["status"].isin(running_statuses)) &
            (apps_df["space_name"].notna())
        ]

        if running_space_apps.empty:
            return pd.DataFrame()

        # Get unique space names
        running_space_names = running_space_apps["space_name"].unique().tolist()

        # Get full space details
        all_spaces_df = self.list_spaces(domain_id, space_type_filter)

        if all_spaces_df.empty:
            return pd.DataFrame()

        # Filter to only running spaces
        running_spaces_df = all_spaces_df[all_spaces_df["space_name"].isin(running_space_names)]

        # Add running app info
        app_info = []
        for space_name in running_spaces_df["space_name"]:
            space_apps = running_space_apps[running_space_apps["space_name"] == space_name]
            app_types = space_apps["app_type"].tolist()
            app_info.append(", ".join(app_types))

        running_spaces_df = running_spaces_df.copy()
        running_spaces_df["running_app_types"] = app_info

        logger.info(f"Found {len(running_spaces_df)} running space(s).")
        return running_spaces_df


def format_spaces_to_dataframe(spaces_response: list) -> pd.DataFrame:
    """
    Convert a list of space information to a formatted DataFrame.

    Parameters
    ----------
    spaces_response : list
        List of space dictionaries.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame.
    """
    return pd.DataFrame(spaces_response)


if __name__ == "__main__":

    start_time = time.time()

    logger.info("Loading config for SageMaker resource management...")

    config_path = project_root_directory / "config/sagemaker_resource_manager.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    region_name = config.get("region_name")
    domain_id = config["domain_id"]
    space_type_filter = config.get("space_type_filter")
    action = config.get("action", "list")  # list, tag, list_with_tags

    logger.info("Config loaded.")
    logger.info(f"Action: {action}")
    logger.info(f"Domain ID: {domain_id}")

    # Initialize the manager
    manager = SageMakerSpaceManager(region_name=region_name)

    if action == "list_domains":
        # List all domains
        domains_df = manager.list_domains()
        print("\n=== SageMaker Studio Domains ===")
        print(domains_df.to_string(index=False))

    elif action == "list":
        # List spaces in a domain
        spaces_df = manager.list_spaces(domain_id, space_type_filter)
        # spaces_df.sort_values(by="space_name", inplace=True)

        spaces_df_print = spaces_df.drop(columns=["space_arn"])
        print("\n=== SageMaker Spaces ===")
        print(spaces_df_print.to_string(index=False))

        # Export to CSV if configured
        output_filepath = config.get("output_filepath")
        if output_filepath:
            output_path = project_root_directory / output_filepath
            output_path.parent.mkdir(parents=True, exist_ok=True)
            spaces_df.to_csv(output_path, index=False)
            logger.info(f"Exported spaces list to: {output_path}")

    elif action == "list_apps":
        # List running apps
        apps_df = manager.list_apps(domain_id)
        print("\n=== SageMaker Apps (Running Instances) ===")
        print(apps_df.to_string(index=False))

    elif action == "list_with_tags":
        # List spaces with their tags
        spaces_df = manager.list_spaces_with_tags(domain_id, space_type_filter)
        
        spaces_df_print = spaces_df.drop(columns=["space_arn"])
        print("\n=== SageMaker Spaces ===")
        print(spaces_df_print.to_string(index=False))

        # Export to CSV if configured
        output_filepath = config.get("output_filepath", False)
        if output_filepath:
            output_path = project_root_directory / output_filepath
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert tags dict to string for CSV export
            spaces_df_export = spaces_df.copy()
            spaces_df_export["tags"] = spaces_df_export["tags"].apply(str)
            spaces_df_export.to_csv(output_path, index=False)
            logger.info(f"Exported spaces list to: {output_path}")

    elif action == "tag":
        # Add/update tags on spaces
        tags_to_apply = config.get("tags", {})
        space_names = config.get("space_names")  # Optional: specific spaces to tag

        if not tags_to_apply:
            logger.error("No tags specified in config. Please add 'tags' to the config file.")
        else:
            results_df = manager.bulk_tag_spaces(
                domain_id=domain_id,
                tags=tags_to_apply,
                space_type_filter=space_type_filter,
                space_names=space_names,
            )
            print("\n=== Tagging Results ===")
            print(results_df.to_string(index=False))

    elif action == "tag_single":
        # Tag a single space
        space_name = config.get("space_name")
        tags_to_apply = config.get("tags", {})

        if not space_name or not tags_to_apply:
            logger.error("Please specify 'space_name' and 'tags' in the config file.")
        else:
            space_arn = manager.get_space_arn(domain_id, space_name)
            if space_arn:
                success = manager.add_or_update_tags(space_arn, tags_to_apply)
                print(f"\nTagging {'successful' if success else 'failed'} for space: {space_name}")
            else:
                logger.error(f"Could not find ARN for space: {space_name}")

    elif action == "get_tags":
        # Get tags for a specific space
        space_name = config.get("space_name")

        if not space_name:
            logger.error("Please specify 'space_name' in the config file.")
        else:
            space_arn = manager.get_space_arn(domain_id, space_name)
            if space_arn:
                tags = manager.get_resource_tags(space_arn)
                print(f"\n=== Tags for {space_name} ===")
                for key, value in tags.items():
                    print(f"  {key}: {value}")
            else:
                logger.error(f"Could not find ARN for space: {space_name}")

    elif action == "list_running":
        # List only spaces with running apps
        running_df = manager.list_running_spaces(domain_id, space_type_filter)
        print("\n=== Running SageMaker Spaces ===")
        if running_df.empty:
            print("No running spaces found.")
        else:
            print(running_df.to_string(index=False))

    elif action == "stop":
        # Stop a single space
        space_name = config.get("space_name")
        wait_for_deletion = config.get("wait_for_deletion", False)

        if not space_name:
            logger.error("Please specify 'space_name' in the config file.")
        else:
            result = manager.stop_space(
                domain_id=domain_id,
                space_name=space_name,
                wait_for_deletion=wait_for_deletion,
            )
            print(f"\n=== Stop Result for {space_name} ===")
            print(f"  Success: {result['success']}")
            print(f"  Apps stopped: {result['apps_stopped']}")
            if result.get("stopped_apps"):
                print("  Stopped apps:")
                for app in result["stopped_apps"]:
                    print(f"    - {app['app_name']} ({app['app_type']})")
            if result.get("failed_apps"):
                print("  Failed apps:")
                for app in result["failed_apps"]:
                    print(f"    - {app['app_name']} ({app['app_type']})")

    elif action == "stop_bulk":
        # Stop multiple spaces in bulk
        space_names = config.get("space_names")
        stop_all_running = config.get("stop_all_running", False)
        wait_for_deletion = config.get("wait_for_deletion", False)

        if not space_names and not stop_all_running:
            logger.error(
                "Please specify 'space_names' or set 'stop_all_running: true' in the config file. "
                "This is a safety measure to prevent accidental bulk stops."
            )
        else:
            results_df = manager.bulk_stop_spaces(
                domain_id=domain_id,
                space_names=space_names,
                space_type_filter=space_type_filter,
                stop_all_running=stop_all_running,
                wait_for_deletion=wait_for_deletion,
            )
            print("\n=== Bulk Stop Results ===")
            if results_df.empty:
                print("No spaces were stopped.")
            else:
                # Display summary
                for _, row in results_df.iterrows():
                    status = "✓" if row["success"] else "✗"
                    print(f"  {status} {row['space_name']}: {row['apps_stopped']} app(s) stopped")

    else:
        logger.error(
            f"Unknown action: {action}. Valid actions: "
            "list_domains, list, list_apps, list_with_tags, list_running, "
            "tag, tag_single, get_tags, stop, stop_bulk"
        )

    end_time = time.time()

    logger.info("Finished.")
    logger.info(f"Time used = {end_time - start_time:.2f} seconds.")

