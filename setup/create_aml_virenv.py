# Script: create_environment.py

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment


def create_custom_environment():
    # Authenticate using DefaultAzureCredential
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    # Initialize MLClient
    ml_client = MLClient.from_config(credential=credential)

    # Define custom environment details
    custom_env_name = "Aml-env"
    custom_job_env = Environment(
        name=custom_env_name,
        description="Custom environment for Insurance model job",
        tags={"scikit-learn": "1.0.2"},
        conda_file="conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    # Create or update the custom environment
    custom_env = ml_client.environments.create_or_update(custom_job_env)

    # Print information about the registered environment
    print(f"{custom_env.name} is registered to workspace. Version is {custom_env.version}")


if __name__ == "__main__":
    create_custom_environment()
