"""
This Python script is to create a new Azureml managed Batch processing endpoint
for hosting the deployment for a model in production
in the Azureml  workspace.
"""

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import BatchEndpoint
import argparse


# create a batch endpoint
def create_batch_endpoint(endpoint_name, description, ml_client):

    endpoint = BatchEndpoint(
        name=endpoint_name,
        description=description
    )

    ml_client.batch_endpoints.begin_create_or_update(endpoint)
    return endpoint


if __name__ == "__main__":

    # connect to the Azure ML workspace
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")

    ml_client = MLClient.from_config(credential=credential)

    # Pass the argument for the endpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint_name", type=str, default="batch-insurance", help="unique endpoint name")
    parser.add_argument("--description", type=str,
                        default="Batch endpoint for classifying insurance customers",
                        help="endpoint description"
                        )
    args = parser.parse_args()

    endpoint_name = args.endpoint_name
    description = args.description

    new_batch_endpoint = create_batch_endpoint(endpoint_name, description, ml_client)

    print(f"New batch endpoint: {new_batch_endpoint.name} - created in AML workspace")
