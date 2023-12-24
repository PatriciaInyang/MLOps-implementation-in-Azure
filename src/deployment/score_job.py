"""
This Python script is for submitting a scoring job to the batch endpoint
which hosts the deployed model in production.
It first registers the input data which can be a folder containing mini batch files
or a path to a file, in Azure as a data set.
"""
# Register the folder containing the input inference data as a data asset in the workspace

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input
import argparse


def reg_data(data_path, dataset_name, ml_client):
    # Create a Data object for the input data
    customer_dataset_unlabeled = Data(
        path=data_path,
        type=AssetTypes.URI_FOLDER,
        description="An unlabeled dataset for vehicle insurance classification",
        name=dataset_name,
    )
    # Register the data in Azure ML workspace
    ml_client.data.create_or_update(customer_dataset_unlabeled)

    # Retrieve information about the registered data
    customer_dataset_unlabeled = ml_client.data.get(
        name=dataset_name, label="latest"
    )
    return customer_dataset_unlabeled


if __name__ == "__main__":
    # connect to the Azure ML workspace
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    ml_client = MLClient.from_config(credential=credential)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./inf_data", help="path to input data")
    parser.add_argument("--score_output", type=str, default="predictions.csv", help="path to output scores")
    args = parser.parse_args()

    # Define dataset name
    dataset_name = "customer-data-folder"

    # Register the input data and retrieve information about the registered data
    Registered_data = reg_data(args.data_path, dataset_name, ml_client)

    # Create an Input object for the scoring job
    input = Input(type=AssetTypes.URI_FOLDER, path=Registered_data.id)

    # Invoke the scoring job on the batch endpoint
    job = ml_client.batch_endpoints.invoke(endpoint_name='batch-insurance', input=input)
    ml_client.jobs.stream(job.name)
    # Refresh the job status
    job = ml_client.jobs.get(job.name)

    # Download the output scores
    ml_client.jobs.download(name=job.name, download_path=".", output_name=args.score_output)

    print("Job completed. Results downloaded as csv file.")
