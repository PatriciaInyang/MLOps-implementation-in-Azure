"""
This Python script is to create an Azureml Batch deployment for an MLflow model
to a batch processing endpoint hosting the model in production.
Note that a custom scoring script (batch_driver.py) is used for the inferencing.
"""

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.ai.ml.entities import CodeConfiguration
import argparse
import datetime


# Get the registered model name from the model registeration step
def read_model_name_from_file(file_path):
    with open(file_path, 'r') as file:
        model_name = file.read().strip()
    return model_name


# Get the latest version of the model already registered in azureml workspace
def get_model(model_name):
    latest_model_version = max([int(m.version) for m in ml_client.models.list(name=model_name)])

    # picking the model to deploy. Here we use the latest version of our registered model
    model = ml_client.models.get(name=model_name, version=latest_model_version)
    return model


# Generate a unique name for each model deployment
def DeploymentNameGenerator(base_name="insurance-classifier"):
    deployment_name = base_name + datetime.datetime.now().strftime("%m%d%H%M")
    return deployment_name


def create_batch_deployment(deployment_name, description, model, env, ml_client):
    deployment = BatchDeployment(
        name=deployment_name,
        description=description,
        endpoint_name='batch-insurance',
        model=model,
        code_configuration=CodeConfiguration(
            code="code",
            scoring_script="batch_driver.py"
        ),
        compute="VM1",
        environment=env,
        instance_count=1,
        max_concurrency_per_instance=2,
        mini_batch_size=1,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.csv",
        retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
        logging_level="info"
    )

    ml_client.batch_deployments.begin_create_or_update(deployment).result()

    return deployment


def set_default_deployment(endpoint_name, deployment):
    endpoint = ml_client.batch_endpoints.get(endpoint_name)
    endpoint.defaults = {}

    endpoint.defaults["deployment_name"] = deployment.name

    ml_client.batch_endpoints.begin_create_or_update(endpoint)


if __name__ == "__main__":

    # connect to the Azure ML workspace
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")

    ml_client = MLClient.from_config(credential=credential)

    # Pass the argument for the endpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_reg_output", type=str, default="script_outputs/registered_model.txt",
                        help="model reg name file")
    parser.add_argument("--description", type=str,
                        default="Batch processing deployment  for classifying insurance customers",
                        help="deployment description")
    parser.add_argument("--model_deploy_output", type=str, default="script_outputs/deployment_outcome.txt",
                        help="outcome message file for deployment script")
    args = parser.parse_args()

    # Get the registered model name from the model registration step
    model_name = read_model_name_from_file(args.model_reg_output)

    if model_name != "No model was registered":

        # Define deployment parameters
        deployment_name = DeploymentNameGenerator(base_name="insurance-classifier")
        description = args.description

        # Get the registered model from the workspace
        model = get_model(model_name)

        # Get the registered environment from the workspace
        env = ml_client.environments.get(name="Aml-env", version="7")

        new_batch_deployment = create_batch_deployment(deployment_name, description, model, env, ml_client)

        default_deployment = set_default_deployment("batch-insurance", new_batch_deployment)

        # Write the message for the outcome of model deployment script
        with open(args.model_deploy_output, 'w') as file:
            file.write(f"New model {new_batch_deployment.name} has been deployed to the batch endpoint")
    else:
        # Write the message for the outcome of model deployment script
        with open(args.model_deploy_output, 'w') as file:
            file.write("model deployment was skipped since the default deployment has better performance (Recall)")
