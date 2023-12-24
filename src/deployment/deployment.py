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


# Get the latest version of the model already registered in azureml workspace
def get_model(model_name):
    latest_model_version = max([int(m.version) for m in ml_client.models.list(name=model_name)])

    # picking the model to deploy. Here we use the latest version of our registered model
    model = ml_client.models.get(name=model_name, version=latest_model_version)
    return model


class DeploymentNameGenerator:
    def __init__(self):
        self.counter = 0

    def get_next_name(self, base_name="classifier-insurance"):
        self.counter += 1
        deployment_name = f"{base_name}{self.counter}"
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
    parser.add_argument("--model_name", type=str, default="Vehicle_insur_model", help="name of registered model")
    parser.add_argument("--description", type=str,
                        default="Batch processing deployment  for classifying insurance customers",
                        help="deployment description"
                        )
    args = parser.parse_args()

    # define deployment parameters
    name_generator = DeploymentNameGenerator()

    # Define deployment parameters
    deployment_name = name_generator.get_next_name()
    description = args.description

    # Get the registered model from the workspace
    model = get_model(args.model_name)

    # Get the registered environment from the workspace
    env = ml_client.environments.get(name="Aml-env", version="7")

    new_batch_deployment = create_batch_deployment(deployment_name, description, model, env, ml_client)

    default_deployment = set_default_deployment("batch-insurance", new_batch_deployment)

    print(f"New model {new_batch_deployment.name} has been deployed to the batch endpoint ")
