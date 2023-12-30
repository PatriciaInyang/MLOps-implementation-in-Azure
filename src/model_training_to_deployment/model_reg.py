"""
This Python script is to register the trained machine learning model
in the Azureml workspace.
"""
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import argparse
import pandas as pd


# Get the deployment_decision from the validation step
def read_deployment_decision(validation_result_path):
    with open(validation_result_path, 'r') as file:
        deployment_decision = file.read().strip()
        print(deployment_decision)
    return deployment_decision


def get_evaluation_recall(metrics_file_path):
    metrics_df = pd.read_csv(metrics_file_path)
    recall_value = round(metrics_df['Recall'].iloc[0], 3)
    print(recall_value)
    return recall_value


def register_model(model_path, model_name, ml_client, model_description, recall_value):
    # Register the model
    get_model = Model(path=model_path,
                      name=model_name,
                      tags={"Type": "Sklearn"},
                      description=model_description,
                      type=AssetTypes.MLFLOW_MODEL,
                      properties={"Recall": recall_value}
                      )
    ml_client.models.create_or_update(get_model)
    return get_model


if __name__ == "__main__":
    # Load the Azure ML workspace
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_result", type=str, default="script_outputs/validation_result.txt",
                        help="Path to the save model file")
    parser.add_argument("--metrics_file", type=str, dest="metrics_file",
                        default="script_outputs/evaluation_metrics.csv",
                        help="Path to training metrics file")
    parser.add_argument("--model", type=str, dest="model", default="model_path", help="Path to the save model file")
    parser.add_argument("--model_reg_output", type=str, dest="model_reg_output",
                        default="script_outputs/registered_model.txt",
                        help="Path to file with model name")
    args = parser.parse_args()

    # Get the deployment_decision from the validation step
    deployment_decision = read_deployment_decision(args.validation_result)

    #
    if deployment_decision == "Yes,Model should be deployed":
        # Get the newly trained model recall to be included as properties when registering model
        recall_value = get_evaluation_recall(args.metrics_file)

        # define variables for the model registration function
        model_path = args.model
        model_name = "Vehicle_insurance_model"
        model_description = 'Cross-selling model for Vehicle insurance'

        # Register the newly trained model in azureml workspace
        registered_model = register_model(model_path, model_name, ml_client, model_description, recall_value)

        print(f"Model registered: {registered_model.name} - Version {registered_model.version}")

        # Write the model name to the output file
        with open(args.model_reg_output, 'w') as file:
            file.write(registered_model.name)
    else:
        print("skip model registration. Models will only be registered to the workspace if fit for deployment")
        # Write the model name to the output file
        with open(args.model_reg_output, 'w') as file:
            file.write("No model was registered")
