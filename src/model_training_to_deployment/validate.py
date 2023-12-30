import pandas as pd
import argparse
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


def get_last_model_recall(ml_client):
    latest_model_version = max([int(m.version) for m in ml_client.models.list(name="Vehicle_insurance_model")])

    # picking the model to deploy. Here we use the latest version of our registered model
    model = ml_client.models.get(name="Vehicle_insurance_model", version=latest_model_version)

    # Extract the 'Recall' value from the properties. Assuming a default value of 0.0 if 'Recall' is not present
    deployed_model_recall_score = float(model.properties.get('Recall', 0.0))

    print(f"Registered insurance cross-selling model version {latest_model_version} "
          f"has performance Recall Value: {deployed_model_recall_score}")
    return deployed_model_recall_score


def get_evaluation_recall(metrics_file_path):
    metrics_df = pd.read_csv(metrics_file_path)
    New_model_recall_score = round(metrics_df['Recall'].iloc[0], 3)
    print(New_model_recall_score)
    return New_model_recall_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_file", type=str, dest="metrics_file",
                        default="script_outputs/evaluation_metrics.csv",
                        help="Path to training metrics file")
    parser.add_argument("--validation_result", type=str, default="script_outputs/validation_result.txt",
                        help="Path to the save model file")
    args = parser.parse_args()

    # Connect to the workspace
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Get the recall value of the last registered model which was deployed as default in endpoint
    deployed_model_recall = get_last_model_recall(ml_client)

    # Get the recall value of newly trained model from the training step
    retrained_model_recall_value = get_evaluation_recall(args.metrics_file)

    # Validate criteria to check if the new model should be deployed as the dafault model in production
    if retrained_model_recall_value >= deployed_model_recall:
        # If criteria is met, proceed to register the model in workspace and deploy to batch endpoint
        deployment_decision = "Yes, Model should be deployed"
        # Write the model name to the output file
        with open(args.validation_result, 'w') as file:
            file.write(deployment_decision)
    else:
        # If criteria is not met, do not register the model in workspace nor deploy to batch endpoint
        deployment_decision = "No, Model should not be deployed"
        # Write the model name to the output file
        with open(args.validation_result, 'w') as file:
            file.write(deployment_decision)


if __name__ == "__main__":

    main()
