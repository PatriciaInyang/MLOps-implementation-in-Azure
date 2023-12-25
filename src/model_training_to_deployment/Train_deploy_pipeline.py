"""
This Python script is the code for a pipeline to automate the process of training and
registering the trained model in Azureml workspace.
The main function first connects to the workspaace, then registers the training data to
the workspace for versioning and tracking.

The pipeline is created using the Azuremel python sdk v2 components.

The components are loaded from configured yaml file that reference
the train.py and model_reg.py scripts.
"""
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input
from azure.ai.ml.entities import Data
import datetime


# Register the training data as a data asset in Azureml Workspace
def Reg_training_data(ml_client, my_path='train.csv'):
    Label = datetime.datetime.now().strftime("%m%d%H%M")
    my_data = Data(path=my_path,
                   type=AssetTypes.URI_FILE,
                   description="Raw Cross-sell insurance dataset",
                   name="data",
                   version=Label
                   )
    ml_client.data.create_or_update(my_data)
    return Label


def main():
    # Connect to the workspace
    credential = DefaultAzureCredential()
    # credential.get_token("https://management.azure.com/.default")
    ml_client = MLClient.from_config(credential=credential)

    # Call the Data registeration function
    Label = Reg_training_data(ml_client, my_path='train.csv')

    # Create components for the model training and model_reg steps
    parent_dir = "."
    train_model_component = load_component(source=parent_dir + "/train.yml")
    validate_model_component = load_component(source=parent_dir + "/validate.yml")
    reg_model_component = load_component(source=parent_dir + "/model_reg.yml")
    deploy_model_component = load_component(source=parent_dir + "/deploy_model.yml")

    # Create the pipeline using the @pipeline function to automate the training and model registration components
    @pipeline()
    def model_train_to_deployment(pipeline_job_input):
        # 1st step - Training and evaluate model
        train_model = train_model_component(input_data=pipeline_job_input)

        # 2nd step - Validate criteria to check if the model should be deployed
        validate_model = validate_model_component(metrics_file=train_model.outputs.metrics_file)

        # 3rd step - Register model and version it
        register_model = reg_model_component(model=train_model.outputs.model,
                                             metrics_file=train_model.outputs.metrics_file,
                                             validation_result=validate_model.outputs.validation_result,
                                             )
        # 4th step - Deploy model to batch-insurance endpoint and make it the default deployment
        deploy_model = deploy_model_component(model_reg_output=register_model.outputs.model_reg_output)

        return {"automated_model_training_to_deployment_pipeline": deploy_model.outputs.model_deploy_output}

    # Define input data  using the data asset registered in the workspace
    pipeline_job_input = Input(type=AssetTypes.URI_FILE, path=f"azureml:data:{Label}")

    # Call up the pipeline function
    pipeline_job = model_train_to_deployment(pipeline_job_input)
    pipeline_job.settings.default_compute = "VM1"

    # Submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(pipeline_job, experiment_name="Training_to_deployment")


if __name__ == "__main__":
    main()
