import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
import time
import datetime
import os
import argparse
from sklearn.model_selection import train_test_split
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input, dsl
from azure.ai.ml import Output
from azure.ai.ml.entities import PipelineJob
from IPython.display import HTML, display

"""
This script configures the Responsible AI dashboard in Azureml workspace.
Explanability and interpretability of any newly trained model can be gotten by
running this script after retraining a new model.
The AzureML RAI allows a maximum limit of 5000 rows data

1. sample_and_save_data: fucntion randomly selects a sample of 5000 customer data
from each of the test and training data split that was used to train the model.

2. save_data_as_parquet: convert the sampled data to tabular and save as parquet

3. create_azureml_resources: gets the credentials for the azureml and connects to the workspace.

4. Reg_test_train_data: Registers the data as MLtable data assets in the azure wokspace.

5. get_latest_model: gets details of the latest version of the model registerd in Azureml

6. get_RAI_inbuilt_components: gets the inbuilt RAI components from azureml.

7. rai_decision_pipeline:constructs the pipeline which returns athe RAI dashboard

8. submit_and_wait: submits the pipeline job that actually creates the RAI dahbaord and its content in azureml workspace

"""


def sample_and_save_data(train_csv_path, test_csv_path):
    # Read data from CSV files
    df1 = pd.read_csv(train_csv_path)
    df2 = pd.read_csv(test_csv_path)

    # Split the data into train and test sets
    Init_train, RAI_train_sample = train_test_split(df1, test_size=5000, stratify=df1['Response'], random_state=2)
    Init_test, RAI_test_sample = train_test_split(df2, test_size=5000, stratify=df2['Response'], random_state=2)

    # Create 'train-data' and 'test-data' directories if they don't exist
    os.makedirs('train-data', exist_ok=True)
    os.makedirs('test-data', exist_ok=True)

    # Save the sampled data to CSV files in the respective directories
    RAI_test_sample.to_csv('test-data/RAI_test_sample.csv', index=False)
    RAI_train_sample.to_csv('train-data/RAI_train_sample.csv', index=False)

    return RAI_train_sample, RAI_test_sample


def save_data_as_parquet(train_csv_path, test_csv_path):

    # Convert data to table
    table_training = pa.Table.from_pandas(train_csv_path)
    table_test = pa.Table.from_pandas(test_csv_path)

    # Write tables out to parquet
    pq.write_table(table_training, "train-data/insurance-training.parquet", version="1.0")
    pq.write_table(table_test, "test-data/insurance-test.parquet", version="1.0")


def create_azureml_resources(registry_name="azureml"):
    credential = DefaultAzureCredential()

    # Get workspace credential token automatically .
    credential.get_token("https://management.azure.com/.default")

    ml_client = MLClient.from_config(credential=credential)

    # Get handle to azureml registry for the RAI built-in components
    ml_client_registry = MLClient(
        credential=credential,
        subscription_id=ml_client.subscription_id,
        resource_group_name=ml_client.resource_group_name,
        registry_name=registry_name,
    )
    return ml_client, ml_client_registry


def Reg_test_train_data(train_data_path, test_data_path, data_version):

    # Assign unique names to register any new test and train as a data asset in the azureml workspace
    input_train_data = "crosssell_train_mltable-" + datetime.datetime.now().strftime("%m%d%H%M%f")
    input_test_data = "crosssell_test_mltable-" + datetime.datetime.now().strftime("%m%d%H%M%f")

    # Register the sampled tabular training data in azureml workspace
    train_data = Data(
        path=train_data_path,
        type=AssetTypes.MLTABLE,
        description="RAI insurance training data",
        name=input_train_data,
        version=data_version,
    )
    ml_client.data.create_or_update(train_data)

    # Register the sampled tabular test data in azureml workspace
    test_data = Data(
        path=test_data_path,
        type=AssetTypes.MLTABLE,
        description="RAI insurance test data",
        name=input_test_data,
        version=data_version,
    )
    ml_client.data.create_or_update(test_data)

    # Configure the registered data as input for the RAI decision pipeline job
    insurance_train_pq = Input(type="mltable",
                               path=f"azureml:{input_train_data}:{data_version}",
                               mode="download",
                               )
    insurance_test_pq = Input(type="mltable",
                              path=f"azureml:{input_test_data}:{data_version}",
                              mode="download",
                              )
    return insurance_train_pq, insurance_test_pq


def get_latest_model(ml_client, model_name):
    latest_model_version = max([int(m.version) for m in ml_client.models.list(name=model_name)])

    # picking the model to deploy. Here we use the latest version of our registered model
    model = ml_client.models.get(name=model_name, version=latest_model_version)

    # Define the names and references variables to the model
    modelname = model.name
    expected_model_id = f"{modelname}:{latest_model_version}"
    azureml_model_id = f"azureml:{expected_model_id}"
    return model, expected_model_id, azureml_model_id


def get_RAI_inbuilt_components(ml_client_registry):

    rai_constructor_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_insight_constructor", label="latest"
    )

    version = rai_constructor_component.version
    print("The current version of RAI built-in components is: " + version)

    rai_erroranalysis_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_erroranalysis", version=version
    )

    rai_explanation_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_explanation", version=version
    )

    rai_gather_component = ml_client_registry.components.get(
        name="microsoft_azureml_rai_tabular_insight_gather", version=version
    )
    return rai_constructor_component, rai_erroranalysis_component, rai_explanation_component, rai_gather_component


def submit_and_wait(ml_client, pipeline_job) -> PipelineJob:
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    assert created_job is not None

    print("Pipeline job can be accessed in the following URL:")
    display(HTML('<a href="{0}">{0}</a>'.format(created_job.studio_url)))

    while created_job.status not in [
        "Completed",
        "Failed",
        "Canceled",
        "NotResponding",
    ]:
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        print("Latest status : {0}".format(created_job.status))
    assert created_job.status == "Completed"
    return created_job


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, dest="test_data", default="test_data.csv", help="Path to test_data")
    parser.add_argument("--train_data", type=str, dest="train_data", default="train_data.csv", help="Path train_data")
    args = parser.parse_args()

    RAI_train_sample, RAI_test_sample = sample_and_save_data(args.train_data, args.test_data)

    # Call the function to read and process the data
    save_data_as_parquet(RAI_train_sample, RAI_test_sample)

    # Call the function to create Azure ML resources
    ml_client, ml_client_registry = create_azureml_resources(registry_name="azureml")

    train_data_path = "train-data/"
    test_data_path = "test-data/"
    data_version = "1"

    insurance_train_pq, insurance_test_pq = Reg_test_train_data(train_data_path, test_data_path, data_version)

    # Define model name
    model_name = "Vehicle_insur_model"

    # Call the function to get the latest model
    model, expected_model_id, azureml_model_id = get_latest_model(ml_client, model_name)

    # Call the function to get the inbuilt RAI components
    (
     rai_constructor_component,
     rai_erroranalysis_component,
     rai_explanation_component,
     rai_gather_component
    ) = get_RAI_inbuilt_components(ml_client_registry)

    # define variables to construct the RAI Insights dashboard
    target_column_name = "Response"
    train_data = insurance_train_pq
    test_data = insurance_test_pq

    # Create the Rai decision pipeline
    @dsl.pipeline(compute="VM1",
                  description="RAI insights on insurance data",
                  experiment_name="RAI_insights_insurance_crossell")
    def rai_decision_pipeline(target_column_name, train_data, test_data):

        # Create the RAI insights dashboard
        create_rai_job = rai_constructor_component(
            title="RAI dashboard insurance",
            task_type="classification",
            model_info=expected_model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=azureml_model_id),
            train_dataset=train_data,
            test_dataset=test_data,
            categorical_column_names='["Region_Code", "Policy_Sales_Channel"]',
            target_column_name=target_column_name,
        )
        create_rai_job.set_limits(timeout=300)

        # Add error analysis
        error_job = rai_erroranalysis_component(rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard)
        error_job.set_limits(timeout=300)

        # Add explanations
        explanation_job = rai_explanation_component(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            comment="add explanation"
            )
        explanation_job.set_limits(timeout=300)

        # Combine insights
        rai_gather_job = rai_gather_component(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_3=error_job.outputs.error_analysis,
            insight_4=explanation_job.outputs.explanation,
        )
        rai_gather_job.set_limits(timeout=300)

        # Set dashboard mode to upload
        rai_gather_job.outputs.dashboard.mode = "upload"

        # Return the dashboard path
        return {"dashboard": rai_gather_job.outputs.dashboard}

    # Call the function to construct the RAI decision pipeline
    insights_pipeline_job = rai_decision_pipeline(target_column_name, train_data, test_data)

    # Workaround to enable the download
    rand_path = str(uuid.uuid4())
    insights_pipeline_job.outputs.dashboard = Output(
        path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
        mode="upload",
        type="uri_folder",
    )

    # Call the function to submit and wait for the pipeline job
    insights_job = submit_and_wait(ml_client, insights_pipeline_job)
