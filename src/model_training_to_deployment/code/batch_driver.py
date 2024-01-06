import os
import json
import mlflow
import pandas as pd
from typing import List, Any, Union


def Label_encode(data):
    # Map categorical features with string values to numerical values
    data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
    data['Vehicle_Age'] = data['Vehicle_Age'].replace({'< 1 Year': 1, '1-2 Year': 2, '> 2 Years': 3})
    data['Vehicle_Damage'] = data['Vehicle_Damage'].replace({'No': 0, 'Yes': 1})

    # Change the channels and region columns data type to string
    data[["Policy_Sales_Channel", "Region_Code"]] = data[["Policy_Sales_Channel", "Region_Code"]].astype(str)

    # Load the save mapping from the model training to apply to the inference data
    with open('Region_channel_mapping.json', 'r') as file:
        load_combined_mapping = json.load(file)
    # Extract individual mappings
    policy_channel_mapping = load_combined_mapping["policy_channel_mapping"]
    region_mapping = load_combined_mapping["region_mapping"]

    # Apply the mapping to data columns Policy_Sales_Channel and Region_Code
    data["Policy_Sales_Channel"] = data["Policy_Sales_Channel"].map(policy_channel_mapping)
    data["Region_Code"] = data["Region_Code"].map(region_mapping)

    return data


def init():
    global model

    # get the path to the registered model file and load it
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "INPUT_model")

    # Load the model, it's input types and output names
    model = mlflow.pyfunc.load_model(model_path)
    print("Model loaded successfully")


def run(mini_batch: List[str]) -> Union[List[Any], pd.DataFrame]:
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")
    resultList = []

    for file_path in mini_batch:
        data = pd.read_csv(file_path)

        # Preprocess the input data
        encoded_data = Label_encode(data)

        # Make predictions using the model
        pred = model.predict(encoded_data)

        # Make a data frame of the prediction scores
        df = pd.DataFrame(pred, columns=["predictions"])
        df["file"] = os.path.basename(file_path)
        df = pd.concat([data, df], axis=1)
        resultList.extend([df.columns.tolist()] + df.values.tolist())

    return pd.DataFrame(resultList)
