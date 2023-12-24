import os
import mlflow
import pandas as pd
from typing import List, Any, Union


def Label_encode(data):
    # Map categorical features with string values to numerical values
    data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
    data['Vehicle_Age'] = data['Vehicle_Age'].replace({'< 1 Year': 1, '1-2 Year': 2, '> 2 Years': 3})
    data['Vehicle_Damage'] = data['Vehicle_Damage'].replace({'No': 0, 'Yes': 1})

    # Calculate frequency counts for Policy_Sales_Channel and Region_Code
    policy_channel_counts = data["Policy_Sales_Channel"].value_counts(normalize=True)
    region_counts = data["Region_Code"].value_counts(normalize=True)

    # Define a threshold for frequency (0.01)
    threshold = 0.01

    # Filter and replace values below the threshold with "other" for Policy_Sales_Channel
    data["Policy_Sales_Channel"] = data["Policy_Sales_Channel"].apply(
        lambda x: x if policy_channel_counts[x] >= threshold else "other")

    # Filter and replace values below the threshold with "other" for Region_Code
    data["Region_Code"] = data["Region_Code"].apply(
        lambda x: x if region_counts[x] >= threshold else "other")

    # Change the data type of the columns to string
    data[["Policy_Sales_Channel", "Region_Code"]] = data[["Policy_Sales_Channel", "Region_Code"]].astype(str)

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
