import os
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

    # List of most occuring sales regions
    Sales_region_from_training = ['28.0', '3.0', '11.0', '41.0', '33.0', '6.0', '35.0', '50.0',
                                  '15.0', '45.0', '8.0', '36.0', '30.0', '47.0', '48.0', '39.0',
                                  '37.0', '2.0', '29.0', '46.0', '13.0', '18.0', '21.0', '10.0', '14.0']

    # List of most occuring sales channels
    Sales_Channel_from_training = ['26.0', '152.0', '160.0', '124.0', '156.0', '157.0', '122.0', '154.0', '151.0']

    # Replace values not in the Sales_region_from_training list with 'other'
    data['Region_Code'] = data['Region_Code'].apply(lambda x: x if x in Sales_region_from_training else 'other')

    # Replace values not in the Sales_Channel_from_training list with 'other'
    data['Policy_Sales_Channel'] = data['Policy_Sales_Channel'].apply(
                                   lambda x: x if x in Sales_Channel_from_training else 'other')

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
