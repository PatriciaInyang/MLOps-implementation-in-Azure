"""
This Python script is designed to automate the process of training and
evaluating four classification models to predict customers interested
in vehicle insurance. The model with the highest recall will be selected as
the best model. This is crucial when dealing with imbalanced datasets.
Key Steps in the Script:
1) Data Loading: loading the training data from a specified CSV file.
2) Data Preprocessing: label encoding some categorical features and
   group Policy channels nd region codes with low frequecy.
3) Data Splitting: function splits the data into training and testing sets,
   considering data stratification to maintain class distribution.
4) Checks for data imbalance and performs random undersampling if required.
5) Models_Training_evaluation: defines a set of machine learning models,
   Logistic Regression, Random Forest, Gradient Boosting, k-Nearest Neighbors.
   * It then trains each model on the training data.
   * Evaluates their performance on the test data.
6) Save evaluation metrics to a file.
7) Output ROC curves plot and confusion matrices to output directory.
8) Save_model_with_highest_recall: The script saves the best-performing model
The saved model can be later used for making predictions on new data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
import random
import mlflow
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

seed = 40
random.seed(seed)
np.random.seed(seed)


def Load_data(file_path):

    # Load the training data
    data = pd.read_csv(file_path, index_col=0)
    return data


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

    # Create mappings based on frequency threshold
    policy_channel_mapping = {}
    for channel, count in policy_channel_counts.items():
        if count >= threshold:
            policy_channel_mapping[channel] = str(channel)
        else:
            policy_channel_mapping[channel] = "other"

    region_mapping = {}
    for region, count in region_counts.items():
        if count >= threshold:
            region_mapping[region] = str(region)
        else:
            region_mapping[region] = "other"

    # Apply the mapping to data columns Policy_Sales_Channel and Region_Code
    data["Policy_Sales_Channel"] = data["Policy_Sales_Channel"].map(policy_channel_mapping)
    data["Region_Code"] = data["Region_Code"].map(region_mapping)

    # Combine both mappings into a dictionary
    combined_mapping = {"policy_channel_mapping": policy_channel_mapping, "region_mapping": region_mapping}

    # Save the combined mapping to a file to be used in the scoring pipeline directory
    folder_path = "code"
    file_path = os.path.join(folder_path, 'Region_channel_mapping.json')
    with open(file_path, 'w') as file:
        json.dump(combined_mapping, file)

    return data


def Split_data(data, test_data_path):
    # Split the data into features and target variable
    X = data.drop('Response', axis=1)
    y = data['Response']

    # Create an instance of StratifiedShuffleSplit
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

    # Use stratified split method to generate the train and test indices
    for train_idx, test_idx in stratified_split.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Return X_train, X_test, y_train, and y_test
    print("Training data shape without resampling is ", X_train.shape, X_test.shape)

    # get current your script directory
    current_script_directory = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to RAI folder
    RAI_folder = os.path.abspath(os.path.join(current_script_directory, "..", "RAI"))

    # Combine the test data back into one file
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save the test dataset to CSV
    test_data.to_csv(os.path.join(RAI_folder, test_data_path), index=False)

    print(f"Test data saved to: {os.path.join(RAI_folder, test_data_path)}")
    return X_train, X_test, y_train, y_test


def Balance_data(X_train, y_train, train_data_path, sampling_strategy=1, random_state=seed):
    # Balance the data using undersampling
    under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_undersampled, y_train_undersampled = under_sampler.fit_resample(X_train, y_train)

    print("Training data shape after resampling is ", X_train_undersampled.shape, y_train_undersampled.shape)

    # Combine the balanced train data back in one file
    train_data = pd.concat([X_train_undersampled,  y_train_undersampled], axis=1)

    # get current your script directory
    current_script_directory = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to RAI folder
    RAI_folder = os.path.abspath(os.path.join(current_script_directory, "..", "RAI"))

    # Save the balanced training dataset to CSV
    train_data.to_csv(os.path.join(RAI_folder, train_data_path), index=False)

    print(f"Balanced train data saved to: {os.path.join(RAI_folder, train_data_path)}")

    return X_train_undersampled, y_train_undersampled


def Train_model(X_train, y_train, model):
    categorical_columns = ["Region_Code", "Policy_Sales_Channel"]
    onehot_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Numerical columns for standardization
    numeric_columns = ["Age", "Annual_Premium", "Vintage"]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # Create a single ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', onehot_transformer, categorical_columns)])

    # Feature selection
    feature_selector = SelectKBest(score_func=mutual_info_classif, k=36)

    full_pipeline = Pipeline([('preprocessor', preprocessor), ('feature_selector', feature_selector), ('model', model)])

    # Fit the pipeline on your training data and use it for prediction
    full_pipeline.fit(X_train, y_train)

    return full_pipeline


def Evaluate_model(full_pipeline, X_test, y_test, model_name):
    # Predict using the trained model
    y_pred = full_pipeline.predict(X_test)

    # Evaluate the model and calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Calculate ROC curve and AUC score
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    # Store the metrics in the DataFrame
    evaluation_metrics = {'Model': model_name, 'Accuracy': accuracy,
                          'Precision': precision,
                          'Recall': recall, 'F1 Score': f1,
                          'AUC Score': auc_score}

    # Calculate confusion matrix and display it
    confusion = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(
        confusion,
        display_labels=['Not interested', 'Interested']
    )
    # Log metrics to MLflow
    mlflow.log_metric("testing_accuracy_score", accuracy)
    mlflow.log_metric("testing_f1_score", f1)
    mlflow.log_metric("testing_precision_score", precision)
    mlflow.log_metric("testing_recall_score", recall)
    mlflow.log_metric("testing_roc_auc", auc_score)

    return evaluation_metrics, cm_display, fpr, tpr, auc_score


def Models_Training_evaluation(X_train, X_test, y_train, y_test, models):

    # Create an empty dictionary to store trained models
    trained_models = {}

    # Create an empty list to store ROC curve data
    all_evaluation_metrics = []

    # Create an empty list to store ROC curve data
    roc_curve_data = {}

    # Initialize a dictionary to store confusion matrix displays
    cm_displays = {}

    for model_name, model in models.items():
        full_pipeline = Train_model(X_train, y_train, model)
        trained_models[model_name] = full_pipeline

        # Evaluate the model and calculate metrics and ROC data
        evaluation_metrics, cm_display, fpr, tpr, auc_score = Evaluate_model(
            full_pipeline, X_test, y_test, model_name
        )

        # Store the confusion matrix display, ROC data and evaluation metric
        all_evaluation_metrics.append(evaluation_metrics)
        roc_curve_data[model_name] = (fpr, tpr, auc_score)
        cm_displays[model_name] = (cm_display)
    return trained_models, all_evaluation_metrics, roc_curve_data, cm_displays


def Save_metrics_tofile(all_evaluation_metrics, metrics_file, output_dir):
    all_evaluation_metrics_df = pd.DataFrame(
        all_evaluation_metrics,
        columns=[
            'Model',
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score',
            'AUC Score'
        ]
    )
    # Save the metrics to a CSV file
    # Create a directory to save ROC and confusion matrix plots
    os.makedirs(output_dir, exist_ok=True)
    all_evaluation_metrics_df.to_csv(os.path.join(output_dir, metrics_file), index=False)
    return all_evaluation_metrics_df


def Saveout_ROC_curve(roc_curve_data, output_dir):

    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, auc_score) in roc_curve_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()


def Saveout_Confusion_Matrices(cm_displays, output_dir):

    # Create a single figure for confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (model_name, cm_display) in enumerate(cm_displays.items()):
        ax = axes[i // 2, i % 2]
        cm_display.plot(cmap='Blues', values_format='.0f',
                        xticks_rotation='horizontal', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()


def Save_model_with_highest_recall(all_evaluation_metrics_df, ML_model_dir, model_file_name, trained_models):
    # Find the model with the highest recall
    highest_recall_model = all_evaluation_metrics_df.loc[all_evaluation_metrics_df['Recall'].idxmax()]
    model_name = highest_recall_model['Model']

    model = trained_models[model_name]

    # Save the model to a joblib file
    joblib.dump(model, model_file_name)

    mlflow.sklearn.save_model(sk_model=model, path=ML_model_dir)

    print(f"The model with the highest recall is {model_name} and has been saved to {ML_model_dir}")


def main():
    """Main function of the script."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, dest="input_data", default="train.csv", help="Path to data file")
    parser.add_argument("--model", type=str, dest="model", default="model_path", help="Path to the save model file")
    parser.add_argument("--test_data", type=str, default="test_data.csv", help="Path to save test_data")
    parser.add_argument("--train_data", type=str, default="train_data.csv", help="Path to save train_data")
    parser.add_argument("--metrics_file", type=str, dest="metrics_file", default="evaluation_metrics.csv",
                        help="Path to training metrics file")
    args = parser.parse_args()

    # 1) Load the data from the specified file
    data = Load_data(args.input_data)
    print('1) The dataset has been imported')

    # 2) Preprocess the data
    data = Label_encode(data)
    print('2) Label encoding done for gender, Vehicle_damage, Vehicle age, Policy_Channels and Region_code')

    X_train, X_test, y_train, y_test = Split_data(data, args.test_data)
    print('3) Train/Test split completed')

    # 3) Check for dataset imbalance and balance it if needed
    imbalance_threshold = 0.2  # Adjust the threshold as needed
    positive_class_ratio = sum(y_train) / len(y_train)
    if positive_class_ratio < imbalance_threshold:
        print('4) Data is an imbalanced data. Use Random undersampling technique to balance data')
        X_train, y_train = Balance_data(X_train, y_train, args.train_data, sampling_strategy=1, random_state=seed)

    # 4) Define the models to be trained and evaluated
    models = {'Logistic Regression': LogisticRegression(random_state=seed)}

    # 5) Train the models and evaluate their performance.
    # Start Logging
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")

        # enable autologging
        mlflow.sklearn.autolog()

        trained_models, evaluation_results, roc_curve_data, cm_displays = Models_Training_evaluation(
            X_train, X_test, y_train, y_test, models
        )
        num_models = len(models)
        print(f'5) {num_models} classification Models have been trained and evaluated on the test data.')

        # 6) Define the file paths for saving metrics and plots
        output_dir = 'script_outputs'
        all_evaluation_metrics_df = Save_metrics_tofile(evaluation_results, args.metrics_file, output_dir)
        print(f'6) See the result of the evaluation metrics in {args.metrics_file}.')

        # 7) Save out the ROC curves and confusion matrices for the models performances.
        Saveout_ROC_curve(roc_curve_data, output_dir)
        Saveout_Confusion_Matrices(cm_displays, output_dir)
        print(f'7) See the {output_dir} for the ROC curves plot and confusion matrices')

        # 8) Save the best model with the highest recall
        # Specify the directory where the model will be saved
        ML_model_path = args.model
        Save_model_with_highest_recall(all_evaluation_metrics_df, ML_model_path, "best_model.pkl", trained_models)


if __name__ == '__main__':
    main()
