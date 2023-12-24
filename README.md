# Implementation of MLOps in Microsoft Azure - for an Insurance Use case. 

## Introduction
MLOps involves the use of DevOps principles to automate the end-to-end process of 
Machine learning model development, model deployment to production, continuous monitoring and retraining.
The goal this project is to successfully deploy a ML model to production and monitor its performance.

The sample data is an open sourced Health insurance cross sell prediction data.
The data is used to build a model which predicts whether the policyholders (customers) of Health insurance
from past year will also be interested in Vehicle Insurance provided by the company. 

# Pre-requisites to run the code
You need an Azure subscription and create an Azure machine learining workspace, python sdk version 2 is mostly used.
configuration file needed in the project directory when the code is ran locally, in order to conect to the AML workspace.

## Model development
The experimentation phase was initially carried out in a notebook for building a ML classification model.
- Exploratory Data analysis 
- Data cleaning and preprocessing
- Experimenting with various data methods of handling data imbalance.
- Model training and evaluation was done on four classification model.
- The best performing model based on the highest Recall is selected and saved in a .pkl file

## Preparation for deployment
In line with DevOps best practices, the notebook was refactored into production ready scripts contained in the src directory.
- **train.py:** training script the Gradient boosting classifier model while using MLflow for autologging and tracking metrics .
- **model_reg.py:**  script for registering the trained model in Azureml workspace.
- **Train_reg_pipeline.py:** script for creating a pipeline to automate the model training and registration component using the train.yml and model_reg.yml files.

- The conda.yml file specifies the virtual environment dependencies required to execute the code for this project.
  This configuration has been utilized to register and build a Docker container image within the Azure ML workspace.
- Github workflow is used to run flake8 linting and standardisation of codes whenever a pull request to created.
- All pipeline jobs, metrics, model registry, environment registery can be viewed in Azureml studio workspace.

## Batch Deployment and Inferencing
The directory "/src/deployment" contains the files needed to deploy the model to production. 
- "code" folder contains the custom batch scoring script named "batch_driver.py" which read in the input data, carryout a label encoding preprocessing step, and passed the transformed data for prediction to the model.
- "inf_data" is the folder that contains mini_btach files with input data that is fed to the model.
- Create_batchendpoint is the script for configuring and setting up and Azureml managed batch endpoint.
- deployment.py is the code for deploying a model to the endpoint.
- score_job.py contains code for sumbitting a scoring job.

## RAI dashboard for model explainability and interpretability
The directory "/src/RAI" contains the script which constructs the RAI decision pipeline and submits a pipeline job for the RAI dashboard to be created in the workplace when a new model is trained and registered.








