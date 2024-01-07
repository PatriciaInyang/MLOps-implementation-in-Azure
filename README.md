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

## Automated Model training to deployment
In line with DevOps best practices, the notebook was refactored into production ready scripts contained in the src directory.
- **train.py:** training script the Gradient boosting classifier model while using MLflow for autologging and tracking metrics.
- **validate.py:** Modelvalidation script which assesses if the the trained model meets the redeployment criteria
- **model_reg.py:**  script for registering a trained model in Azureml workspace. only Models that pass the validation gets registered.
- **deploy_model.py:** script for deploying model to a batch endpoint. Included in the deployment configuration is a custom batch scoring script with path "code/batch_driver.py"
  which reads in the input data, carryout a label encoding preprocessing step, and passes the transformed data for prediction to the model.
  
- **Train_deploy_pipeline.py:** script for combining the four steps above to runa pipeline job that automate the model training to deployment
  using AML components configured in the train.yml, model_reg.yml, validate.yml, and deploy_model.yml files.

## RAI dashboard for model explainability and interpretability
The directory "/src/RAI" contains the script which constructs the RAI decision pipeline and submits a pipeline job for the RAI dashboard to be created in the workplace when a new model is registered.

## Batch Inferencing
The directory "/src/inference" contains for submiting inference requests to the batch endpoint. 
- "inf_data" is the folder that contains mini_batch files with input data that is fed to the model.
- score_job.py contains code for sumbitting a scoring job.

## GitHub action Workflows
The .github\workflows path automate workflow for executing the contains CI/CD checks and jobs explained below:
- flake8 workflow for linting and standardisation of codes whenever a pull request to created.
- score_prod.yml: for triggering a automated submission of scoring job to the model in production.
- Deploy_train_prod.yml: for automating model retraining and redeployment in production

## setup
This directory contains stand alone scripts or cofiguration files for setting up requires resources in AML workspace
- The conda.yml file specifies the virtual environment dependencies required to execute the code for this project.
  This configuration has been utilized to register and build a Docker container image within the Azure ML workspace.
- Create_batchendpoint is the script for configuring and setting up and Azureml managed batch endpoint.

All pipeline jobs, metrics, model registry, environment registery can be viewed in Azureml studio workspace.

## Flow diagram of how the MLOps was designed and how the pipelines are connected

![MLOps implementation flow daigram](https://github.com/PatriciaInyang/MLOps-implementation-in-Azure/assets/115210334/f13e3bf8-f12a-4e18-be5f-e980a0399fa1)



