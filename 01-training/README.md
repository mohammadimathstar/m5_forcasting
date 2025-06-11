# virtual environment

First you need to create the virtual environment with the packages in `requirements.txt`


# Configuring MLflow ()

To start running mlflow:

mlflow server --backend-store-uri sqlite:///mlflow.db


# Configuring Prefect (orchestration)

prefect init

prefect server start

prefect worker start -p mypool3 -t process

prefect deploy pipeline.py:m5_pipeline_with_tuning -n m5deploy -p mypool3

prefect deployment run m5_pipeline_with_tuning/m5deploy


# to see cache in prefect
prefect deployment ls
prefect deployment delete <name>/<deploy_name>
