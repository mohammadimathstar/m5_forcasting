# virtual environment

First you need to create the virtual environment with the packages in `requirements.txt`


# Configuring MLflow ()

To start running mlflow:

mlflow server --backend-store-uri sqlite:///mlflow.db


# Configuring Prefect (orchestration)

prefect init

prefect server start

prefect worker start -p mypool2 -t process

prefect deploy pipeline.py:m5_pipeline -n m5deployment -p mypool2

prefect deployment run m5-pipeline/m5deployment