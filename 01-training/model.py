import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from .config import MLFLOW_TRACKING_URI
from prefect import task
from sklearn.metrics import root_mean_squared_error
from .metrics import smape, mase, smape_lgb_metric


@task(name="Train_model", log_prints=True)
def train_model(X_train, y_train, X_valid, y_valid):
    """
    Train LightGBM model on training data.

    Args:
        X_train (pd.DataFrame): Features for training
        y_train (pd.Series): Target values for training
        X_valid (pd.DataFrame): Features for validation
        y_valid (pd.Series): Target values for validation

    Returns:
        lgb.Booster: Trained LightGBM model
    """
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("m5_forecasting")


    with mlflow.start_run():
        params = {
            'objective': 'regression',
            'metric': 'None', #'rmse'
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 128,
            'verbose': -1
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            feval=smape_lgb_metric,
            num_boost_round=500,
            valid_names=['train', 'valid'],
            # early_stopping_rounds=50
        )

        y_pred = model.predict(X_valid)
        smape_val = smape(y_valid.values, y_pred)
        mase_val = mase(y_valid.values, y_pred, y_valid.values[:-1])  # naive lag-1

        rmse = root_mean_squared_error(y_valid, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("sMAPE", smape_val)
        mlflow.log_metric("MASE", mase_val)
        print(f"RMSE: {rmse}")

        print("Saving model...")
        mlflow.log_params(params)
        mlflow.lightgbm.log_model(model, artifact_path="model")

        return model
