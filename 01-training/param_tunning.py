from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import lightgbm as lgb
# import numpy as np
from .metrics import smape, smape_lgb_metric, mase
from sklearn.metrics import root_mean_squared_error

import mlflow
# import mlflow.lightgbm
from prefect import task
from .config import MLFLOW_TRACKING_URI


def objective(params, X_train, y_train, X_valid, y_valid):
    model_params = {
        'objective': 'regression',
        'metric': 'None',
        'verbose': -1,
        **params
    }
        
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(
        model_params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        #feval=lambda preds, data: ('sMAPE', smape(data.get_label(), preds), False),
        feval=smape_lgb_metric,
    )

    y_pred = model.predict(X_valid)
    
    smape_val = smape(y_valid.values, y_pred)
    mase_val = mase(y_valid.values, y_pred, y_valid.values[:-1])  # naive lag-1
    rmse = root_mean_squared_error(y_valid, y_pred)

    return {'smape': smape_val, 'mase': mase_val, 'rmse': rmse, 'status': STATUS_OK}


@task(name="Run_hyperopt", log_prints=True)
def run_hyperopt(X_train, y_train, X_valid, y_valid, max_evals=30):
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'num_leaves': hp.quniform('num_leaves', 31, 256, 1),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 100, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
        'lambda_l1': hp.uniform('lambda_l1', 0.0, 5.0),
        'lambda_l2': hp.uniform('lambda_l2', 0.0, 5.0)
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("m5")


    trials = Trials()

    def wrapped(params):
        # convert floats from quniform to int where needed
        params['num_leaves'] = int(params['num_leaves'])
        params['min_data_in_leaf'] = int(params['min_data_in_leaf'])

        
        with mlflow.start_run():
            mlflow.log_params(params)
            result = objective(params, X_train, y_train, X_valid, y_valid)
        
            mlflow.log_metric("rmse", result['rmse'])
            mlflow.log_metric("sMAPE", result['smape'])
            mlflow.log_metric("MASE", result['mase'])
        return {'loss': result['mase'], 'status': STATUS_OK}

    best = fmin(fn=wrapped, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best

