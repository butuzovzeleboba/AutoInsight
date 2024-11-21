import mlflow
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from constants import FEATURE_SELECTED, PARAMS_CATBOOST_TUNED

from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_absolute_percentage_error, root_mean_squared_error
)


df_train = pd.read_csv('../data/train_data.csv')
df_test = pd.read_csv('../data/test_data.csv')
df_full = pd.read_csv('../data/ml_data.csv')
df_for_preds = pd.read_csv('../data/cars_data_clear.csv')


X_train = df_train.drop(['price', 'text'], axis=1)[FEATURE_SELECTED]
y_train = df_train['price']

X_test = df_test.drop(['price', 'text'], axis=1)[FEATURE_SELECTED]
y_test = df_test['price']

def eval_metrics(y_test, y_pred):
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return rmse, mae, mape, r2

mlflow.set_tracking_uri('../mlruns')

mlflow.set_experiment('With text 10 classes feature selected, params tuned')

with mlflow.start_run():

    model = CatBoostRegressor(**PARAMS_CATBOOST_TUNED, verbose=500)
    
    mlflow.log_params(PARAMS_CATBOOST_TUNED)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse, mae, mape, r2 = eval_metrics(y_test, y_pred)
    
    print('Catboost model')
    print(f'  RMSE: {rmse}')
    print(f'  MAE: {mae}')
    print(f'  R2: {r2}')
    print(f'  mape: {mape}')

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('mape', mape)

    # Пример входных данных для логирования
    input_example = X_test.sample(5)

    # Логирование модели с примером входных данных
    mlflow.catboost.log_model(model, 'catboost_base', input_example=input_example)

preds = model.predict(df_train.drop(['price', 'text'], axis=1)[FEATURE_SELECTED])
df_for_preds['price_predicted'] = preds
df_for_preds.to_csv('../data/cars_data_predictons.csv', index=False)
