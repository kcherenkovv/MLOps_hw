import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

#Загрузка предобработанных тестовых данных
df_test_scaled = pd.read_csv('test/df_test_scaled.csv')

#Выделяем признаки и целевую метку
X_test = df_test_scaled.drop('Precipitation', axis=1)
y_test = df_test_scaled['Precipitation']

#Загрузка обученной модели
model = joblib.load('models/linear_regression_model.pkl')

#Предсказание на тестовых данных
y_test_pred = model.predict(X_test)
predict = pd.DataFrame(y_test_pred)

print("--- Metrics:")
print(f"--- MAE - {mean_absolute_error(y_test, predict)}")
print(f"--- MSE - {mean_squared_error(y_test, predict)}")
print(f"--- RMSE - {mean_squared_error(y_test, predict)**0.5}")
print(f"--- r2 - {r2_score(y_test, predict.values, multioutput='variance_weighted')}")